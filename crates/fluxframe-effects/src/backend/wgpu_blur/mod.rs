//! GPU blur backend driven by `wgpu` compute shaders.
//!
//! Stage 7 step 3 — full separable box-blur pipeline:
//!
//! * Probe acquires a Vulkan adapter + device, compiles the two
//!   compute pipelines (`cs_blur_h`, `cs_blur_v` from `blur.wgsl`)
//!   and allocates the uniform buffer.  All heavy GPU setup happens
//!   once, in [`WgpuBlurBackend::probe`].
//! * Prepare allocates the per-frame-size GPU resources: two
//!   RGBA8 storage textures (ping-pong), one staging buffer for
//!   readback, and the two bind groups (`A→B`, `B→A`).
//! * Blur uploads RGB→RGBA into texture `A`, dispatches H+V pairs
//!   for the requested number of `passes` (ping-pong, final result
//!   lands back in `A`), copies into the staging buffer, blocks on
//!   `Device::poll(PollType::Wait)` for readback, and strips
//!   RGBA→RGB into `dst`.
//!
//! Design notes (locked in `doc/plan/stage-7-wgpu-blur.md`):
//!
//! * **Sync seam.** All async `wgpu` APIs (adapter/device request,
//!   buffer mapping) are wrapped with [`pollster::block_on`].  No
//!   tokio anywhere in the workspace.
//! * **`Send` only.** `wgpu::Device` / `Queue` etc. are `Send` but
//!   not `Sync`.  Matches the `BlurBackend: Send` bound; the effect
//!   chain runs on one worker thread.
//! * **Boundary mode.** Clamp-to-edge, matching the CPU
//!   `box_blur_rgb` reference so the integration test can compare
//!   outputs byte-for-byte (±1 LSB tolerance for float-vs-integer
//!   rounding).
//! * **Sampled input + storage output.** Read-only storage textures
//!   require a wgpu feature flag on some adapters; binding the input
//!   as a sampled `texture_2d<f32>` and reading with `textureLoad`
//!   stays on the core feature set.

use std::sync::mpsc;

use fluxframe_core::error::EffectError;

use crate::backend::blur::BlurBackend;

/// Human-readable component string used in error reasons so they
/// match the rest of the effect ecosystem ("blur backend (wgpu)").
const COMPONENT: &str = "blur backend (wgpu)";

/// Workgroup tile size in WGSL (`@workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)`).
/// `8` (64 threads) is the conservative default for Intel Xe iGPUs;
/// keep the WGSL literal in sync with this constant.
const WORKGROUP_SIZE: u32 = 8;

/// Size of the uniform buffer in bytes — `BlurParams { width, height,
/// radius, _pad }` = 4 × `u32` = 16 bytes, 16-byte aligned by default.
const UNIFORM_BUFFER_SIZE: u64 = 16;

/// Inline the WGSL source so a stripped binary still carries the
/// shader.  Using `include_str!` (rather than `include_wgsl!`) keeps
/// the file readable in IDEs and avoids the proc-macro round-trip.
const BLUR_SHADER_WGSL: &str = include_str!("blur.wgsl");

/// Box-blur backend backed by `wgpu` compute shaders.
///
/// Construct via [`Self::probe`] — it returns `Err` if the host has
/// no usable Vulkan adapter, which the factory uses as the fall-back
/// trigger to skip wgpu and pick the CPU candidate.  Direct
/// construction is intentionally not provided.
pub struct WgpuBlurBackend {
    /// Stable adapter handle, kept alive for the lifetime of the
    /// backend.  Used by the boot-time info line for diagnostics;
    /// holds no significant cost beyond a refcount bump.
    #[allow(
        dead_code,
        reason = "Adapter is kept alive so probe-time `info!` diagnostics survive on demand;\
                  Stage 7+ may add adapter.get_info() lookups for the metrics reporter."
    )]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Bind-group layout shared by both passes and both directions
    /// (A→B and B→A).  Captured once in `probe`, reused per
    /// `prepare` to build the two concrete bind groups.
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_h: wgpu::ComputePipeline,
    pipeline_v: wgpu::ComputePipeline,
    /// 16-byte uniform buffer holding `(width, height, radius, _pad)`
    /// for the current `blur()` call.  Allocated in `probe`; updated
    /// by `queue.write_buffer` at the top of every `blur`.
    uniform_buf: wgpu::Buffer,
    // ---- Per-prepare state, populated by `prepare(w, h)`: ----
    textures: Option<TextureSet>,
    prepared_dims: (u32, u32),
    /// CPU-side staging for RGB→RGBA padding before `queue.write_texture`.
    /// Reused across `blur()` calls — size matches the prepared
    /// resolution.
    upload_buf: Vec<u8>,
}

/// Per-`prepare` GPU resources.  Reallocated whenever `prepare` is
/// called with new dimensions; kept in an `Option` on the parent so
/// teardown is just `self.textures = None`.
struct TextureSet {
    /// Initial input + final output (ping-pong lands result here).
    /// Held by the bind groups via the views; named here only to
    /// anchor the texture's lifetime to the [`TextureSet`].
    tex_a: wgpu::Texture,
    /// Intermediate "other" texture; same lifetime-anchor role as
    /// [`Self::tex_a`].
    #[allow(
        dead_code,
        reason = "Anchored here so the bind groups' captured views remain valid for the \
                  lifetime of the TextureSet."
    )]
    tex_b: wgpu::Texture,
    /// Views captured upfront so `BindGroup`s can borrow them.
    #[allow(
        dead_code,
        reason = "Held to keep view lifetimes tied to the texture set; the bind groups \
                  capture the views internally."
    )]
    view_a: wgpu::TextureView,
    #[allow(
        dead_code,
        reason = "Same as `view_a` — referenced via the bind groups, kept here to anchor \
                  lifetime."
    )]
    view_b: wgpu::TextureView,
    /// `(input=A, output=B)` — used by H pass on iteration `i` and
    /// then again by the next-iteration H pass if `passes > 1`.
    bind_group_a_to_b: wgpu::BindGroup,
    /// `(input=B, output=A)` — used by V pass.
    bind_group_b_to_a: wgpu::BindGroup,
    /// Readback target.  Size = `staging_bytes_per_row * height`.
    staging_buf: wgpu::Buffer,
    /// Row stride enforced by `copy_texture_to_buffer` — must be a
    /// multiple of `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes).
    staging_bytes_per_row: u32,
}

impl WgpuBlurBackend {
    /// Probe the host for a usable Vulkan adapter and return a fully
    /// initialised backend with compiled pipelines.  See module
    /// docstring for the failure modes and the success-log shape.
    pub fn probe() -> Result<Self, EffectError> {
        let instance = create_instance();
        let adapter = request_adapter(&instance)?;
        let info = adapter.get_info();
        let (device, queue) = request_device(&adapter)?;
        log_adapter(&info);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluxframe blur shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER_WGSL.into()),
        });
        let bind_group_layout = create_bind_group_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluxframe blur pipeline layout"),
            // wgpu 29: each set slot is `Option<&BindGroupLayout>` so
            // shaders can leave individual sets unused.  We use set 0
            // only.
            bind_group_layouts: &[Some(&bind_group_layout)],
            // No `var<immediate>` data — the BlurParams uniform is a
            // regular UBO via bind group binding 2.
            immediate_size: 0,
        });
        let pipeline_h = create_pipeline(&device, &pipeline_layout, &shader_module, "cs_blur_h");
        let pipeline_v = create_pipeline(&device, &pipeline_layout, &shader_module, "cs_blur_v");
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluxframe blur uniform buf"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            adapter,
            device,
            queue,
            bind_group_layout,
            pipeline_h,
            pipeline_v,
            uniform_buf,
            textures: None,
            prepared_dims: (0, 0),
            upload_buf: Vec::new(),
        })
    }
}

fn create_instance() -> wgpu::Instance {
    // Pin Vulkan backend explicitly: on Linux this is the only path
    // we care about, and constraining the search avoids accidentally
    // binding to a software (`llvmpipe`) ICD when a proper hardware
    // ICD is also installed.
    let desc = wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    };
    wgpu::Instance::new(desc)
}

fn request_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, EffectError> {
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        // false = allow iGPU candidates; we want them.
        force_fallback_adapter: false,
    };
    pollster::block_on(instance.request_adapter(&options)).map_err(|e| EffectError::PrepareFailed {
        name: COMPONENT.into(),
        reason: format!("no Vulkan adapter available: {e}"),
    })
}

fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), EffectError> {
    let desc = wgpu::DeviceDescriptor {
        label: Some("fluxframe wgpu blur device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    };
    pollster::block_on(adapter.request_device(&desc)).map_err(|e| EffectError::PrepareFailed {
        name: COMPONENT.into(),
        reason: format!("Vulkan adapter rejected device request: {e}"),
    })
}

fn log_adapter(info: &wgpu::AdapterInfo) {
    tracing::info!(
        backend = "wgpu",
        vendor = info.vendor,
        device_id = info.device,
        device_type = ?info.device_type,
        device_name = %info.name,
        driver = %info.driver,
        driver_info = %info.driver_info,
        "wgpu blur backend probe succeeded",
    );
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fluxframe blur bind group layout"),
        entries: &[
            // 0: sampled input — `texture_2d<f32>` via `textureLoad`.
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 1: write-only storage output.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // 2: BlurParams uniform.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(UNIFORM_BUFFER_SIZE),
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Padding-aware row stride for `copy_texture_to_buffer` destinations:
/// must be a multiple of `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256).
fn padded_bytes_per_row(width: u32) -> u32 {
    let unpadded = width * 4;
    let modulo = unpadded % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    if modulo == 0 {
        unpadded
    } else {
        unpadded + (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - modulo)
    }
}

/// Pad packed RGB into packed RGBA with alpha = 255.  Both buffers
/// MUST have matching pixel counts (`src.len() / 3 == dst.len() / 4`).
fn pad_rgb_to_rgba(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len() % 3, 0);
    debug_assert_eq!(dst.len() % 4, 0);
    debug_assert_eq!(src.len() / 3, dst.len() / 4);
    for (rgb, rgba) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        rgba[0] = rgb[0];
        rgba[1] = rgb[1];
        rgba[2] = rgb[2];
        rgba[3] = 0xFF;
    }
}

/// Strip packed RGBA from a padded staging readback into packed RGB.
/// `src` is `bytes_per_row * height` bytes with possible row
/// padding; `dst` is `width * height * 3` bytes packed.
fn strip_rgba_to_rgb(src: &[u8], dst: &mut [u8], width: u32, height: u32, bytes_per_row: u32) {
    let w = width as usize;
    let h = height as usize;
    let bpr = bytes_per_row as usize;
    debug_assert_eq!(dst.len(), w * h * 3);
    debug_assert!(src.len() >= bpr * h);
    for y in 0..h {
        let row_src = &src[y * bpr..y * bpr + w * 4];
        let row_dst = &mut dst[y * w * 3..(y + 1) * w * 3];
        for (rgba, rgb) in row_src.chunks_exact(4).zip(row_dst.chunks_exact_mut(3)) {
            rgb[0] = rgba[0];
            rgb[1] = rgba[1];
            rgb[2] = rgba[2];
        }
    }
}

/// `(width, height, radius, _pad)` packed little-endian for the
/// `BlurParams` uniform.  Manual byte-conversion avoids dragging in
/// `bytemuck` just for one 16-byte write.
fn pack_uniform(width: u32, height: u32, radius: u32) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&width.to_le_bytes());
    buf[4..8].copy_from_slice(&height.to_le_bytes());
    buf[8..12].copy_from_slice(&radius.to_le_bytes());
    // bytes [12..16] stay zero — `_pad` in the WGSL struct.
    buf
}

impl WgpuBlurBackend {
    fn allocate_textures(&self, width: u32, height: u32) -> TextureSet {
        let texture_desc_for = |label: &'static str| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let tex_a = self.device.create_texture(&texture_desc_for("blur tex A"));
        let tex_b = self.device.create_texture(&texture_desc_for("blur tex B"));
        let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());

        let make_bind_group =
            |label: &'static str, input: &wgpu::TextureView, output: &wgpu::TextureView| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(output),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.uniform_buf.as_entire_binding(),
                        },
                    ],
                })
            };
        let bind_group_a_to_b = make_bind_group("blur bg A->B", &view_a, &view_b);
        let bind_group_b_to_a = make_bind_group("blur bg B->A", &view_b, &view_a);

        let staging_bytes_per_row = padded_bytes_per_row(width);
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur staging buf"),
            size: u64::from(staging_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        TextureSet {
            tex_a,
            tex_b,
            view_a,
            view_b,
            bind_group_a_to_b,
            bind_group_b_to_a,
            staging_buf,
            staging_bytes_per_row,
        }
    }

    fn dispatch_blur(&self, textures: &TextureSet, width: u32, height: u32, passes: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blur encoder"),
            });
        let workgroups_x = width.div_ceil(WORKGROUP_SIZE);
        let workgroups_y = height.div_ceil(WORKGROUP_SIZE);
        for _ in 0..passes {
            // H pass: A → B
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blur H"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_h);
                pass.set_bind_group(0, &textures.bind_group_a_to_b, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            // V pass: B → A
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blur V"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_v);
                pass.set_bind_group(0, &textures.bind_group_b_to_a, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
        }
        // Final result is back in tex_a after each H+V pair, so we
        // always copy from tex_a regardless of the loop count.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &textures.staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(textures.staging_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
    }

    fn read_back(
        &self,
        textures: &TextureSet,
        dst: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let slice = textures.staging_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Channel error means the receiver was dropped before
            // the callback ran — impossible in our sync flow.
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: format!("device poll failed: {e}"),
            })?;
        let map_result = rx.recv().map_err(|e| EffectError::ProcessFailed {
            name: COMPONENT.into(),
            reason: format!("map callback channel closed: {e}"),
        })?;
        map_result.map_err(|e| EffectError::ProcessFailed {
            name: COMPONENT.into(),
            reason: format!("staging buffer map failed: {e}"),
        })?;
        {
            let data = slice.get_mapped_range();
            strip_rgba_to_rgb(&data, dst, width, height, textures.staging_bytes_per_row);
        }
        textures.staging_buf.unmap();
        Ok(())
    }
}

impl BlurBackend for WgpuBlurBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn prepare(&mut self, width: u32, height: u32) -> Result<(), EffectError> {
        if width < 2 || height < 2 {
            return Err(EffectError::PrepareFailed {
                name: COMPONENT.into(),
                reason: format!("frame {width}x{height} too small; blur needs at least 2x2"),
            });
        }
        // Drop the previous texture set first so the GPU can recycle
        // the memory while we allocate the new one.
        self.textures = None;
        let textures = self.allocate_textures(width, height);
        self.upload_buf = vec![0u8; (width as usize) * (height as usize) * 4];
        self.textures = Some(textures);
        self.prepared_dims = (width, height);
        Ok(())
    }

    fn blur(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        radius: u32,
        passes: u32,
    ) -> Result<(), EffectError> {
        if (width, height) != self.prepared_dims {
            return Err(EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: format!(
                    "frame {width}x{height} differs from prepared {}x{}",
                    self.prepared_dims.0, self.prepared_dims.1
                ),
            });
        }
        let expected = (width as usize) * (height as usize) * 3;
        if src.len() != expected || dst.len() != expected {
            return Err(EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: format!(
                    "buffer length mismatch: src={}, dst={}, expected={expected}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        // Passthrough shortcut (matches `box_blur_rgb` behaviour).
        if radius == 0 || passes == 0 {
            dst.copy_from_slice(src);
            return Ok(());
        }

        // Stash a copy of the input dims out of `self.prepared_dims`
        // borrow scope so the `&mut self` borrow inside `dispatch_blur`
        // doesn't fight an active `&self.textures` reference.
        let textures = self
            .textures
            .as_ref()
            .ok_or_else(|| EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: "blur called before prepare".into(),
            })?;

        // 1. Pad RGB→RGBA on the CPU side, into the cached upload buffer.
        pad_rgb_to_rgba(src, &mut self.upload_buf);
        // 2. Update the uniform with current dims + radius.
        self.queue
            .write_buffer(&self.uniform_buf, 0, &pack_uniform(width, height, radius));
        // 3. Upload the RGBA frame to texture A.
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.upload_buf,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // 4. Dispatch the H+V passes (ping-pong) and copy to staging.
        self.dispatch_blur(textures, width, height, passes);
        // 5. Block on the GPU + strip RGBA→RGB into the caller buffer.
        self.read_back(textures, dst, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probe must succeed on this developer machine (Intel Arc iGPU +
    /// Mesa Vulkan ICD).  `#[ignore]` so CI without a GPU does not fail;
    /// run locally with `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires Vulkan ICD on the host (Mesa or vendor driver)"]
    fn probe_succeeds_on_this_machine() {
        let backend = WgpuBlurBackend::probe().expect("Vulkan adapter expected on this host");
        assert_eq!(backend.name(), "wgpu");
    }

    /// Trait-object form is what factories return; smoke-check that
    /// the type compiles in that shape.  Does not actually probe.
    #[test]
    fn type_satisfies_blur_backend_trait_object() {
        fn accepts<T: BlurBackend + Send>(_t: &T) {}
        fn proof(b: &WgpuBlurBackend) {
            accepts(b);
        }
        // Reference both fns so the underscore-stripped versions are
        // not themselves "unused".  Type-only check; no instance of
        // `WgpuBlurBackend` is constructed here.
        let _ = proof as fn(&WgpuBlurBackend);
    }

    #[test]
    fn padded_bytes_per_row_is_aligned() {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        for w in [1u32, 4, 63, 64, 65, 256, 257, 1280, 1920] {
            let padded = padded_bytes_per_row(w);
            assert!(padded >= w * 4);
            assert_eq!(
                padded % align,
                0,
                "row stride {padded} not aligned for w={w}"
            );
        }
    }

    #[test]
    fn pad_then_strip_round_trips() {
        let w = 4u32;
        let h = 3u32;
        let src: Vec<u8> = (0..(w * h * 3) as u8).collect();
        let mut padded_buf = vec![0u8; (w * h * 4) as usize];
        pad_rgb_to_rgba(&src, &mut padded_buf);
        // Emulate the staging layout: this small width has padding.
        let bpr = padded_bytes_per_row(w);
        let mut staging = vec![0u8; (bpr * h) as usize];
        for y in 0..h as usize {
            let row_offset = y * bpr as usize;
            let src_row_offset = y * (w * 4) as usize;
            staging[row_offset..row_offset + (w * 4) as usize]
                .copy_from_slice(&padded_buf[src_row_offset..src_row_offset + (w * 4) as usize]);
        }
        let mut roundtrip = vec![0u8; src.len()];
        strip_rgba_to_rgb(&staging, &mut roundtrip, w, h, bpr);
        assert_eq!(roundtrip, src);
    }

    #[test]
    fn pack_uniform_layout() {
        let buf = pack_uniform(1280, 720, 21);
        assert_eq!(&buf[0..4], &1280u32.to_le_bytes());
        assert_eq!(&buf[4..8], &720u32.to_le_bytes());
        assert_eq!(&buf[8..12], &21u32.to_le_bytes());
        assert_eq!(&buf[12..16], &[0u8; 4]);
    }
}
