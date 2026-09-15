//! Top-level [`fluxframe_core::VideoEffect`] orchestrating the
//! segmented-composite pipeline: segmentation → mask chain → bg/fg
//! chains → alpha composite.
//!
//! The composite owns no parsing logic — it accepts pre-configured
//! sub-effects via [`CompositeEffect::new`]. The TOML-side wiring
//! (parsing `[mask]`/`[background]`/`[foreground]` into chains) lives
//! in [`crate::composite::builder`] which the CLI invokes once at
//! startup.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use fluxframe_core::metrics::StageKey;
use fluxframe_core::plane::{
    FramePlane, MaskEffect, MaskPlane, PlaneEffect, PostEffect, SubchainKind,
};
use fluxframe_core::traits::{RawEffectParams, VideoEffect};

use crate::composite::segmentation::{SegmentationBase, SegmentationOutcome};
use crate::processing::compose::alpha_composite_rgb_in_place;
use crate::processing::resize_mask_bilinear;

/// Scope under which the composite's own stages are reported.
const SCOPE: &str = "composite";

/// Time `f` and publish the elapsed wall-clock under `key` through the
/// frame's telemetry sink (a no-op sink costs one `Instant::now`
/// pair).  The closure receives `ctx` back so the borrow of
/// `ctx.telemetry` after the call does not clash with the stage's own
/// use of the context.
#[inline]
fn timed<R>(ctx: &mut FrameContext, key: StageKey, f: impl FnOnce(&mut FrameContext) -> R) -> R {
    let start = std::time::Instant::now();
    let r = f(ctx);
    ctx.telemetry.record_stage(key, start.elapsed());
    r
}

/// Build a `ProcessFailed` error for this effect with the given
/// `reason`. Kept private so the effect name stays in one place.
fn process_err(reason: impl Into<String>) -> EffectError {
    EffectError::ProcessFailed {
        name: CompositeEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

/// The composite effect glues a [`SegmentationBase`] together with
/// three ordered chains (mask post-processing, background pipeline,
/// foreground pipeline) and performs the final alpha-composite.
///
/// Lifecycle is the standard [`VideoEffect`] one. `configure` is a
/// no-op at this level because each sub-effect is pre-configured by
/// the builder; calling it with non-empty params is rejected so a
/// mistaken TOML cannot silently produce a half-configured composite.
pub struct CompositeEffect {
    segmentation: SegmentationBase,
    mask_chain: Vec<Slot<dyn MaskEffect>>,
    bg_chain: Vec<Slot<dyn PlaneEffect>>,
    fg_chain: Vec<Slot<dyn PlaneEffect>>,
    /// Post-composite mask-aware chain. Each effect sees the already-
    /// blended frame plus a read-only view of the upscaled mask.
    post_chain: Vec<Slot<dyn PostEffect>>,
    /// Mask at frame resolution after the implicit bilinear upscale.
    mask_full: Vec<f32>,
    /// Frame-sized scratch holding the background plane through the
    /// background chain. The foreground chain runs in place on
    /// `frame.data`, so no `fg_plane` scratch is needed.
    bg_plane: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
    /// Snapshot of the `ProcessingContext` from the last successful
    /// `prepare()`. Used by Stage 13's `set_chain` flow to prepare
    /// newly-injected sub-effects against the right resolution.
    /// `None` before the first `prepare()`.
    processing_ctx: Option<ProcessingContext>,
}

/// Typed envelope of the four possible sub-chain payloads handed to
/// [`CompositeEffect::replace_subchain`].
///
/// Each variant matches the corresponding sub-chain slot inside the
/// composite. Variants own the new chain; on success they are swapped
/// in atomically (the old chain is dropped on the swap-out vector).
pub enum SubChainPayload {
    /// Replacement mask post-processing chain.
    Mask(Vec<Slot<dyn MaskEffect>>),
    /// Replacement background plane chain.
    Background(Vec<Slot<dyn PlaneEffect>>),
    /// Replacement foreground plane chain.
    Foreground(Vec<Slot<dyn PlaneEffect>>),
    /// Replacement post-composite chain.
    Post(Vec<Slot<dyn PostEffect>>),
}

/// A configured sub-effect together with its enable flag.
///
/// The flag lives next to the effect, so a sub-chain swap
/// ([`CompositeEffect::replace_subchain`]) replaces effects and flags in
/// one step and the two can never drift apart.
pub struct Slot<E: ?Sized> {
    effect: Box<E>,
    enabled: bool,
}

impl<E: ?Sized> Slot<E> {
    /// Pair a configured `effect` with its initial enable flag.
    #[must_use]
    pub fn new(effect: Box<E>, enabled: bool) -> Self {
        Self { effect, enabled }
    }

    /// The wrapped effect.
    #[must_use]
    pub fn effect(&self) -> &E {
        &self.effect
    }

    /// Whether the effect runs per frame.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl CompositeEffect {
    /// Effect name as registered in the top-level effect registry.
    pub const NAME: &'static str = "composite";

    /// Construct from pre-configured sub-components.
    #[must_use]
    pub fn new(
        segmentation: SegmentationBase,
        mask_chain: Vec<Slot<dyn MaskEffect>>,
        bg_chain: Vec<Slot<dyn PlaneEffect>>,
        fg_chain: Vec<Slot<dyn PlaneEffect>>,
        post_chain: Vec<Slot<dyn PostEffect>>,
    ) -> Self {
        Self {
            segmentation,
            mask_chain,
            bg_chain,
            fg_chain,
            post_chain,
            mask_full: Vec::new(),
            bg_plane: Vec::new(),
            frame_w: 0,
            frame_h: 0,
            processing_ctx: None,
        }
    }

    /// Borrow the snapshot of [`ProcessingContext`] captured by the
    /// last successful `prepare()`. `None` before the first prepare.
    /// Used by Stage 13's `set_chain` flow to prepare newly-injected
    /// sub-effects against the right resolution.
    ///
    /// Crate-private: external callers go through
    /// [`Self::replace_subchain`] which reads the snapshot internally.
    #[must_use]
    #[allow(
        dead_code,
        reason = "kept as crate-private accessor for future tests / debug"
    )]
    pub(crate) fn processing_ctx(&self) -> Option<&ProcessingContext> {
        self.processing_ctx.as_ref()
    }

    /// Borrow the underlying [`SegmentationBase`]. Used by the
    /// Stage 15 supervisor's lifecycle wrapper (`ManagedComposite`
    /// in the CLI crate) to drive engine unload/reload — see
    /// [`SegmentationBase::take_engine`] /
    /// [`SegmentationBase::install_engine`]. Not part of the normal
    /// effect lifecycle; production effects should never need it.
    #[must_use]
    pub fn segmentation_mut(&mut self) -> &mut SegmentationBase {
        &mut self.segmentation
    }

    /// Read-only counterpart to [`Self::segmentation_mut`]. Used by
    /// `ManagedComposite` to query [`SegmentationBase::engine_loaded`]
    /// without taking a mutable borrow across the worker loop.
    #[must_use]
    pub fn segmentation(&self) -> &SegmentationBase {
        &self.segmentation
    }

    /// Replace one of the composite's sub-chains. New effects must
    /// already have been `configure()`d by the caller. This call
    /// drives `prepare()` on each new effect against the stored
    /// [`ProcessingContext`] and swaps the new chain in. On failure
    /// the old chain is left intact.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::PrepareFailed`] when prepare on a new
    /// effect fails, or when the composite has not been prepared yet
    /// (no stored `ProcessingContext`).
    pub fn replace_subchain(&mut self, payload: SubChainPayload) -> Result<(), EffectError> {
        let ctx = self.processing_ctx.clone();
        match payload {
            SubChainPayload::Mask(v) => replace_chain(&mut self.mask_chain, v, ctx.as_ref()),
            SubChainPayload::Background(v) => replace_chain(&mut self.bg_chain, v, ctx.as_ref()),
            SubChainPayload::Foreground(v) => replace_chain(&mut self.fg_chain, v, ctx.as_ref()),
            SubChainPayload::Post(v) => replace_chain(&mut self.post_chain, v, ctx.as_ref()),
        }
    }

    /// Enable or disable every effect named `name` in `section`.
    ///
    /// A disabled effect keeps its configuration and prepared resources
    /// but is skipped per frame. Re-enabling resets its temporal state
    /// first ([`MaskEffect::reset_state`]), so it does not resume from
    /// stale history. Returns whether any flag actually changed.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] when the sub-chain has no
    /// effect named `name`.
    pub fn set_effect_enabled(
        &mut self,
        section: SubchainKind,
        name: &str,
        enabled: bool,
    ) -> Result<bool, EffectError> {
        match section {
            SubchainKind::Mask => set_enabled_in(&mut self.mask_chain, name, enabled, section),
            SubchainKind::Background => set_enabled_in(&mut self.bg_chain, name, enabled, section),
            SubchainKind::Foreground => set_enabled_in(&mut self.fg_chain, name, enabled, section),
            SubchainKind::Post => set_enabled_in(&mut self.post_chain, name, enabled, section),
        }
    }

    /// Names of the disabled effects of `section`, in chain order.
    #[must_use]
    pub fn disabled_effects(&self, section: SubchainKind) -> Vec<&str> {
        fn disabled<E: ?Sized + SubEffectAdapter>(chain: &[Slot<E>]) -> Vec<&str> {
            chain
                .iter()
                .filter(|slot| !slot.enabled)
                .map(|slot| slot.effect.name_str())
                .collect()
        }
        match section {
            SubchainKind::Mask => disabled(&self.mask_chain),
            SubchainKind::Background => disabled(&self.bg_chain),
            SubchainKind::Foreground => disabled(&self.fg_chain),
            SubchainKind::Post => disabled(&self.post_chain),
        }
    }
}

/// Internal sub-chain swap helper. Generic over the four sub-effect
/// traits via [`SubEffectAdapter`] so the same body serves
/// mask/bg/fg/post.
fn replace_chain<E: ?Sized + SubEffectAdapter>(
    target: &mut Vec<Slot<E>>,
    mut new_chain: Vec<Slot<E>>,
    ctx: Option<&ProcessingContext>,
) -> Result<(), EffectError> {
    let Some(ctx) = ctx else {
        return Err(EffectError::PrepareFailed {
            name: CompositeEffect::NAME.to_string(),
            reason: "replace_subchain called before composite was prepared".into(),
        });
    };
    // Prepare new effects first; on failure the old chain stays.
    // Disabled effects are prepared too, so enabling one later is instant.
    for slot in &mut new_chain {
        slot.effect.prepare_mut(ctx)?;
    }
    // Drop old chain (each effect's Drop releases its resources).
    target.clear();
    *target = new_chain;
    Ok(())
}

/// Unified adapter trait that normalises `name()`, `configure()` and
/// `prepare()` over the three sub-effect dyn types
/// (`MaskEffect`/`PlaneEffect`/`PostEffect`). Object-safe; used by
/// both [`replace_chain`] and [`reconfigure_in`] so the same body
/// serves every sub-chain kind.
///
/// `pub(crate)` so other modules inside the crate can drive
/// sub-effects uniformly without re-implementing the dispatch.
pub(crate) trait SubEffectAdapter {
    /// Stable name of the underlying effect.
    fn name_str(&self) -> &str;
    /// Apply user-supplied configuration. Routes to the effect's
    /// `configure()` method.
    fn configure_mut(&mut self, params: RawEffectParams) -> Result<(), EffectError>;
    /// Run `prepare()` against the negotiated context.
    fn prepare_mut(&mut self, ctx: &ProcessingContext) -> Result<(), EffectError>;
    /// Drop temporal state. Routes to the effect's `reset_state()`.
    fn reset_state_mut(&mut self);
}

impl SubEffectAdapter for dyn MaskEffect {
    fn name_str(&self) -> &str {
        self.name()
    }
    fn configure_mut(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        self.configure(params)
    }
    fn prepare_mut(&mut self, ctx: &ProcessingContext) -> Result<(), EffectError> {
        self.prepare(ctx)
    }
    fn reset_state_mut(&mut self) {
        self.reset_state();
    }
}

impl SubEffectAdapter for dyn PlaneEffect {
    fn name_str(&self) -> &str {
        self.name()
    }
    fn configure_mut(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        self.configure(params)
    }
    fn prepare_mut(&mut self, ctx: &ProcessingContext) -> Result<(), EffectError> {
        self.prepare(ctx)
    }
    fn reset_state_mut(&mut self) {
        self.reset_state();
    }
}

impl SubEffectAdapter for dyn PostEffect {
    fn name_str(&self) -> &str {
        self.name()
    }
    fn configure_mut(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        self.configure(params)
    }
    fn prepare_mut(&mut self, ctx: &ProcessingContext) -> Result<(), EffectError> {
        self.prepare(ctx)
    }
    fn reset_state_mut(&mut self) {
        self.reset_state();
    }
}

impl VideoEffect for CompositeEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
        // Sub-effects are configured by the builder before construction;
        // this entry point exists only to satisfy the `VideoEffect`
        // contract.
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        if context.format != PixelFormat::Rgb {
            return Err(EffectError::PrepareFailed {
                name: Self::NAME.to_string(),
                reason: format!("composite requires RGB input, got {:?}", context.format),
            });
        }

        self.segmentation.prepare(context)?;
        // Disabled effects are prepared too, so enabling one is instant.
        for slot in &mut self.mask_chain {
            slot.effect.prepare(context)?;
        }
        for slot in &mut self.bg_chain {
            slot.effect.prepare(context)?;
        }
        for slot in &mut self.fg_chain {
            slot.effect.prepare(context)?;
        }
        for slot in &mut self.post_chain {
            slot.effect.prepare(context)?;
        }

        self.frame_w = context.width;
        self.frame_h = context.height;
        let frame_pixels = (context.width as usize) * (context.height as usize);
        let frame_bytes = frame_pixels * 3;
        self.mask_full = vec![0.0_f32; frame_pixels];
        self.bg_plane = vec![0_u8; frame_bytes];
        self.processing_ctx = Some(context.clone());
        Ok(())
    }

    fn process(
        &mut self,
        frame: &mut VideoFrame,
        ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if frame.format != PixelFormat::Rgb {
            return Err(process_err(format!(
                "unexpected pixel format {:?}",
                frame.format
            )));
        }
        if frame.width != self.frame_w || frame.height != self.frame_h {
            return Err(process_err(format!(
                "frame {}x{} differs from prepared {}x{}",
                frame.width, frame.height, self.frame_w, self.frame_h
            )));
        }

        // 1. Segmentation produces a mask at model resolution.
        let seg_result = timed(ctx, StageKey::new(SCOPE, "seg"), |ctx| {
            self.segmentation.process(frame, ctx)
        });
        let (mask_w, mask_h) = match seg_result {
            SegmentationOutcome::Ok { width, height } => (width, height),
            SegmentationOutcome::Fallback => {
                ctx.fallback_active = true;
                return Ok(());
            }
            SegmentationOutcome::Fatal(err) => return Err(err),
        };

        // 2. Mask chain at model resolution.
        timed(ctx, StageKey::new(SCOPE, "mask_chain"), |ctx| {
            let mut mask_plane = MaskPlane::new(self.segmentation.mask_mut(), mask_w, mask_h);
            for slot in self.mask_chain.iter_mut().filter(|slot| slot.enabled) {
                let effect = &mut slot.effect;
                timed(
                    ctx,
                    StageKey::new(SubchainKind::Mask.as_str(), effect.name()),
                    |ctx| effect.process(&mut mask_plane, ctx),
                )?;
            }
            Ok::<(), EffectError>(())
        })?;

        // 3. Resize mask to frame resolution. The composite owns the
        //    implicit upscale — mask effects never resize.  Timed
        //    separately from the chain: this is the frame-res part.
        timed(ctx, StageKey::new(SCOPE, "mask_resize"), |_| {
            resize_mask_bilinear(
                self.segmentation.mask(),
                mask_w,
                mask_h,
                &mut self.mask_full,
                self.frame_w,
                self.frame_h,
            );
        });

        // 4. Promote the frame buffer to Owned so we can mutate it
        //    directly as the foreground plane.
        frame.data.make_owned();
        let Some(frame_bytes) = frame.data.as_mut_owned() else {
            return Err(process_err(
                "frame buffer not owned after make_owned (FrameBuffer invariant broken)",
            ));
        };

        // 5. Snapshot the original frame into `bg_plane`. This is the
        //    only full-frame copy in the composite hot path.
        timed(ctx, StageKey::new(SCOPE, "bg_copy"), |_| {
            self.bg_plane.copy_from_slice(frame_bytes);
        });

        // 6. Background chain runs on the snapshot.
        timed(ctx, StageKey::new(SCOPE, "bg_chain"), |ctx| {
            let mut background = FramePlane::new(&mut self.bg_plane, self.frame_w, self.frame_h);
            for slot in self.bg_chain.iter_mut().filter(|slot| slot.enabled) {
                let effect = &mut slot.effect;
                timed(
                    ctx,
                    StageKey::new(SubchainKind::Background.as_str(), effect.name()),
                    |ctx| effect.process(&mut background, ctx),
                )?;
            }
            Ok::<(), EffectError>(())
        })?;

        // 7. Foreground chain runs in place on `frame.data`.
        timed(ctx, StageKey::new(SCOPE, "fg_chain"), |ctx| {
            let mut foreground = FramePlane::new(frame_bytes, self.frame_w, self.frame_h);
            for slot in self.fg_chain.iter_mut().filter(|slot| slot.enabled) {
                let effect = &mut slot.effect;
                timed(
                    ctx,
                    StageKey::new(SubchainKind::Foreground.as_str(), effect.name()),
                    |ctx| effect.process(&mut foreground, ctx),
                )?;
            }
            Ok::<(), EffectError>(())
        })?;

        // 8. Composite bg into fg/frame in place using the parallel
        //    helper.
        timed(ctx, StageKey::new(SCOPE, "compose"), |_| {
            alpha_composite_rgb_in_place(frame_bytes, &self.bg_plane, &self.mask_full);
        });

        // 9. Post chain. Each post-effect mutates the composed frame
        //    while reading the upscaled mask. Split-borrow so the
        //    mask buffer can be passed as `&MaskPlane` without
        //    clashing with `&mut self.post_chain`.
        timed(ctx, StageKey::new(SCOPE, "post_chain"), |ctx| {
            let mut composed = FramePlane::new(frame_bytes, self.frame_w, self.frame_h);
            let mask_plane = MaskPlane::new(&mut self.mask_full, self.frame_w, self.frame_h);
            for slot in self.post_chain.iter_mut().filter(|slot| slot.enabled) {
                let effect = &mut slot.effect;
                timed(
                    ctx,
                    StageKey::new(SubchainKind::Post.as_str(), effect.name()),
                    |ctx| effect.process(&mut composed, &mask_plane, ctx),
                )?;
            }
            Ok::<(), EffectError>(())
        })?;

        Ok(())
    }

    fn reconfigure_named_effect(
        &mut self,
        section: SubchainKind,
        name: &str,
        params: RawEffectParams,
    ) -> Result<(), EffectError> {
        match section {
            SubchainKind::Mask => reconfigure_in(self.mask_chain.iter_mut(), name, params, section),
            SubchainKind::Background => {
                reconfigure_in(self.bg_chain.iter_mut(), name, params, section)
            }
            SubchainKind::Foreground => {
                reconfigure_in(self.fg_chain.iter_mut(), name, params, section)
            }
            SubchainKind::Post => reconfigure_in(self.post_chain.iter_mut(), name, params, section),
        }
    }
}

/// Generic helper: walk a sub-chain by name and call `configure` on
/// the matching effect. Returns `InvalidConfig` when no effect in the
/// chain matches `name`. Generic over the sub-effect trait so the
/// same body serves all four sub-chains.
fn reconfigure_in<'a, E, I>(
    iter: I,
    name: &str,
    params: RawEffectParams,
    section: SubchainKind,
) -> Result<(), EffectError>
where
    I: Iterator<Item = &'a mut Slot<E>>,
    E: 'a + ?Sized + SubEffectAdapter,
{
    for slot in iter {
        if slot.effect.name_str() == name {
            return slot.effect.configure_mut(params);
        }
    }
    Err(no_such_effect(name, section))
}

/// Set the flag of every slot named `name`; see
/// [`CompositeEffect::set_effect_enabled`].
fn set_enabled_in<E: ?Sized + SubEffectAdapter>(
    chain: &mut [Slot<E>],
    name: &str,
    enabled: bool,
    section: SubchainKind,
) -> Result<bool, EffectError> {
    let mut found = false;
    let mut changed = false;
    for slot in chain
        .iter_mut()
        .filter(|slot| slot.effect.name_str() == name)
    {
        found = true;
        if slot.enabled != enabled {
            if enabled {
                slot.effect.reset_state_mut();
            }
            slot.enabled = enabled;
            changed = true;
        }
    }
    if found {
        Ok(changed)
    } else {
        Err(no_such_effect(name, section))
    }
}

fn no_such_effect(name: &str, section: SubchainKind) -> EffectError {
    EffectError::InvalidConfig {
        name: CompositeEffect::NAME.to_string(),
        reason: format!("no effect named '{name}' in the '{section}' sub-chain"),
        hint: None,
    }
}

// End-to-end test for `CompositeEffect::process` lives in
// `tests/composite_e2e.rs` (added in Stage 9.6 once the builder is
// in place — it needs the sidecar TOML loader or a test-only
// shortcut into `SegmentationBase`).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::segmentation::SegmentationConfig;

    /// Composite whose background chain holds `names`, all enabled.
    /// Never prepared: flag handling needs no model.
    fn composite_with_background(names: &[&str]) -> CompositeEffect {
        let registry = crate::plane_effects::default_registry();
        let bg = names
            .iter()
            .map(|name| Slot::new(registry.build(name).expect("registered"), true))
            .collect();
        let segmentation = SegmentationBase::new(SegmentationConfig {
            model: "/tmp/unused.onnx".into(),
            model_config: None,
            fallback_threshold: 3,
        });
        CompositeEffect::new(segmentation, Vec::new(), bg, Vec::new(), Vec::new())
    }

    #[test]
    fn set_effect_enabled_toggles_every_slot_with_the_name() {
        let mut c = composite_with_background(&["color_fill", "vignette", "color_fill"]);
        let bg = SubchainKind::Background;

        assert_eq!(
            c.set_effect_enabled(bg, "color_fill", false).ok(),
            Some(true)
        );
        assert_eq!(c.disabled_effects(bg), vec!["color_fill", "color_fill"]);
        assert_eq!(
            c.set_effect_enabled(bg, "color_fill", false).ok(),
            Some(false),
            "repeating the same flag changes nothing"
        );

        assert_eq!(
            c.set_effect_enabled(bg, "color_fill", true).ok(),
            Some(true)
        );
        assert!(c.disabled_effects(bg).is_empty());
    }

    #[test]
    fn set_effect_enabled_rejects_unknown_name_or_wrong_section() {
        let mut c = composite_with_background(&["color_fill"]);
        for (section, name) in [
            (SubchainKind::Background, "blur"),
            (SubchainKind::Foreground, "color_fill"),
        ] {
            let err = c
                .set_effect_enabled(section, name, false)
                .expect_err("no such effect");
            assert!(format!("{err}").contains("no effect named"), "{err}");
        }
        assert!(c.disabled_effects(SubchainKind::Background).is_empty());
    }
}
