// Separable box blur (RGBA8Unorm).  Two entry points — one per axis;
// the host code dispatches them alternately and ping-pongs between
// two storage textures for `passes > 1`.  Boundary policy is
// clamp-to-edge so the result matches the CPU `box_blur_rgb`
// reference (the integration test relies on this).
//
// Naive O(radius) reads per pixel — fine for the radii FluxFrame
// actually uses (≤ 21 on the downscaled buffer, ≤ ~5 after
// `blur_downscale=4`).  A workgroup-shared running sum would be
// faster at huge radii but adds workgroup-boundary handling that
// is not worth the complexity here.

struct BlurParams {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
};

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: BlurParams;

@compute @workgroup_size(8, 8, 1)
fn cs_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let r = i32(params.radius);
    let w_last = i32(params.width) - 1;
    let y = i32(gid.y);
    let x_centre = i32(gid.x);
    var acc = vec4<f32>(0.0);
    var n = 0.0;
    for (var dx = -r; dx <= r; dx = dx + 1) {
        let sx = clamp(x_centre + dx, 0, w_last);
        acc = acc + textureLoad(input_tex, vec2<i32>(sx, y), 0);
        n = n + 1.0;
    }
    textureStore(output_tex, vec2<i32>(x_centre, y), acc / n);
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let r = i32(params.radius);
    let h_last = i32(params.height) - 1;
    let x = i32(gid.x);
    let y_centre = i32(gid.y);
    var acc = vec4<f32>(0.0);
    var n = 0.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        let sy = clamp(y_centre + dy, 0, h_last);
        acc = acc + textureLoad(input_tex, vec2<i32>(x, sy), 0);
        n = n + 1.0;
    }
    textureStore(output_tex, vec2<i32>(x, y_centre), acc / n);
}
