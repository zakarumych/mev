
struct VertOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    bary: vec3<f32>,
}

struct Constants {
    angle: f32,
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<uniform> pc: Constants;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertOutput {

    let x = (-0.5 + (f32(index) * 0.5));
    let y = (-(sqrt(3.0) / 6.0) + f32(index == 1u) * sqrt(3.0) / 2.0);

    let a = pc.angle * 6.28318530717958647692528676655900577;

    let ca = cos(a);
    let sa = sin(a);

    let output = VertOutput(
        vec4<f32>((ca * x + sa * y) / f32(pc.width) * f32(pc.height), (ca * y - sa * x), 0.0, 1.0),
        vec3<f32>(f32(index == 0u), f32(index == 1u), f32(index == 2u))
    );
    return output;
}

fn cbrt_vec3(v: vec3<f32>) -> vec3<f32> {
    return sign(v) * pow(abs(v), vec3<f32>(1.0 / 3.0));
}

fn rgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
    let lms = mat3x3<f32>(
        0.4122214708, 0.5363325363, 0.0514459929,
        0.2119034982, 0.6806995451, 0.1073969566,
        0.0883024619, 0.2817188376, 0.6299787005
    ) * rgb;

    let lms_ = cbrt_vec3(lms);

    return mat3x3<f32>(
        0.2104542553, 0.7936177850, -0.0040720468,
        1.9779984951, -2.4285922050, 0.4505937099,
        0.0259040371, 0.7827717662, -0.8086757660
    ) * lms_;
}

fn oklab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    let lms_ = mat3x3<f32>(
        1.0, 0.3963377774, 0.2158037573,
        1.0, -0.1055613458, -0.0638541728,
        1.0, -0.0894841775, -1.2914855480
    ) * lab;

    let lms = lms_ * lms_ * lms_;

    return mat3x3<f32>(
        4.0767416621, -3.3077115913, 0.2309699292,
        -1.2684380046, 2.6097574011, -0.3413193965,
        -0.0041960863, -0.7034186147, 1.7076147010
    ) * lms;
}

@fragment
fn fs_main(@location(0) bary: vec3<f32>) -> @location(0) vec4<f32> {
    let colors = mat3x3<f32>(rgb_to_oklab(vec3<f32>(1.0, 1.0, 0.0)), rgb_to_oklab(vec3<f32>(0.0, 1.0, 1.0)), rgb_to_oklab(vec3<f32>(1.0, 0.0, 1.0)));

    let interpolated_colors = colors * bary;

    var rgb: vec3<f32> = oklab_to_rgb(interpolated_colors);

    return vec4<f32>(rgb, 1.0);
}
