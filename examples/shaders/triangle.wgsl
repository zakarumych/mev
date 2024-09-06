
struct VertOutput {
    @builtin(position)
    position: vec4<f32>,

    @location(0)
    color: vec3<f32>,
}

struct Constants {
    angle: f32,
    width: u32,
    height: u32,
}

var<push_constant> pc: Constants;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertOutput {
    let colors = mat3x3<f32>(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0));
    var rgb: vec3<f32> = vec3<f32>(colors[0][index], colors[1][index], colors[2][index]);

    let x = (-0.5 + (f32(index) * 0.5));
    let y = (-(sqrt(3.0) / 6.0) + f32(index == 1u) * sqrt(3.0) / 2.0);

    let a = pc.angle * 6.28318530717958647692528676655900577;

    let ca = cos(a);
    let sa = sin(a);

    let output = VertOutput(
        vec4<f32>((ca * x + sa * y) / f32(pc.width) * f32(pc.height), (ca * y - sa * x), 0.0, 1.0),
        rgb,
    );
    return output;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
