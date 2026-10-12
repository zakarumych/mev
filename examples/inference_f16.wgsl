enable f16;
alias Scalar = f16;
alias Input = array<Scalar, 2>;
alias Encoded = array<Scalar, 2>;
alias NetworkOutput = array<Scalar, 3>;
alias Output = array<Scalar, 3>;
alias Target = array<Scalar, 3>;

struct NnConfig {
    sample_count: u32,
    workgroup_count: u32,
    parameter_count: u32,
    subgroup_capacity: u32,
    tape_stride: u32,
    dense_gradient_offset: u32,
    aux_gradient_offset: u32,
    loss_offset: u32,
    step: u32,
    trainable_count: u32,
    encoded_size: u32,
    output_size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
};

fn nn_activation(x: Scalar) -> Scalar { return tanh(x); }

fn nn_activation_derivative(x: Scalar, y: Scalar) -> Scalar { return 1.0 - y * y; }

fn nn_encode(input: Input, sample: u32) -> Encoded { var r: Encoded; for (var i = 0u; i < 2u; i++) { r[i] = input[i]; } return r; }fn nn_encode_backward(input: Input, encoded: Encoded, grad: Encoded, sample: u32) {}
fn nn_output(values: NetworkOutput, sample: u32) -> Output { var r: Output; for (var i = 0u; i < 3u; i++) { r[i] = values[i]; } return r; }fn nn_output_backward(values: NetworkOutput, prediction: Output, grad: Output, sample: u32) -> NetworkOutput { var r: NetworkOutput; for (var i = 0u; i < 3u; i++) { r[i] = grad[i]; } return r; }
@group(0) @binding(0) var<uniform> config: NnConfig;
@group(0) @binding(1) var<storage,read> parameters: array<f16>;
@group(0) @binding(2) var<storage,read> inputs: array<Scalar>;
@group(0) @binding(3) var<storage,read_write> outputs: array<Scalar>;
var<workgroup> nn_cache: array<Scalar, 12>;
@compute @workgroup_size(64)fn nn_inference(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let sample = gid.x; let valid = sample < config.sample_count; var input: Input; if valid { for (var i = 0u; i < 2u; i++) { input[i] = inputs[sample * 2u + i]; } } var a: array<Scalar,3>; var next: array<Scalar,3>; var encoded: Encoded; if valid { encoded = nn_encode(input, sample);for (var i = 0u; i < 2u; i++) { a[i] = encoded[i]; } }for (var k = lid.x; k < 9u; k += 64u) { nn_cache[k] = Scalar(parameters[0u + k]); } workgroupBarrier();
    if valid { for (var j = 0u; j < 3u; j++) { var z = nn_cache[6u + j];for (var k = 0u; k < 2u; k++) { z += a[k] * nn_cache[j * 2u + k]; }next[j] = z; } }
    if valid { for (var j = 0u; j < 3u; j++) { let z = next[j];a[j] = nn_activation(z); } } workgroupBarrier();
    for (var k = lid.x; k < 12u; k += 64u) { nn_cache[k] = Scalar(parameters[9u + k]); } workgroupBarrier();
    if valid { for (var j = 0u; j < 3u; j++) { var z = nn_cache[9u + j];for (var k = 0u; k < 3u; k++) { z += a[k] * nn_cache[j * 3u + k]; }next[j] = z; } }
    if valid { for (var j = 0u; j < 3u; j++) { let z = next[j];a[j] = z; } } workgroupBarrier();
    if valid { var values: NetworkOutput;for (var i = 0u; i < 3u; i++) { values[i] = a[i]; }let prediction = nn_output(values, sample);for (var i = 0u; i < 3u; i++) { outputs[sample * 3u + i] = prediction[i]; } }
}