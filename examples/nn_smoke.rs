//! Headless numerical verification for `mev::kernels::nn`.
//!
//! Run with `cargo run --example nn_smoke`. The example selects a compute-capable
//! queue, runs inference and training, copies GPU-owned state into host-readable
//! staging buffers, and checks results against CPU forward, finite-difference,
//! and Adam references. It also exercises custom WGSL hooks, trainable auxiliary
//! state, sliced bindings, empty batches, and validation failures.

use std::{error::Error, fmt};

use mev::kernels::nn::{
    Activation, Adam, AuxiliaryDesc, AuxiliaryMode, Batch, Encoding, HashGrid, HashGridLevel, Loss,
    Nn, NnDesc, Optimizer, Output, Precision,
};
use mev::{
    Buffer, BufferDesc, BufferInitDesc, BufferUsage, Device, DeviceDesc, Features, PipelineStages,
    Queue, QueueFlags,
};

const SLICE_PAD: usize = 256;
const SENTINEL: u8 = 0xA7;

#[derive(Debug)]
struct SmokeError(String);

impl fmt::Display for SmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SmokeError {}

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(SmokeError(message.into()))
}

struct SlicedBuffer {
    buffer: Buffer,
    payload_bytes: usize,
}

impl SlicedBuffer {
    fn initialized_bytes(device: &Device, payload: &[u8], usage: BufferUsage, name: &str) -> Self {
        let mut bytes = vec![SENTINEL; SLICE_PAD + payload.len() + SLICE_PAD];
        bytes[SLICE_PAD..SLICE_PAD + payload.len()].copy_from_slice(payload);
        Self {
            buffer: device.new_buffer_init(BufferInitDesc {
                data: &bytes,
                usage: usage | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                name,
            }),
            payload_bytes: payload.len(),
        }
    }

    fn initialized(device: &Device, values: &[f32], usage: BufferUsage, name: &str) -> Self {
        Self::initialized_bytes(device, bytemuck::cast_slice(values), usage, name)
    }

    fn zeroed(device: &Device, float_count: usize, usage: BufferUsage, name: &str) -> Self {
        Self::initialized(device, &vec![0.0; float_count], usage, name)
    }

    fn slice(&self) -> mev::BufferSlice<'_> {
        self.buffer.slice(SLICE_PAD..SLICE_PAD + self.payload_bytes)
    }
}

fn read_bytes(
    device: &Device,
    queue: &mut Queue,
    source: mev::BufferSlice<'_>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let size = source.size();
    let mut staging = device.new_buffer(BufferDesc {
        name: "NN smoke readback",
        size,
        usage: BufferUsage::TRANSFER_DST | BufferUsage::HOST_READ,
    });
    let mut encoder = queue.new_command_encoder();
    encoder.barrier(
        PipelineStages::COMPUTE_SHADER | PipelineStages::TRANSFER,
        PipelineStages::TRANSFER,
    );
    encoder.copy().copy_buffer_to_buffer(&source, &staging);
    queue.submit_checkpoint([encoder.finish()])?;
    queue.wait_idle()?;
    let mut bytes = vec![0; size];
    staging.read(0, &mut bytes)?;
    Ok(bytes)
}

fn read_f32(
    device: &Device,
    queue: &mut Queue,
    source: mev::BufferSlice<'_>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = read_bytes(device, queue, source)?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn assert_sentinels(
    device: &Device,
    queue: &mut Queue,
    sliced: &SlicedBuffer,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let bytes = read_bytes(device, queue, (&sliced.buffer).into())?;
    if bytes[..SLICE_PAD].iter().any(|&x| x != SENTINEL)
        || bytes[SLICE_PAD + sliced.payload_bytes..]
            .iter()
            .any(|&x| x != SENTINEL)
    {
        return Err(fail(format!(
            "{label}: sliced binding changed sentinel padding"
        )));
    }
    Ok(())
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * e.abs();
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "{label}[{index}]: actual={a}, expected={e}, tolerance={tolerance}"
        );
    }
}

#[derive(Clone)]
struct CpuDense {
    input_size: usize,
    layers: Vec<usize>,
}

impl CpuDense {
    fn parameter_count(&self) -> usize {
        let mut input = self.input_size;
        let mut count = 0;
        for &output in &self.layers {
            count += output * (input + 1);
            input = output;
        }
        count
    }

    fn forward_one_f64(&self, parameters: &[f32], input: &[f32]) -> Vec<f64> {
        let mut values: Vec<f64> = input.iter().map(|&value| value as f64).collect();
        let mut offset = 0;
        for (layer_index, &output_size) in self.layers.iter().enumerate() {
            let input_size = values.len();
            let mut next = vec![0.0f64; output_size];
            for output in 0..output_size {
                let row = offset + output * input_size;
                let mut value = parameters[offset + output_size * input_size + output] as f64;
                for input in 0..input_size {
                    value += parameters[row + input] as f64 * values[input];
                }
                next[output] = if layer_index + 1 == self.layers.len() {
                    value
                } else {
                    value.tanh()
                };
            }
            offset += output_size * (input_size + 1);
            values = next;
        }
        values
    }

    fn forward(&self, parameters: &[f32], inputs: &[f32]) -> Vec<f32> {
        inputs
            .chunks_exact(self.input_size)
            .flat_map(|input| self.forward_one_f64(parameters, input))
            .map(|value| value as f32)
            .collect()
    }

    fn mean_loss(&self, parameters: &[f32], inputs: &[f32], targets: &[f32]) -> f64 {
        let predictions: Vec<f64> = inputs
            .chunks_exact(self.input_size)
            .flat_map(|input| self.forward_one_f64(parameters, input))
            .collect();
        let samples = inputs.len() / self.input_size;
        predictions
            .iter()
            .zip(targets)
            .map(|(&prediction, &target)| {
                let residual = prediction - target as f64;
                0.5 * residual * residual
            })
            .sum::<f64>()
            / samples as f64
    }

    fn finite_difference_gradient(
        &self,
        parameters: &[f32],
        inputs: &[f32],
        targets: &[f32],
    ) -> Vec<f32> {
        let epsilon = 1.0e-3f32;
        let mut plus = parameters.to_vec();
        let mut minus = parameters.to_vec();
        (0..parameters.len())
            .map(|index| {
                plus[index] += epsilon;
                minus[index] -= epsilon;
                let gradient = (self.mean_loss(&plus, inputs, targets)
                    - self.mean_loss(&minus, inputs, targets))
                    / (2.0 * epsilon as f64);
                plus[index] = parameters[index];
                minus[index] = parameters[index];
                gradient as f32
            })
            .collect()
    }
}

fn adam_step(values: &mut [f32], moments: &mut [f32], gradient: &[f32], settings: Adam, step: u32) {
    for index in 0..values.len() {
        let m = settings.beta1 * moments[index * 2] + (1.0 - settings.beta1) * gradient[index];
        let v = settings.beta2 * moments[index * 2 + 1]
            + (1.0 - settings.beta2) * gradient[index] * gradient[index];
        moments[index * 2] = m;
        moments[index * 2 + 1] = v;
        let corrected_m = m / (1.0 - settings.beta1.powi(step as i32));
        let corrected_v = v / (1.0 - settings.beta2.powi(step as i32));
        values[index] -=
            settings.learning_rate * corrected_m / (corrected_v.sqrt() + settings.epsilon);
    }
}

fn deterministic(count: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32 * 0.731 + phase).sin() * scale)
        .collect()
}

fn reduced_dense_gradient(nn: &Nn, batch: Batch, workgroup_size: u32, words: &[u32]) -> Vec<f32> {
    let groups = batch.sample_count.div_ceil(workgroup_size);
    let capacity = nn.subgroup_capacity();
    let count = nn.parameter_count() as usize;
    let mut gradient = vec![0.0; count];
    for group in 0..groups {
        let subgroups = words[group as usize];
        assert!(
            subgroups > 0 && subgroups <= capacity,
            "workgroup {group} reported invalid subgroup count {subgroups}, capacity {capacity}"
        );
        for subgroup in 0..subgroups {
            let base = groups as usize + ((group * capacity + subgroup) as usize * count);
            for parameter in 0..count {
                gradient[parameter] += f32::from_bits(words[base + parameter]);
            }
        }
    }
    for value in &mut gradient {
        *value /= batch.sample_count as f32;
    }
    gradient
}

fn custom_forward_one(
    parameters: &[f32],
    trainable: &[f32],
    readonly: f32,
    input: &[f32],
    sample: usize,
) -> [f64; 2] {
    let encoded = [
        input[0] as f64 * trainable[0] as f64,
        input[1] as f64 + trainable[1 + sample % 2] as f64 + readonly as f64,
        input[0] as f64 - input[1] as f64,
    ];
    let mut hidden = [0.0f64; 3];
    for output in 0..3 {
        let row = output * 3;
        let raw = parameters[9 + output] as f64
            + parameters[row] as f64 * encoded[0]
            + parameters[row + 1] as f64 * encoded[1]
            + parameters[row + 2] as f64 * encoded[2];
        hidden[output] = raw / (1.0 + raw.abs());
    }
    let mut raw = [0.0f64; 2];
    for output in 0..2 {
        let row = 12 + output * 3;
        raw[output] = parameters[18 + output] as f64
            + parameters[row] as f64 * hidden[0]
            + parameters[row + 1] as f64 * hidden[1]
            + parameters[row + 2] as f64 * hidden[2];
    }
    [raw[0] + 0.25 * raw[1], raw[0] - raw[1]]
}

fn custom_outputs(
    parameters: &[f32],
    trainable: &[f32],
    readonly: f32,
    inputs: &[f32],
) -> Vec<f32> {
    inputs
        .chunks_exact(2)
        .enumerate()
        .flat_map(|(sample, input)| {
            custom_forward_one(parameters, trainable, readonly, input, sample)
        })
        .map(|value| value as f32)
        .collect()
}

fn custom_mean_loss(
    parameters: &[f32],
    trainable: &[f32],
    readonly: f32,
    inputs: &[f32],
    targets: &[f32],
) -> f64 {
    inputs
        .chunks_exact(2)
        .enumerate()
        .map(|(sample, input)| custom_forward_one(parameters, trainable, readonly, input, sample))
        .zip(targets.chunks_exact(2))
        .map(|(prediction, target)| {
            let a = prediction[0] - target[0] as f64;
            let b = prediction[1] - target[1] as f64;
            0.35 * a * a + 0.65 * b * b
        })
        .sum::<f64>()
        / (inputs.len() / 2) as f64
}

fn custom_finite_difference(
    parameters: &[f32],
    trainable: &[f32],
    readonly: f32,
    inputs: &[f32],
    targets: &[f32],
) -> Vec<f32> {
    let epsilon = 1.0e-3f32;
    let mut values = [parameters, trainable].concat();
    (0..values.len())
        .map(|index| {
            values[index] += epsilon;
            let plus = custom_mean_loss(
                &values[..parameters.len()],
                &values[parameters.len()..],
                readonly,
                inputs,
                targets,
            );
            values[index] -= 2.0 * epsilon;
            let minus = custom_mean_loss(
                &values[..parameters.len()],
                &values[parameters.len()..],
                readonly,
                inputs,
                targets,
            );
            values[index] += epsilon;
            ((plus - minus) / (2.0 * epsilon as f64)) as f32
        })
        .collect()
}

fn builtin_case(device: &Device, queue: &mut Queue) -> Result<(), Box<dyn Error>> {
    let cpu = CpuDense {
        input_size: 3,
        layers: vec![4, 3, 2],
    };
    let adam = Adam {
        learning_rate: 0.01,
        beta1: 0.8,
        beta2: 0.93,
        epsilon: 1.0e-6,
    };
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 3 }], &[4, 3, 2]);
    desc.activation = Activation::Tanh;
    desc.optimizer = Optimizer::Adam(adam);
    desc.workgroup_size = 64;
    let initial = deterministic(cpu.parameter_count(), 0.19, 0.4);
    let wrong = Nn::new(device, desc.clone(), &initial[..initial.len() - 1]);
    assert!(wrong.is_err(), "wrong initial weight count must fail");

    let mut nn = Nn::new(device, desc, &initial)?;
    let batch = Batch::new(67);
    let inputs = deterministic(batch.sample_count as usize * 3, 0.8, 0.2);
    let targets = deterministic(batch.sample_count as usize * 2, 0.55, 1.7);
    let input = SlicedBuffer::initialized(device, &inputs, BufferUsage::STORAGE, "NN input");
    let target = SlicedBuffer::initialized(device, &targets, BufferUsage::STORAGE, "NN target");
    let output = SlicedBuffer::zeroed(
        device,
        batch.sample_count as usize * 2,
        BufferUsage::STORAGE,
        "NN output",
    );
    let post_update_output = SlicedBuffer::zeroed(
        device,
        batch.sample_count as usize * 2,
        BufferUsage::STORAGE,
        "NN post-update output",
    );

    let epoch = nn.inference(queue, input.slice(), output.slice(), &[], batch)?;
    assert!(epoch.is_some(), "nonempty inference must submit");
    let gpu_output = read_f32(device, queue, output.slice())?;
    assert_close(
        "multi-workgroup inference",
        &gpu_output,
        &cpu.forward(&initial, &inputs),
        2.0e-5,
        2.0e-5,
    );

    let empty_before = read_f32(device, queue, nn.parameters().into())?;
    assert!(
        nn.training(
            queue,
            input.buffer.slice(SLICE_PAD..SLICE_PAD),
            target.buffer.slice(SLICE_PAD..SLICE_PAD),
            output.buffer.slice(SLICE_PAD..SLICE_PAD),
            &[],
            Batch::new(0),
        )?
        .is_none(),
        "empty training must not submit"
    );
    let empty_after = read_f32(device, queue, nn.parameters().into())?;
    assert_eq!(
        empty_before, empty_after,
        "empty training changed parameters"
    );

    let undersized = input.buffer.slice(SLICE_PAD..SLICE_PAD + 4);
    assert!(
        nn.inference(queue, undersized, output.slice(), &[], batch)
            .is_err(),
        "undersized input must fail"
    );

    let mut cpu_parameters = initial;
    let mut cpu_moments = vec![0.0; cpu_parameters.len() * 2];
    for step in 1..=3 {
        let step_inputs: Vec<f32> = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| value + step as f32 * 0.013 * ((index % 3) as f32 - 1.0))
            .collect();
        let step_targets: Vec<f32> = targets
            .iter()
            .enumerate()
            .map(|(index, value)| value - step as f32 * 0.009 * (index % 2) as f32)
            .collect();
        let step_input =
            SlicedBuffer::initialized(device, &step_inputs, BufferUsage::STORAGE, "NN step input");
        let step_target = SlicedBuffer::initialized(
            device,
            &step_targets,
            BufferUsage::STORAGE,
            "NN step target",
        );
        let expected_output = cpu.forward(&cpu_parameters, &step_inputs);
        let gradient = cpu.finite_difference_gradient(&cpu_parameters, &step_inputs, &step_targets);
        let mut next_cpu_parameters = cpu_parameters.clone();
        let mut next_cpu_moments = cpu_moments.clone();
        adam_step(
            &mut next_cpu_parameters,
            &mut next_cpu_moments,
            &gradient,
            adam,
            step,
        );
        nn.training(
            queue,
            step_input.slice(),
            step_target.slice(),
            output.slice(),
            &[],
            batch,
        )?;
        nn.inference(
            queue,
            step_input.slice(),
            post_update_output.slice(),
            &[],
            batch,
        )?;
        let gpu_pre_update = read_f32(device, queue, output.slice())?;
        assert_close(
            "training pre-update output",
            &gpu_pre_update,
            &expected_output,
            3.0e-5,
            3.0e-5,
        );
        let gpu_post_update = read_f32(device, queue, post_update_output.slice())?;
        assert_close(
            "immediate post-training inference",
            &gpu_post_update,
            &cpu.forward(&next_cpu_parameters, &step_inputs),
            3.0e-5,
            3.0e-5,
        );
        if step == 1 {
            let tape = nn
                .tape()
                .ok_or_else(|| fail("training did not allocate tape"))?
                .clone();
            assert!(
                nn.training(
                    queue,
                    tape.slice(..3 * 4),
                    step_target.buffer.slice(SLICE_PAD..SLICE_PAD + 2 * 4),
                    output.buffer.slice(SLICE_PAD..SLICE_PAD + 2 * 4),
                    &[],
                    Batch::new(1),
                )
                .is_err(),
                "training input aliasing retained tape must fail"
            );
        }
        let gradient_buffer = nn
            .gradients()
            .ok_or_else(|| fail("training did not allocate gradients"))?;
        let gradient_bytes = read_bytes(device, queue, gradient_buffer.into())?;
        let gradient_words = bytemuck::cast_slice::<u8, u32>(&gradient_bytes);
        let gpu_gradient = reduced_dense_gradient(&nn, batch, 64, gradient_words);
        assert_close(
            "finite-difference gradient",
            &gpu_gradient,
            &gradient,
            2.0e-5,
            1.0e-3,
        );

        let gpu_parameters = read_f32(device, queue, nn.parameters().into())?;
        let gpu_moments = read_f32(device, queue, nn.moments().into())?;
        assert_close(
            "Adam parameters",
            &gpu_parameters,
            &next_cpu_parameters,
            1.0e-5,
            1.0e-4,
        );
        assert_close(
            "Adam moments",
            &gpu_moments,
            &next_cpu_moments,
            2.0e-5,
            1.0e-3,
        );
        cpu_parameters = next_cpu_parameters;
        cpu_moments = next_cpu_moments;
    }

    assert_sentinels(device, queue, &input, "input")?;
    assert_sentinels(device, queue, &target, "target")?;
    assert_sentinels(device, queue, &output, "output")?;
    assert_sentinels(device, queue, &post_update_output, "post-update output")?;
    println!("[PASS] built-in inference, gradients, persistent Adam, validation, slices");
    Ok(())
}

fn custom_case(device: &Device, queue: &mut Queue) -> Result<(), Box<dyn Error>> {
    let mut desc = NnDesc::new(
        &[Encoding::Custom {
            input_arity: 2,
            output_arity: 3,
            source: "fn encode_custom(input:array<Scalar,2>,sample:u32)->array<Scalar,3>{var r:array<Scalar,3>;r[0]=input[0]*Scalar(trainable[0]);r[1]=input[1]+Scalar(trainable[1u+sample%2u])+Scalar(context[0]);r[2]=input[0]-input[1];return r;}\n\
                     fn encode_custom_backward(input:array<Scalar,2>,encoded:array<Scalar,3>,grad:array<Scalar,3>,sample:u32){nn_add_trainable_gradient(0u,grad[0]*input[0]);nn_add_trainable_gradient(1u+sample%2u,grad[1]);}".into(),
            forward: "encode_custom".into(),
            backward: "encode_custom_backward".into(),
        }],
        &[3, 2],
    );
    desc.workgroup_size = 7;
    desc.activation = Activation::Custom(
        "fn nn_activation(x:f32)->f32{return x/(1.0+abs(x));}\n\
         fn nn_activation_derivative(x:f32,y:f32)->f32{let d=1.0+abs(x);return 1.0/(d*d);}"
            .into(),
    );
    desc.output = Output::Custom {
        output_size: 2,
        source: "fn nn_output(values:NetworkOutput,sample:u32)->Output{var r:Output;r[0]=values[0]+0.25*values[1];r[1]=values[0]-values[1];return r;}\n\
                 fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{var r:NetworkOutput;r[0]=grad[0]+grad[1];r[1]=0.25*grad[0]-grad[1];return r;}".into(),
    };
    desc.loss = Loss::Custom {
        target_size: 2,
        source: "fn nn_loss(prediction:Output,target_value:Target)->f32{let a=prediction[0]-target_value[0];let b=prediction[1]-target_value[1];return 0.35*a*a+0.65*b*b;}\n\
                 fn nn_loss_derivative(prediction:Output,target_value:Target)->Output{var r:Output;r[0]=0.7*(prediction[0]-target_value[0]);r[1]=1.3*(prediction[1]-target_value[1]);return r;}".into(),
    };
    desc.auxiliary = vec![
        AuxiliaryDesc::new(
            "trainable",
            "array<f32>",
            AuxiliaryMode::Trainable { elements: 3 },
        ),
        AuxiliaryDesc::new("context", "array<f32>", AuxiliaryMode::ReadOnly),
    ];
    desc.optimizer = Optimizer::Adam(Adam {
        learning_rate: 0.005,
        ..Adam::default()
    });
    let parameter_count = 3 * 4 + 2 * 4;
    let initial_parameters = deterministic(parameter_count, 0.12, 0.7);
    let mut nn = Nn::new(device, desc, &initial_parameters)?;
    let batch = Batch::new(67);
    let input_values = deterministic(batch.sample_count as usize * 2, 0.6, 0.8);
    let target_values = deterministic(batch.sample_count as usize * 2, 0.4, 1.1);
    let initial_trainable = [0.8, -0.15, 0.21];
    let readonly_value = 0.03;
    let expected_output = custom_outputs(
        &initial_parameters,
        &initial_trainable,
        readonly_value,
        &input_values,
    );
    let expected_gradient = custom_finite_difference(
        &initial_parameters,
        &initial_trainable,
        readonly_value,
        &input_values,
        &target_values,
    );
    let input =
        SlicedBuffer::initialized(device, &input_values, BufferUsage::STORAGE, "custom input");
    let target = SlicedBuffer::initialized(
        device,
        &target_values,
        BufferUsage::STORAGE,
        "custom target",
    );
    let output = SlicedBuffer::zeroed(
        device,
        batch.sample_count as usize * 2,
        BufferUsage::STORAGE,
        "custom output",
    );
    let trainable = SlicedBuffer::initialized(
        device,
        &initial_trainable,
        BufferUsage::STORAGE,
        "custom trainable",
    );
    let replacement = SlicedBuffer::initialized(
        device,
        &[0.8, -0.15, 0.21],
        BufferUsage::STORAGE,
        "custom replacement",
    );
    let readonly = SlicedBuffer::initialized(
        device,
        &[readonly_value],
        BufferUsage::STORAGE,
        "custom readonly",
    );

    nn.training(
        queue,
        input.slice(),
        target.slice(),
        output.slice(),
        &[trainable.slice(), readonly.slice()],
        batch,
    )?;
    let gpu_output = read_f32(device, queue, output.slice())?;
    assert_close(
        "custom pre-update output",
        &gpu_output,
        &expected_output,
        4.0e-5,
        4.0e-5,
    );
    let gradient_buffer = nn
        .gradients()
        .ok_or_else(|| fail("custom training did not allocate gradients"))?;
    let gradient_bytes = read_bytes(device, queue, gradient_buffer.into())?;
    let gradient_words = bytemuck::cast_slice::<u8, u32>(&gradient_bytes);
    let dense_gradient = reduced_dense_gradient(&nn, batch, 7, gradient_words);
    let groups = batch.sample_count.div_ceil(7);
    let auxiliary_offset =
        groups as usize + groups as usize * nn.subgroup_capacity() as usize * parameter_count;
    let auxiliary_gradient: Vec<f32> = gradient_words
        [auxiliary_offset..auxiliary_offset + initial_trainable.len()]
        .iter()
        .map(|&word| f32::from_bits(word) / batch.sample_count as f32)
        .collect();
    assert_close(
        "custom dense finite-difference gradient",
        &dense_gradient,
        &expected_gradient[..parameter_count],
        2.0e-5,
        1.0e-3,
    );
    assert_close(
        "custom auxiliary finite-difference gradient",
        &auxiliary_gradient,
        &expected_gradient[parameter_count..],
        2.0e-5,
        1.0e-3,
    );
    let mut expected_values =
        [initial_parameters.as_slice(), initial_trainable.as_slice()].concat();
    let mut expected_moments = vec![0.0; expected_values.len() * 2];
    adam_step(
        &mut expected_values,
        &mut expected_moments,
        &expected_gradient,
        Adam {
            learning_rate: 0.005,
            ..Adam::default()
        },
        1,
    );
    let gpu_parameters = read_f32(device, queue, nn.parameters().into())?;
    let gpu_trainable = read_f32(device, queue, trainable.slice())?;
    let gpu_moments = read_f32(device, queue, nn.moments().into())?;
    assert_close(
        "custom Adam dense",
        &gpu_parameters,
        &expected_values[..parameter_count],
        1.0e-5,
        1.0e-4,
    );
    assert_close(
        "custom Adam auxiliary",
        &gpu_trainable,
        &expected_values[parameter_count..],
        1.0e-5,
        1.0e-4,
    );
    assert_close(
        "custom Adam moments",
        &gpu_moments,
        &expected_moments,
        2.0e-5,
        1.0e-3,
    );
    assert!(
        nn.training(
            queue,
            input.slice(),
            target.slice(),
            output.slice(),
            &[replacement.slice(), readonly.slice()],
            batch,
        )
        .is_err(),
        "trainable auxiliary rebind must fail"
    );
    let parameters_before_reset = read_f32(device, queue, nn.parameters().into())?;
    nn.reset_optimizer(queue)?;
    let parameters_after_reset = read_f32(device, queue, nn.parameters().into())?;
    let moments_after_reset = read_f32(device, queue, nn.moments().into())?;
    assert_eq!(
        parameters_before_reset, parameters_after_reset,
        "optimizer reset changed dense parameters"
    );
    assert!(
        moments_after_reset.iter().all(|&value| value == 0.0),
        "optimizer reset did not clear every dense and auxiliary moment"
    );
    nn.training(
        queue,
        input.slice(),
        target.slice(),
        output.slice(),
        &[replacement.slice(), readonly.slice()],
        batch,
    )?;
    let updated = read_f32(device, queue, replacement.slice())?;
    assert!(
        updated
            .iter()
            .zip([0.8, -0.15, 0.21])
            .any(|(a, b)| (a - b).abs() > 1.0e-6),
        "custom encoding did not update trainable auxiliary values"
    );
    for (label, buffer) in [
        ("custom input", &input),
        ("custom target", &target),
        ("custom output", &output),
        ("custom trainable", &trainable),
        ("custom replacement", &replacement),
        ("custom readonly", &readonly),
    ] {
        assert_sentinels(device, queue, buffer, label)?;
    }
    println!("[PASS] custom activation/encoding/output/loss and auxiliary rebind/reset");
    Ok(())
}

fn mixed_forward(parameters: &[f32], g: f32, inputs: &[f32]) -> Vec<f32> {
    inputs
        .chunks_exact(5)
        .flat_map(|input| {
            let encoded = [
                input[0] as f64,
                input[1] as f64,
                input[2] as f64 * g as f64,
                (input[2] as f64).powi(2),
                input[2] as f64 + g as f64,
                input[3] as f64 * g as f64 + input[4] as f64,
            ];
            let mut hidden = [0.0f64; 3];
            for output in 0..3 {
                let row = output * 6;
                let mut value = parameters[18 + output] as f64;
                for index in 0..6 {
                    value += parameters[row + index] as f64 * encoded[index];
                }
                hidden[output] = value.tanh();
            }
            let mut result = [0.0f32; 2];
            for output in 0..2 {
                let row = 21 + output * 3;
                let mut value = parameters[27 + output] as f64;
                for index in 0..3 {
                    value += parameters[row + index] as f64 * hidden[index];
                }
                result[output] = value as f32;
            }
            result
        })
        .collect()
}

fn mixed_loss(parameters: &[f32], g: f32, inputs: &[f32], targets: &[f32]) -> f64 {
    mixed_forward(parameters, g, inputs)
        .iter()
        .zip(targets)
        .map(|(&prediction, &target)| {
            let residual = prediction as f64 - target as f64;
            0.5 * residual * residual
        })
        .sum::<f64>()
        / (inputs.len() / 5) as f64
}

fn mixed_gradient(parameters: &[f32], g: f32, inputs: &[f32], targets: &[f32]) -> Vec<f32> {
    let epsilon = 1.0e-3f32;
    let mut values = parameters.to_vec();
    values.push(g);
    (0..values.len())
        .map(|index| {
            values[index] += epsilon;
            let plus = mixed_loss(
                &values[..parameters.len()],
                values[parameters.len()],
                inputs,
                targets,
            );
            values[index] -= 2.0 * epsilon;
            let minus = mixed_loss(
                &values[..parameters.len()],
                values[parameters.len()],
                inputs,
                targets,
            );
            values[index] += epsilon;
            ((plus - minus) / (2.0 * epsilon as f64)) as f32
        })
        .collect()
}

fn mixed_vector_case(
    device: &Device,
    queue: &mut Queue,
    precision: Precision,
) -> Result<(), Box<dyn Error>> {
    let custom_a = Encoding::Custom {
        input_arity: 1,
        output_arity: 3,
        source: "fn encode_a(input:array<Scalar,1>,sample:u32)->array<Scalar,3>{var r:array<Scalar,3>;let g=Scalar(shared_g[0]);r[0]=input[0]*g;r[1]=input[0]*input[0];r[2]=input[0]+g;return r;}\n\
                 fn encode_a_backward(input:array<Scalar,1>,encoded:array<Scalar,3>,grad:array<Scalar,3>,sample:u32){let x=encoded[2]-Scalar(shared_g[0]);nn_add_shared_g_gradient(0u,grad[0]*x+grad[2]);}"
            .into(),
        forward: "encode_a".into(),
        backward: "encode_a_backward".into(),
    };
    let custom_b = Encoding::Custom {
        input_arity: 2,
        output_arity: 1,
        source: "fn encode_b(input:array<Scalar,2>,sample:u32)->array<Scalar,1>{var r:array<Scalar,1>;r[0]=input[0]*Scalar(shared_g[0])+input[1];return r;}\n\
                 fn encode_b_backward(input:array<Scalar,2>,encoded:array<Scalar,1>,grad:array<Scalar,1>,sample:u32){nn_add_shared_g_gradient(0u,grad[0]*input[0]);}"
            .into(),
        forward: "encode_b".into(),
        backward: "encode_b_backward".into(),
    };
    let mut desc = NnDesc::new(
        &[Encoding::Identity { arity: 2 }, custom_a, custom_b],
        &[3, 2],
    );
    desc.precision = precision;
    desc.activation = Activation::Tanh;
    desc.workgroup_size = 7;
    let adam = Adam {
        learning_rate: 0.004,
        beta1: 0.8,
        beta2: 0.93,
        epsilon: 1.0e-6,
    };
    desc.optimizer = Optimizer::Adam(adam);
    desc.auxiliary = vec![AuxiliaryDesc::new(
        "shared_g",
        "array<f32>",
        AuxiliaryMode::Trainable { elements: 1 },
    )];
    let initial = deterministic(29, 0.13, 0.52);
    let mut nn = Nn::new(device, desc, &initial)?;
    assert_eq!(nn.input_size(), 5);
    assert_eq!(nn.encoded_size(), 6);
    assert_eq!(nn.parameter_count(), 29);
    let parameters: Vec<f32> = match precision {
        Precision::F32 => initial.clone(),
        Precision::F16 => initial
            .iter()
            .map(|&value| half::f16::from_f32(value).to_f32())
            .collect(),
    };
    let batch = Batch::new(67);
    let inputs = deterministic(67 * 5, 0.62, 0.27);
    let targets = deterministic(67 * 2, 0.41, 1.33);
    let g = 0.37;
    let expected_output = mixed_forward(&parameters, g, &inputs);
    let gradient = mixed_gradient(&parameters, g, &inputs, &targets);
    let input = SlicedBuffer::initialized(device, &inputs, BufferUsage::STORAGE, "mixed input");
    let target = SlicedBuffer::initialized(device, &targets, BufferUsage::STORAGE, "mixed target");
    let output = SlicedBuffer::zeroed(device, 67 * 2, BufferUsage::STORAGE, "mixed output");
    let auxiliary = SlicedBuffer::initialized(device, &[g], BufferUsage::STORAGE, "mixed g");
    let inference_input_size = nn.inference_input_buffer_size(batch)?;
    let inference_output_size = nn.inference_output_buffer_size(batch)?;
    assert_eq!(
        inference_input_size,
        if precision == Precision::F32 {
            1340
        } else {
            672
        }
    );
    assert_eq!(
        inference_output_size,
        if precision == Precision::F32 {
            536
        } else {
            268
        }
    );
    let inference_inputs: Vec<f32> = match precision {
        Precision::F32 => inputs.clone(),
        Precision::F16 => inputs
            .iter()
            .map(|&value| half::f16::from_f32(value).to_f32())
            .collect(),
    };
    let inference_input_payload = match precision {
        Precision::F32 => bytemuck::cast_slice(&inputs).to_vec(),
        Precision::F16 => half_bytes(&inputs, inference_input_size),
    };
    let inference_input = SlicedBuffer::initialized_bytes(
        device,
        &inference_input_payload,
        BufferUsage::STORAGE,
        "mixed inference input",
    );
    let inference_output = SlicedBuffer::initialized_bytes(
        device,
        &vec![SENTINEL; inference_output_size],
        BufferUsage::STORAGE,
        "mixed inference output",
    );
    nn.inference(
        queue,
        inference_input.slice(),
        inference_output.slice(),
        &[auxiliary.slice()],
        batch,
    )?;
    let inference_bytes = read_bytes(device, queue, inference_output.slice())?;
    let inference_actual = match precision {
        Precision::F32 => bytemuck::cast_slice::<u8, f32>(&inference_bytes).to_vec(),
        Precision::F16 => decode_half(&inference_bytes, 67 * 2),
    };
    let inference_g = match precision {
        Precision::F32 => g,
        Precision::F16 => half::f16::from_f32(g).to_f32(),
    };
    let inference_expected = mixed_forward(&parameters, inference_g, &inference_inputs);
    let (inference_atol, inference_rtol) = match precision {
        Precision::F32 => (3.0e-5, 3.0e-5),
        Precision::F16 => (3.0e-3, 8.0e-3),
    };
    assert_close(
        "mixed-vector inference order",
        &inference_actual,
        &inference_expected,
        inference_atol,
        inference_rtol,
    );
    nn.training(
        queue,
        input.slice(),
        target.slice(),
        output.slice(),
        &[auxiliary.slice()],
        batch,
    )?;
    let gpu_output = read_f32(device, queue, output.slice())?;
    assert_close(
        "mixed-vector forward order",
        &gpu_output,
        &expected_output,
        3.0e-5,
        3.0e-5,
    );
    let gradient_buffer = nn
        .gradients()
        .ok_or_else(|| fail("mixed-vector training did not allocate gradients"))?;
    let gradient_bytes = read_bytes(device, queue, gradient_buffer.into())?;
    let words = bytemuck::cast_slice::<u8, u32>(&gradient_bytes);
    let dense = reduced_dense_gradient(&nn, batch, 7, words);
    let aux_offset = nn.auxiliary_gradient_offset_words(batch)? as usize;
    let aux_gradient = f32::from_bits(words[aux_offset]) / batch.sample_count as f32;
    assert_close(
        "mixed-vector dense gradient",
        &dense,
        &gradient[..29],
        3.0e-5,
        1.5e-3,
    );
    assert_close(
        "mixed-vector summed auxiliary gradient",
        &[aux_gradient],
        &gradient[29..],
        3.0e-5,
        1.5e-3,
    );
    let mut expected_values = [parameters.as_slice(), &[g]].concat();
    let mut expected_moments = vec![0.0; expected_values.len() * 2];
    adam_step(
        &mut expected_values,
        &mut expected_moments,
        &gradient,
        adam,
        1,
    );
    if precision == Precision::F16 {
        for value in &mut expected_values[..29] {
            *value = half::f16::from_f32(*value).to_f32();
        }
    }
    let gpu_parameters = match precision {
        Precision::F32 => read_f32(device, queue, nn.parameters().into())?,
        Precision::F16 => read_half_parameters(device, queue, &nn)?,
    };
    let gpu_auxiliary = read_f32(device, queue, auxiliary.slice())?;
    assert_close(
        "mixed-vector Adam dense",
        &gpu_parameters,
        &expected_values[..29],
        2.0e-5,
        1.0e-3,
    );
    assert_close(
        "mixed-vector Adam auxiliary",
        &gpu_auxiliary,
        &expected_values[29..],
        2.0e-6,
        1.0e-5,
    );
    assert_sentinels(device, queue, &input, "mixed input")?;
    assert_sentinels(device, queue, &target, "mixed target")?;
    assert_sentinels(device, queue, &output, "mixed output")?;
    assert_sentinels(device, queue, &auxiliary, "mixed auxiliary")?;
    assert_sentinels(device, queue, &inference_input, "mixed inference input")?;
    assert_sentinels(device, queue, &inference_output, "mixed inference output")?;
    println!("[PASS] {precision:?} mixed-vector encoding order, gradients, and Adam");
    Ok(())
}

fn hashgrid_forward(
    levels: &[HashGridLevel],
    parameters: &[f32],
    table: &[f32],
    inputs: &[f32],
    precision: Precision,
) -> Vec<f64> {
    let scalar = |value: f64| match precision {
        Precision::F32 => value,
        Precision::F16 => half::f16::from_f64(value).to_f64(),
    };
    inputs
        .chunks_exact(6)
        .flat_map(|input| {
            let coordinate = [input[1], input[2], input[3]];
            let mut encoded = vec![scalar(input[0] as f64)];
            let mut offset = 0;
            for level in levels {
                let vertices = level.resolution + 1;
                let entries = (vertices * vertices).min(level.table_size);
                let plane_size = (entries * level.features) as usize;
                let position = coordinate.map(|value| {
                    scalar(scalar(value as f64).clamp(0.0, 1.0) * level.resolution as f64)
                });
                for feature in 0..level.features as usize {
                    let mut product = 1.0;
                    for (plane, (a, b)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
                        let lower_a = position[a].floor() as u32;
                        let lower_b = position[b].floor() as u32;
                        let fraction_a = scalar(position[a] - lower_a as f64);
                        let fraction_b = scalar(position[b] - lower_b as f64);
                        let mut value = 0.0;
                        for bit_b in 0..2 {
                            for bit_a in 0..2 {
                                let a = (lower_a + bit_a).min(level.resolution);
                                let b = (lower_b + bit_b).min(level.resolution);
                                let entry = if entries == vertices * vertices {
                                    a + b * vertices
                                } else {
                                    (a ^ b.wrapping_mul(2_654_435_761)) % entries
                                };
                                let weight_a = if bit_a == 0 {
                                    scalar(1.0 - fraction_a)
                                } else {
                                    fraction_a
                                };
                                let weight_b = if bit_b == 0 {
                                    scalar(1.0 - fraction_b)
                                } else {
                                    fraction_b
                                };
                                let index = offset
                                    + plane * plane_size
                                    + entry as usize * level.features as usize
                                    + feature;
                                let weighted = scalar(
                                    scalar(table[index] as f64) * scalar(weight_a * weight_b),
                                );
                                value = scalar(value + weighted);
                            }
                        }
                        product = scalar(product * value);
                    }
                    encoded.push(product);
                }
                offset += 3 * plane_size;
            }
            encoded.push(scalar(
                scalar(input[4] as f64) + scalar(2.0 * scalar(input[5] as f64)),
            ));
            let width = encoded.len();
            let mut result = [0.0; 2];
            for output in 0..2 {
                let mut value = scalar(parameters[2 * width + output] as f64);
                for index in 0..width {
                    value = scalar(
                        value
                            + scalar(
                                scalar(parameters[output * width + index] as f64) * encoded[index],
                            ),
                    );
                }
                result[output] = value;
            }
            result
        })
        .collect()
}

fn hashgrid_gradient(
    levels: &[HashGridLevel],
    parameters: &[f32],
    table: &[f32],
    inputs: &[f32],
    targets: &[f32],
) -> Vec<f32> {
    let mut values = [parameters, table].concat();
    let loss = |values: &[f32]| {
        hashgrid_forward(
            levels,
            &values[..parameters.len()],
            &values[parameters.len()..],
            inputs,
            Precision::F32,
        )
        .iter()
        .zip(targets)
        .map(|(&prediction, &target)| 0.5 * (prediction - target as f64).powi(2))
        .sum::<f64>()
            / (inputs.len() / 6) as f64
    };
    (0..values.len())
        .map(|index| {
            let original = values[index];
            values[index] = original + 1.0e-3;
            let upper = values[index];
            let plus = loss(&values);
            values[index] = original - 1.0e-3;
            let lower = values[index];
            let minus = loss(&values);
            values[index] = original;
            ((plus - minus) / (upper as f64 - lower as f64)) as f32
        })
        .collect()
}

fn hashgrid_case(
    device: &Device,
    queue: &mut Queue,
    precision: Precision,
) -> Result<(), Box<dyn Error>> {
    let grid = HashGrid::new(&[
        HashGridLevel {
            resolution: 1,
            features: 2,
            table_size: 4,
        },
        HashGridLevel {
            resolution: 8192,
            features: 1,
            table_size: 3,
        },
    ])?;
    assert_eq!(grid.parameter_count(), 33);
    assert_eq!(grid.parameter_bytes(), 132);
    assert_eq!(grid.output_arity(), 3);
    let custom = Encoding::Custom {
        input_arity: 2,
        output_arity: 1,
        source: "fn tail_encode(input:array<Scalar,2>,sample:u32)->array<Scalar,1>{return array<Scalar,1>(input[0]+Scalar(2.0)*input[1]);}\n\
                 fn tail_backward(input:array<Scalar,2>,encoded:array<Scalar,1>,grad:array<Scalar,1>,sample:u32){}"
            .into(),
        forward: "tail_encode".into(),
        backward: "tail_backward".into(),
    };
    let mut desc = NnDesc::new(
        &[
            Encoding::Identity { arity: 1 },
            Encoding::HashGrid {
                grid: grid.clone(),
                auxiliary: "grid_table".into(),
            },
            custom,
        ],
        &[2],
    );
    desc.precision = precision;
    desc.workgroup_size = 7;
    let adam = Adam {
        learning_rate: 0.003,
        beta1: 0.8,
        beta2: 0.93,
        epsilon: 1.0e-6,
    };
    desc.optimizer = Optimizer::Adam(adam);
    desc.auxiliary = vec![grid.auxiliary("grid_table", true)];
    let initial = deterministic(12, 0.31, 0.73);
    let parameters: Vec<f32> = initial
        .iter()
        .map(|&value| match precision {
            Precision::F32 => value,
            Precision::F16 => half::f16::from_f32(value).to_f32(),
        })
        .collect();
    let mut nn = Nn::new(device, desc.clone(), &initial)?;
    assert_eq!(nn.input_size(), 6);
    assert_eq!(nn.encoded_size(), 5);
    let batch = Batch::new(67);
    let mut inputs = deterministic(67 * 6, 0.7, 0.27);
    for (sample, input) in inputs.chunks_exact_mut(6).enumerate() {
        for axis in 0..3 {
            input[axis + 1] = ((sample * (axis * 4 + 3) + axis) % 19) as f32 / 18.0;
        }
    }
    inputs[1..4].copy_from_slice(&[0.0, 0.0, 0.0]);
    inputs[7..10].copy_from_slice(&[1.0, 1.0, 1.0]);
    inputs[13..16].copy_from_slice(&[-0.25, 1.25, 0.37]);
    let targets = deterministic(67 * 2, 0.37, 1.13);
    let mut table: Vec<f32> = deterministic(33, 0.23, 0.45)
        .into_iter()
        .map(|value| value + 0.65)
        .collect();
    for entry in 0..4 {
        table[entry * 2] = 0.0;
    }
    let gradient = hashgrid_gradient(grid.levels(), &parameters, &table, &inputs, &targets);
    let input = SlicedBuffer::initialized(device, &inputs, BufferUsage::STORAGE, "hashgrid input");
    let target =
        SlicedBuffer::initialized(device, &targets, BufferUsage::STORAGE, "hashgrid target");
    let output = SlicedBuffer::zeroed(device, 134, BufferUsage::STORAGE, "hashgrid output");
    let auxiliary =
        SlicedBuffer::initialized(device, &table, BufferUsage::STORAGE, "hashgrid table");
    let inference_input_size = nn.inference_input_buffer_size(batch)?;
    let inference_output_size = nn.inference_output_buffer_size(batch)?;
    let inference_payload = match precision {
        Precision::F32 => bytemuck::cast_slice(&inputs).to_vec(),
        Precision::F16 => half_bytes(&inputs, inference_input_size),
    };
    let inference_input = SlicedBuffer::initialized_bytes(
        device,
        &inference_payload,
        BufferUsage::STORAGE,
        "hashgrid inference input",
    );
    let inference_output = SlicedBuffer::initialized_bytes(
        device,
        &vec![SENTINEL; inference_output_size],
        BufferUsage::STORAGE,
        "hashgrid inference output",
    );
    let short = auxiliary
        .buffer
        .slice(SLICE_PAD..SLICE_PAD + grid.parameter_bytes() - 4);
    assert!(
        nn.inference(
            queue,
            inference_input.slice(),
            inference_output.slice(),
            std::slice::from_ref(&short),
            batch,
        )
        .is_err(),
        "hashgrid inference accepted an undersized table"
    );
    assert!(
        nn.training(
            queue,
            input.slice(),
            target.slice(),
            output.slice(),
            std::slice::from_ref(&short),
            batch,
        )
        .is_err(),
        "hashgrid training accepted an undersized table"
    );
    nn.inference(
        queue,
        inference_input.slice(),
        inference_output.slice(),
        &[auxiliary.slice()],
        batch,
    )?;
    let inference_bytes = read_bytes(device, queue, inference_output.slice())?;
    let inference_actual = match precision {
        Precision::F32 => bytemuck::cast_slice::<u8, f32>(&inference_bytes).to_vec(),
        Precision::F16 => decode_half(&inference_bytes, 134),
    };
    let inference_expected: Vec<f32> =
        hashgrid_forward(grid.levels(), &parameters, &table, &inputs, precision)
            .into_iter()
            .map(|value| value as f32)
            .collect();
    let tolerance = if precision == Precision::F16 {
        0.002
    } else {
        3.0e-6
    };
    assert_close(
        "hashgrid inference",
        &inference_actual,
        &inference_expected,
        tolerance,
        tolerance,
    );
    nn.training(
        queue,
        input.slice(),
        target.slice(),
        output.slice(),
        &[auxiliary.slice()],
        batch,
    )?;
    let expected_output: Vec<f32> =
        hashgrid_forward(grid.levels(), &parameters, &table, &inputs, Precision::F32)
            .into_iter()
            .map(|value| value as f32)
            .collect();
    assert_close(
        "hashgrid training forward",
        &read_f32(device, queue, output.slice())?,
        &expected_output,
        3.0e-6,
        3.0e-6,
    );
    let gradient_bytes = read_bytes(
        device,
        queue,
        nn.gradients()
            .ok_or_else(|| fail("hashgrid gradients missing"))?
            .into(),
    )?;
    let words = bytemuck::cast_slice::<u8, u32>(&gradient_bytes);
    let mut actual_gradient = reduced_dense_gradient(&nn, batch, 7, words);
    let aux_offset = nn.auxiliary_gradient_offset_words(batch)? as usize;
    actual_gradient.extend(
        words[aux_offset..aux_offset + table.len()]
            .iter()
            .map(|&word| f32::from_bits(word) / batch.sample_count as f32),
    );
    assert_close(
        "hashgrid dense and table gradients",
        &actual_gradient,
        &gradient,
        2.0e-6,
        2.0e-4,
    );
    assert!(
        (0..4).any(|entry| actual_gradient[12 + entry * 2].abs() > 1.0e-4),
        "zero XY plane received no derivative"
    );
    for entry in 0..4 {
        assert_eq!(
            actual_gradient[12 + 8 + entry * 2],
            0.0,
            "YZ derivative must contain zero XY factor"
        );
        assert_eq!(
            actual_gradient[12 + 16 + entry * 2],
            0.0,
            "ZX derivative must contain zero XY factor"
        );
    }
    let mut expected_values = [parameters.as_slice(), table.as_slice()].concat();
    let mut expected_moments = vec![0.0; expected_values.len() * 2];
    adam_step(
        &mut expected_values,
        &mut expected_moments,
        &actual_gradient,
        adam,
        1,
    );
    if precision == Precision::F16 {
        for value in &mut expected_values[..12] {
            *value = half::f16::from_f32(*value).to_f32();
        }
    }
    let gpu_parameters = match precision {
        Precision::F32 => read_f32(device, queue, nn.parameters().into())?,
        Precision::F16 => read_half_parameters(device, queue, &nn)?,
    };
    assert_close(
        "hashgrid Adam dense",
        &gpu_parameters,
        &expected_values[..12],
        2.0e-6,
        2.0e-6,
    );
    assert_close(
        "hashgrid Adam table",
        &read_f32(device, queue, auxiliary.slice())?,
        &expected_values[12..],
        2.0e-6,
        2.0e-6,
    );
    assert_close(
        "hashgrid Adam moments",
        &read_f32(device, queue, nn.moments().into())?,
        &expected_moments,
        2.0e-6,
        2.0e-6,
    );
    desc.auxiliary = vec![grid.auxiliary("grid_table", false)];
    let mut frozen_nn = Nn::new(device, desc, &initial)?;
    let frozen = SlicedBuffer::initialized(device, &table, BufferUsage::STORAGE, "frozen hashgrid");
    assert!(
        frozen_nn
            .inference(
                queue,
                inference_input.slice(),
                inference_output.slice(),
                &[short],
                batch,
            )
            .is_err(),
        "frozen hashgrid accepted an undersized table"
    );
    frozen_nn.inference(
        queue,
        inference_input.slice(),
        inference_output.slice(),
        &[frozen.slice()],
        batch,
    )?;
    let frozen_bytes = read_bytes(device, queue, inference_output.slice())?;
    let frozen_actual = match precision {
        Precision::F32 => bytemuck::cast_slice::<u8, f32>(&frozen_bytes).to_vec(),
        Precision::F16 => decode_half(&frozen_bytes, 134),
    };
    assert_close(
        "frozen hashgrid inference",
        &frozen_actual,
        &inference_expected,
        tolerance,
        tolerance,
    );
    frozen_nn.training(
        queue,
        input.slice(),
        target.slice(),
        output.slice(),
        &[frozen.slice()],
        batch,
    )?;
    assert_eq!(
        read_f32(device, queue, frozen.slice())?,
        table,
        "frozen hashgrid table changed"
    );
    assert_close(
        "frozen hashgrid training",
        &read_f32(device, queue, output.slice())?,
        &expected_output,
        3.0e-6,
        3.0e-6,
    );
    for (label, buffer) in [
        ("hashgrid input", &input),
        ("hashgrid target", &target),
        ("hashgrid output", &output),
        ("hashgrid table", &auxiliary),
        ("hashgrid frozen table", &frozen),
        ("hashgrid inference input", &inference_input),
        ("hashgrid inference output", &inference_output),
    ] {
        assert_sentinels(device, queue, buffer, label)?;
    }
    println!(
        "[PASS] {precision:?} hashgrid dense/hash interpolation, gradients, Adam, and frozen tables"
    );
    Ok(())
}

fn half_bytes(values: &[f32], padded_size: usize) -> Vec<u8> {
    let mut bytes = vec![SENTINEL; padded_size];
    for (index, &value) in values.iter().enumerate() {
        bytes[index * 2..index * 2 + 2]
            .copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
    }
    bytes
}

fn decode_half(bytes: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            half::f16::from_bits(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]))
                .to_f32()
        })
        .collect()
}

fn half_dense_forward(parameters: &[f32], inputs: &[f32], batch: usize) -> Vec<f32> {
    let parameters: Vec<half::f16> = parameters
        .iter()
        .map(|&value| half::f16::from_f32(value))
        .collect();
    let inputs: Vec<half::f16> = inputs
        .iter()
        .map(|&value| half::f16::from_f32(value))
        .collect();
    let mut result = Vec::with_capacity(batch * 3);
    for sample in 0..batch {
        let input = &inputs[sample * 2..sample * 2 + 2];
        let mut hidden = [half::f16::ZERO; 3];
        for output in 0..3 {
            let mut value = parameters[6 + output];
            for index in 0..2 {
                value += parameters[output * 2 + index] * input[index];
            }
            hidden[output] = half::f16::from_f32(value.to_f32().tanh());
        }
        for output in 0..3 {
            let mut value = parameters[18 + output];
            for index in 0..3 {
                value += parameters[9 + output * 3 + index] * hidden[index];
            }
            result.push(value.to_f32());
        }
    }
    result
}

fn read_half_parameters(
    device: &Device,
    queue: &mut Queue,
    nn: &Nn,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = read_bytes(device, queue, nn.parameters().into())?;
    Ok(decode_half(&bytes, nn.parameter_count() as usize))
}

fn f16_case(device: &Device, queue: &mut Queue) -> Result<(), Box<dyn Error>> {
    let mut desc = NnDesc::new(&[Encoding::Identity { arity: 2 }], &[3, 3]);
    desc.precision = Precision::F16;
    desc.activation = Activation::Tanh;
    let parameters = deterministic(21, 0.17, 0.31);
    let nn = Nn::new(device, desc, &parameters)?;
    assert_eq!(nn.precision(), Precision::F16);
    assert_eq!(nn.parameter_buffer_size(), 44, "odd half parameter packing");
    let batch = Batch::new(65);
    let inputs = deterministic(130, 0.7, 0.19);
    let input_size = nn.inference_input_buffer_size(batch)?;
    let output_size = nn.inference_output_buffer_size(batch)?;
    assert_eq!(input_size, 260);
    assert_eq!(
        output_size, 392,
        "odd output count must pad total bytes once"
    );
    let input = SlicedBuffer::initialized_bytes(
        device,
        &half_bytes(&inputs, input_size),
        BufferUsage::STORAGE,
        "F16 inference input",
    );
    let output = SlicedBuffer::initialized_bytes(
        device,
        &vec![SENTINEL; output_size],
        BufferUsage::STORAGE,
        "F16 inference output",
    );
    nn.inference(queue, input.slice(), output.slice(), &[], batch)?;
    let bytes = read_bytes(device, queue, output.slice())?;
    let actual = decode_half(&bytes, 195);
    let expected = half_dense_forward(&parameters, &inputs, 65);
    assert_close(
        "F16 multi-layer inference",
        &actual,
        &expected,
        3.0e-3,
        8.0e-3,
    );
    assert_eq!(
        &bytes[390..],
        &[SENTINEL; 2],
        "F16 output touched total padding"
    );
    assert_sentinels(device, queue, &input, "F16 input")?;
    assert_sentinels(device, queue, &output, "F16 output")?;

    let mut custom_desc = NnDesc::new(
        &[Encoding::Custom {
            input_arity: 1,
            output_arity: 1,
            source: "fn encode_half_custom(input:array<Scalar,1>,sample:u32)->array<Scalar,1>{var r:array<Scalar,1>;r[0]=input[0]+Scalar(context[0]);return r;}\n\
                     fn encode_half_custom_backward(input:array<Scalar,1>,encoded:array<Scalar,1>,grad:array<Scalar,1>,sample:u32){}".into(),
            forward: "encode_half_custom".into(),
            backward: "encode_half_custom_backward".into(),
        }],
        &[2, 1],
    );
    custom_desc.precision = Precision::F16;
    custom_desc.activation = Activation::Custom(
        "fn nn_activation(x:Scalar)->Scalar{return x*x;}\n\
         fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return Scalar(2.0)*x;}"
            .into(),
    );
    custom_desc.output = Output::Custom {
        output_size: 1,
        source: "fn nn_output(values:NetworkOutput,sample:u32)->Output{var r:Output;r[0]=values[0]*Scalar(0.5);return r;}\n\
                 fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{var r:NetworkOutput;r[0]=grad[0]*Scalar(0.5);return r;}"
            .into(),
    };
    custom_desc.auxiliary = vec![AuxiliaryDesc::new(
        "context",
        "array<f32>",
        AuxiliaryMode::ReadOnly,
    )];
    let custom_parameters = [0.4, -0.2, 0.1, 0.3, -0.5, 0.07, -0.09];
    let custom = Nn::new(device, custom_desc, &custom_parameters)?;
    let custom_batch = Batch::new(3);
    let custom_input_values = [0.2, -0.4, 0.8];
    let custom_input = SlicedBuffer::initialized_bytes(
        device,
        &half_bytes(
            &custom_input_values,
            custom.inference_input_buffer_size(custom_batch)?,
        ),
        BufferUsage::STORAGE,
        "F16 custom input",
    );
    let custom_output = SlicedBuffer::initialized_bytes(
        device,
        &vec![SENTINEL; custom.inference_output_buffer_size(custom_batch)?],
        BufferUsage::STORAGE,
        "F16 custom output",
    );
    let context = SlicedBuffer::initialized(device, &[0.125], BufferUsage::STORAGE, "F16 context");
    custom.inference(
        queue,
        custom_input.slice(),
        custom_output.slice(),
        &[context.slice()],
        custom_batch,
    )?;
    let custom_values = decode_half(&read_bytes(device, queue, custom_output.slice())?, 3);
    let custom_half_parameters: Vec<half::f16> = custom_parameters
        .iter()
        .map(|&value| half::f16::from_f32(value))
        .collect();
    let context_half = half::f16::from_f32(0.125);
    let custom_expected: Vec<f32> = custom_input_values
        .iter()
        .map(|&input| {
            let encoded = half::f16::from_f32(input) + context_half;
            let h0 = custom_half_parameters[0] * encoded + custom_half_parameters[2];
            let h1 = custom_half_parameters[1] * encoded + custom_half_parameters[3];
            let h0 = h0 * h0;
            let h1 = h1 * h1;
            let raw = custom_half_parameters[4] * h0
                + custom_half_parameters[5] * h1
                + custom_half_parameters[6];
            (raw * half::f16::from_f32(0.5)).to_f32()
        })
        .collect();
    assert_close(
        "F16 scalar custom hooks",
        &custom_values,
        &custom_expected,
        2.0e-3,
        4.0e-3,
    );

    let adam = Adam {
        learning_rate: 0.001,
        ..Adam::default()
    };
    let mut training_desc = NnDesc::new(&[Encoding::Identity { arity: 1 }], &[1]);
    training_desc.precision = Precision::F16;
    training_desc.optimizer = Optimizer::Adam(adam);
    let mut training = Nn::new(device, training_desc, &[0.0, 8.0])?;
    let train_input =
        SlicedBuffer::initialized(device, &[0.0], BufferUsage::STORAGE, "F16 train input");
    let train_target =
        SlicedBuffer::initialized(device, &[0.0], BufferUsage::STORAGE, "F16 train target");
    let train_output = SlicedBuffer::zeroed(device, 1, BufferUsage::STORAGE, "F16 train output");
    let mut quantized = vec![0.0, 8.0];
    let mut moments = vec![0.0; 4];
    for step in 1..=8 {
        let gradient = [0.0, quantized[1]];
        adam_step(&mut quantized, &mut moments, &gradient, adam, step);
        quantized[1] = half::f16::from_f32(quantized[1]).to_f32();
        training.training(
            queue,
            train_input.slice(),
            train_target.slice(),
            train_output.slice(),
            &[],
            Batch::new(1),
        )?;
        let gpu = read_half_parameters(device, queue, &training)?;
        assert_close("F16 quantized Adam", &gpu, &quantized, 0.0, 0.0);
        assert_eq!(gpu[1], 8.0, "sub-ULP update must round away every step");
    }
    let gpu_tiny_moments = read_f32(device, queue, training.moments().into())?;
    assert_close(
        "F16 sub-ULP Adam moments",
        &gpu_tiny_moments,
        &moments,
        2.0e-6,
        1.0e-5,
    );

    let representable_adam = Adam {
        learning_rate: 0.01,
        beta1: 0.8,
        beta2: 0.93,
        epsilon: 1.0e-6,
    };
    let mut representable_desc = NnDesc::new(
        &[Encoding::Custom {
            input_arity: 1,
            output_arity: 1,
            source: "fn encode_trainable(input:array<Scalar,1>,sample:u32)->array<Scalar,1>{var r:array<Scalar,1>;r[0]=input[0]+Scalar(train_aux[0]);return r;}\n\
                     fn encode_trainable_backward(input:array<Scalar,1>,encoded:array<Scalar,1>,grad:array<Scalar,1>,sample:u32){nn_add_train_aux_gradient(0u,grad[0]);}".into(),
            forward: "encode_trainable".into(),
            backward: "encode_trainable_backward".into(),
        }],
        &[1],
    );
    representable_desc.precision = Precision::F16;
    representable_desc.optimizer = Optimizer::Adam(representable_adam);
    representable_desc.auxiliary = vec![AuxiliaryDesc::new(
        "train_aux",
        "array<f32>",
        AuxiliaryMode::Trainable { elements: 1 },
    )];
    let mut representable = Nn::new(device, representable_desc, &[0.5, -0.25])?;
    let input_value = 0.73123;
    let representable_input = SlicedBuffer::initialized(
        device,
        &[input_value],
        BufferUsage::STORAGE,
        "F16 changing input",
    );
    let representable_output =
        SlicedBuffer::zeroed(device, 1, BufferUsage::STORAGE, "F16 changing output");
    let representable_aux =
        SlicedBuffer::initialized(device, &[0.1], BufferUsage::STORAGE, "F16 trainable aux");
    let mut representable_values = vec![0.5, -0.25, 0.1];
    let mut representable_moments = vec![0.0; 6];
    for (index, &target_value) in [0.2, -0.1, 0.4, -0.3].iter().enumerate() {
        let target = SlicedBuffer::initialized(
            device,
            &[target_value],
            BufferUsage::STORAGE,
            "F16 changing target",
        );
        let encoded = input_value + representable_values[2];
        let prediction = representable_values[0] * encoded + representable_values[1];
        let residual = prediction - target_value;
        let gradient = [
            residual * encoded,
            residual,
            residual * representable_values[0],
        ];
        representable.training(
            queue,
            representable_input.slice(),
            target.slice(),
            representable_output.slice(),
            &[representable_aux.slice()],
            Batch::new(1),
        )?;
        let gpu_prediction = read_f32(device, queue, representable_output.slice())?;
        assert_close(
            "F16 training pre-update output",
            &gpu_prediction,
            &[prediction],
            2.0e-6,
            2.0e-6,
        );
        adam_step(
            &mut representable_values,
            &mut representable_moments,
            &gradient,
            representable_adam,
            index as u32 + 1,
        );
        for value in &mut representable_values[..2] {
            *value = half::f16::from_f32(*value).to_f32();
        }
        let gpu_values = read_half_parameters(device, queue, &representable)?;
        let gpu_aux = read_f32(device, queue, representable_aux.slice())?;
        let gpu_moments = read_f32(device, queue, representable.moments().into())?;
        assert_close(
            "F16 representable Adam parameters",
            &gpu_values,
            &representable_values[..2],
            0.0,
            0.0,
        );
        assert_close(
            "F16 representable Adam auxiliary",
            &gpu_aux,
            &representable_values[2..],
            2.0e-6,
            1.0e-5,
        );
        assert_close(
            "F16 representable Adam moments",
            &gpu_moments,
            &representable_moments,
            2.0e-6,
            1.0e-5,
        );
    }

    let edited = half_bytes(&[0.0, 4.0], training.parameter_buffer_size());
    let mut encoder = queue.new_command_encoder();
    encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
    encoder
        .copy()
        .write_buffer_raw(training.parameters(), &edited);
    queue.submit_checkpoint([encoder.finish()])?;
    training.reset_optimizer(queue)?;
    let mut reset_values = vec![0.0, 4.0];
    let mut reset_moments = vec![0.0; 4];
    adam_step(&mut reset_values, &mut reset_moments, &[0.0, 4.0], adam, 1);
    reset_values[1] = half::f16::from_f32(reset_values[1]).to_f32();
    training.training(
        queue,
        train_input.slice(),
        train_target.slice(),
        train_output.slice(),
        &[],
        Batch::new(1),
    )?;
    let reset_gpu = read_half_parameters(device, queue, &training)?;
    assert_close(
        "F16 reset after external edit",
        &reset_gpu,
        &reset_values,
        0.0,
        0.0,
    );
    let reset_gpu_moments = read_f32(device, queue, training.moments().into())?;
    assert_close(
        "F16 reset Adam moments",
        &reset_gpu_moments,
        &reset_moments,
        2.0e-6,
        1.0e-5,
    );
    println!("[PASS] F16 inference packing, scalar hooks, quantized Adam, and reset");
    Ok(())
}

fn select_device() -> Result<(Device, Queue), Box<dyn Error>> {
    let instance = mev::Instance::load()?;
    let selection = instance
        .capabilities()
        .devices
        .iter()
        .enumerate()
        .find_map(|(device_index, device)| {
            device
                .families
                .iter()
                .position(|family| family.queue_flags.contains(QueueFlags::COMPUTE))
                .map(|family_index| (device_index, family_index as u32))
        })
        .ok_or_else(|| fail("no compute-capable GPU queue family found"))?;
    let (device, queues) = instance.new_device(DeviceDesc {
        idx: selection.0,
        queues: &[selection.1],
        features: Features::empty(),
    })?;
    let queue = queues
        .into_iter()
        .next()
        .ok_or_else(|| fail("device returned no requested compute queue"))?;
    Ok((device, queue))
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .ok();
    let (device, mut queue) = select_device()?;
    builtin_case(&device, &mut queue)?;
    custom_case(&device, &mut queue)?;
    mixed_vector_case(&device, &mut queue, Precision::F32)?;
    mixed_vector_case(&device, &mut queue, Precision::F16)?;
    hashgrid_case(&device, &mut queue, Precision::F32)?;
    hashgrid_case(&device, &mut queue, Precision::F16)?;
    f16_case(&device, &mut queue)?;
    println!("All NN smoke checks passed.");
    Ok(())
}
