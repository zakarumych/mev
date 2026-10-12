//! Small dense neural networks evaluated and trained on the GPU.
//!
//! Each sample packs its input vectors in descriptor order. Each vector is encoded
//! separately, and the concatenated encoded vectors feed the first dense layer.
//! [`Precision::F16`] uses half-precision inference
//! input, output, parameters, cache, and arithmetic; training input, output, targets,
//! tape, gradients, moments, and trainable auxiliary parameters remain `f32`. Other
//! auxiliary layouts stay as declared. Dense parameters contain
//! each layer's row-major weights followed by its biases. [`Activation`] applies only to hidden layers;
//! [`Output`] transforms the final layer. See `examples/nn_smoke.rs` for a complete
//! construction, inference, training, and readback flow.
//!
//! Custom WGSL receives fixed-size `Input`, `Encoded`, `NetworkOutput`, `Output`, and
//! `Target` array aliases. `Input` and `Encoded` describe a complete sample; custom
//! encoding hooks use arrays with their own vector arities. `Scalar` is `f16` in
//! half-precision inference and `f32` in training; supplied WGSL is inserted verbatim.
//! Auxiliary declarations and prelude types must keep one data layout across all
//! passes. Hooks run only for valid samples and must not use workgroup or subgroup
//! collectives. Encoding and output hooks may read auxiliaries. A
//! [`AuxiliaryMode::ReadWrite`] hook may also write its binding, so the caller must
//! make custom writes collision-free. Training leaves trainable auxiliaries unchanged
//! through forward and backward; generated `nn_add_NAME_gradient(index, delta)`
//! helpers atomically scatter their gradients, and Adam performs the only parameter
//! mutation.
//!
//! Squared error is `0.5 * sum((prediction - target)^2)` per sample. Loss derivative
//! hooks return an unnormalized derivative. Adam divides dense and auxiliary gradients
//! by the actual sample count once. Each successful nonempty training submission takes
//! one Adam step. F16 training widens dense parameters when loading each layer; Adam
//! reads and widens each current half value before writing its half-precision update.
//! [`Nn::reset_optimizer`] clears moments and permits trainable
//! auxiliary slices to be rebound.
//!
//! Calls accept a compute-capable queue from the construction device and require
//! [`crate::Features::BUFFER_WRITE_COMMAND`] for command-ordered uploads. The caller
//! must synchronize use across distinct queues. Retained gradients contain one subgroup-count word per
//! workgroup, followed by `subgroup_capacity * parameter_count` dense-gradient words
//! per workgroup, packed trainable auxiliary gradients, and one loss word per sample.
//! Only slots below each recorded subgroup count contain dense gradients.

mod hashgrid;
mod shader;

pub use self::hashgrid::{HashGrid, HashGridLevel};

use std::{error::Error as StdError, fmt};

use crate::{
    ArgumentGroupLayout, ArgumentKind, ArgumentLayout, Buffer, BufferDesc, BufferInitDesc,
    BufferSlice, BufferUsage, ComputePipeline, ComputePipelineDesc, Device, DeviceError, Extent3,
    InitMapped, LibraryDesc, LibraryInput, PipelineError, PipelineStages, Queue,
    ShaderLibraryError, ShaderStages,
};

/// Dense parameter and inference arithmetic precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    /// `f32` parameters and inference arithmetic.
    F32,
    /// IEEE binary16 parameters and inference arithmetic.
    F16,
}

/// Hidden-layer activation.
#[derive(Clone, Debug, Default)]
pub enum Activation {
    #[default]
    Identity,
    Relu,
    Sigmoid,
    Tanh,
    /// WGSL defining `nn_activation(x: Scalar) -> Scalar` and
    /// `nn_activation_derivative(x: Scalar, y: Scalar) -> Scalar`.
    Custom(String),
}

/// Converts one input vector to an encoded vector.
#[derive(Clone, Debug)]
pub enum Encoding {
    /// Copies the input vector without changing its arity.
    Identity { arity: u32 },
    /// Factorized XY, YZ, and ZX hash-grid features for a three-dimensional input.
    ///
    /// `auxiliary` names an `array<f32>` read-only or trainable binding holding
    /// the grid parameters. Multiple grids may use the same binding prefix.
    HashGrid { grid: HashGrid, auxiliary: String },
    /// Verbatim WGSL and names of the vector's forward and backward functions.
    ///
    /// With input arity `I` and output arity `O`, the signatures are
    /// `forward(input: array<Scalar, I>, sample: u32) -> array<Scalar, O>` and
    /// `backward(input: array<Scalar, I>, encoded: array<Scalar, O>,
    /// grad: array<Scalar, O>, sample: u32)`. The backward function returns nothing
    /// and may accumulate auxiliary gradients with `nn_add_NAME_gradient`.
    /// Function names must be identifiers outside the reserved `nn_` prefix.
    /// Functions may be supplied in [`NnDesc::prelude`], leaving `source` empty.
    Custom {
        input_arity: u32,
        output_arity: u32,
        source: String,
        forward: String,
        backward: String,
    },
}

impl Encoding {
    pub const fn input_arity(&self) -> u32 {
        match self {
            Self::Identity { arity } => *arity,
            Self::HashGrid { grid, .. } => grid.input_arity(),
            Self::Custom { input_arity, .. } => *input_arity,
        }
    }

    pub const fn output_arity(&self) -> u32 {
        match self {
            Self::Identity { arity } => *arity,
            Self::HashGrid { grid, .. } => grid.output_arity(),
            Self::Custom { output_arity, .. } => *output_arity,
        }
    }
}

/// Converts the last dense layer's values to one output sample.
#[derive(Clone, Debug, Default)]
pub enum Output {
    #[default]
    Identity,
    Activation(Activation),
    /// WGSL defining `nn_output(values: NetworkOutput, sample: u32) -> Output` and
    /// `nn_output_backward(values: NetworkOutput, prediction: Output, grad: Output,
    /// sample: u32) -> NetworkOutput`.
    Custom {
        output_size: u32,
        source: String,
    },
}

/// Per-sample training loss.
#[derive(Clone, Debug, Default)]
pub enum Loss {
    #[default]
    SquaredError,
    /// WGSL defining `nn_loss(prediction: Output, target_value: Target) -> f32` and
    /// `nn_loss_derivative(prediction: Output, target_value: Target) -> Output`.
    Custom { target_size: u32, source: String },
}

/// Adam optimizer settings.
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Optimizer used by training submissions.
#[derive(Clone, Copy, Debug)]
pub enum Optimizer {
    Adam(Adam),
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::Adam(Adam::default())
    }
}

/// Access and training behavior of an auxiliary binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuxiliaryMode {
    Uniform,
    ReadOnly,
    ReadWrite,
    Trainable { elements: u32 },
}

/// Descriptor for one auxiliary shader binding.
#[derive(Clone, Debug)]
pub struct AuxiliaryDesc {
    pub name: String,
    pub wgsl_type: String,
    pub mode: AuxiliaryMode,
}

impl AuxiliaryDesc {
    pub fn new(name: impl Into<String>, wgsl_type: impl Into<String>, mode: AuxiliaryMode) -> Self {
        Self {
            name: name.into(),
            wgsl_type: wgsl_type.into(),
            mode,
        }
    }
}

/// Network construction descriptor.
#[derive(Clone, Debug)]
pub struct NnDesc {
    /// Ordered input vectors and their encoders. Every arity must be positive.
    pub inputs: Vec<Encoding>,
    pub layers: Vec<u32>,
    pub activation: Activation,
    pub output: Output,
    pub loss: Loss,
    pub optimizer: Optimizer,
    pub precision: Precision,
    pub auxiliary: Vec<AuxiliaryDesc>,
    pub workgroup_size: u32,
    pub prelude: String,
}

impl NnDesc {
    pub fn new(inputs: &[Encoding], layers: &[u32]) -> Self {
        Self {
            inputs: inputs.to_vec(),
            layers: layers.to_vec(),
            activation: Activation::default(),
            output: Output::default(),
            loss: Loss::default(),
            optimizer: Optimizer::default(),
            precision: Precision::default(),
            auxiliary: Vec::new(),
            workgroup_size: 64,
            prelude: String::new(),
        }
    }

    pub const fn precision(&self) -> Precision {
        self.precision
    }
}

/// Number of samples in one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    pub sample_count: u32,
}

impl Batch {
    pub const fn new(sample_count: u32) -> Self {
        Self { sample_count }
    }

    pub const fn is_empty(self) -> bool {
        self.sample_count == 0
    }
}

impl From<u32> for Batch {
    fn from(sample_count: u32) -> Self {
        Self::new(sample_count)
    }
}

/// Failure to construct or run a network.
#[derive(Debug)]
pub enum NnError {
    InvalidDescriptor(String),
    InvalidInitialWeights { expected: usize, actual: usize },
    InvalidBinding(String),
    AuxiliaryBindingChanged { index: usize, name: String },
    Shader(ShaderLibraryError),
    Pipeline(PipelineError),
    Device(DeviceError),
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(message) => write!(f, "invalid NN descriptor: {message}"),
            Self::InvalidInitialWeights { expected, actual } => write!(
                f,
                "invalid initial weight count: expected {expected}, got {actual}"
            ),
            Self::InvalidBinding(message) => write!(f, "invalid NN binding: {message}"),
            Self::AuxiliaryBindingChanged { index, name } => write!(
                f,
                "trainable auxiliary binding {index} ({name}) changed; reset optimizer first"
            ),
            Self::Shader(error) => write!(f, "NN shader error: {error}"),
            Self::Pipeline(error) => write!(f, "NN pipeline error: {error}"),
            Self::Device(error) => write!(f, "NN device error: {error}"),
        }
    }
}

impl StdError for NnError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Shader(error) => Some(error),
            Self::Pipeline(error) => Some(error),
            Self::Device(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ShaderLibraryError> for NnError {
    fn from(error: ShaderLibraryError) -> Self {
        Self::Shader(error)
    }
}

impl From<PipelineError> for NnError {
    fn from(error: PipelineError) -> Self {
        Self::Pipeline(error)
    }
}

impl From<DeviceError> for NnError {
    fn from(error: DeviceError) -> Self {
        Self::Device(error)
    }
}

/// Error returned by neural-network construction and execution.
pub type Error = NnError;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NnConfig {
    sample_count: u32,
    workgroup_count: u32,
    parameter_count: u32,
    subgroup_capacity: u32,
    tape_stride: u32,
    dense_gradient_offset: u32,
    auxiliary_gradient_offset: u32,
    loss_offset: u32,
    step: u32,
    trainable_count: u32,
    encoded_size: u32,
    output_size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct LayerSpec {
    pub input_size: u32,
    pub output_size: u32,
    pub parameter_offset: u32,
    pub tape_offset: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct TrainableAuxSpec {
    pub auxiliary_index: usize,
    pub element_offset: u32,
    pub elements: u32,
}

/// GPU-resident network and optimizer state.
pub struct Nn {
    device: Device,
    desc: NnDesc,
    inference_pipeline: ComputePipeline,
    training_forward_pipeline: ComputePipeline,
    backward_pipeline: ComputePipeline,
    adam_pipeline: ComputePipeline,
    parameters: Buffer,
    config: Buffer,
    moments: Buffer,
    tape: Option<Buffer>,
    gradients: Option<Buffer>,
    batch_capacity: u32,
    step: u32,
    input_size: u32,
    encoded_size: u32,
    network_output_size: u32,
    output_size: u32,
    target_size: u32,
    parameter_count: u32,
    tape_stride: u32,
    subgroup_capacity: u32,
    trainable_count: u32,
    trainable_auxiliary: Vec<TrainableAuxSpec>,
    trainable_bindings: Vec<Option<BufferSlice<'static>>>,
    auxiliary_minimum_sizes: Vec<usize>,
    limits: crate::generic::ComputeKernelLimits,
}

impl Nn {
    /// Creates a network from dense parameters in layer order.
    ///
    /// Each layer contributes `input_width * output_width` row-major weights, then
    /// `output_width` biases. Initial `f32` values are converted to binary16 when
    /// [`NnDesc::precision`] is [`Precision::F16`].
    pub fn new(device: &Device, desc: NnDesc, initial_weights: &[f32]) -> Result<Self, NnError> {
        let validated = validate_desc(device, &desc)?;
        if initial_weights.len() != validated.parameter_count as usize {
            return Err(NnError::InvalidInitialWeights {
                expected: validated.parameter_count as usize,
                actual: initial_weights.len(),
            });
        }

        let spec = ShaderSpec {
            desc: &desc,
            input_size: validated.input_size,
            encoded_size: validated.encoded_size,
            network_output_size: validated.network_output_size,
            output_size: validated.output_size,
            target_size: validated.target_size,
            parameter_count: validated.parameter_count,
            max_layer_width: validated.max_layer_width,
            max_layer_parameter_count: validated.max_layer_parameter_count,
            subgroup_capacity: validated.subgroup_capacity,
            tape_stride: validated.tape_stride,
            layers: &validated.layers,
            trainable_auxiliary: &validated.trainable_auxiliary,
        };
        let generated = generate_shader(&spec)?;
        let auxiliary_layout = auxiliary_layout(&desc);
        let forward_auxiliary_access = auxiliary_read_only(&desc, false);
        let adam_auxiliary_access = auxiliary_read_only(&desc, true);
        let mut inference_access = vec![false, true, true, false];
        inference_access.extend_from_slice(&forward_auxiliary_access);
        let mut training_forward_access = vec![false, true, true, false, false];
        training_forward_access.extend_from_slice(&forward_auxiliary_access);
        let mut backward_access = vec![false, true, true, true, true, false];
        backward_access.extend_from_slice(&forward_auxiliary_access);
        let mut adam_access = vec![false, false, false, false];
        adam_access.extend_from_slice(&adam_auxiliary_access);
        let inference_pipeline = make_pipeline(
            device,
            "NN inference",
            &generated.inference,
            "nn_inference",
            desc.workgroup_size,
            4,
            &auxiliary_layout,
            &inference_access,
        )?;
        let training_forward_pipeline = make_pipeline(
            device,
            "NN training forward",
            &generated.training_forward,
            "nn_training_forward",
            desc.workgroup_size,
            5,
            &auxiliary_layout,
            &training_forward_access,
        )?;
        let backward_pipeline = make_pipeline(
            device,
            "NN backward",
            &generated.backward,
            "nn_backward",
            desc.workgroup_size,
            6,
            &auxiliary_layout,
            &backward_access,
        )?;
        let adam_pipeline = make_pipeline(
            device,
            "NN Adam",
            &generated.adam,
            "nn_adam",
            desc.workgroup_size,
            4,
            &auxiliary_layout,
            &adam_access,
        )?;

        let parameter_usage =
            BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;
        let parameters = match desc.precision {
            Precision::F32 => device.new_buffer_init(BufferInitDesc {
                name: "NN parameters",
                data: initial_weights,
                usage: parameter_usage,
            }),
            Precision::F16 => {
                let mut values: Vec<u16> = initial_weights
                    .iter()
                    .map(|&value| half::f16::from_f32(value).to_bits())
                    .collect();
                if values.len() % 2 != 0 {
                    values.push(0);
                }
                device.new_buffer_init(BufferInitDesc {
                    name: "NN parameters",
                    data: values.as_slice(),
                    usage: parameter_usage,
                })
            }
        };
        let config = device.new_buffer_init(BufferInitDesc {
            name: "NN config",
            data: std::slice::from_ref(&<NnConfig as bytemuck::Zeroable>::zeroed()),
            usage: BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST,
        });
        let moment_size = checked_bytes(
            checked_mul(
                checked_add(
                    validated.parameter_count,
                    validated.trainable_count,
                    "moment count",
                )?,
                2,
                "moment count",
            )?,
            "moment buffer size",
        )?;
        let moments = device.new_buffer_init(BufferInitDesc {
            name: "NN Adam moments",
            data: InitMapped {
                size: moment_size.max(4),
                write: |bytes: &mut [u8]| bytes.fill(0),
            },
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        });

        Ok(Self {
            device: device.clone(),
            desc,
            inference_pipeline,
            training_forward_pipeline,
            backward_pipeline,
            adam_pipeline,
            parameters,
            config,
            moments,
            tape: None,
            gradients: None,
            batch_capacity: 0,
            step: 0,
            input_size: validated.input_size,
            encoded_size: validated.encoded_size,
            network_output_size: validated.network_output_size,
            output_size: validated.output_size,
            target_size: validated.target_size,
            parameter_count: validated.parameter_count,
            tape_stride: validated.tape_stride,
            subgroup_capacity: validated.subgroup_capacity,
            trainable_count: validated.trainable_count,
            trainable_bindings: vec![None; validated.trainable_auxiliary.len()],
            trainable_auxiliary: validated.trainable_auxiliary,
            auxiliary_minimum_sizes: validated.auxiliary_minimum_sizes,
            limits: validated.limits,
        })
    }

    /// Dense weights followed by biases for each layer in descriptor order.
    /// F16 storage is padded to a 4-byte size when the logical element count is odd.
    /// Reset the optimizer after modifying this buffer outside [`Nn::training`].
    pub fn parameters(&self) -> &Buffer {
        &self.parameters
    }

    /// Interleaved Adam `m, v` values for dense then trainable auxiliary parameters.
    pub fn moments(&self) -> &Buffer {
        &self.moments
    }

    /// Saved encoded, per-layer, and predicted values for the current scratch capacity.
    pub fn tape(&self) -> Option<&Buffer> {
        self.tape.as_ref()
    }

    /// Subgroup counts, dense gradients, auxiliary gradients, and per-sample losses.
    pub fn gradients(&self) -> Option<&Buffer> {
        self.gradients.as_ref()
    }

    pub const fn parameter_count(&self) -> u32 {
        self.parameter_count
    }

    /// Number of packed scalars in one raw input sample.
    pub const fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Number of concatenated encoded scalars entering the first dense layer.
    pub const fn encoded_size(&self) -> u32 {
        self.encoded_size
    }

    pub fn parameter_buffer_size(&self) -> usize {
        self.parameters.size()
    }

    pub const fn precision(&self) -> Precision {
        self.desc.precision
    }

    /// Required packed inference input bytes, including F16's 4-byte tail padding.
    pub fn inference_input_buffer_size(&self, batch: Batch) -> Result<usize, NnError> {
        checked_scalar_bytes(
            checked_mul(self.input_size, batch.sample_count, "inference input size")?,
            "inference input size",
            self.desc.precision,
        )
    }

    /// Required packed inference output bytes, including F16's 4-byte tail padding.
    pub fn inference_output_buffer_size(&self, batch: Batch) -> Result<usize, NnError> {
        checked_scalar_bytes(
            checked_mul(
                self.output_size,
                batch.sample_count,
                "inference output size",
            )?,
            "inference output size",
            self.desc.precision,
        )
    }

    pub const fn moment_buffer_size(&self) -> usize {
        (self.parameter_count as usize + self.trainable_count as usize) * 8
    }

    pub const fn tape_stride(&self) -> u32 {
        self.tape_stride
    }

    pub const fn subgroup_capacity(&self) -> u32 {
        self.subgroup_capacity
    }

    pub const fn trainable_auxiliary_count(&self) -> u32 {
        self.trainable_count
    }

    pub fn tape_buffer_size(&self, batch: Batch) -> Result<usize, NnError> {
        checked_bytes(
            checked_mul(self.tape_stride, batch.sample_count, "tape size")?,
            "tape size",
        )
    }

    pub fn gradient_buffer_size(&self, batch: Batch) -> Result<usize, NnError> {
        if batch.is_empty() {
            return Ok(0);
        }
        let groups = div_ceil(batch.sample_count, self.desc.workgroup_size);
        checked_bytes(
            self.gradient_layout(groups, batch.sample_count)?.1,
            "gradient size",
        )
    }

    pub const fn workgroup_count(&self, batch: Batch) -> u32 {
        div_ceil(batch.sample_count, self.desc.workgroup_size)
    }

    pub const fn dense_gradient_offset_words(&self, batch: Batch) -> u32 {
        self.workgroup_count(batch)
    }

    pub fn auxiliary_gradient_offset_words(&self, batch: Batch) -> Result<u32, NnError> {
        self.auxiliary_gradient_offset(self.workgroup_count(batch))
    }

    pub fn loss_offset_words(&self, batch: Batch) -> Result<u32, NnError> {
        checked_add(
            self.auxiliary_gradient_offset_words(batch)?,
            self.trainable_count,
            "loss offset",
        )
    }

    /// Submits inference and returns its checkpoint epoch, or `None` for an empty batch.
    /// Input and output elements use [`Self::precision`].
    pub fn inference(
        &self,
        queue: &mut Queue,
        input: BufferSlice<'_>,
        output: BufferSlice<'_>,
        auxiliary: &[BufferSlice<'_>],
        batch: Batch,
    ) -> Result<Option<u64>, NnError> {
        self.validate_queue(queue)?;
        if batch.is_empty() {
            return Ok(None);
        }
        let groups = self.validate_call(&input, None, &output, auxiliary, batch)?;
        let config = self.config(groups, batch.sample_count, self.step, false)?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        {
            let mut copy = encoder.copy();
            copy.write_buffer(&self.config, &config);
        }
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        encoder.barrier(
            PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.inference_pipeline);
            let mut bindings = vec![
                (ArgumentKind::UniformBuffer, (&self.config).into()),
                (ArgumentKind::StorageBuffer, (&self.parameters).into()),
                (ArgumentKind::StorageBuffer, input),
                (ArgumentKind::StorageBuffer, output),
            ];
            self.append_auxiliary(&mut bindings, auxiliary);
            compute.with_buffer_arguments(0, &bindings);
            compute.dispatch(Extent3::new(groups, 1, 1));
        }
        Ok(Some(queue.submit_checkpoint([encoder.finish()])?))
    }

    /// Submits forward, backward, and Adam passes in one checkpoint.
    ///
    /// Input, targets, and output are packed `f32` even for an F16 network. `output`
    /// contains predictions made before this call's parameter update.
    pub fn training(
        &mut self,
        queue: &mut Queue,
        input: BufferSlice<'_>,
        targets: BufferSlice<'_>,
        output: BufferSlice<'_>,
        auxiliary: &[BufferSlice<'_>],
        batch: Batch,
    ) -> Result<Option<u64>, NnError> {
        self.validate_queue(queue)?;
        if batch.is_empty() {
            return Ok(None);
        }
        let groups = self.validate_call(&input, Some(&targets), &output, auxiliary, batch)?;
        let pending_bindings = self.validate_trainable_identity(auxiliary)?;
        self.ensure_capacity(batch.sample_count)?;
        let tape = self
            .tape
            .as_ref()
            .ok_or_else(|| NnError::InvalidDescriptor("missing tape".into()))?;
        let gradients = self
            .gradients
            .as_ref()
            .ok_or_else(|| NnError::InvalidDescriptor("missing gradients".into()))?;
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| NnError::InvalidDescriptor("Adam step exceeds u32 range".into()))?;
        let config = self.config(groups, batch.sample_count, next_step, true)?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        {
            let mut copy = encoder.copy();
            copy.write_buffer(&self.config, &config);
        }
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        encoder.barrier(
            PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.training_forward_pipeline);
            let mut bindings = vec![
                (ArgumentKind::UniformBuffer, (&self.config).into()),
                (ArgumentKind::StorageBuffer, (&self.parameters).into()),
                (ArgumentKind::StorageBuffer, input.clone()),
                (ArgumentKind::StorageBuffer, output),
                (ArgumentKind::StorageBuffer, tape.into()),
            ];
            self.append_auxiliary(&mut bindings, auxiliary);
            compute.with_buffer_arguments(0, &bindings);
            compute.dispatch(Extent3::new(groups, 1, 1));
        }
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        if self.trainable_count != 0 {
            let auxiliary_offset = self.auxiliary_gradient_offset(groups)? as usize * 4;
            let auxiliary_end = auxiliary_offset + self.trainable_count as usize * 4;
            encoder
                .copy()
                .fill_buffer(gradients.slice(auxiliary_offset..auxiliary_end), 0);
        }
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        encoder.barrier(
            PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.backward_pipeline);
            let mut bindings = vec![
                (ArgumentKind::UniformBuffer, (&self.config).into()),
                (ArgumentKind::StorageBuffer, (&self.parameters).into()),
                (ArgumentKind::StorageBuffer, input),
                (ArgumentKind::StorageBuffer, targets),
                (ArgumentKind::StorageBuffer, tape.into()),
                (ArgumentKind::StorageBuffer, gradients.into()),
            ];
            self.append_auxiliary(&mut bindings, auxiliary);
            compute.with_buffer_arguments(0, &bindings);
            compute.dispatch(Extent3::new(groups, 1, 1));
        }
        encoder.barrier(
            PipelineStages::COMPUTE_SHADER,
            PipelineStages::COMPUTE_SHADER,
        );
        {
            let update_count =
                checked_add(self.parameter_count, self.trainable_count, "update count")?;
            let update_groups = div_ceil(update_count, self.desc.workgroup_size);
            let mut compute = encoder.compute();
            compute.with_pipeline(&self.adam_pipeline);
            let mut bindings = vec![
                (ArgumentKind::UniformBuffer, (&self.config).into()),
                (ArgumentKind::StorageBuffer, (&self.parameters).into()),
                (ArgumentKind::StorageBuffer, gradients.into()),
                (ArgumentKind::StorageBuffer, (&self.moments).into()),
            ];
            self.append_auxiliary(&mut bindings, auxiliary);
            compute.with_buffer_arguments(0, &bindings);
            compute.dispatch(Extent3::new(update_groups, 1, 1));
        }
        let epoch = queue.submit_checkpoint([encoder.finish()])?;
        self.step = next_step;
        for (slot, binding) in self.trainable_bindings.iter_mut().zip(pending_bindings) {
            if slot.is_none() {
                *slot = Some(binding);
            }
        }
        Ok(Some(epoch))
    }

    /// Clears Adam moments and releases trainable auxiliary binding identities.
    pub fn reset_optimizer(&mut self, queue: &mut Queue) -> Result<Option<u64>, NnError> {
        self.validate_queue(queue)?;
        let mut encoder = queue.new_command_encoder();
        encoder.barrier(PipelineStages::COMPUTE_SHADER, PipelineStages::TRANSFER);
        encoder.copy().fill_buffer(&self.moments, 0);
        encoder.barrier(PipelineStages::TRANSFER, PipelineStages::COMPUTE_SHADER);
        let epoch = queue.submit_checkpoint([encoder.finish()])?;
        self.step = 0;
        self.trainable_bindings.fill(None);
        Ok(Some(epoch))
    }

    fn validate_queue(&self, queue: &Queue) -> Result<(), NnError> {
        if queue.device() != &self.device {
            return Err(NnError::InvalidBinding(
                "queue belongs to a different device".into(),
            ));
        }
        Ok(())
    }

    fn ensure_capacity(&mut self, sample_count: u32) -> Result<(), NnError> {
        if sample_count <= self.batch_capacity {
            return Ok(());
        }
        let groups = div_ceil(sample_count, self.desc.workgroup_size);
        let tape_size = checked_bytes(
            checked_mul(self.tape_stride, sample_count, "tape size")?,
            "tape size",
        )?;
        let gradient_size = checked_bytes(
            self.gradient_layout(groups, sample_count)?.1,
            "gradient size",
        )?;
        if tape_size > self.limits.max_storage_buffer_range
            || gradient_size > self.limits.max_storage_buffer_range
        {
            return Err(NnError::InvalidBinding(
                "batch scratch exceeds device storage binding range".into(),
            ));
        }
        self.tape = Some(self.device.new_buffer(BufferDesc {
            name: "NN tape",
            size: tape_size,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
        }));
        self.gradients = Some(self.device.new_buffer(BufferDesc {
            name: "NN gradients and losses",
            size: gradient_size,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        }));
        self.batch_capacity = sample_count;
        Ok(())
    }

    fn append_auxiliary<'a>(
        &self,
        bindings: &mut Vec<(ArgumentKind, BufferSlice<'a>)>,
        auxiliary: &[BufferSlice<'a>],
    ) {
        bindings.extend(
            self.desc
                .auxiliary
                .iter()
                .zip(auxiliary)
                .map(|(desc, slice)| {
                    let kind = if desc.mode == AuxiliaryMode::Uniform {
                        ArgumentKind::UniformBuffer
                    } else {
                        ArgumentKind::StorageBuffer
                    };
                    (kind, slice.clone())
                }),
        );
    }

    fn validate_call(
        &self,
        input: &BufferSlice<'_>,
        targets: Option<&BufferSlice<'_>>,
        output: &BufferSlice<'_>,
        auxiliary: &[BufferSlice<'_>],
        batch: Batch,
    ) -> Result<u32, NnError> {
        let groups = div_ceil(batch.sample_count, self.desc.workgroup_size);
        if groups > self.limits.max_dispatch_count {
            return Err(NnError::InvalidBinding(format!(
                "batch needs {groups} workgroups, device limit is {}",
                self.limits.max_dispatch_count
            )));
        }
        let inference = targets.is_none();
        let io_precision = if inference {
            self.desc.precision
        } else {
            Precision::F32
        };
        validate_slice(
            input,
            checked_scalar_bytes(
                checked_mul(self.input_size, batch.sample_count, "input size")?,
                "input size",
                io_precision,
            )?,
            ArgumentKind::StorageBuffer,
            &self.limits,
            "input",
        )?;
        validate_slice(
            output,
            checked_scalar_bytes(
                checked_mul(self.output_size, batch.sample_count, "output size")?,
                "output size",
                io_precision,
            )?,
            ArgumentKind::StorageBuffer,
            &self.limits,
            "output",
        )?;
        if self.desc.precision == Precision::F16 && inference {
            if input.size() % 4 != 0 || output.size() % 4 != 0 {
                return Err(NnError::InvalidBinding(
                    "F16 inference input and output slice sizes must be multiples of 4 bytes"
                        .into(),
                ));
            }
        }
        if auxiliary.len() != self.desc.auxiliary.len() {
            return Err(NnError::InvalidBinding(format!(
                "expected {} auxiliary bindings, got {}",
                self.desc.auxiliary.len(),
                auxiliary.len()
            )));
        }
        for (index, (desc, slice)) in self.desc.auxiliary.iter().zip(auxiliary).enumerate() {
            let (kind, required) = match desc.mode {
                AuxiliaryMode::Uniform => (ArgumentKind::UniformBuffer, 1),
                AuxiliaryMode::ReadOnly | AuxiliaryMode::ReadWrite => {
                    (ArgumentKind::StorageBuffer, 1)
                }
                AuxiliaryMode::Trainable { elements } => (
                    ArgumentKind::StorageBuffer,
                    checked_bytes(elements, "trainable auxiliary size")?,
                ),
            };
            validate_slice(
                slice,
                required.max(self.auxiliary_minimum_sizes[index]),
                kind,
                &self.limits,
                &format!("auxiliary {index} ({})", desc.name),
            )?;
            if self.desc.precision == Precision::F16
                && kind == ArgumentKind::StorageBuffer
                && slice.size() % 4 != 0
            {
                return Err(NnError::InvalidBinding(format!(
                    "auxiliary {index} ({}) storage slice size must be a multiple of 4 bytes for F16",
                    desc.name
                )));
            }
            if matches!(desc.mode, AuxiliaryMode::Trainable { .. }) && slice.size() != required {
                return Err(NnError::InvalidBinding(format!(
                    "trainable auxiliary {index} ({}) must be exactly {required} bytes",
                    desc.name
                )));
            }
        }
        if let Some(targets) = targets {
            validate_slice(
                targets,
                checked_bytes(
                    checked_mul(self.target_size, batch.sample_count, "target size")?,
                    "target size",
                )?,
                ArgumentKind::StorageBuffer,
                &self.limits,
                "targets",
            )?;
            reject_overlap(output, targets, "output", "targets")?;
        }
        reject_overlap(output, input, "output", "input")?;
        for (index, (desc, slice)) in self.desc.auxiliary.iter().zip(auxiliary).enumerate() {
            reject_overlap(output, slice, "output", &format!("auxiliary {index}"))?;
            if matches!(
                desc.mode,
                AuxiliaryMode::ReadWrite | AuxiliaryMode::Trainable { .. }
            ) {
                reject_overlap(slice, input, &format!("auxiliary {index}"), "input")?;
                if let Some(targets) = targets {
                    reject_overlap(slice, targets, &format!("auxiliary {index}"), "targets")?;
                }
            }
            for (other_index, (other_desc, other)) in self
                .desc
                .auxiliary
                .iter()
                .zip(auxiliary)
                .enumerate()
                .skip(index + 1)
            {
                if !matches!(desc.mode, AuxiliaryMode::Uniform | AuxiliaryMode::ReadOnly)
                    || !matches!(
                        other_desc.mode,
                        AuxiliaryMode::Uniform | AuxiliaryMode::ReadOnly
                    )
                {
                    reject_overlap(
                        slice,
                        other,
                        &format!("auxiliary {index}"),
                        &format!("auxiliary {other_index}"),
                    )?;
                }
            }
        }
        self.reject_internal_write_alias(output, "output")?;
        for (index, (desc, slice)) in self.desc.auxiliary.iter().zip(auxiliary).enumerate() {
            if matches!(
                desc.mode,
                AuxiliaryMode::ReadWrite | AuxiliaryMode::Trainable { .. }
            ) {
                self.reject_internal_write_alias(slice, &format!("auxiliary {index}"))?;
            }
        }
        if let Some(targets) = targets {
            self.reject_training_read_alias(input, "input")?;
            self.reject_training_read_alias(targets, "targets")?;
            for (index, slice) in auxiliary.iter().enumerate() {
                self.reject_training_read_alias(slice, &format!("auxiliary {index}"))?;
            }
        }
        Ok(groups)
    }

    fn reject_training_read_alias(
        &self,
        slice: &BufferSlice<'_>,
        label: &str,
    ) -> Result<(), NnError> {
        reject_overlap(slice, &(&self.moments).into(), label, "NN moments")?;
        if let Some(tape) = &self.tape {
            reject_overlap(slice, &tape.into(), label, "NN tape")?;
        }
        if let Some(gradients) = &self.gradients {
            reject_overlap(slice, &gradients.into(), label, "NN gradients")?;
        }
        Ok(())
    }

    fn reject_internal_write_alias(
        &self,
        slice: &BufferSlice<'_>,
        label: &str,
    ) -> Result<(), NnError> {
        reject_overlap(slice, &(&self.parameters).into(), label, "NN parameters")?;
        reject_overlap(slice, &(&self.config).into(), label, "NN config")?;
        reject_overlap(slice, &(&self.moments).into(), label, "NN moments")?;
        if let Some(tape) = &self.tape {
            reject_overlap(slice, &tape.into(), label, "NN tape")?;
        }
        if let Some(gradients) = &self.gradients {
            reject_overlap(slice, &gradients.into(), label, "NN gradients")?;
        }
        Ok(())
    }

    fn validate_trainable_identity(
        &self,
        auxiliary: &[BufferSlice<'_>],
    ) -> Result<Vec<BufferSlice<'static>>, NnError> {
        let mut pending = Vec::with_capacity(self.trainable_auxiliary.len());
        for (binding_index, spec) in self.trainable_auxiliary.iter().enumerate() {
            let binding = auxiliary[spec.auxiliary_index].clone().into_owned();
            if let Some(expected) = &self.trainable_bindings[binding_index] {
                if expected != &binding {
                    return Err(NnError::AuxiliaryBindingChanged {
                        index: spec.auxiliary_index,
                        name: self.desc.auxiliary[spec.auxiliary_index].name.clone(),
                    });
                }
            }
            pending.push(binding);
        }
        Ok(pending)
    }

    fn gradient_layout(&self, groups: u32, samples: u32) -> Result<(u32, u32), NnError> {
        let dense_slots = checked_mul(
            checked_mul(groups, self.subgroup_capacity, "dense gradient slots")?,
            self.parameter_count,
            "dense gradient slots",
        )?;
        let auxiliary_offset = checked_add(groups, dense_slots, "gradient layout")?;
        let loss_offset = checked_add(auxiliary_offset, self.trainable_count, "gradient layout")?;
        Ok((
            auxiliary_offset,
            checked_add(loss_offset, samples, "gradient layout")?,
        ))
    }

    fn auxiliary_gradient_offset(&self, groups: u32) -> Result<u32, NnError> {
        Ok(self.gradient_layout(groups, 0)?.0)
    }

    fn config(
        &self,
        groups: u32,
        samples: u32,
        step: u32,
        training: bool,
    ) -> Result<NnConfig, NnError> {
        let auxiliary_gradient_offset = if training {
            self.auxiliary_gradient_offset(groups)?
        } else {
            0
        };
        let Optimizer::Adam(adam) = self.desc.optimizer;
        Ok(NnConfig {
            sample_count: samples,
            workgroup_count: groups,
            parameter_count: self.parameter_count,
            subgroup_capacity: self.subgroup_capacity,
            tape_stride: self.tape_stride,
            dense_gradient_offset: if training { groups } else { 0 },
            auxiliary_gradient_offset,
            loss_offset: if training {
                checked_add(
                    auxiliary_gradient_offset,
                    self.trainable_count,
                    "loss offset",
                )?
            } else {
                0
            },
            step,
            trainable_count: self.trainable_count,
            encoded_size: self.encoded_size,
            output_size: self.output_size,
            learning_rate: adam.learning_rate,
            beta1: adam.beta1,
            beta2: adam.beta2,
            epsilon: adam.epsilon,
        })
    }
}

struct ValidatedDesc {
    input_size: u32,
    encoded_size: u32,
    network_output_size: u32,
    output_size: u32,
    target_size: u32,
    parameter_count: u32,
    max_layer_width: u32,
    max_layer_parameter_count: u32,
    subgroup_capacity: u32,
    tape_stride: u32,
    trainable_count: u32,
    layers: Vec<LayerSpec>,
    trainable_auxiliary: Vec<TrainableAuxSpec>,
    auxiliary_minimum_sizes: Vec<usize>,
    limits: crate::generic::ComputeKernelLimits,
}

fn validate_inputs(inputs: &[Encoding]) -> Result<(u32, u32), NnError> {
    if inputs.is_empty() {
        return Err(NnError::InvalidDescriptor(
            "at least one input vector is required".into(),
        ));
    }
    let mut input_size = 0;
    let mut encoded_size = 0;
    for (index, encoding) in inputs.iter().enumerate() {
        if encoding.input_arity() == 0 || encoding.output_arity() == 0 {
            return Err(NnError::InvalidDescriptor(format!(
                "input vector {index} must have nonzero input and output arities"
            )));
        }
        if let Encoding::Custom {
            forward, backward, ..
        } = encoding
        {
            for name in [forward, backward] {
                if !valid_identifier(name)
                    || name == "_"
                    || name.starts_with("__")
                    || name.starts_with("nn_")
                {
                    return Err(NnError::InvalidDescriptor(format!(
                        "invalid or reserved encoding function name `{name}` for input vector {index}"
                    )));
                }
            }
        }
        input_size = checked_add(input_size, encoding.input_arity(), "input size")?;
        encoded_size = checked_add(encoded_size, encoding.output_arity(), "encoded size")?;
    }
    Ok((input_size, encoded_size))
}

fn validate_hashgrid_bindings(desc: &NnDesc) -> Result<Vec<usize>, NnError> {
    let mut minimum_sizes = vec![1; desc.auxiliary.len()];
    for (input_index, encoding) in desc.inputs.iter().enumerate() {
        let Encoding::HashGrid { grid, auxiliary } = encoding else {
            continue;
        };
        grid.validate_precision(desc.precision)?;
        let Some((index, binding)) = desc
            .auxiliary
            .iter()
            .enumerate()
            .find(|(_, binding)| binding.name == *auxiliary)
        else {
            return Err(NnError::InvalidDescriptor(format!(
                "hash grid input {input_index} names missing auxiliary `{auxiliary}`"
            )));
        };
        if binding.wgsl_type.trim() != "array<f32>" {
            return Err(NnError::InvalidDescriptor(format!(
                "hash grid auxiliary `{auxiliary}` must have WGSL type array<f32>"
            )));
        }
        match binding.mode {
            AuxiliaryMode::ReadOnly => {}
            AuxiliaryMode::Trainable { elements } if elements >= grid.parameter_count() => {}
            AuxiliaryMode::Trainable { .. } => {
                return Err(NnError::InvalidDescriptor(format!(
                    "hash grid auxiliary `{auxiliary}` needs at least {} trainable elements",
                    grid.parameter_count()
                )));
            }
            AuxiliaryMode::Uniform | AuxiliaryMode::ReadWrite => {
                return Err(NnError::InvalidDescriptor(format!(
                    "hash grid auxiliary `{auxiliary}` must be read-only or trainable"
                )));
            }
        }
        minimum_sizes[index] = minimum_sizes[index].max(grid.parameter_bytes());
    }
    Ok(minimum_sizes)
}

fn validate_desc(device: &Device, desc: &NnDesc) -> Result<ValidatedDesc, NnError> {
    let (input_size, encoded_size) = validate_inputs(&desc.inputs)?;
    if desc.layers.is_empty() || desc.layers.contains(&0) {
        return Err(NnError::InvalidDescriptor(
            "at least one nonzero layer width is required".into(),
        ));
    }
    if desc.workgroup_size == 0 {
        return Err(NnError::InvalidDescriptor(
            "workgroup size must be nonzero".into(),
        ));
    }
    let network_output_size = desc.layers[desc.layers.len() - 1];
    let output_size = match &desc.output {
        Output::Identity | Output::Activation(_) => network_output_size,
        Output::Custom { output_size, .. } if *output_size != 0 => *output_size,
        Output::Custom { .. } => {
            return Err(NnError::InvalidDescriptor(
                "output size must be nonzero".into(),
            ));
        }
    };
    let target_size = match &desc.loss {
        Loss::SquaredError => output_size,
        Loss::Custom { target_size, .. } if *target_size != 0 => *target_size,
        Loss::Custom { .. } => {
            return Err(NnError::InvalidDescriptor(
                "target size must be nonzero".into(),
            ));
        }
    };
    if let Output::Activation(Activation::Custom(output)) = &desc.output {
        match &desc.activation {
            Activation::Custom(hidden) if hidden == output => {}
            _ => {
                return Err(NnError::InvalidDescriptor(
                    "a custom output activation must match the hidden custom activation; use Output::Custom for distinct functions".into(),
            ));
            }
        }
    }
    let Optimizer::Adam(adam) = desc.optimizer;
    if !adam.learning_rate.is_finite()
        || adam.learning_rate <= 0.0
        || !adam.beta1.is_finite()
        || !(0.0..1.0).contains(&adam.beta1)
        || !adam.beta2.is_finite()
        || !(0.0..1.0).contains(&adam.beta2)
        || !adam.epsilon.is_finite()
        || adam.epsilon <= 0.0
    {
        return Err(NnError::InvalidDescriptor("invalid Adam settings".into()));
    }
    let limits = device.compute_kernel_limits();
    if desc.precision == Precision::F16 && !limits.shader_f16 {
        return Err(NnError::InvalidDescriptor(
            "device does not support f16 shaders".into(),
        ));
    }
    if desc.workgroup_size > limits.max_workgroup_size {
        return Err(NnError::InvalidDescriptor(format!(
            "workgroup size {} exceeds device limit {}",
            desc.workgroup_size, limits.max_workgroup_size
        )));
    }
    if !limits.subgroup_arithmetic {
        return Err(NnError::InvalidDescriptor(
            "device does not support subgroup arithmetic".into(),
        ));
    }
    let subgroup_capacity = limits
        .max_subgroups_per_workgroup
        .unwrap_or(desc.workgroup_size)
        .min(desc.workgroup_size);
    let max_layer_width = desc.layers.iter().copied().fold(encoded_size, u32::max);
    let mut max_layer_parameter_count = 0;
    let mut cached_input = encoded_size;
    for &cached_output in &desc.layers {
        max_layer_parameter_count = max_layer_parameter_count.max(checked_mul(
            checked_add(cached_input, 1, "workgroup cache")?,
            cached_output,
            "workgroup cache",
        )?);
        cached_input = cached_output;
    }
    let workgroup_bytes = checked_bytes(max_layer_parameter_count, "workgroup cache")?;
    if workgroup_bytes > limits.max_workgroup_memory {
        return Err(NnError::InvalidDescriptor(format!(
            "layer cache needs {workgroup_bytes} bytes, device limit is {}",
            limits.max_workgroup_memory
        )));
    }
    let mut parameter_count = 0;
    let mut tape_offset = encoded_size;
    let mut layer_input = encoded_size;
    let mut layers = Vec::with_capacity(desc.layers.len());
    for &layer_output in &desc.layers {
        let layer_parameters = checked_mul(
            checked_add(layer_input, 1, "layer parameter count")?,
            layer_output,
            "layer parameter count",
        )?;
        layers.push(LayerSpec {
            input_size: layer_input,
            output_size: layer_output,
            parameter_offset: parameter_count,
            tape_offset,
        });
        parameter_count = checked_add(parameter_count, layer_parameters, "parameter count")?;
        tape_offset = checked_add(
            tape_offset,
            checked_mul(layer_output, 2, "tape stride")?,
            "tape stride",
        )?;
        layer_input = layer_output;
    }
    let tape_stride = checked_add(tape_offset, output_size, "tape stride")?;
    let mut names = std::collections::HashSet::new();
    let mut trainable_count = 0;
    let mut trainable_auxiliary = Vec::new();
    for (index, auxiliary) in desc.auxiliary.iter().enumerate() {
        if !valid_identifier(&auxiliary.name)
            || auxiliary.name.starts_with("nn_")
            || auxiliary.name == "Scalar"
        {
            return Err(NnError::InvalidDescriptor(format!(
                "invalid or reserved auxiliary name `{}`",
                auxiliary.name
            )));
        }
        if !names.insert(auxiliary.name.as_str()) {
            return Err(NnError::InvalidDescriptor(format!(
                "duplicate auxiliary name `{}`",
                auxiliary.name
            )));
        }
        if auxiliary.wgsl_type.trim().is_empty() {
            return Err(NnError::InvalidDescriptor(format!(
                "auxiliary `{}` has an empty WGSL type",
                auxiliary.name
            )));
        }
        if let AuxiliaryMode::Trainable { elements } = auxiliary.mode {
            if elements == 0 || auxiliary.wgsl_type.trim() != "array<f32>" {
                return Err(NnError::InvalidDescriptor(format!(
                    "trainable auxiliary `{}` must have nonzero elements and WGSL type array<f32>",
                    auxiliary.name
                )));
            }
            trainable_auxiliary.push(TrainableAuxSpec {
                auxiliary_index: index,
                element_offset: trainable_count,
                elements,
            });
            trainable_count = checked_add(trainable_count, elements, "trainable auxiliary count")?;
        }
    }
    let auxiliary_minimum_sizes = validate_hashgrid_bindings(desc)?;
    if auxiliary_minimum_sizes
        .iter()
        .any(|&size| size > limits.max_storage_buffer_range)
    {
        return Err(NnError::InvalidDescriptor(
            "hash grid parameter buffer exceeds storage binding range".into(),
        ));
    }
    let auxiliary_count = u32::try_from(desc.auxiliary.len())
        .map_err(|_| NnError::InvalidDescriptor("too many auxiliary bindings".into()))?;
    if 6 + auxiliary_count > limits.max_buffer_bindings {
        return Err(NnError::InvalidDescriptor(
            "backward pass exceeds buffer binding limit".into(),
        ));
    }
    let storage_auxiliary = desc
        .auxiliary
        .iter()
        .filter(|aux| aux.mode != AuxiliaryMode::Uniform)
        .count() as u32;
    if 5 + storage_auxiliary > limits.max_storage_buffers {
        return Err(NnError::InvalidDescriptor(
            "backward pass exceeds storage buffer limit".into(),
        ));
    }
    let uniform_auxiliary = auxiliary_count - storage_auxiliary;
    if 1 + uniform_auxiliary > limits.max_uniform_buffers {
        return Err(NnError::InvalidDescriptor(
            "pass exceeds uniform buffer limit".into(),
        ));
    }
    if checked_scalar_bytes(parameter_count, "parameter buffer size", desc.precision)?
        > limits.max_storage_buffer_range
    {
        return Err(NnError::InvalidDescriptor(
            "parameter buffer exceeds storage binding range".into(),
        ));
    }
    let update_count = checked_add(parameter_count, trainable_count, "update count")?;
    if div_ceil(update_count, desc.workgroup_size) > limits.max_dispatch_count {
        return Err(NnError::InvalidDescriptor(
            "Adam pass exceeds device dispatch limit".into(),
        ));
    }
    let moment_words = checked_mul(update_count, 2, "moment count")?;
    if checked_bytes(moment_words, "moment buffer size")? > limits.max_storage_buffer_range {
        return Err(NnError::InvalidDescriptor(
            "moment buffer exceeds storage binding range".into(),
        ));
    }
    if std::mem::size_of::<NnConfig>() > limits.max_uniform_buffer_range {
        return Err(NnError::InvalidDescriptor(
            "NN config exceeds uniform binding range".into(),
        ));
    }
    Ok(ValidatedDesc {
        input_size,
        encoded_size,
        network_output_size,
        output_size,
        target_size,
        parameter_count,
        max_layer_width,
        max_layer_parameter_count,
        subgroup_capacity,
        tape_stride,
        trainable_count,
        layers,
        trainable_auxiliary,
        auxiliary_minimum_sizes,
        limits,
    })
}

fn make_pipeline(
    device: &Device,
    name: &str,
    source: &str,
    entry: &str,
    workgroup_size: u32,
    core_binding_count: usize,
    auxiliary_layout: &[ArgumentLayout],
    storage_read_only: &[bool],
) -> Result<ComputePipeline, NnError> {
    let library = device.new_shader_library(LibraryDesc {
        name,
        input: LibraryInput::wgsl(source),
    })?;
    let mut bindings = Vec::with_capacity(core_binding_count + auxiliary_layout.len());
    bindings.push(argument_layout(ArgumentKind::UniformBuffer));
    bindings.extend((1..core_binding_count).map(|_| argument_layout(ArgumentKind::StorageBuffer)));
    bindings.extend_from_slice(auxiliary_layout);
    let groups = [ArgumentGroupLayout {
        arguments: &bindings,
    }];
    let desc = ComputePipelineDesc {
        name,
        shader: library.entry(entry),
        work_group_size: [workgroup_size, 1, 1],
        constants: 0,
        arguments: &groups,
    };
    #[cfg(mev_backend = "webgpu")]
    let pipeline = device.new_compute_pipeline_with_buffer_access(desc, storage_read_only)?;
    #[cfg(not(mev_backend = "webgpu"))]
    let pipeline = {
        let _ = storage_read_only;
        device.new_compute_pipeline(desc)?
    };
    Ok(pipeline)
}

fn auxiliary_layout(desc: &NnDesc) -> Vec<ArgumentLayout> {
    desc.auxiliary
        .iter()
        .map(|auxiliary| {
            argument_layout(if auxiliary.mode == AuxiliaryMode::Uniform {
                ArgumentKind::UniformBuffer
            } else {
                ArgumentKind::StorageBuffer
            })
        })
        .collect()
}

fn auxiliary_read_only(desc: &NnDesc, adam: bool) -> Vec<bool> {
    desc.auxiliary
        .iter()
        .map(|auxiliary| match auxiliary.mode {
            AuxiliaryMode::Uniform | AuxiliaryMode::ReadWrite => false,
            AuxiliaryMode::ReadOnly => true,
            AuxiliaryMode::Trainable { .. } => !adam,
        })
        .collect()
}

fn argument_layout(kind: ArgumentKind) -> ArgumentLayout {
    ArgumentLayout {
        kind,
        size: None,
        stages: ShaderStages::COMPUTE,
    }
}

fn validate_slice(
    slice: &BufferSlice<'_>,
    required_size: usize,
    kind: ArgumentKind,
    limits: &crate::generic::ComputeKernelLimits,
    label: &str,
) -> Result<(), NnError> {
    let (usage, alignment, max_range) = match kind {
        ArgumentKind::UniformBuffer => (
            BufferUsage::UNIFORM,
            limits.uniform_offset_alignment,
            limits.max_uniform_buffer_range,
        ),
        ArgumentKind::StorageBuffer => (
            BufferUsage::STORAGE,
            limits.storage_offset_alignment,
            limits.max_storage_buffer_range,
        ),
        _ => unreachable!("NN only binds buffers"),
    };
    if !slice.buffer().usage().contains(usage) {
        return Err(NnError::InvalidBinding(format!(
            "{label} lacks {usage:?} usage"
        )));
    }
    if alignment != 0 && slice.offset() % alignment != 0 {
        return Err(NnError::InvalidBinding(format!(
            "{label} offset {} is not aligned to {alignment}",
            slice.offset()
        )));
    }
    if slice.size() < required_size {
        return Err(NnError::InvalidBinding(format!(
            "{label} is {} bytes, needs at least {required_size}",
            slice.size()
        )));
    }
    if slice.size() > max_range {
        return Err(NnError::InvalidBinding(format!(
            "{label} is {} bytes, device binding limit is {max_range}",
            slice.size()
        )));
    }
    Ok(())
}

fn reject_overlap(
    left: &BufferSlice<'_>,
    right: &BufferSlice<'_>,
    left_name: &str,
    right_name: &str,
) -> Result<(), NnError> {
    if left.buffer() == right.buffer()
        && left.offset() < right.offset() + right.size()
        && right.offset() < left.offset() + left.size()
    {
        return Err(NnError::InvalidBinding(format!(
            "{left_name} overlaps {right_name}"
        )));
    }
    Ok(())
}

fn valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('_' | 'a'..='z' | 'A'..='Z'))
        && chars.all(|character| matches!(character, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9'))
}

fn checked_add(left: u32, right: u32, label: &str) -> Result<u32, NnError> {
    left.checked_add(right)
        .ok_or_else(|| NnError::InvalidDescriptor(format!("{label} exceeds u32 range")))
}

fn checked_mul(left: u32, right: u32, label: &str) -> Result<u32, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| NnError::InvalidDescriptor(format!("{label} exceeds u32 range")))
}

fn checked_bytes(words: u32, label: &str) -> Result<usize, NnError> {
    (words as usize)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| NnError::InvalidDescriptor(format!("{label} exceeds usize range")))
}

fn checked_scalar_bytes(
    elements: u32,
    label: &str,
    precision: Precision,
) -> Result<usize, NnError> {
    let bytes = (elements as usize)
        .checked_mul(match precision {
            Precision::F32 => 4,
            Precision::F16 => 2,
        })
        .ok_or_else(|| NnError::InvalidDescriptor(format!("{label} exceeds usize range")))?;
    Ok(bytes
        .checked_add(3)
        .ok_or_else(|| NnError::InvalidDescriptor(format!("{label} exceeds usize range")))?
        & !3)
}

const fn div_ceil(value: u32, divisor: u32) -> u32 {
    value / divisor + (value % divisor != 0) as u32
}

/// Validated, backend-independent shader inputs.
pub(crate) struct ShaderSpec<'a> {
    pub desc: &'a NnDesc,
    pub input_size: u32,
    pub encoded_size: u32,
    pub network_output_size: u32,
    pub output_size: u32,
    pub target_size: u32,
    pub parameter_count: u32,
    pub max_layer_width: u32,
    pub max_layer_parameter_count: u32,
    pub subgroup_capacity: u32,
    pub tape_stride: u32,
    pub layers: &'a [LayerSpec],
    pub trainable_auxiliary: &'a [TrainableAuxSpec],
}

pub(crate) use shader::generate as generate_shader;

#[cfg(test)]
mod tests {
    use super::{
        AuxiliaryDesc, AuxiliaryMode, Encoding, HashGrid, HashGridLevel, NnDesc, NnError,
        Precision, validate_hashgrid_bindings, validate_inputs,
    };

    fn custom(input_arity: u32, output_arity: u32) -> Encoding {
        Encoding::Custom {
            input_arity,
            output_arity,
            source: String::new(),
            forward: "encode_vector".into(),
            backward: "encode_vector_backward".into(),
        }
    }

    #[test]
    fn ordered_inputs_aggregate_raw_and_encoded_arities() {
        let inputs = [
            Encoding::Identity { arity: 2 },
            custom(3, 5),
            Encoding::Identity { arity: 1 },
            custom(2, 1),
        ];
        let desc = NnDesc::new(&inputs, &[4, 1]);
        assert_eq!(desc.inputs.len(), 4);
        assert!(matches!(validate_inputs(&desc.inputs), Ok((8, 9))));
        assert_eq!(desc.inputs[1].input_arity(), 3);
        assert_eq!(desc.inputs[1].output_arity(), 5);
    }

    #[test]
    fn input_vectors_require_positive_arities() {
        for inputs in [
            vec![],
            vec![Encoding::Identity { arity: 0 }],
            vec![custom(0, 1)],
            vec![custom(1, 0)],
            vec![Encoding::Identity { arity: 1 }, custom(1, 0)],
        ] {
            assert!(matches!(
                validate_inputs(&inputs),
                Err(NnError::InvalidDescriptor(_))
            ));
        }
    }

    #[test]
    fn input_vector_aggregate_overflow_is_rejected() {
        for (inputs, label) in [
            (
                [custom(u32::MAX, 1), Encoding::Identity { arity: 1 }],
                "input size",
            ),
            (
                [custom(1, u32::MAX), Encoding::Identity { arity: 1 }],
                "encoded size",
            ),
        ] {
            match validate_inputs(&inputs) {
                Err(NnError::InvalidDescriptor(message)) => {
                    assert!(message.contains(label), "{message}");
                }
                result => panic!("expected {label} overflow, got {result:?}"),
            }
        }
    }

    #[test]
    fn custom_encoding_hook_names_require_unreserved_identifiers() {
        for name in [
            "",
            "_",
            "__encode",
            "0encode",
            "encode()",
            "a b",
            "nn_encode",
        ] {
            for forward in [true, false] {
                let mut encoding = custom(1, 2);
                if let Encoding::Custom {
                    forward: forward_name,
                    backward: backward_name,
                    ..
                } = &mut encoding
                {
                    *if forward { forward_name } else { backward_name } = name.into();
                }
                assert!(
                    matches!(
                        validate_inputs(&[encoding]),
                        Err(NnError::InvalidDescriptor(_))
                    ),
                    "accepted invalid hook name {name:?}"
                );
            }
        }
        assert!(matches!(
            validate_inputs(&[custom(1, 2), custom(1, 2)]),
            Ok((2, 4))
        ));
    }

    fn grid(resolution: u32, features: u32) -> Result<HashGrid, NnError> {
        HashGrid::new(&[HashGridLevel {
            resolution,
            features,
            table_size: 4,
        }])
    }

    #[test]
    fn hashgrid_binding_requires_named_f32_parameter_storage() -> Result<(), NnError> {
        let grid = grid(2, 2)?;
        let mut desc = NnDesc::new(
            &[Encoding::HashGrid {
                grid: grid.clone(),
                auxiliary: "features".into(),
            }],
            &[1],
        );
        assert!(validate_hashgrid_bindings(&desc).is_err());
        for binding in [
            grid.auxiliary("wrong_name", false),
            AuxiliaryDesc::new("features", "array<f16>", AuxiliaryMode::ReadOnly),
            AuxiliaryDesc::new("features", "array<f32>", AuxiliaryMode::Uniform),
            AuxiliaryDesc::new("features", "array<f32>", AuxiliaryMode::ReadWrite),
            AuxiliaryDesc::new(
                "features",
                "array<f32>",
                AuxiliaryMode::Trainable { elements: 23 },
            ),
        ] {
            desc.auxiliary = vec![binding];
            assert!(validate_hashgrid_bindings(&desc).is_err());
        }
        for mode in [
            AuxiliaryMode::ReadOnly,
            AuxiliaryMode::Trainable { elements: 24 },
            AuxiliaryMode::Trainable { elements: 25 },
        ] {
            desc.auxiliary = vec![AuxiliaryDesc::new("features", "array<f32>", mode)];
            assert_eq!(validate_hashgrid_bindings(&desc)?, [96]);
        }
        Ok(())
    }

    #[test]
    fn shared_hashgrid_binding_uses_largest_required_prefix() -> Result<(), NnError> {
        let small = grid(1, 1)?;
        let large = grid(2, 3)?;
        for reverse in [false, true] {
            let mut inputs = [
                Encoding::HashGrid {
                    grid: small.clone(),
                    auxiliary: "shared".into(),
                },
                Encoding::HashGrid {
                    grid: large.clone(),
                    auxiliary: "shared".into(),
                },
            ];
            if reverse {
                inputs.reverse();
            }
            let mut desc = NnDesc::new(&inputs, &[1]);
            desc.auxiliary = vec![
                AuxiliaryDesc::new("other", "array<f32>", AuxiliaryMode::ReadOnly),
                large.auxiliary("shared", false),
            ];
            assert_eq!(validate_hashgrid_bindings(&desc)?, [1, 144]);
            assert_eq!(validate_inputs(&inputs)?, (6, 4));
        }
        Ok(())
    }

    #[test]
    fn descriptor_precision_applies_to_hashgrid_resolution() -> Result<(), NnError> {
        let grid = grid(2049, 1)?;
        let mut desc = NnDesc::new(
            &[Encoding::HashGrid {
                grid: grid.clone(),
                auxiliary: "features".into(),
            }],
            &[1],
        );
        desc.auxiliary = vec![grid.auxiliary("features", false)];
        assert_eq!(validate_hashgrid_bindings(&desc)?, [48]);
        desc.precision = Precision::F16;
        assert!(validate_hashgrid_bindings(&desc).is_err());
        Ok(())
    }
}
