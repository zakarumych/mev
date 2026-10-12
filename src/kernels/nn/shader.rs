use std::fmt::Write as _;

mod hashgrid;

use super::{Activation, AuxiliaryMode, Encoding, Loss, NnError, Output, Precision, ShaderSpec};

pub(crate) struct GeneratedShaders {
    pub inference: String,
    pub training_forward: String,
    pub backward: String,
    pub adam: String,
}

pub(crate) fn generate(spec: &ShaderSpec<'_>) -> Result<GeneratedShaders, NnError> {
    validate_custom_activations(spec)?;
    let mut inference_types = Vec::new();
    let mut training_types = None;
    if spec
        .desc
        .inputs
        .iter()
        .any(|encoding| matches!(encoding, Encoding::Custom { .. }))
    {
        let provisional = forward_shader(spec, false, &[]);
        inference_types =
            reflect_encoding_types(spec, &provisional, spec.desc.precision == Precision::F16)?;
        if spec.desc.precision == Precision::F16 {
            let provisional = forward_shader(spec, true, &[]);
            training_types = Some(reflect_encoding_types(spec, &provisional, false)?);
        }
    }
    let training_types = training_types.as_deref().unwrap_or(&inference_types);
    let generated = GeneratedShaders {
        inference: forward_shader(spec, false, &inference_types),
        training_forward: forward_shader(spec, true, training_types),
        backward: backward_shader(spec, training_types),
        adam: adam_shader(spec),
    };

    eprintln!("Generated inference shader:\n{}", generated.inference);
    eprintln!(
        "Generated training forward shader:\n{}",
        generated.training_forward
    );
    eprintln!("Generated backward shader:\n{}", generated.backward);
    eprintln!("Generated Adam shader:\n{}", generated.adam);

    if spec.desc.precision == Precision::F16 && !spec.desc.auxiliary.is_empty() {
        validate_auxiliary_types(spec, &generated)?;
    }
    Ok(generated)
}

struct EncodingTypes {
    forward_input: String,
    backward_input: String,
    backward_encoded: String,
    backward_gradient: String,
}

fn reflect_encoding_types(
    spec: &ShaderSpec<'_>,
    source: &str,
    half_arithmetic: bool,
) -> Result<Vec<Option<EncodingTypes>>, NnError> {
    let module =
        naga::front::wgsl::parse_str(source).map_err(crate::ShaderLibraryError::ParseWgsl)?;
    let width = if half_arithmetic { 2 } else { 4 };
    let mut types = Vec::with_capacity(spec.desc.inputs.len());
    for (index, encoding) in spec.desc.inputs.iter().enumerate() {
        let Encoding::Custom {
            input_arity,
            output_arity,
            forward,
            backward,
            ..
        } = encoding
        else {
            types.push(None);
            continue;
        };
        let find_function = |name: &str| {
            module
                .functions
                .iter()
                .find(|(_, function)| function.name.as_deref() == Some(name))
                .map(|(_, function)| function)
                .ok_or_else(|| {
                    NnError::InvalidDescriptor(format!(
                        "input encoding {index} function {name} is not defined"
                    ))
                })
        };
        let forward_function = find_function(forward)?;
        let backward_function = find_function(backward)?;
        let invalid_forward = || {
            NnError::InvalidDescriptor(format!(
                "input encoding {index} function {forward} must take (array<Scalar, {input_arity}>, u32) and return array<Scalar, {output_arity}>"
            ))
        };
        let invalid_backward = || {
            NnError::InvalidDescriptor(format!(
                "input encoding {index} function {backward} must take (array<Scalar, {input_arity}>, array<Scalar, {output_arity}>, array<Scalar, {output_arity}>, u32) and return nothing"
            ))
        };
        let [forward_input, forward_sample] = forward_function.arguments.as_slice() else {
            return Err(invalid_forward());
        };
        let Some(forward_result) = &forward_function.result else {
            return Err(invalid_forward());
        };
        let forward_input = encoding_array_type(&module, forward_input.ty, *input_arity, width)
            .ok_or_else(invalid_forward)?;
        if !is_u32(&module, forward_sample.ty)
            || encoding_array_type(&module, forward_result.ty, *output_arity, width).is_none()
        {
            return Err(invalid_forward());
        }
        let [
            backward_input,
            backward_encoded,
            backward_gradient,
            backward_sample,
        ] = backward_function.arguments.as_slice()
        else {
            return Err(invalid_backward());
        };
        if backward_function.result.is_some() || !is_u32(&module, backward_sample.ty) {
            return Err(invalid_backward());
        }
        types.push(Some(EncodingTypes {
            forward_input,
            backward_input: encoding_array_type(&module, backward_input.ty, *input_arity, width)
                .ok_or_else(invalid_backward)?,
            backward_encoded: encoding_array_type(
                &module,
                backward_encoded.ty,
                *output_arity,
                width,
            )
            .ok_or_else(invalid_backward)?,
            backward_gradient: encoding_array_type(
                &module,
                backward_gradient.ty,
                *output_arity,
                width,
            )
            .ok_or_else(invalid_backward)?,
        }));
    }
    Ok(types)
}

fn is_u32(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    module.types[ty].inner == naga::TypeInner::Scalar(naga::Scalar::U32)
}

fn encoding_array_type(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    arity: u32,
    width: u8,
) -> Option<String> {
    let naga::TypeInner::Array {
        base,
        size: naga::ArraySize::Constant(size),
        ..
    } = module.types[ty].inner
    else {
        return None;
    };
    if size.get() != arity
        || module.types[base].inner
            != naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Float,
                width,
            })
    {
        return None;
    }
    // Naga compares array element handles, so callback aliases must remain intact.
    Some(module.types[ty].name.clone().unwrap_or_else(|| {
        let scalar =
            module.types[base]
                .name
                .as_deref()
                .unwrap_or(if width == 2 { "f16" } else { "f32" });
        format!("array<{scalar},{arity}>")
    }))
}

fn parameter_scalar(spec: &ShaderSpec<'_>) -> &'static str {
    match spec.desc.precision {
        Precision::F32 => "f32",
        Precision::F16 => "f16",
    }
}

fn validate_auxiliary_types(
    spec: &ShaderSpec<'_>,
    generated: &GeneratedShaders,
) -> Result<(), NnError> {
    let inference = naga::front::wgsl::parse_str(&generated.inference)
        .map_err(crate::ShaderLibraryError::ParseWgsl)?;
    let training = naga::front::wgsl::parse_str(&generated.training_forward)
        .map_err(crate::ShaderLibraryError::ParseWgsl)?;
    for auxiliary in &spec.desc.auxiliary {
        let inference_type = inference
            .global_variables
            .iter()
            .find(|(_, variable)| variable.name.as_deref() == Some(&auxiliary.name))
            .map(|(_, variable)| variable.ty);
        let training_type = training
            .global_variables
            .iter()
            .find(|(_, variable)| variable.name.as_deref() == Some(&auxiliary.name))
            .map(|(_, variable)| variable.ty);
        let matching = match (inference_type, training_type) {
            (Some(left), Some(right)) => same_buffer_type(&inference, left, &training, right),
            _ => false,
        };
        if !matching {
            return Err(NnError::InvalidDescriptor(format!(
                "auxiliary resource {} must have the same type and layout in inference and training; use explicit scalar types in resource declarations",
                auxiliary.name
            )));
        }
    }
    Ok(())
}

fn same_buffer_type(
    left_module: &naga::Module,
    left: naga::Handle<naga::Type>,
    right_module: &naga::Module,
    right: naga::Handle<naga::Type>,
) -> bool {
    use naga::TypeInner;
    match (
        &left_module.types[left].inner,
        &right_module.types[right].inner,
    ) {
        (TypeInner::Scalar(left), TypeInner::Scalar(right))
        | (TypeInner::Atomic(left), TypeInner::Atomic(right)) => left == right,
        (
            TypeInner::Vector {
                size: left_size,
                scalar: left_scalar,
            },
            TypeInner::Vector {
                size: right_size,
                scalar: right_scalar,
            },
        ) => left_size == right_size && left_scalar == right_scalar,
        (
            TypeInner::Matrix {
                columns: left_columns,
                rows: left_rows,
                scalar: left_scalar,
            },
            TypeInner::Matrix {
                columns: right_columns,
                rows: right_rows,
                scalar: right_scalar,
            },
        ) => {
            left_columns == right_columns && left_rows == right_rows && left_scalar == right_scalar
        }
        (
            TypeInner::Array {
                base: left_base,
                size: left_size,
                stride: left_stride,
            },
            TypeInner::Array {
                base: right_base,
                size: right_size,
                stride: right_stride,
            },
        ) => {
            let same_size = match (left_size, right_size) {
                (naga::ArraySize::Constant(left), naga::ArraySize::Constant(right)) => {
                    left == right
                }
                (naga::ArraySize::Dynamic, naga::ArraySize::Dynamic) => true,
                _ => false,
            };
            same_size
                && left_stride == right_stride
                && same_buffer_type(left_module, *left_base, right_module, *right_base)
        }
        (
            TypeInner::Struct {
                members: left_members,
                span: left_span,
            },
            TypeInner::Struct {
                members: right_members,
                span: right_span,
            },
        ) => {
            left_span == right_span
                && left_members.len() == right_members.len()
                && left_members.iter().zip(right_members).all(|(left, right)| {
                    left.offset == right.offset
                        && same_buffer_type(left_module, left.ty, right_module, right.ty)
                })
        }
        _ => false,
    }
}

fn validate_custom_activations(spec: &ShaderSpec<'_>) -> Result<(), NnError> {
    if let (Activation::Custom(hidden), Output::Activation(Activation::Custom(output))) =
        (&spec.desc.activation, &spec.desc.output)
    {
        if hidden != output {
            return Err(NnError::InvalidDescriptor(
                "hidden and output custom activations must use the same WGSL source".into(),
            ));
        }
    }
    Ok(())
}

fn common(
    spec: &ShaderSpec<'_>,
    auxiliary_base: u32,
    writable_aux: bool,
    atomic_helpers: bool,
    half_arithmetic: bool,
) -> String {
    let mut s = String::new();
    if spec.desc.precision == Precision::F16 {
        s.push_str("enable f16;\n");
    }
    writeln!(
        s,
        "alias Scalar = {};",
        if half_arithmetic { "f16" } else { "f32" }
    )
    .unwrap();
    writeln!(s, "alias Input = array<Scalar, {}>;", spec.input_size).unwrap();
    writeln!(s, "alias Encoded = array<Scalar, {}>;", spec.encoded_size).unwrap();
    writeln!(
        s,
        "alias NetworkOutput = array<Scalar, {}>;",
        spec.network_output_size
    )
    .unwrap();
    writeln!(s, "alias Output = array<Scalar, {}>;", spec.output_size).unwrap();
    writeln!(s, "alias Target = array<Scalar, {}>;", spec.target_size).unwrap();
    s.push_str(
        "struct NnConfig {\n\
         sample_count:u32, workgroup_count:u32, parameter_count:u32, subgroup_capacity:u32,\n\
         tape_stride:u32, dense_gradient_offset:u32, aux_gradient_offset:u32, loss_offset:u32,\n\
         step:u32, trainable_count:u32, encoded_size:u32, output_size:u32,\n\
         learning_rate:f32, beta1:f32, beta2:f32, epsilon:f32,\n\
         };\n",
    );
    s.push_str(&spec.desc.prelude);
    s.push('\n');
    for (binding, aux) in spec.desc.auxiliary.iter().enumerate() {
        let address = match aux.mode {
            AuxiliaryMode::Uniform => "uniform",
            AuxiliaryMode::ReadOnly if !writable_aux => "storage, read",
            AuxiliaryMode::ReadOnly => "storage, read",
            AuxiliaryMode::ReadWrite => "storage, read_write",
            AuxiliaryMode::Trainable { .. } if writable_aux => "storage, read_write",
            AuxiliaryMode::Trainable { .. } => "storage, read",
        };
        writeln!(
            s,
            "@group(0) @binding({}) var<{address}> {}: {};",
            auxiliary_base + binding as u32,
            aux.name,
            aux.wgsl_type
        )
        .unwrap();
    }
    let mut offset = 0u32;
    for aux in &spec.desc.auxiliary {
        if let AuxiliaryMode::Trainable { elements } = aux.mode {
            if atomic_helpers {
                writeln!(
                    s,
                    "fn nn_add_{}_gradient(index:u32, delta:Scalar) {{",
                    aux.name
                )
                .unwrap();
                writeln!(
                    s,
                    "  let slot = config.aux_gradient_offset + {offset}u + index;"
                )
                .unwrap();
                s.push_str("  var old = atomicLoad(&gradients[slot]);\n  loop {\n    let value = bitcast<f32>(old) + delta;\n    let result = atomicCompareExchangeWeak(&gradients[slot], old, bitcast<u32>(value));\n    if result.exchanged { break; }\n    old = result.old_value;\n  }\n}\n");
            } else {
                writeln!(
                    s,
                    "fn nn_add_{}_gradient(index:u32, delta:Scalar) {{}}",
                    aux.name
                )
                .unwrap();
            }
            offset += elements;
        }
    }
    s
}

fn custom_sources(spec: &ShaderSpec<'_>, include_loss: bool) -> String {
    let mut s = String::new();
    let mut encoding_sources = Vec::new();
    for encoding in &spec.desc.inputs {
        if let Encoding::Custom { source, .. } = encoding {
            if !encoding_sources.contains(&source.as_str()) {
                s.push_str(source);
                s.push('\n');
                encoding_sources.push(source.as_str());
            }
        }
    }
    if let Activation::Custom(source) = &spec.desc.activation {
        s.push_str(source);
        s.push('\n');
    }
    if let Output::Activation(Activation::Custom(source)) = &spec.desc.output {
        if !matches!(&spec.desc.activation, Activation::Custom(hidden) if hidden == source) {
            s.push_str(source);
            s.push('\n');
        }
    }
    if let Output::Custom { source, .. } = &spec.desc.output {
        s.push_str(source);
        s.push('\n');
    }
    if include_loss {
        if let Loss::Custom { source, .. } = &spec.desc.loss {
            s.push_str(source);
            s.push('\n');
        }
    }
    s
}

fn builtin_functions(
    spec: &ShaderSpec<'_>,
    include_loss: bool,
    encoding_types: &[Option<EncodingTypes>],
) -> String {
    let mut s = String::new();
    match &spec.desc.activation {
        Activation::Identity => s.push_str("fn nn_activation(x:Scalar)->Scalar{return x;} fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return 1.0;}\n"),
        Activation::Relu => s.push_str("fn nn_activation(x:Scalar)->Scalar{return max(x,0.0);} fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return select(Scalar(0.0),Scalar(1.0),x>0.0);}\n"),
        Activation::Sigmoid => s.push_str("fn nn_activation(x:Scalar)->Scalar{return 1.0/(1.0+exp(-x));} fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return y*(1.0-y);}\n"),
        Activation::Tanh => s.push_str("fn nn_activation(x:Scalar)->Scalar{return tanh(x);} fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return 1.0-y*y;}\n"),
        Activation::Custom(_) => {}
    }
    s.push_str(&encoding_functions(spec, encoding_types));
    match &spec.desc.output {
        Output::Identity => writeln!(s,"fn nn_output(values:NetworkOutput,sample:u32)->Output{{var r:Output; for(var i=0u;i<{}u;i++){{r[i]=values[i];}} return r;}} fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{{var r:NetworkOutput; for(var i=0u;i<{}u;i++){{r[i]=grad[i];}} return r;}}",spec.output_size,spec.network_output_size).unwrap(),
        Output::Activation(a) => {
            let (f, d) = match a {
                Activation::Identity => ("x", "1.0"),
                Activation::Relu => ("max(x,0.0)", "select(Scalar(0.0),Scalar(1.0),x>0.0)"),
                Activation::Sigmoid => ("1.0/(1.0+exp(-x))", "y*(1.0-y)"),
                Activation::Tanh => ("tanh(x)", "1.0-y*y"),
                Activation::Custom(_) => ("nn_activation(x)", "nn_activation_derivative(x,y)"),
            };
            writeln!(s,"fn nn_output_activation(x:Scalar)->Scalar{{return {f};}} fn nn_output_activation_derivative(x:Scalar,y:Scalar)->Scalar{{return {d};}}").unwrap();
            writeln!(s,"fn nn_output(values:NetworkOutput,sample:u32)->Output{{var r:Output; for(var i=0u;i<{}u;i++){{r[i]=nn_output_activation(values[i]);}} return r;}} fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{{var r:NetworkOutput; for(var i=0u;i<{}u;i++){{r[i]=grad[i]*nn_output_activation_derivative(values[i],prediction[i]);}} return r;}}",spec.output_size,spec.network_output_size).unwrap();
        }
        Output::Custom { .. } => {}
    }
    if include_loss {
        match &spec.desc.loss {
            Loss::SquaredError => writeln!(s,"fn nn_loss(prediction:Output,target_value:Target)->f32{{var v=0.0;for(var i=0u;i<{}u;i++){{let d=prediction[i]-target_value[i];v+=0.5*d*d;}}return v;}} fn nn_loss_derivative(prediction:Output,target_value:Target)->Output{{var r:Output;for(var i=0u;i<{}u;i++){{r[i]=prediction[i]-target_value[i];}}return r;}}",spec.output_size,spec.output_size).unwrap(),
            Loss::Custom { .. } => {}
        }
    }
    s
}

fn encoding_functions(spec: &ShaderSpec<'_>, encoding_types: &[Option<EncodingTypes>]) -> String {
    let mut s = String::new();
    for (index, encoding) in spec.desc.inputs.iter().enumerate() {
        if let Encoding::HashGrid { grid, auxiliary } = encoding {
            let trainable = spec.desc.auxiliary.iter().any(|binding| {
                binding.name == *auxiliary
                    && matches!(binding.mode, AuxiliaryMode::Trainable { .. })
            });
            s.push_str(&hashgrid::functions(index, grid, auxiliary, trainable));
        }
    }
    s.push_str("fn nn_encode(input:Input,sample:u32)->Encoded{var result:Encoded;\n");
    let mut input_offset = 0u32;
    let mut output_offset = 0u32;
    for (index, encoding) in spec.desc.inputs.iter().enumerate() {
        let input_arity = encoding.input_arity();
        let output_arity = encoding.output_arity();
        match encoding {
            Encoding::Identity { arity } => {
                writeln!(s, "for(var i=0u;i<{arity}u;i++){{result[{output_offset}u+i]=input[{input_offset}u+i];}}").unwrap();
            }
            Encoding::HashGrid { .. } => {
                writeln!(s, "{{var values:array<Scalar,3>;for(var i=0u;i<3u;i++){{values[i]=input[{input_offset}u+i];}}let encoded_values=nn_hashgrid_{index}_forward(values);for(var i=0u;i<{output_arity}u;i++){{result[{output_offset}u+i]=encoded_values[i];}}}}").unwrap();
            }
            Encoding::Custom { forward, .. } => {
                let input_type = encoding_types
                    .get(index)
                    .and_then(Option::as_ref)
                    .map(|types| types.forward_input.clone())
                    .unwrap_or_else(|| format!("array<Scalar,{input_arity}>"));
                writeln!(s, "{{var values:{input_type};for(var i=0u;i<{input_arity}u;i++){{values[i]=input[{input_offset}u+i];}}let encoded_values={forward}(values,sample);for(var i=0u;i<{output_arity}u;i++){{result[{output_offset}u+i]=encoded_values[i];}}}}").unwrap();
            }
        }
        input_offset += input_arity;
        output_offset += output_arity;
    }
    s.push_str("return result;}\nfn nn_encode_backward(input:Input,encoded:Encoded,grad:Encoded,sample:u32){\n");
    input_offset = 0;
    output_offset = 0;
    for (index, encoding) in spec.desc.inputs.iter().enumerate() {
        let input_arity = encoding.input_arity();
        let output_arity = encoding.output_arity();
        if let Encoding::Custom { backward, .. } = encoding {
            let (input_type, encoded_type, gradient_type) = encoding_types
                .get(index)
                .and_then(Option::as_ref)
                .map(|types| {
                    (
                        types.backward_input.clone(),
                        types.backward_encoded.clone(),
                        types.backward_gradient.clone(),
                    )
                })
                .unwrap_or_else(|| {
                    (
                        format!("array<Scalar,{input_arity}>"),
                        format!("array<Scalar,{output_arity}>"),
                        format!("array<Scalar,{output_arity}>"),
                    )
                });
            writeln!(s, "{{var values:{input_type};var encoded_values:{encoded_type};var gradients:{gradient_type};for(var i=0u;i<{input_arity}u;i++){{values[i]=input[{input_offset}u+i];}}for(var i=0u;i<{output_arity}u;i++){{encoded_values[i]=encoded[{output_offset}u+i];gradients[i]=grad[{output_offset}u+i];}}{backward}(values,encoded_values,gradients,sample);}}").unwrap();
        } else if matches!(encoding, Encoding::HashGrid { .. }) {
            writeln!(s, "{{var values:array<Scalar,3>;var gradients:array<Scalar,{output_arity}>;for(var i=0u;i<3u;i++){{values[i]=input[{input_offset}u+i];}}for(var i=0u;i<{output_arity}u;i++){{gradients[i]=grad[{output_offset}u+i];}}nn_hashgrid_{index}_backward(values,gradients);}}").unwrap();
        }
        input_offset += input_arity;
        output_offset += output_arity;
    }
    s.push_str("}\n");
    s
}

fn forward_shader(
    spec: &ShaderSpec<'_>,
    training: bool,
    encoding_types: &[Option<EncodingTypes>],
) -> String {
    let half_arithmetic = !training && spec.desc.precision == Precision::F16;
    let mut s = common(
        spec,
        if training { 5 } else { 4 },
        false,
        false,
        half_arithmetic,
    );
    s.push_str(&custom_sources(spec, false));
    s.push_str(&builtin_functions(spec, false, encoding_types));
    writeln!(s, "@group(0) @binding(0) var<uniform> config:NnConfig;\n@group(0) @binding(1) var<storage,read> parameters:array<{}>;\n@group(0) @binding(2) var<storage,read> inputs:array<Scalar>;\n@group(0) @binding(3) var<storage,read_write> outputs:array<Scalar>;", parameter_scalar(spec)).unwrap();
    if training {
        s.push_str("@group(0) @binding(4) var<storage,read_write> tape:array<f32>;\n");
    }
    writeln!(
        s,
        "var<workgroup> nn_cache:array<Scalar, {}>;",
        max_layer_parameters(spec)
    )
    .unwrap();
    let entry = if training {
        "nn_training_forward"
    } else {
        "nn_inference"
    };
    writeln!(s, "@compute @workgroup_size({}) fn {entry}(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){{", spec.desc.workgroup_size).unwrap();
    write!(s,"let sample=gid.x; let valid=sample<config.sample_count; var input:Input; if valid {{for(var i=0u;i<{}u;i++){{input[i]=inputs[sample*{}u+i];}}}} var a:array<Scalar,",spec.input_size,spec.input_size).unwrap();
    write!(
        s,
        "{}>; var next:array<Scalar,{}>;",
        spec.max_layer_width, spec.max_layer_width
    )
    .unwrap();
    write!(s," var encoded:Encoded; if valid {{encoded=nn_encode(input,sample);for(var i=0u;i<{}u;i++){{a[i]=encoded[i];}}}}",spec.encoded_size).unwrap();
    let mut tape_offset = 0u32;
    if training {
        s.push_str(" let tape_base=sample*config.tape_stride;");
        writeln!(
            s,
            " if valid {{for(var i=0u;i<{}u;i++){{tape[tape_base+i]=encoded[i];}}}}",
            spec.encoded_size
        )
        .unwrap();
        tape_offset += spec.encoded_size;
    }
    let mut parameter_offset = 0u32;
    let mut input_width = spec.encoded_size;
    for (layer_index, &output_width) in spec.desc.layers.iter().enumerate() {
        let final_layer = layer_index + 1 == spec.desc.layers.len();
        let weight_count = input_width * output_width;
        let layer_parameters = weight_count + output_width;
        writeln!(
            s,
            "for(var k=lid.x;k<{}u;k+={}u){{nn_cache[k]=Scalar(parameters[{}u+k]);}} workgroupBarrier();",
            layer_parameters, spec.desc.workgroup_size, parameter_offset
        )
        .unwrap();
        writeln!(s,"if valid {{for(var j=0u;j<{}u;j++){{var z=nn_cache[{}u+j];for(var k=0u;k<{}u;k++){{z+=a[k]*nn_cache[j*{}u+k];}}next[j]=z;}}}}",output_width,weight_count,input_width,input_width).unwrap();
        if training {
            writeln!(s,"if valid {{for(var j=0u;j<{}u;j++){{let z=next[j];tape[tape_base+{}u+j]=z;let y={};tape[tape_base+{}u+{}u+j]=y;a[j]=y;}}}} workgroupBarrier();",output_width,tape_offset,if final_layer{"z"}else{"nn_activation(z)"},tape_offset,output_width).unwrap();
            tape_offset += output_width * 2;
        } else {
            writeln!(
                s,
                "if valid {{for(var j=0u;j<{}u;j++){{let z=next[j];a[j]={};}}}} workgroupBarrier();",
                output_width,
                if final_layer { "z" } else { "nn_activation(z)" }
            )
            .unwrap();
        }
        parameter_offset += output_width * (input_width + 1);
        input_width = output_width;
    }
    writeln!(s,"if valid {{var values:NetworkOutput;for(var i=0u;i<{}u;i++){{values[i]=a[i];}}let prediction=nn_output(values,sample);for(var i=0u;i<{}u;i++){{outputs[sample*{}u+i]=prediction[i];{} }} }} }}",spec.network_output_size,spec.output_size,spec.output_size,if training {format!("tape[tape_base+{}u+i]=prediction[i];",tape_offset)} else {String::new()}).unwrap();
    s
}

fn max_layer_parameters(spec: &ShaderSpec<'_>) -> u32 {
    spec.max_layer_parameter_count
}

fn backward_shader(spec: &ShaderSpec<'_>, encoding_types: &[Option<EncodingTypes>]) -> String {
    let mut s = common(spec, 6, false, true, false);
    s.push_str(&custom_sources(spec, true));
    s.push_str(&builtin_functions(spec, true, encoding_types));
    writeln!(s, "@group(0) @binding(0) var<uniform> config:NnConfig;\n@group(0) @binding(1) var<storage,read> parameters:array<{}>;\n@group(0) @binding(2) var<storage,read> inputs:array<f32>;\n@group(0) @binding(3) var<storage,read> targets:array<f32>;\n@group(0) @binding(4) var<storage,read> tape:array<f32>;\n@group(0) @binding(5) var<storage,read_write> gradients:array<atomic<u32>>;", parameter_scalar(spec)).unwrap();
    writeln!(
        s,
        "var<workgroup> nn_cache:array<f32,{}>;",
        max_layer_parameters(spec)
    )
    .unwrap();
    writeln!(s,"@compute @workgroup_size({}) fn nn_backward(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>,@builtin(subgroup_id) subgroup_id:u32,@builtin(num_subgroups) num_subgroups:u32){{",spec.desc.workgroup_size).unwrap();
    write!(s,"let sample=gid.x;let valid=sample<config.sample_count;let elected=lid.x==subgroupMin(lid.x);let tape_base=sample*config.tape_stride;if lid.x==0u{{atomicStore(&gradients[wid.x],num_subgroups);}}var input:Input;var encoded:Encoded;var prediction:Output;var target_value:Target;var grad:array<f32,{}>;var previous:array<f32,{}>;",spec.max_layer_width,spec.max_layer_width).unwrap();
    writeln!(s,"if valid {{for(var i=0u;i<{}u;i++){{input[i]=inputs[sample*{}u+i];}}for(var i=0u;i<{}u;i++){{encoded[i]=tape[tape_base+i];}}for(var i=0u;i<{}u;i++){{prediction[i]=tape[tape_base+config.tape_stride-config.output_size+i];}}for(var i=0u;i<{}u;i++){{target_value[i]=targets[sample*{}u+i];}}let loss_grad=nn_loss_derivative(prediction,target_value);var values:NetworkOutput;",spec.input_size,spec.input_size,spec.encoded_size,spec.output_size,spec.target_size,spec.target_size).unwrap();
    let mut layer_tape = spec.encoded_size;
    for (i, &width) in spec.desc.layers.iter().enumerate() {
        if i + 1 == spec.desc.layers.len() {
            writeln!(
                s,
                "for(var j=0u;j<{}u;j++){{values[j]=tape[tape_base+{}u+j];}}",
                width, layer_tape
            )
            .unwrap();
        }
        layer_tape += width * 2;
    }
    writeln!(s,"let output_grad=nn_output_backward(values,prediction,loss_grad,sample);for(var i=0u;i<{}u;i++){{grad[i]=output_grad[i];}}atomicStore(&gradients[config.loss_offset+sample],bitcast<u32>(nn_loss(prediction,target_value)));}}",spec.network_output_size).unwrap();
    let mut param_offsets = Vec::new();
    let mut p = 0u32;
    let mut iw = spec.encoded_size;
    let mut tape_offsets = Vec::new();
    let mut t = spec.encoded_size;
    for &ow in &spec.desc.layers {
        param_offsets.push(p);
        tape_offsets.push(t);
        p += ow * (iw + 1);
        t += ow * 2;
        iw = ow;
    }
    for li in (0..spec.desc.layers.len()).rev() {
        let ow = spec.desc.layers[li];
        let input_width = if li == 0 {
            spec.encoded_size
        } else {
            spec.desc.layers[li - 1]
        };
        let final_layer = li + 1 == spec.desc.layers.len();
        if !final_layer {
            writeln!(s,"if valid {{for(var j=0u;j<{}u;j++){{let z=tape[tape_base+{}u+j];let y=tape[tape_base+{}u+{}u+j];grad[j]*=nn_activation_derivative(z,y);}}}}",ow,tape_offsets[li],tape_offsets[li],ow).unwrap();
        }
        writeln!(
            s,
            "for(var i=0u;i<{}u;i++){{previous[i]=0.0;}}",
            input_width
        )
        .unwrap();
        let weight_count = input_width * ow;
        writeln!(
            s,
            "for(var k=lid.x;k<{}u;k+={}u){{nn_cache[k]=f32(parameters[{}u+k]);}}workgroupBarrier();",
            weight_count + ow,
            spec.desc.workgroup_size,
            param_offsets[li]
        )
        .unwrap();
        for j in 0..ow {
            s.push_str("{\n");
            let prev_expr = if li == 0 {
                "encoded[i]".to_string()
            } else {
                format!(
                    "tape[tape_base+{}u+{}u+i]",
                    tape_offsets[li - 1],
                    spec.desc.layers[li - 1]
                )
            };
            writeln!(s,"for(var i=0u;i<{}u;i++){{var contribution=0.0;if valid{{contribution=grad[{}u]*{};previous[i]+=grad[{}u]*nn_cache[{}u*{}u+i];}}let sum=subgroupAdd(contribution);if elected{{atomicStore(&gradients[config.dense_gradient_offset+(wid.x*config.subgroup_capacity+subgroup_id)*config.parameter_count+{}u+{}u*{}u+i],bitcast<u32>(sum));}}}}",input_width,j,prev_expr,j,j,input_width,param_offsets[li],j,input_width).unwrap();
            writeln!(s,"var bias_contribution=0.0;if valid{{bias_contribution=grad[{}u];}}let bias_sum=subgroupAdd(bias_contribution);if elected{{atomicStore(&gradients[config.dense_gradient_offset+(wid.x*config.subgroup_capacity+subgroup_id)*config.parameter_count+{}u+{}u+{}u],bitcast<u32>(bias_sum));}}",j,param_offsets[li],weight_count,j).unwrap();
            s.push_str("}\n");
        }
        s.push_str("workgroupBarrier();\n");
        writeln!(
            s,
            "for(var i=0u;i<{}u;i++){{grad[i]=previous[i];}}",
            input_width
        )
        .unwrap();
    }
    writeln!(s,"if valid {{var encoded_grad:Encoded;for(var i=0u;i<{}u;i++){{encoded_grad[i]=grad[i];}}nn_encode_backward(input,encoded,encoded_grad,sample);}}\n}}",spec.encoded_size).unwrap();
    s
}

fn adam_shader(spec: &ShaderSpec<'_>) -> String {
    let half_parameters = spec.desc.precision == Precision::F16;
    let mut s = common(spec, 4, true, false, false);
    writeln!(s,"@group(0) @binding(0) var<uniform> config:NnConfig;\n@group(0) @binding(1) var<storage,read_write> parameters:array<{}>;\n@group(0) @binding(2) var<storage,read_write> gradients:array<atomic<u32>>;\n@group(0) @binding(3) var<storage,read_write> moments:array<f32>;", parameter_scalar(spec)).unwrap();
    write!(s,"fn nn_adam_update(value:f32,gradient:f32,index:u32)->f32{{let m=config.beta1*moments[index*2u]+(1.0-config.beta1)*gradient;let v=config.beta2*moments[index*2u+1u]+(1.0-config.beta2)*gradient*gradient;moments[index*2u]=m;moments[index*2u+1u]=v;let mf=m/(1.0-pow(config.beta1,f32(config.step)));let vf=v/(1.0-pow(config.beta2,f32(config.step)));return value-config.learning_rate*mf/(sqrt(vf)+config.epsilon);}}\n@compute @workgroup_size({}) fn nn_adam(@builtin(global_invocation_id) gid:vec3<u32>){{let index=gid.x;if index>=config.parameter_count+config.trainable_count||config.sample_count==0u{{return;}}if index<config.parameter_count{{var sum=0.0;for(var w=0u;w<config.workgroup_count;w++){{let count=atomicLoad(&gradients[w]);for(var sg=0u;sg<count;sg++){{sum+=bitcast<f32>(atomicLoad(&gradients[config.dense_gradient_offset+(w*config.subgroup_capacity+sg)*config.parameter_count+index]));}}}}",spec.desc.workgroup_size).unwrap();
    if half_parameters {
        s.push_str("let value=f32(parameters[index]);let updated=nn_adam_update(value,sum/f32(config.sample_count),index);parameters[index]=f16(updated);");
    } else {
        s.push_str("parameters[index]=nn_adam_update(parameters[index],sum/f32(config.sample_count),index);");
    }
    s.push_str("return;}let auxiliary_index=index-config.parameter_count;\n");
    let mut off = 0u32;
    for aux in &spec.desc.auxiliary {
        if let AuxiliaryMode::Trainable { elements } = aux.mode {
            writeln!(s,"if auxiliary_index>={}u&&auxiliary_index<{}u{{let i=auxiliary_index-{}u;let g=bitcast<f32>(atomicLoad(&gradients[config.aux_gradient_offset+auxiliary_index]))/f32(config.sample_count);{}[i]=nn_adam_update({}[i],g,index);return;}}",off,off+elements,off,aux.name,aux.name).unwrap();
            off += elements;
        }
    }
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ShaderLanguage,
        generic::parse_shader,
        kernels::nn::{HashGrid, HashGridLevel, LayerSpec, NnDesc, TrainableAuxSpec},
    };

    fn validate_all(spec: &ShaderSpec<'_>) {
        let generated = generate(spec).expect("valid shader descriptor");
        for source in [
            generated.inference,
            generated.training_forward,
            generated.backward,
            generated.adam,
        ] {
            let (module, info, _) = parse_shader(
                source.as_bytes(),
                Some("generated-nn.wgsl"),
                ShaderLanguage::Wgsl,
            )
            .expect("generated WGSL must parse and validate");
            #[cfg(mev_backend = "vulkan")]
            naga::back::spv::write_vec(&module, &info, &Default::default(), None)
                .expect("generated WGSL must compile to SPIR-V");
            #[cfg(not(mev_backend = "vulkan"))]
            let _ = (module, info);
        }
    }

    fn with_spec<T>(desc: &NnDesc, output_size: u32, f: impl FnOnce(&ShaderSpec<'_>) -> T) -> T {
        let (input_size, encoded_size) =
            super::super::validate_inputs(&desc.inputs).expect("valid input encodings");
        let mut input = encoded_size;
        let mut parameter_count = 0;
        let mut max_layer_width = encoded_size;
        let mut max_layer_parameter_count = 0;
        let mut tape_offset = encoded_size;
        let mut layers = Vec::new();
        for &output in &desc.layers {
            let layer_parameter_count = input * output + output;
            layers.push(LayerSpec {
                input_size: input,
                output_size: output,
                parameter_offset: parameter_count,
                tape_offset,
            });
            parameter_count += layer_parameter_count;
            max_layer_width = max_layer_width.max(output);
            max_layer_parameter_count = max_layer_parameter_count.max(layer_parameter_count);
            tape_offset += 2 * output;
            input = output;
        }
        let mut trainable_auxiliary = Vec::new();
        let mut element_offset = 0;
        for (auxiliary_index, auxiliary) in desc.auxiliary.iter().enumerate() {
            if let AuxiliaryMode::Trainable { elements } = auxiliary.mode {
                trainable_auxiliary.push(TrainableAuxSpec {
                    auxiliary_index,
                    element_offset,
                    elements,
                });
                element_offset += elements;
            }
        }
        let shader_spec = ShaderSpec {
            desc,
            input_size,
            encoded_size,
            network_output_size: *desc.layers.last().expect("layers are nonempty"),
            output_size,
            target_size: output_size,
            parameter_count,
            max_layer_width,
            max_layer_parameter_count,
            subgroup_capacity: desc.workgroup_size,
            tape_stride: tape_offset + output_size,
            layers: &layers,
            trainable_auxiliary: &trainable_auxiliary,
        };
        f(&shader_spec)
    }

    fn validate_desc(desc: &NnDesc, output_size: u32) {
        with_spec(desc, output_size, validate_all);
    }

    #[test]
    fn validates_builtin_variants_and_multiple_widths() {
        for precision in [Precision::F32, Precision::F16] {
            for activation in [
                Activation::Identity,
                Activation::Relu,
                Activation::Sigmoid,
                Activation::Tanh,
            ] {
                let mut desc = NnDesc::new(&[Encoding::Identity { arity: 3 }], &[5, 3, 1]);
                desc.precision = precision;
                desc.activation = activation.clone();
                desc.output = Output::Activation(activation);
                desc.workgroup_size = 7;
                validate_desc(&desc, 1);
                if precision == Precision::F16 {
                    let shaders = with_spec(&desc, 1, generate).expect("valid descriptor");
                    assert_half_inference(&shaders.inference);
                }
            }
        }
    }

    fn assert_half_inference(source: &str) {
        let (module, info, _) =
            parse_shader(source.as_bytes(), None, ShaderLanguage::Wgsl).expect("valid inference");
        let half = naga::TypeInner::Scalar(naga::Scalar {
            kind: naga::ScalarKind::Float,
            width: 2,
        });
        for name in ["parameters", "inputs", "outputs", "nn_cache"] {
            let (_, variable) = module
                .global_variables
                .iter()
                .find(|(_, variable)| variable.name.as_deref() == Some(name))
                .expect("required storage or cache");
            let naga::TypeInner::Array { base, stride, .. } = module.types[variable.ty].inner
            else {
                panic!("{name} must be a scalar array");
            };
            assert_eq!(module.types[base].inner, half, "{name} must contain f16");
            assert_eq!(stride, 2, "{name} must use packed f16");
        }
        let assert_function = |function: &naga::Function, info: &naga::valid::FunctionInfo| {
            let mut operations = 0;
            for (handle, expression) in function.expressions.iter() {
                if matches!(
                    expression,
                    naga::Expression::Binary { .. }
                        | naga::Expression::Unary { .. }
                        | naga::Expression::Math { .. }
                ) {
                    let ty = info[handle].ty.inner_with(&module.types);
                    let scalar = match ty {
                        naga::TypeInner::Scalar(scalar)
                        | naga::TypeInner::Vector { scalar, .. }
                        | naga::TypeInner::Matrix { scalar, .. } => Some(scalar),
                        _ => None,
                    };
                    if let Some(scalar) =
                        scalar.filter(|scalar| scalar.kind == naga::ScalarKind::Float)
                    {
                        assert_eq!(
                            scalar.width, 2,
                            "inference arithmetic must stay f16: {expression:?}"
                        );
                        operations += 1;
                    }
                }
            }
            operations
        };
        for (handle, function) in module.functions.iter() {
            assert_function(function, &info[handle]);
        }
        assert!(
            assert_function(&module.entry_points[0].function, info.get_entry_point(0)) > 0,
            "dense inference must contain native f16 arithmetic"
        );
    }

    #[test]
    fn validates_half_scalar_custom_hooks_without_rewriting() {
        let mut desc = NnDesc::new(&[Encoding::Identity { arity: 3 }], &[5, 3, 1]);
        desc.precision = Precision::F16;
        desc.prelude = "struct Settings { scale:f32, };".into();
        desc.auxiliary = vec![
            super::super::AuxiliaryDesc::new("settings", "Settings", AuxiliaryMode::Uniform),
            super::super::AuxiliaryDesc::new("lookup", "array<f32>", AuxiliaryMode::ReadOnly),
            super::super::AuxiliaryDesc::new(
                "gain",
                "array<f32>",
                AuxiliaryMode::Trainable { elements: 3 },
            ),
        ];
        let encoding = "alias Position=array<Scalar,3>;fn encode_position(input:Position,sample:u32)->Position{var r:Position;for(var i=0u;i<3u;i++){r[i]=input[i]*Scalar(gain[i])*Scalar(lookup[sample])*Scalar(settings.scale);}return r;}fn backprop_position(input:Position,encoded:Position,grad:Position,sample:u32){for(var i=0u;i<3u;i++){nn_add_gain_gradient(i,grad[i]*input[i]*Scalar(lookup[sample])*Scalar(settings.scale));}}";
        let activation = "fn nn_activation(x:Scalar)->Scalar{return x*x;}fn nn_activation_derivative(x:Scalar,y:Scalar)->Scalar{return 2.0*x;}";
        let output = "fn nn_output(values:NetworkOutput,sample:u32)->Output{var r:Output;r[0]=values[0]*0.25;return r;}fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{var r:NetworkOutput;r[0]=grad[0]*0.25;return r;}";
        let loss = "fn nn_loss(prediction:Output,target_value:Target)->f32{let d=prediction[0]-target_value[0];return 0.5*d*d;}fn nn_loss_derivative(prediction:Output,target_value:Target)->Output{var r:Output;r[0]=prediction[0]-target_value[0];return r;}";
        desc.inputs = vec![Encoding::Custom {
            input_arity: 3,
            output_arity: 3,
            source: encoding.into(),
            forward: "encode_position".into(),
            backward: "backprop_position".into(),
        }];
        desc.activation = Activation::Custom(activation.into());
        desc.output = Output::Custom {
            output_size: 1,
            source: output.into(),
        };
        desc.loss = Loss::Custom {
            target_size: 1,
            source: loss.into(),
        };
        validate_desc(&desc, 1);
        let shaders = with_spec(&desc, 1, generate).expect("valid descriptor");
        for source in [
            &shaders.inference,
            &shaders.training_forward,
            &shaders.backward,
        ] {
            for custom in [encoding, activation, output] {
                assert!(source.contains(custom), "custom source must stay verbatim");
            }
        }
        assert!(shaders.backward.contains(loss));
        assert_half_inference(&shaders.inference);
    }

    #[test]
    fn rejects_auxiliary_types_depending_on_pass_scalar() {
        for (prelude, resource_type) in [
            ("", "array<Scalar>"),
            (
                "alias Value = vec3<Scalar>; struct Values { data:array<Value>, };",
                "Values",
            ),
            ("struct Values { matrix:mat2x3<Scalar>, };", "Values"),
            ("struct Values { @size(16) value:Scalar, };", "Values"),
        ] {
            let mut desc = NnDesc::new(&[Encoding::Identity { arity: 3 }], &[1]);
            desc.precision = Precision::F16;
            desc.prelude = prelude.into();
            desc.auxiliary = vec![super::super::AuxiliaryDesc::new(
                "lookup",
                resource_type,
                AuxiliaryMode::ReadOnly,
            )];
            let error = with_spec(&desc, 1, generate)
                .err()
                .expect("resource layout mismatch");
            assert!(matches!(error, NnError::InvalidDescriptor(_)), "{error}");
        }
    }

    #[test]
    fn validates_custom_vector_hooks_and_auxiliary_bindings() {
        let mut desc = NnDesc::new(&[Encoding::Identity { arity: 3 }], &[4, 3, 2]);
        desc.prelude = "struct Settings { scale:f32, };".into();
        desc.auxiliary = vec![
            super::super::AuxiliaryDesc::new("settings", "Settings", AuxiliaryMode::Uniform),
            super::super::AuxiliaryDesc::new("lookup", "array<f32>", AuxiliaryMode::ReadOnly),
            super::super::AuxiliaryDesc::new(
                "scratch",
                "array<atomic<u32>>",
                AuxiliaryMode::ReadWrite,
            ),
            super::super::AuxiliaryDesc::new(
                "gain",
                "array<f32>",
                AuxiliaryMode::Trainable { elements: 4 },
            ),
        ];
        desc.inputs = vec![Encoding::Custom { input_arity: 3, output_arity: 4, source: "alias RawInput=array<f32,3>;alias Features=array<f32,4>;fn encode_features(input:RawInput,sample:u32)->Features{var r:Features;r[0]=input[0];r[1]=input[1];r[2]=input[2];r[3]=lookup[sample]*gain[0]*settings.scale;atomicAdd(&scratch[0],1u);return r;}fn backprop_features(input:RawInput,encoded:Features,grad:Features,sample:u32){nn_add_gain_gradient(0u,grad[3]*lookup[sample]*settings.scale);}".into(), forward: "encode_features".into(), backward: "backprop_features".into() }];
        desc.activation = Activation::Custom("fn nn_activation(x:f32)->f32{return x*x;}fn nn_activation_derivative(x:f32,y:f32)->f32{return 2.0*x;}".into());
        desc.output = Output::Custom { output_size: 2, source: "fn nn_output(values:NetworkOutput,sample:u32)->Output{var r:Output;r[0]=values[0]+values[1];r[1]=values[1]-values[0];return r;}fn nn_output_backward(values:NetworkOutput,prediction:Output,grad:Output,sample:u32)->NetworkOutput{var r:NetworkOutput;r[0]=grad[0]-grad[1];r[1]=grad[0]+grad[1];return r;}".into() };
        desc.loss = Loss::Custom { target_size: 2, source: "fn nn_loss(prediction:Output,target_value:Target)->f32{let a=prediction[0]-target_value[0];let b=prediction[1]-target_value[1];return 0.5*(a*a+b*b);}fn nn_loss_derivative(prediction:Output,target_value:Target)->Output{var r:Output;r[0]=prediction[0]-target_value[0];r[1]=prediction[1]-target_value[1];return r;}".into() };
        desc.workgroup_size = 7;
        validate_desc(&desc, 2);
    }

    #[test]
    fn validates_ordered_vectors_and_shared_encoding_source() {
        let source = "alias Pair=array<Scalar,2>;alias Triple=array<Scalar,3>;alias Single=array<Scalar,1>;\nfn expand_pair(input:Pair,sample:u32)->Triple{var r:Triple;r[0]=input[0]*Scalar(gain[0]);r[1]=input[1];r[2]=input[0]+input[1];return r;}\nfn backprop_pair(input:Pair,encoded:Triple,grad:Triple,sample:u32){nn_add_gain_gradient(0u,grad[0]*input[0]);}\nfn reduce_triple(input:Triple,sample:u32)->Single{var r:Single;r[0]=(input[0]+input[1]+input[2])*Scalar(gain[1]);return r;}\nfn backprop_triple(input:Triple,encoded:Single,grad:Single,sample:u32){nn_add_gain_gradient(1u,grad[0]*(input[0]+input[1]+input[2]));}";
        let inputs = [
            Encoding::Identity { arity: 2 },
            Encoding::Custom {
                input_arity: 2,
                output_arity: 3,
                source: source.into(),
                forward: "expand_pair".into(),
                backward: "backprop_pair".into(),
            },
            Encoding::Custom {
                input_arity: 3,
                output_arity: 1,
                source: source.into(),
                forward: "reduce_triple".into(),
                backward: "backprop_triple".into(),
            },
        ];
        for precision in [Precision::F32, Precision::F16] {
            let mut desc = NnDesc::new(&inputs, &[5, 3, 1]);
            desc.precision = precision;
            desc.workgroup_size = 7;
            desc.auxiliary = vec![super::super::AuxiliaryDesc::new(
                "gain",
                "array<f32>",
                AuxiliaryMode::Trainable { elements: 2 },
            )];
            validate_desc(&desc, 1);
            let shaders = with_spec(&desc, 1, generate).expect("valid descriptor");
            for shader in [
                &shaders.inference,
                &shaders.training_forward,
                &shaders.backward,
            ] {
                assert_eq!(
                    shader.matches(source).count(),
                    1,
                    "shared encoding source appears once verbatim"
                );
            }
            if precision == Precision::F16 {
                assert_half_inference(&shaders.inference);
            }
        }
    }

    #[test]
    fn validates_hashgrids_mixed_with_identity_and_custom_vectors() -> Result<(), NnError> {
        let first = HashGrid::new(&[
            HashGridLevel {
                resolution: 1,
                features: 2,
                table_size: 64,
            },
            HashGridLevel {
                resolution: 3,
                features: 3,
                table_size: 7,
            },
            HashGridLevel {
                resolution: 8192,
                features: 1,
                table_size: 8,
            },
        ])?;
        let second = HashGrid::new(&[HashGridLevel {
            resolution: 2,
            features: 1,
            table_size: 9,
        }])?;
        let inputs = [
            Encoding::Identity { arity: 1 },
            Encoding::HashGrid {
                grid: first.clone(),
                auxiliary: "first_grid".into(),
            },
            Encoding::Custom {
                input_arity: 2,
                output_arity: 1,
                source: "fn encode_pair(input:array<Scalar,2>,sample:u32)->array<Scalar,1>{return array<Scalar,1>(input[0]+input[1]);}fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,1>,grad:array<Scalar,1>,sample:u32){}".into(),
                forward: "encode_pair".into(),
                backward: "backprop_pair".into(),
            },
            Encoding::HashGrid {
                grid: second.clone(),
                auxiliary: "second_grid".into(),
            },
        ];
        for precision in [Precision::F32, Precision::F16] {
            for trainable in [false, true] {
                let mut desc = NnDesc::new(&inputs, &[5, 2]);
                desc.precision = precision;
                desc.workgroup_size = 7;
                desc.auxiliary = vec![
                    first.auxiliary("first_grid", trainable),
                    second.auxiliary("second_grid", !trainable),
                ];
                validate_desc(&desc, 2);
                if precision == Precision::F16 {
                    let shaders = with_spec(&desc, 2, generate)?;
                    assert_half_inference(&shaders.inference);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn validates_distinct_aliases_for_each_encoding_argument() {
        let source = "alias ForwardScalar=Scalar;alias BackwardScalar=Scalar;alias SavedScalar=Scalar;alias GradientScalar=Scalar;\nfn encode_pair(input:array<ForwardScalar,2>,sample:u32)->array<ForwardScalar,3>{var r:array<ForwardScalar,3>;r[0]=input[0];r[1]=input[1];r[2]=input[0]+input[1];return r;}\nfn backprop_pair(input:array<BackwardScalar,2>,encoded:array<SavedScalar,3>,grad:array<GradientScalar,3>,sample:u32){}";
        for precision in [Precision::F32, Precision::F16] {
            let inputs = [Encoding::Custom {
                input_arity: 2,
                output_arity: 3,
                source: source.into(),
                forward: "encode_pair".into(),
                backward: "backprop_pair".into(),
            }];
            let mut desc = NnDesc::new(&inputs, &[1]);
            desc.precision = precision;
            validate_desc(&desc, 1);
        }
    }

    #[test]
    fn rejects_custom_encoding_signature_arity_mismatches() {
        let forward_valid =
            "fn encode_pair(input:array<Scalar,2>,sample:u32)->array<Scalar,2>{return input;}";
        let backward_valid = "fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,2>,grad:array<Scalar,2>,sample:u32){}";
        let cases = [
            (
                "fn encode_pair(input:array<Scalar,3>,sample:u32)->array<Scalar,2>{return array<Scalar,2>();}",
                backward_valid,
            ),
            (
                "fn encode_pair(input:array<Scalar,2>,sample:u32)->array<Scalar,3>{return array<Scalar,3>();}",
                backward_valid,
            ),
            (
                forward_valid,
                "fn backprop_pair(input:array<Scalar,3>,encoded:array<Scalar,2>,grad:array<Scalar,2>,sample:u32){}",
            ),
            (
                forward_valid,
                "fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,2>,grad:array<Scalar,3>,sample:u32){}",
            ),
            (
                forward_valid,
                "fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,3>,grad:array<Scalar,2>,sample:u32){}",
            ),
            (
                forward_valid,
                "fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,2>,grad:array<Scalar,2>,sample:i32){}",
            ),
            (
                forward_valid,
                "fn backprop_pair(input:array<Scalar,2>,encoded:array<Scalar,2>,grad:array<Scalar,2>,sample:u32)->Scalar{return 0.0;}",
            ),
        ];
        for precision in [Precision::F32, Precision::F16] {
            for (forward, backward) in cases {
                let inputs = [Encoding::Custom {
                    input_arity: 2,
                    output_arity: 2,
                    source: format!("{forward}\n{backward}"),
                    forward: "encode_pair".into(),
                    backward: "backprop_pair".into(),
                }];
                let mut desc = NnDesc::new(&inputs, &[1]);
                desc.precision = precision;
                assert!(
                    matches!(
                        with_spec(&desc, 1, generate),
                        Err(NnError::InvalidDescriptor(_))
                    ),
                    "encoding hook types must match the declared arities"
                );
            }
        }
    }
}
