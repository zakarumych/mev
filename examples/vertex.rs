use mev::{Arguments, VertexBinding};

const WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> transform: mat4x4<f32>;

fn main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return transform * vec4<f32>(input.position, 1.0);
}
"#;

#[derive(VertexBinding)]
struct MyVertexBuffer {
    color: mev::vec4,
    uv: mev::vec2,
    position: mev::vec3,
}

#[derive(Arguments)]
struct MyArguments {
    #[mev(vertex)]
    transform: mev::Buffer,
}

fn make_pipeline(device: &mev::Device) -> mev::RenderPipeline {
    let vertex_layouts = vec![MyVertexBuffer::LAYOUT];
    let mut vertex_attributes = vec![];

    vertex_attributes.extend(MyVertexBuffer::descs(0).position);
    vertex_attributes.extend(MyVertexBuffer::descs(0).color);
    vertex_attributes.extend(MyVertexBuffer::descs(0).uv);

    device
        .new_render_pipeline(mev::RenderPipelineDesc {
            name: "my_pipeline",
            vertex_shader: device
                .new_shader_library(mev::LibraryDesc {
                    name: "my_shader",
                    input: mev::LibraryInput::wgsl(WGSL),
                })
                .unwrap()
                .entry("main"),
            vertex_layouts,
            vertex_attributes,
            primitive_topology: mev::PrimitiveTopology::Triangle,
            raster: None,
            constants: 0,
            arguments: &[MyArguments::LAYOUT],
        })
        .unwrap()
}

fn make_pipeline_old(device: &mev::Device) -> mev::RenderPipeline {
    device
        .new_render_pipeline(mev::RenderPipelineDesc {
            name: "my_pipeline",
            vertex_shader: device
                .new_shader_library(mev::LibraryDesc {
                    name: "my_shader",
                    input: mev::LibraryInput::wgsl(WGSL),
                })
                .unwrap()
                .entry("main"),
            vertex_layouts: vec![mev::VertexLayoutDesc {
                step_mode: mev::VertexStepMode::Vertex,
                stride: 4 * 4 + 4 * 2 + 4 * 3,
            }],
            vertex_attributes: vec![
                mev::VertexAttributeDesc {
                    buffer_index: 0,
                    format: mev::VertexFormat::Float32x3,
                    offset: 4 * 4 + 4 * 2, // position after color and uv
                },
                mev::VertexAttributeDesc {
                    buffer_index: 0,
                    format: mev::VertexFormat::Float32x4,
                    offset: 0, // color at the start
                },
                mev::VertexAttributeDesc {
                    buffer_index: 0,
                    format: mev::VertexFormat::Float32x2,
                    offset: 4 * 4, // uv after color
                },
            ],
            primitive_topology: mev::PrimitiveTopology::Triangle,
            raster: None,
            constants: 0,
            arguments: &[mev::ArgumentGroupLayout {
                arguments: &[mev::ArgumentLayout {
                    kind: mev::ArgumentKind::UniformBuffer,
                    size: 1,
                    stages: mev::ShaderStages::VERTEX,
                }],
            }],
        })
        .unwrap()
}

fn main() {
    return;
    make_pipeline_old(todo!());
    make_pipeline(todo!());
}
