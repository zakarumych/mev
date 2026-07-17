#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use mev::{Arguments, AutoDeviceRepr, VertexBinding};
use winit::application::ApplicationHandler;

const WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Uniforms {
    transform: mat4x4<f32>,
    projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_pos = uniforms.transform * vec4<f32>(input.position, 1.0);
    let clip_pos = uniforms.projection * world_pos;
    return VertexOutput(clip_pos, input.color);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit, VertexBinding)]
struct MyVertexBuffer {
    position: mev::vec3,
    color: mev::vec4,
    uv: mev::vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, AutoDeviceRepr)]
struct Uniforms {
    transform: mev::mat4x4,
    projection: mev::mat4x4,
}

#[derive(Arguments)]
struct MyArguments {
    #[mev(uniform)]
    #[mev(vertex)]
    uniforms: mev::Buffer,
}

struct VertexApp {
    start: Instant,
    last_format: Option<mev::PixelFormat>,
    vertex_buffer: mev::Buffer,
    index_buffer: mev::Buffer,
    index_count: u32,
    uniform_buffer: mev::Buffer,
    pipeline: Option<mev::RenderPipeline>,
    surface: Option<mev::Surface>,
    queue: mev::Queue,
    window: Option<winit::window::Window>,
}

impl ApplicationHandler for VertexApp {
    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.queue.wait_idle().unwrap();
        self.pipeline = None;
        self.surface = None;
        self.window = None;
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.queue.wait_idle().unwrap();
        self.pipeline = None;
        self.surface = None;
        self.window = None;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::Resized(size) => {
                let surface = self.surface.as_mut().unwrap();
                surface.preferred_extent(mev::Extent2::new(size.width, size.height));
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.render();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            #[allow(unused_mut)]
            let mut attributes = winit::window::WindowAttributes::default();

            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;
                let canvas = web_sys::window()
                    .unwrap()
                    .document()
                    .unwrap()
                    .get_element_by_id("mev_example")
                    .unwrap()
                    .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
                    .unwrap();

                attributes = attributes.with_canvas(Some(canvas));
            }

            let window = event_loop.create_window(attributes).unwrap();
            let mut surface = self.queue.new_surface(&window, &window).unwrap();

            let size = window.inner_size();
            surface.preferred_extent(mev::Extent2::new(size.width, size.height));
            surface.preferred_usage(mev::ImageUsage::TARGET);

            self.window = Some(window);
            self.surface = Some(surface);
        }

        self.window.as_ref().unwrap().request_redraw();
    }
}

impl VertexApp {
    fn render(&mut self) {
        let mut frame = self.surface.as_mut().unwrap().next_frame().unwrap();

        let target_format = frame.image().format();
        let target_extent = frame.image().extent();
        let angle_x = self.start.elapsed().as_secs_f32() * 0.5;
        let angle_y = self.start.elapsed().as_secs_f32() * 0.7;

        if self.pipeline.is_none() || self.last_format != Some(target_format) {
            let library = self
                .queue
                .new_shader_library(mev::LibraryDesc {
                    name: "main",
                    input: mev::LibraryInput::wgsl(WGSL),
                })
                .unwrap();

            let vertex_layouts = vec![MyVertexBuffer::LAYOUT];
            let mut vertex_attributes = vec![];

            vertex_attributes.extend(MyVertexBuffer::descs(0).position);
            vertex_attributes.extend(MyVertexBuffer::descs(0).color);
            vertex_attributes.extend(MyVertexBuffer::descs(0).uv);

            let pipeline = self
                .queue
                .new_render_pipeline(mev::RenderPipelineDesc {
                    name: "main",
                    vertex_shader: mev::Shader {
                        library: library.clone(),
                        entry: "vs_main".into(),
                    },
                    vertex_layouts,
                    vertex_attributes,
                    primitive_topology: mev::PrimitiveTopology::Triangle,
                    raster: Some(mev::RasterDesc {
                        fragment_shader: Some(mev::Shader {
                            library,
                            entry: "fs_main".into(),
                        }),
                        color_targets: vec![mev::ColorTargetDesc {
                            format: target_format,
                            blend: Some(mev::BlendDesc::default()),
                        }],
                        depth_stencil: None,
                        front_face: mev::FrontFace::default(),
                        culling: mev::Culling::Back,
                    }),
                    constants: 0,
                    arguments: &[MyArguments::LAYOUT],
                })
                .unwrap();

            self.pipeline = Some(pipeline);
            self.last_format = Some(target_format);
        }

        let pipeline = self.pipeline.as_ref().unwrap();

        // Build rotation matrices for the spinning cube.
        let cos_x = angle_x.cos();
        let sin_x = angle_x.sin();
        let cos_y = angle_y.cos();
        let sin_y = angle_y.sin();

        let rot_x = mev::mat4x4(
            mev::vec4(1.0, 0.0, 0.0, 0.0),
            mev::vec4(0.0, cos_x, sin_x, 0.0),
            mev::vec4(0.0, -sin_x, cos_x, 0.0),
            mev::vec4(0.0, 0.0, 0.0, 1.0),
        );

        let rot_y = mev::mat4x4(
            mev::vec4(cos_y, 0.0, -sin_y, 0.0),
            mev::vec4(0.0, 1.0, 0.0, 0.0),
            mev::vec4(sin_y, 0.0, cos_y, 0.0),
            mev::vec4(0.0, 0.0, 0.0, 1.0),
        );

        let translate = mev::mat4x4(
            mev::vec4(1.0, 0.0, 0.0, 0.0),
            mev::vec4(0.0, 1.0, 0.0, 0.0),
            mev::vec4(0.0, 0.0, 1.0, 0.0),
            mev::vec4(0.0, 0.0, -3.0, 1.0),
        );

        let transform = translate * rot_y * rot_x;

        // Perspective projection matrix using Vulkan/Metal conventions:
        // NDC depth range is [0, 1] (not OpenGL's [-1, 1]), and NDC y points
        // down, so the y scale is negated to keep the image right-side up
        // and preserve the winding order used for back-face culling.
        let extent = target_extent.into_2d();
        let aspect = extent.width() as f32 / extent.height() as f32;
        let fov = 1.0f32;
        let near = 0.1f32;
        let far = 100.0f32;
        let f = 1.0 / (fov / 2.0).tan();

        let projection = mev::mat4x4(
            mev::vec4(f / aspect, 0.0, 0.0, 0.0),
            mev::vec4(0.0, -f, 0.0, 0.0),
            mev::vec4(0.0, 0.0, far / (near - far), -1.0),
            mev::vec4(0.0, 0.0, (far * near) / (near - far), 0.0),
        );

        let uniforms = Uniforms {
            transform,
            projection,
        };

        self.queue
            .sync_frame(&mut frame, mev::PipelineStages::COLOR_OUTPUT);
        let mut encoder = self.queue.new_command_encoder();
        encoder.init_image(
            mev::PipelineStages::empty(),
            mev::PipelineStages::COLOR_OUTPUT,
            frame.image(),
        );

        {
            let mut copy = encoder.copy();

            copy.barrier(
                mev::PipelineStages::VERTEX_SHADER,
                mev::PipelineStages::TRANSFER,
            );
            copy.write_buffer(&self.uniform_buffer, &uniforms);
            copy.barrier(
                mev::PipelineStages::TRANSFER,
                mev::PipelineStages::VERTEX_SHADER,
            );
        }

        {
            let mut render = encoder.render(mev::RenderPassDesc {
                name: "main",
                color_attachments: &[
                    mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)
                ],
                depth_stencil_attachment: None,
            });

            render.with_viewport(mev::Offset3::ZERO, target_extent.into_3d().cast_as_f32());
            render.with_scissor(mev::Offset2::ZERO, extent);
            render.with_pipeline(pipeline);
            render.with_arguments(
                0,
                &MyArguments {
                    uniforms: self.uniform_buffer.clone(),
                },
            );
            render.bind_vertex_buffers(0, &[&self.vertex_buffer]);
            render.bind_index_buffer(&self.index_buffer);
            render.draw_indexed(0, 0..self.index_count, 0..1);
        }

        encoder.present(frame, mev::PipelineStages::COLOR_OUTPUT);
        let cbuf = encoder.finish();

        self.window.as_ref().unwrap().pre_present_notify();
        self.queue.submit_checkpoint([cbuf]).unwrap();
    }
}

fn cube_geometry() -> (Vec<MyVertexBuffer>, Vec<u32>) {
    let size = 0.5;
    let vertices = vec![
        // Front face (z+)
        MyVertexBuffer {
            position: mev::vec3(-size, -size, size),
            color: mev::vec4(1.0, 0.0, 0.0, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, -size, size),
            color: mev::vec4(1.0, 0.5, 0.0, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, size),
            color: mev::vec4(1.0, 1.0, 0.0, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, size, size),
            color: mev::vec4(0.5, 1.0, 0.0, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
        // Back face (z-)
        MyVertexBuffer {
            position: mev::vec3(-size, -size, -size),
            color: mev::vec4(0.0, 1.0, 0.0, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, size, -size),
            color: mev::vec4(0.0, 1.0, 0.5, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, -size),
            color: mev::vec4(0.0, 1.0, 1.0, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, -size, -size),
            color: mev::vec4(0.0, 0.5, 1.0, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
        // Top face (y+)
        MyVertexBuffer {
            position: mev::vec3(-size, size, -size),
            color: mev::vec4(0.0, 0.0, 1.0, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, size, size),
            color: mev::vec4(0.5, 0.0, 1.0, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, size),
            color: mev::vec4(1.0, 0.0, 1.0, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, -size),
            color: mev::vec4(1.0, 0.0, 0.5, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
        // Bottom face (y-)
        MyVertexBuffer {
            position: mev::vec3(-size, -size, -size),
            color: mev::vec4(1.0, 1.0, 0.0, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, -size, -size),
            color: mev::vec4(1.0, 1.0, 0.5, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, -size, size),
            color: mev::vec4(1.0, 1.0, 1.0, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, -size, size),
            color: mev::vec4(1.0, 0.5, 1.0, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
        // Right face (x+)
        MyVertexBuffer {
            position: mev::vec3(size, -size, -size),
            color: mev::vec4(0.5, 1.0, 1.0, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, -size),
            color: mev::vec4(0.0, 1.0, 1.0, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, size, size),
            color: mev::vec4(0.5, 0.5, 1.0, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(size, -size, size),
            color: mev::vec4(1.0, 0.5, 0.5, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
        // Left face (x-)
        MyVertexBuffer {
            position: mev::vec3(-size, -size, -size),
            color: mev::vec4(0.5, 0.5, 0.5, 1.0),
            uv: mev::vec2(0.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, -size, size),
            color: mev::vec4(0.5, 0.5, 1.0, 1.0),
            uv: mev::vec2(1.0, 0.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, size, size),
            color: mev::vec4(0.5, 1.0, 0.5, 1.0),
            uv: mev::vec2(1.0, 1.0),
        },
        MyVertexBuffer {
            position: mev::vec3(-size, size, -size),
            color: mev::vec4(1.0, 0.5, 0.5, 1.0),
            uv: mev::vec2(0.0, 1.0),
        },
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, // Front
        4, 5, 6, 4, 6, 7, // Back
        8, 9, 10, 8, 10, 11, // Top
        12, 13, 14, 12, 14, 15, // Bottom
        16, 17, 18, 16, 18, 19, // Right
        20, 21, 22, 20, 22, 23, // Left
    ];

    (vertices, indices)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let instance = mev::Instance::load().expect("Failed to init graphics");

    let (_device, mut queues) = instance
        .new_device(mev::DeviceDesc {
            idx: 0,
            queues: &[0],
            features: mev::Features::SURFACE,
        })
        .unwrap();
    let queue = queues.pop().unwrap();

    let (vertices, indices) = cube_geometry();

    let vertex_buffer = queue.new_buffer_init(mev::BufferInitDesc {
        data: bytemuck::cast_slice(&vertices),
        name: "vertices",
        usage: mev::BufferUsage::VERTEX,
    });

    let index_buffer = queue.new_buffer_init(mev::BufferInitDesc {
        data: bytemuck::cast_slice(&indices),
        name: "indices",
        usage: mev::BufferUsage::INDEX,
    });

    let uniform_buffer = queue.new_buffer(mev::BufferDesc {
        name: "uniforms",
        size: std::mem::size_of::<Uniforms>(),
        usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
    });

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = VertexApp {
        start: Instant::now(),
        last_format: None,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        uniform_buffer,
        pipeline: None,
        surface: None,
        queue,
        window: None,
    };

    let _ = event_loop.run_app(&mut app);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(tracing_web::MakeWebConsoleWriter::new());

    tracing_subscriber::registry().with(fmt_layer).init();

    wasm_bindgen_futures::spawn_local(async {
        let instance = mev::Instance::load_async()
            .await
            .expect("Failed to init graphics");

        let (_device, mut queues) = instance
            .new_device_async(mev::DeviceDesc {
                idx: 0,
                queues: &[0],
                features: mev::Features::SURFACE,
            })
            .await
            .unwrap();
        let queue = queues.pop().unwrap();

        let (vertices, indices) = cube_geometry();

        let vertex_buffer = queue.new_buffer_init(mev::BufferInitDesc {
            data: bytemuck::cast_slice(&vertices),
            name: "vertices",
            usage: mev::BufferUsage::VERTEX,
        });

        let index_buffer = queue.new_buffer_init(mev::BufferInitDesc {
            data: bytemuck::cast_slice(&indices),
            name: "indices",
            usage: mev::BufferUsage::INDEX,
        });

        let uniform_buffer = queue.new_buffer(mev::BufferDesc {
            name: "uniforms",
            size: std::mem::size_of::<Uniforms>(),
            usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
        });

        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        let mut app = VertexApp {
            start: Instant::now(),
            last_format: None,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            uniform_buffer,
            pipeline: None,
            surface: None,
            queue,
            window: None,
        };

        let _ = event_loop.run_app(&mut app);
    });
}
