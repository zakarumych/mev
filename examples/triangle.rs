#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use mev::{Arguments, DeviceRepr};
use winit::application::ApplicationHandler;

#[derive(Arguments)]
struct TriangleArguments {
    #[mev(uniform)]
    #[mev(vertex)]
    pc: mev::Buffer,
}

struct TriangleApp {
    queue: mev::Queue,
    window: Option<winit::window::Window>,
    surface: Option<mev::Surface>,
    last_format: Option<mev::PixelFormat>,
    pipeline: Option<mev::RenderPipeline>,
    pc: mev::Buffer,
    start: Instant,
}

impl ApplicationHandler for TriangleApp {
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
            let surface = self.queue.new_surface(&window, &window).unwrap();

            self.window = Some(window);
            self.surface = Some(surface);
        }

        self.window.as_ref().unwrap().request_redraw();
    }
}

impl TriangleApp {
    fn render(&mut self) {
        let mut frame = self.surface.as_mut().unwrap().next_frame().unwrap();

        let target_format = frame.image().format();
        let target_extent = frame.image().extent();
        let angle = self.start.elapsed().as_secs_f32() * 0.1;

        if self.pipeline.is_none() || self.last_format != Some(target_format) {
            let library = self
                .queue
                .new_shader_library(mev::LibraryDesc {
                    name: "main",
                    input: mev::include_library!(Wgsl "shaders/triangle.wgsl"),
                })
                .unwrap();

            let pipeline = self
                .queue
                .new_render_pipeline(mev::RenderPipelineDesc {
                    name: "main",
                    vertex_shader: mev::Shader {
                        library: library.clone(),
                        entry: "vs_main".into(),
                    },
                    vertex_attributes: vec![],
                    vertex_layouts: vec![],
                    primitive_topology: mev::PrimitiveTopology::Triangle,
                    raster: Some(mev::RasterDesc {
                        fragment_shader: Some(mev::Shader {
                            library: library,
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
                    arguments: &[TriangleArguments::LAYOUT],
                    constants: TriangleConstants::SIZE,
                })
                .unwrap();

            self.pipeline = Some(pipeline);
            self.last_format = Some(target_format);
        }

        let pipeline = self.pipeline.as_ref().unwrap();

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

            copy.write_buffer(
                &self.pc,
                &TriangleConstants {
                    angle,
                    width: target_extent.width(),
                    height: target_extent.height(),
                },
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
            render.with_scissor(mev::Offset2::ZERO, target_extent.into_2d());
            render.with_pipeline(pipeline);
            render.with_arguments(
                0,
                &TriangleArguments {
                    pc: self.pc.clone(),
                },
            );
            render.draw(0..3, 0..1);
        }

        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);
        let cbuf = encoder.finish();

        self.window.as_ref().unwrap().pre_present_notify();
        self.queue.submit_checkpoint([cbuf]).unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    mev::match_backend! {
        metal => {
            println!("Metal backend");
        }
        vulkan => {
            println!("Vulkan backend");
        }
        webgpu => {
            println!("WebGPU backend");
        }
    }

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

    let pc = queue.new_buffer(mev::BufferDesc {
        name: "triangle_constants",
        size: TriangleConstants::SIZE,
        usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
    });

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = TriangleApp {
        queue,
        window: None,
        surface: None,
        last_format: None,
        pipeline: None,
        start: Instant::now(),
        pc,
    };

    let _ = event_loop.run_app(&mut app);
}

// #[cfg(target_arch = "wasm32")]
// fn get_canvas_element() -> Option<web_sys::HtmlCanvasElement> {
//     use eframe::wasm_bindgen::JsCast;

//     let document = web_sys::window()?.document()?;
//     let canvas = document.get_element_by_id("egui_snarl_demo")?;
//     canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok()
// }

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false) // Only partially supported across browsers
        .without_time() // std::time is not available in browsers
        .with_writer(tracing_web::MakeWebConsoleWriter::new()); // write events to the console

    tracing_subscriber::registry().with(fmt_layer).init();

    mev::match_backend! {
        metal => {
            tracing::info!("mev Metal backend");
        }
        vulkan => {
            tracing::info!("mev Vulkan backend");
        }
        webgpu => {
            tracing::info!("mev WebGPU backend");
        }
    }

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

        let pc = queue.new_buffer(mev::BufferDesc {
            name: "triangle_constants",
            size: TriangleConstants::SIZE,
            usage: mev::BufferUsage::UNIFORM | mev::BufferUsage::TRANSFER_DST,
        });

        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        let mut app = TriangleApp {
            queue,
            window: None,
            surface: None,
            last_format: None,
            pipeline: None,
            start: Instant::now(),
            pc,
        };

        let _ = event_loop.run_app(&mut app);
    });
}

#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod, mev::AutoDeviceRepr)]
#[repr(C)]
pub struct TriangleConstants {
    pub angle: f32,
    pub width: u32,
    pub height: u32,
}
