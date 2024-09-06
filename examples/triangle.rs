use std::time::Instant;

use mev::DeviceRepr;
use winit::application::ApplicationHandler;

struct TriangleApp {
    queue: mev::Queue,
    window: Option<winit::window::Window>,
    surface: Option<mev::Surface>,
    last_format: Option<mev::PixelFormat>,
    pipeline: Option<mev::RenderPipeline>,
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
            winit::event::WindowEvent::RedrawRequested => {
                self.render();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(winit::window::Window::default_attributes())
                .unwrap();
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
                    input: mev::include_library!(
                        "shaders/triangle.wgsl" as mev::ShaderLanguage::Wgsl
                    ),
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
                    arguments: &[],
                    constants: TriangleConstants::SIZE,
                })
                .unwrap();

            self.pipeline = Some(pipeline);
            self.last_format = Some(target_format);
        }

        let pipeline = self.pipeline.as_ref().unwrap();

        let mut encoder = self.queue.new_command_encoder().unwrap();
        encoder.init_image(mev::PipelineStages::empty(), mev::PipelineStages::FRAGMENT_SHADER, frame.image());
        {
            let mut render = encoder.render(mev::RenderPassDesc {
                name: "main",
                color_attachments: &[mev::AttachmentDesc::new(frame.image()).clear(mev::ClearColor::DARK_GRAY)],
                depth_stencil_attachment: None,
            });

            render.with_viewport(mev::Offset3::ZERO, target_extent.into_3d().cast_as_f32());
            render.with_scissor(mev::Offset2::ZERO, target_extent.into_2d());
            render.with_pipeline(pipeline);
            render.with_constants(&TriangleConstants {
                angle,
                width: target_extent.width(),
                height: target_extent.height(),
            });
            render.draw(0..3, 0..1);
        }

        self.queue.sync_frame(&mut frame, mev::PipelineStages::FRAGMENT_SHADER);
        encoder.present(frame, mev::PipelineStages::FRAGMENT_SHADER);
        let cbuf = encoder.finish().unwrap();
        self.queue.submit([cbuf], true).unwrap();
    
    }
}

fn main() {
    let instance = mev::Instance::load().expect("Failed to init graphics");

    let (_device, mut queues) = instance
        .create(mev::DeviceDesc {
            idx: 0,
            queues: &[0],
            features: mev::Features::SURFACE,
        })
        .unwrap();
    let queue = queues.pop().unwrap();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = TriangleApp {
        queue,
        window: None,
        surface: None,
        last_format: None,
        pipeline: None,
        start: Instant::now(),
    };
    let _ = event_loop.run_app(&mut app);
}

#[derive(mev::DeviceRepr)]
pub struct TriangleConstants {
    pub angle: f32,
    pub width: u32,
    pub height: u32,
}
