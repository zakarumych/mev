#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use winit::application::ApplicationHandler;

struct DummyApp {
    window: Option<winit::window::Window>,
}

impl ApplicationHandler for DummyApp {
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

            self.window = Some(window);
        }

        self.window.as_ref().unwrap().request_redraw();
    }
}

impl DummyApp {
    fn render(&mut self) {
        self.window.as_ref().unwrap().pre_present_notify();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = DummyApp { window: None };

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

    wasm_bindgen_futures::spawn_local(async {
        let instance = mev::Instance::load_async()
            .await
            .expect("Failed to init graphics");

        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        let mut app = DummyApp { window: None };

        let _ = event_loop.run_app(&mut app);
    });
}
