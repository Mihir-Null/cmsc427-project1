use std::sync::Arc;

use winit::dpi::PhysicalPosition;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use crate::state::State;

pub async fn run() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("CMSC427 Project 1 – Scene Explorer")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720u32))
            .build(&event_loop)
            .expect("failed to create window"),
    );

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.body())
            .and_then(|b| {
                window
                    .canvas()
                    .and_then(|canvas| b.append_child(&canvas).ok())
            });
    }

    let mut state = State::new(Arc::clone(&window)).await;

    #[cfg(not(target_arch = "wasm32"))]
    let mut last_time = std::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let mut last_time_ms = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let mut last_cursor: Option<PhysicalPosition<f64>> = None;

    let event_handler =
        move |event: Event<()>, elwt: &winit::event_loop::EventLoopWindowTarget<()>| {
            elwt.set_control_flow(ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::Resized(s) => {
                            state.resize(s);
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {}
                        WindowEvent::CursorLeft { .. } => {
                            last_cursor = None;
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            if let Some(prev) = last_cursor {
                                state
                                    .camera
                                    .process_mouse_motion(position.x - prev.x, position.y - prev.y);
                            }
                            last_cursor = Some(position);
                        }
                        WindowEvent::KeyboardInput {
                            event:
                                winit::event::KeyEvent {
                                    physical_key: PhysicalKey::Code(key),
                                    state: ks,
                                    ..
                                },
                            ..
                        } => {
                            if key == KeyCode::Escape {
                                elwt.exit();
                            }
                            state.camera.process_key(key, ks);
                        }
                        WindowEvent::RedrawRequested => match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let sz = state.size;
                                state.resize(sz);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                elwt.exit();
                            }
                            Err(wgpu::SurfaceError::Timeout) => {}
                        },
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    #[cfg(not(target_arch = "wasm32"))]
                    let dt = {
                        let now = std::time::Instant::now();
                        let dt = now.duration_since(last_time).as_secs_f32();
                        last_time = now;
                        dt.min(0.1)
                    };
                    #[cfg(target_arch = "wasm32")]
                    let dt = {
                        let now = web_sys::window()
                            .and_then(|w| w.performance())
                            .map(|p| p.now())
                            .unwrap_or(last_time_ms);
                        let dt = ((now - last_time_ms) / 1000.0) as f32;
                        last_time_ms = now;
                        dt.min(0.1)
                    };
                    state.update(dt);
                    window.request_redraw();
                }
                _ => {}
            }
        };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(event_handler).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn(event_handler);
    }
}
