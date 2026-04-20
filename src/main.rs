#[cfg(not(target_arch = "wasm32"))]
mod app;
#[cfg(not(target_arch = "wasm32"))]
mod camera;
#[cfg(not(target_arch = "wasm32"))]
mod geometry;
#[cfg(not(target_arch = "wasm32"))]
mod state;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    pollster::block_on(app::run());
}

#[cfg(target_arch = "wasm32")]
fn main() {}
