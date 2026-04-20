mod app;
mod camera;
mod geometry;
mod state;

fn main() {
    env_logger::init();
    pollster::block_on(app::run());
}
