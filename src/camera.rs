use glam::{Mat4, Vec3};
use std::collections::HashSet;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

pub const PLAYER_HALF_HEIGHT: f32 = 0.85;
pub const MOVE_SPEED: f32 = 6.0; // world units/sec
pub const MOUSE_SENSITIVITY: f32 = 0.0035; // radians/pixel
pub const CAMERA_DISTANCE: f32 = 7.0;
pub const CAMERA_TARGET_HEIGHT: f32 = 1.2;
pub const MIN_PITCH: f32 = -1.05;
pub const MAX_PITCH: f32 = 0.45;

pub struct Camera {
    /// Camera eye position sent to the shader for view/projection and specular lighting.
    pub position: Vec3,
    /// Player position on the ground plane. The rendered player mesh follows this.
    pub player_position: Vec3,
    /// Third-person orbit yaw in radians. 0 faces world -Z.
    pub yaw: f32,
    /// Third-person orbit pitch in radians. Negative values look down from above.
    pub pitch: f32,
    held_keys: HashSet<KeyCode>,
}

impl Camera {
    pub fn new(x: f32, z: f32) -> Self {
        let mut camera = Self {
            position: Vec3::ZERO,
            player_position: Vec3::new(x, PLAYER_HALF_HEIGHT, z),
            yaw: 0.0,
            pitch: -0.35,
            held_keys: HashSet::new(),
        };
        camera.update_camera_position();
        camera
    }

    pub fn process_key(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.held_keys.insert(key);
            }
            ElementState::Released => {
                self.held_keys.remove(&key);
            }
        }
    }

    pub fn process_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw += delta_x as f32 * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - delta_y as f32 * MOUSE_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
        self.update_camera_position();
    }

    pub fn update(&mut self, dt: f32) {
        let mut movement = Vec3::ZERO;
        let forward = self.horizontal_forward();
        let right = Vec3::new(-forward.z, 0.0, forward.x);

        if self.held_keys.contains(&KeyCode::KeyW) {
            movement += forward;
        }
        if self.held_keys.contains(&KeyCode::KeyS) {
            movement -= forward;
        }
        if self.held_keys.contains(&KeyCode::KeyA) {
            movement -= right;
        }
        if self.held_keys.contains(&KeyCode::KeyD) {
            movement += right;
        }

        if movement.length_squared() > 0.0 {
            self.player_position += movement.normalize() * MOVE_SPEED * dt;
            self.player_position.y = PLAYER_HALF_HEIGHT;
        }
        self.update_camera_position();
    }

    /// Horizontal player-forward vector from yaw. yaw=0 -> (0,0,-1), yaw=pi/2 -> (1,0,0).
    pub fn horizontal_forward(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    fn camera_forward(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            -self.yaw.cos() * cos_pitch,
        )
        .normalize()
    }

    fn target_position(&self) -> Vec3 {
        self.player_position + Vec3::Y * CAMERA_TARGET_HEIGHT
    }

    fn update_camera_position(&mut self) {
        self.position = self.target_position() - self.camera_forward() * CAMERA_DISTANCE;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target_position(), Vec3::Y)
    }

    /// View-projection matrix with wgpu depth correction (GL [-1,1] → wgpu [0,1]).
    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4 * 1.5, aspect, 0.1, 500.0);
        // Remaps z: z_wgpu = z_gl * 0.5 + 0.5
        let correction = Mat4::from_cols(
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        correction * proj * self.view_matrix()
    }
}
