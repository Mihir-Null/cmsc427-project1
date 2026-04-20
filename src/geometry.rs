use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Place at (x,y,z) with identity rotation and uniform scale s.
    pub fn with_scale(x: f32, y: f32, z: f32, s: f32) -> Self {
        Self {
            position: Vec3::new(x, y, z),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(s),
        }
    }

    /// Place at (x,y,z) with per-axis scale.
    pub fn with_scale3(x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32) -> Self {
        Self {
            position: Vec3::new(x, y, z),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(sx, sy, sz),
        }
    }

    pub fn to_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Flat XZ plane centered at origin, facing +Y.
pub fn plane(size: f32, uv_scale: f32) -> Mesh {
    let h = size / 2.0;
    let vertices = vec![
        Vertex {
            position: [-h, 0.0, -h],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.0, 0.0],
        },
        Vertex {
            position: [h, 0.0, -h],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [uv_scale, 0.0],
        },
        Vertex {
            position: [h, 0.0, h],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [uv_scale, uv_scale],
        },
        Vertex {
            position: [-h, 0.0, h],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.0, uv_scale],
        },
    ];
    Mesh {
        vertices,
        indices: vec![0, 3, 2, 0, 2, 1],
    }
}

/// Box with explicit half-extents hw/hh/hd centered at origin.
pub fn box_mesh(hw: f32, hh: f32, hd: f32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
        (
            [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                [hw, -hh, hd],
                [-hw, -hh, hd],
                [-hw, -hh, -hd],
                [hw, -hh, -hd],
            ],
            [0.0, -1.0, 0.0],
        ),
        (
            [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]],
            [0.0, 0.0, 1.0],
        ),
        (
            [
                [hw, -hh, -hd],
                [-hw, -hh, -hd],
                [-hw, hh, -hd],
                [hw, hh, -hd],
            ],
            [0.0, 0.0, -1.0],
        ),
        (
            [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]],
            [1.0, 0.0, 0.0],
        ),
        (
            [
                [-hw, -hh, -hd],
                [-hw, -hh, hd],
                [-hw, hh, hd],
                [-hw, hh, -hd],
            ],
            [-1.0, 0.0, 0.0],
        ),
    ];

    for (quad_pos, normal) in &faces {
        let base = vertices.len() as u32;
        for (pos, uv) in quad_pos.iter().zip(uvs.iter()) {
            vertices.push(Vertex {
                position: *pos,
                normal: *normal,
                tex_coords: *uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh { vertices, indices }
}

pub fn cube() -> Mesh {
    box_mesh(0.5, 0.5, 0.5)
}

/// UV sphere centered at origin.
pub fn sphere(radius: f32, stacks: u32, slices: u32) -> Mesh {
    use std::f32::consts::PI;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for stack in 0..=stacks {
        let phi = PI * (stack as f32 / stacks as f32);
        let (sin_phi, cos_phi) = (phi.sin(), phi.cos());
        for slice in 0..=slices {
            let theta = 2.0 * PI * (slice as f32 / slices as f32);
            let (sin_theta, cos_theta) = (theta.sin(), theta.cos());
            let (x, y, z) = (cos_theta * sin_phi, cos_phi, sin_theta * sin_phi);
            vertices.push(Vertex {
                position: [x * radius, y * radius, z * radius],
                normal: [x, y, z],
                tex_coords: [slice as f32 / slices as f32, stack as f32 / stacks as f32],
            });
        }
    }
    for stack in 0..stacks {
        for slice in 0..slices {
            let a = stack * (slices + 1) + slice;
            let b = (stack + 1) * (slices + 1) + slice;
            indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    Mesh { vertices, indices }
}

/// Vertical cylinder with flat caps.
pub fn cylinder(radius: f32, height: f32, slices: u32) -> Mesh {
    use std::f32::consts::TAU;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let hy = height / 2.0;

    // Side
    for i in 0..=slices {
        let theta = TAU * (i as f32 / slices as f32);
        let (s, c) = (theta.sin(), theta.cos());
        let u = i as f32 / slices as f32;
        vertices.push(Vertex {
            position: [c * radius, -hy, s * radius],
            normal: [c, 0.0, s],
            tex_coords: [u, 1.0],
        });
        vertices.push(Vertex {
            position: [c * radius, hy, s * radius],
            normal: [c, 0.0, s],
            tex_coords: [u, 0.0],
        });
    }
    for i in 0..slices {
        let b = i * 2;
        indices.extend_from_slice(&[b, b + 1, b + 3, b, b + 3, b + 2]);
    }

    // Caps
    for &(y, ny) in &[(-hy, -1.0f32), (hy, 1.0f32)] {
        let center = vertices.len() as u32;
        vertices.push(Vertex {
            position: [0.0, y, 0.0],
            normal: [0.0, ny, 0.0],
            tex_coords: [0.5, 0.5],
        });
        let ring_start = vertices.len() as u32;
        for i in 0..slices {
            let theta = TAU * (i as f32 / slices as f32);
            let (s, c) = (theta.sin(), theta.cos());
            vertices.push(Vertex {
                position: [c * radius, y, s * radius],
                normal: [0.0, ny, 0.0],
                tex_coords: [0.5 + 0.5 * c, 0.5 + 0.5 * s],
            });
        }
        for i in 0..slices {
            let curr = ring_start + i;
            let next = ring_start + (i + 1) % slices;
            if ny > 0.0 {
                indices.extend_from_slice(&[center, curr, next]);
            } else {
                indices.extend_from_slice(&[center, next, curr]);
            }
        }
    }
    Mesh { vertices, indices }
}
