use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::camera::Camera;
use crate::geometry::{box_mesh, cube, cylinder, plane, sphere, Mesh, Transform, Vertex};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

// ── Uniform structs (must match WGSL byte-for-byte) ──────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4], // 64 bytes
    position: [f32; 3],       // 12 bytes
    _pad: f32,                //  4 bytes → total 80
}

impl CameraUniform {
    pub fn from_camera(cam: &Camera, aspect: f32) -> Self {
        Self {
            view_proj: cam.view_proj_matrix(aspect).to_cols_array_2d(),
            position: cam.position.to_array(),
            _pad: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuLight {
    position: [f32; 3],
    _pad0: f32,
    color: [f32; 3],
    intensity: f32,
}

impl GpuLight {
    pub fn new(pos: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position: pos.to_array(),
            _pad0: 0.0,
            color: color.to_array(),
            intensity,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LightsUniform {
    lights: [GpuLight; 3],   //  96 bytes
    ambient_color: [f32; 3], //  12 bytes
    ambient_intensity: f32,  //   4 bytes → total 112
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ModelUniform {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
}

impl ModelUniform {
    pub fn from_transform(t: &Transform) -> Self {
        let m = t.to_matrix();
        let n = m.inverse().transpose();
        Self {
            model: m.to_cols_array_2d(),
            normal_matrix: n.to_cols_array_2d(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MaterialUniform {
    base_color: [f32; 4],
    specular_color: [f32; 3],
    shininess: f32,
    use_texture: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
}

impl MaterialUniform {
    pub fn flat(r: f32, g: f32, b: f32, spec: [f32; 3], shininess: f32) -> Self {
        Self {
            base_color: [r, g, b, 1.0],
            specular_color: spec,
            shininess,
            use_texture: 0,
            _pad_a: 0,
            _pad_b: 0,
            _pad_c: 0,
        }
    }
    pub fn textured(spec: [f32; 3], shininess: f32) -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            specular_color: spec,
            shininess,
            use_texture: 1,
            _pad_a: 0,
            _pad_b: 0,
            _pad_c: 0,
        }
    }
}

// ── GPU mesh ──────────────────────────────────────────────────────────────────

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl GpuMesh {
    fn upload(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VB"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("IB"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
        }
    }
}

// ── Scene object ──────────────────────────────────────────────────────────────

struct SceneObject {
    mesh: GpuMesh,
    pub transform: Transform,
    model_buffer: wgpu::Buffer,
    _material_buffer: wgpu::Buffer,
    per_obj_bg: wgpu::BindGroup,
    texture_bg: wgpu::BindGroup,
}

impl SceneObject {
    fn update_transform(&self, queue: &wgpu::Queue) {
        let u = ModelUniform::from_transform(&self.transform);
        queue.write_buffer(&self.model_buffer, 0, bytemuck::cast_slice(&[u]));
    }
}

// ── State ─────────────────────────────────────────────────────────────────────

pub struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    render_pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    per_obj_bgl: wgpu::BindGroupLayout,
    texture_bgl: wgpu::BindGroupLayout,

    pub camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    lights_buffer: wgpu::Buffer,
    lights_bind_group: wgpu::BindGroup,

    _checker_tex: wgpu::Texture,
    checker_view: wgpu::TextureView,
    checker_sampler: wgpu::Sampler,
    _white_tex: wgpu::Texture,
    white_view: wgpu::TextureView,
    white_sampler: wgpu::Sampler,

    scene_objects: Vec<SceneObject>,
    player_object: SceneObject,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();

        // 1. Instance → 2. Surface → 3. Adapter → 4. Device+Queue
        // The public web build uses WGPU's WebGL2 fallback path. Some browsers reject
        // wgpu 0.20's older WebGPU limit names during requestDevice negotiation.
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::GL;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(|e| format!("failed to create surface: {e:?}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: cfg!(target_arch = "wasm32"),
            })
            .await
            .ok_or_else(|| {
                "no compatible GPU adapter found. Check that WebGPU/WebGL is enabled in this browser."
                    .to_string()
            })?;

        log::info!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| format!("failed to create GPU device: {e:?}"))?;

        // 5. Surface config
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| "surface reported no supported color formats".to_string())?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // 6. Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // 7. Bind group layouts
        let uniform_entry =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            )],
        });
        let lights_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lights BGL"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::FRAGMENT)],
        });
        let per_obj_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PerObj BGL"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 8. Pipeline layout + 9. Render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl, &lights_bgl, &per_obj_bgl, &texture_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Main Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // no backface culling for simplicity
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // 10. Depth texture
        let depth_view = Self::create_depth_texture(&device, &config);

        // 11. Camera uniform
        let camera = Camera::new(0.0, 5.0);
        let cam_data =
            CameraUniform::from_camera(&camera, config.width as f32 / config.height as f32);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[cam_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 12. Lights uniform
        let lights_data = LightsUniform {
            lights: [
                GpuLight::new(Vec3::new(5.0, 15.0, 5.0), Vec3::new(1.0, 0.97, 0.9), 2.5),
                GpuLight::new(Vec3::new(-4.0, 3.0, -6.0), Vec3::new(1.0, 0.6, 0.2), 8.0),
                GpuLight::new(Vec3::new(8.0, 5.0, -3.0), Vec3::new(0.4, 0.7, 1.0), 6.0),
            ],
            ambient_color: [0.15, 0.15, 0.2],
            ambient_intensity: 1.0,
        };
        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lights Buffer"),
            contents: bytemuck::cast_slice(&[lights_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lights BG"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buffer.as_entire_binding(),
            }],
        });

        // 13. Textures
        let (checker_tex, checker_view, checker_sampler) =
            Self::create_checkerboard_texture(&device, &queue, 256, 16);
        let (white_tex, white_view, white_sampler) =
            Self::create_solid_texture(&device, &queue, [255, 255, 255, 255]);

        // 14. Scene objects
        let make = |mesh: &Mesh,
                    t: Transform,
                    mat: MaterialUniform,
                    tv: &wgpu::TextureView,
                    s: &wgpu::Sampler| {
            Self::create_scene_object(&device, mesh, t, mat, &per_obj_bgl, &texture_bgl, tv, s)
        };

        let ground = make(
            &plane(40.0, 8.0),
            Transform::new(Vec3::ZERO, glam::Quat::IDENTITY, Vec3::ONE),
            MaterialUniform::textured([0.1, 0.1, 0.1], 4.0),
            &checker_view,
            &checker_sampler,
        );

        let barn = make(
            &box_mesh(1.5, 1.25, 2.0),
            Transform::with_scale3(-4.0, 1.25, -6.0, 1.0, 1.0, 1.0),
            MaterialUniform::flat(0.75, 0.1, 0.1, [0.6, 0.4, 0.4], 64.0),
            &white_view,
            &white_sampler,
        );

        let trunk = make(
            &cylinder(0.25, 2.5, 16),
            Transform::with_scale3(3.5, 1.25, -5.5, 1.0, 1.0, 1.0),
            MaterialUniform::flat(0.4, 0.25, 0.1, [0.05, 0.05, 0.05], 4.0),
            &white_view,
            &white_sampler,
        );

        let canopy = make(
            &sphere(1.4, 12, 20),
            Transform::with_scale3(3.5, 3.3, -5.5, 1.0, 1.0, 1.0),
            MaterialUniform::flat(0.1, 0.55, 0.15, [0.05, 0.1, 0.05], 4.0),
            &white_view,
            &white_sampler,
        );

        let boulder = make(
            &sphere(0.8, 10, 16),
            Transform::with_scale3(-1.5, 0.4, -9.0, 1.0, 0.5, 1.0),
            MaterialUniform::flat(0.5, 0.5, 0.52, [0.4, 0.4, 0.4], 24.0),
            &white_view,
            &white_sampler,
        );

        let crate_obj = make(
            &cube(),
            Transform::with_scale(4.0, 0.5, -3.5, 1.0),
            MaterialUniform::textured([0.3, 0.25, 0.2], 16.0),
            &checker_view,
            &checker_sampler,
        );

        let player_obj = make(
            &box_mesh(0.28, 0.85, 0.45),
            Transform::with_scale(0.0, 0.85, 5.0, 1.0),
            MaterialUniform::flat(0.2, 0.4, 1.0, [0.6, 0.8, 1.0], 96.0),
            &white_view,
            &white_sampler,
        );

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            depth_view,
            per_obj_bgl,
            texture_bgl,
            camera,
            camera_buffer,
            camera_bind_group,
            lights_buffer,
            lights_bind_group,
            _checker_tex: checker_tex,
            checker_view,
            checker_sampler,
            _white_tex: white_tex,
            white_view,
            white_sampler,
            scene_objects: vec![ground, barn, trunk, canopy, boulder, crate_obj],
            player_object: player_obj,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = Self::create_depth_texture(&self.device, &self.config);
    }

    pub fn update(&mut self, dt: f32) {
        self.camera.update(dt);
        let aspect = self.config.width as f32 / self.config.height as f32;
        let cam_data = CameraUniform::from_camera(&self.camera, aspect);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[cam_data]));

        let p = self.camera.player_position;
        self.player_object.transform =
            Transform::new(p, Quat::from_rotation_y(self.camera.yaw), Vec3::ONE);
        self.player_object.update_transform(&self.queue);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.47,
                            g: 0.65,
                            b: 0.87,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.lights_bind_group, &[]);

            for obj in self
                .scene_objects
                .iter()
                .chain(std::iter::once(&self.player_object))
            {
                pass.set_bind_group(2, &obj.per_obj_bg, &[]);
                pass.set_bind_group(3, &obj.texture_bg, &[]);
                pass.set_vertex_buffer(0, obj.mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(obj.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..obj.mesh.num_indices, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }

    fn create_scene_object(
        device: &wgpu::Device,
        mesh: &Mesh,
        transform: Transform,
        material: MaterialUniform,
        per_obj_bgl: &wgpu::BindGroupLayout,
        texture_bgl: &wgpu::BindGroupLayout,
        tex_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> SceneObject {
        let gpu_mesh = GpuMesh::upload(device, mesh);

        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model Buffer"),
            contents: bytemuck::cast_slice(&[ModelUniform::from_transform(&transform)]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: bytemuck::cast_slice(&[material]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let per_obj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PerObj BG"),
            layout: per_obj_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        });
        let texture_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture BG"),
            layout: texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        SceneObject {
            mesh: gpu_mesh,
            transform,
            model_buffer,
            _material_buffer: material_buffer,
            per_obj_bg,
            texture_bg,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_checkerboard_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        check_size: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let mut pixels = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let i = ((y * size + x) * 4) as usize;
                let (r, g, b) = if ((x / check_size) + (y / check_size)) % 2 == 0 {
                    (34, 110, 34)
                } else {
                    (60, 179, 60)
                };
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = b;
                pixels[i + 3] = 255;
            }
        }
        Self::upload_texture(device, queue, &pixels, size, size, "Checkerboard")
    }

    fn create_solid_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: [u8; 4],
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        Self::upload_texture(device, queue, &rgba, 1, 1, "Solid")
    }

    fn upload_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        (texture, view, sampler)
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}
