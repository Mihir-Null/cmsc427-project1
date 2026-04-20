// ============================================================
// CMSC427 Project 1 – Main WGSL Shader
//
// Bind Groups:
//   Group 0: Camera (view-projection + position)
//   Group 1: Lights (3 point lights + ambient)
//   Group 2: Per-object (model matrix + material)
//   Group 3: Texture (2D texture + sampler)
// ============================================================

struct CameraUniform {
    view_proj : mat4x4<f32>,
    position  : vec3<f32>,
    _pad      : f32,
}

struct Light {
    position  : vec3<f32>,
    _pad0     : f32,
    color     : vec3<f32>,
    intensity : f32,
}

struct LightsUniform {
    lights           : array<Light, 3>,
    ambient_color    : vec3<f32>,
    ambient_intensity: f32,
}

struct ModelUniform {
    model         : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
}

struct MaterialUniform {
    base_color     : vec4<f32>,
    specular_color : vec3<f32>,
    shininess      : f32,
    use_texture    : u32,
    _pad_a         : u32,
    _pad_b         : u32,
    _pad_c         : u32,
}

@group(0) @binding(0) var<uniform> camera   : CameraUniform;
@group(1) @binding(0) var<uniform> lights   : LightsUniform;
@group(2) @binding(0) var<uniform> model_uni: ModelUniform;
@group(2) @binding(1) var<uniform> material : MaterialUniform;
@group(3) @binding(0) var t_diffuse : texture_2d<f32>;
@group(3) @binding(1) var s_diffuse : sampler;

struct VertexInput {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) tex_coords : vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos     : vec4<f32>,
    @location(0)       world_pos    : vec3<f32>,
    @location(1)       world_normal : vec3<f32>,
    @location(2)       tex_coords   : vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let world_pos4 = model_uni.model * vec4<f32>(in.position, 1.0);
    let world_normal = normalize(
        (model_uni.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz
    );
    var out: VertexOutput;
    out.clip_pos    = camera.view_proj * world_pos4;
    out.world_pos   = world_pos4.xyz;
    out.world_normal = world_normal;
    out.tex_coords  = in.tex_coords;
    return out;
}

fn point_light(
    light     : Light,
    world_pos : vec3<f32>,
    N         : vec3<f32>,
    V         : vec3<f32>,
    base_rgb  : vec3<f32>,
    spec_rgb  : vec3<f32>,
    shininess : f32,
) -> vec3<f32> {
    let L_vec = light.position - world_pos;
    let dist  = length(L_vec);
    let L     = L_vec / dist;
    let atten = light.intensity / (1.0 + 0.09 * dist + 0.032 * dist * dist);
    let diff  = max(dot(N, L), 0.0);
    let H     = normalize(L + V);
    let spec  = pow(max(dot(N, H), 0.0), shininess);
    return (diff * base_rgb + spec * spec_rgb) * light.color * atten;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var base_color: vec4<f32>;
    if material.use_texture != 0u {
        base_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    } else {
        base_color = material.base_color;
    }

    let N = normalize(in.world_normal);
    let V = normalize(camera.position - in.world_pos);

    var color = lights.ambient_color * lights.ambient_intensity * base_color.rgb;

    color += point_light(lights.lights[0], in.world_pos, N, V,
                         base_color.rgb, material.specular_color, material.shininess);
    color += point_light(lights.lights[1], in.world_pos, N, V,
                         base_color.rgb, material.specular_color, material.shininess);
    color += point_light(lights.lights[2], in.world_pos, N, V,
                         base_color.rgb, material.specular_color, material.shininess);

    return vec4<f32>(color, base_color.a);
}
