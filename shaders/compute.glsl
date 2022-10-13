#include "shared.inl"

DAXA_USE_PUSH_CONSTANT(ComputePush)

#define SUBSAMPLES 2

f32vec3 hsv2rgb(f32vec3 c)
{
    f32vec4 k = f32vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    f32vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

#define INPUT push_constant.gpu_input

f32vec3 mandelbrot_colored(f32vec2 pixel_p)
{
    f32vec2 uv = pixel_p / f32vec2(INPUT.frame_dim.xy);
    uv = (uv - 0.5) * f32vec2(f32(INPUT.frame_dim.x) / f32(INPUT.frame_dim.y), 1);
    f32 time = INPUT.time;
    f32 scale = INPUT.zoom;
    f32vec2 center = INPUT.view_origin;
    u32 max_steps = INPUT.max_steps;
    f32vec2 z = uv * scale * 2 + center;
    f32vec2 c = z;
    u32 i = 0;
    for (; i < max_steps; ++i)
    {
        f32vec2 z_ = z;
        z.x = z_.x * z_.x - z_.y * z_.y;
        z.y = 2.0 * z_.x * z_.y;
        z += c;
        if (dot(z, z) > 256 * 256)
            break;
    }
    f32vec3 col = f32vec3(0, 0, 0);
    if (i != max_steps)
    {
        f32 l = i;
        f32 sl = l - log2(log2(dot(z, z))) + 4.0;
        sl = pow(sl * 0.01, 1.0);
        col = hsv2rgb(f32vec3(sl, 1, 1));
    }
    return col;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main()
{
    u32vec3 pixel_i = gl_GlobalInvocationID.xyz;
    if (pixel_i.x >= INPUT.frame_dim.x || pixel_i.y >= INPUT.frame_dim.y)
        return;
    f32vec3 col = f32vec3(0, 0, 0);
    for (i32 yi = 0; yi < SUBSAMPLES; ++yi)
    {
        for (i32 xi = 0; xi < SUBSAMPLES; ++xi)
        {
            f32vec2 offset = f32vec2(xi, yi) / f32(SUBSAMPLES);
            col += mandelbrot_colored(f32vec2(pixel_i.xy) + offset);
        }
    }
    col *= 1.0 / f32(SUBSAMPLES * SUBSAMPLES);
    
    imageStore(
        daxa_GetRWImage(image2D, rgba32f, push_constant.image_id),
        i32vec2(pixel_i.xy),
        f32vec4(col, 1));
}
