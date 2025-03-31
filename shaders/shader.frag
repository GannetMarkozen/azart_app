#version 450 core

layout (location = 0) out vec4 frag_color;

layout (location = 0) in vec2 uv;

layout (set = 0, binding = 0) uniform sampler2D image;

void main() {
    frag_color = texture(image, uv);
}