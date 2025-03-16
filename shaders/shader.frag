#version 450 core

// Bindings.
//layout (set = 0, binding = 0) uniform sampler2D image;

layout (location = 0) out vec4 frag_color;

layout (location = 0) in vec2 uv;

void main() {
    frag_color = vec4(uv, 0.0, 1.0);
    //frag_color = texture(image, uv);
}
