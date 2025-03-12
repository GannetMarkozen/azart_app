#version 450 core

layout (location = 0) out vec3 out_color;

void main() {
    const vec2 positions[3] = vec2[](
        vec2(0.0, -0.5),  // top
        vec2(0.5, 0.5),  // right
        vec2(-0.5, 0.5)   // left
    );

    vec3 color = vec3(0.0, 0.0, 0.0);
    color[gl_VertexIndex] = 1.0;

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    out_color = color;
}
