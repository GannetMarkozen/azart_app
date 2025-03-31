#version 450 core

#extension GL_EXT_nonuniform_qualifier : enable

layout (location = 0) out vec2 out_uv;

void main() {
    const vec2 positions[3] = vec2[](
        vec2(0.0, -0.5),  // top
        vec2(0.5, 0.5),  // right
        vec2(-0.5, 0.5)   // left
    );
    
    const vec2 uvs[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    out_uv = uvs[gl_VertexIndex];
}