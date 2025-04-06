
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_debug_printf : enable

//#include "global_bindings.glsl"

#define GLOBAL_SET_INDEX 4

layout (binding = 0) uniform ViewMatrices {
    mat4 model;
    mat4 view;
    mat4 proj;
} view;

//~
// Vertex attributes.
in vec3 pos;
in vec2 uv;
//~

out vec2 out_uv;

void main() {
    /*const vec2 positions[3] = vec2[](
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
    out_uv = uvs[gl_VertexIndex];*/

    const float OFFSET = 2.5;
    const vec3 actual_pos = pos + vec3(0.0, gl_InstanceIndex * OFFSET - (OFFSET / 2), 0.0);
    gl_Position = view.proj * view.view * view.model * vec4(actual_pos, 1.0);
    out_uv = uv;
}