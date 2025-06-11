
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_debug_printf : enable
#extension VK_KHR_shader_draw_parameters : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_multiview : enable
#extension GL_EXT_multiview2 : enable

//#include "global_bindings.glsl"

#define GLOBAL_SET_INDEX 4

layout (binding = 0, scalar) uniform GlobalUbo {
    //mat4 model;
    mat4 views[2];
    //mat4 projs[2];
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

    //gl_Position = view.projs[gl_ViewIndex] * view.model * vec4(pos, 1.0);
    gl_Position = view.views[gl_ViewIndex] * vec4(pos, 1.0);
    out_uv = uv;
}