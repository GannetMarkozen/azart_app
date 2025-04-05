#define GLOBAL_SET_INDEX 4

layout (set = GLOBAL_SET_INDEX) uniform ViewMatrices {
	mat4 view;
	mat4 proj;
} view;