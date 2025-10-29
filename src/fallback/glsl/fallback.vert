#version 450

layout(location = 0) out vec2 vUv;

// Full-screen triangle covering the viewport without an index buffer.
const vec2 POSITIONS[3] = vec2[](
	vec2(-1.0, -1.0),
	vec2(-1.0, 3.0),
	vec2(3.0, -1.0)
);

void main() {
	vec2 position = POSITIONS[gl_VertexIndex];
	gl_Position = vec4(position, 0.0, 1.0);

	// Convert clip-space position to UV while flipping Y for texture sampling.
	float u = position.x * 0.5 + 0.5;
	float v = 1.0 - (position.y * 0.5 + 0.5);
	vUv = vec2(u, v);
}
