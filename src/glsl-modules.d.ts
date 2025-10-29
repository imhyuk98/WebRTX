declare module '*.vert' {
	const source: string;
	export default source;
}

declare module '*.frag' {
	const source: string;
	export default source;
}

declare module '*.comp' {
	const source: string;
	export default source;
}

declare module '*.glsl' {
	const source: string;
	export default source;
}

declare module '@webgpu/glslang' {
	interface GlslangInstance {
		compileGLSL(source: string, stage: 'vertex' | 'fragment' | 'compute', enableTransformFeedback: boolean): Uint32Array;
	}

	type GlslangFactory = () => Promise<GlslangInstance>;

	const createGlslang: GlslangFactory;
	export default createGlslang;
}

declare module '@webgpu/glslang/dist/web-devel-onefile/glslang.js' {
	interface GlslangInstance {
		compileGLSL(source: string, stage: 'vertex' | 'fragment' | 'compute', enableTransformFeedback: boolean): Uint32Array;
	}

	type GlslangFactory = () => Promise<GlslangInstance>;

	const createGlslang: GlslangFactory;
	export default createGlslang;
}
