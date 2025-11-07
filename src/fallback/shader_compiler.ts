type ShaderStage = 'vertex' | 'fragment' | 'compute';

interface GlslangCompiler {
	compileGLSL(source: string, stage: ShaderStage, enableTransformFeedback: boolean): Uint32Array;
}

type NagaModule = typeof import('../../naga/pkg');

let glslangInstance: GlslangCompiler | null = null;
let nagaInstance: NagaModule | null = null;

async function loadGlslang(): Promise<GlslangCompiler> {
	if (glslangInstance) {
		return glslangInstance;
	}
	const module = await import('@webgpu/glslang/dist/web-devel-onefile/glslang.js');
	const factoryCandidate = (module as { default?: unknown }).default ?? (module as unknown);
	if (typeof factoryCandidate !== 'function') {
		throw new Error('[WebRTX][Fallback] glslang module did not provide a factory function.');
	}
	const factory = factoryCandidate as () => Promise<GlslangCompiler>;
	const instance = await factory();
	if (!instance) {
		throw new Error('[WebRTX][Fallback] Failed to initialize glslang.');
	}
	glslangInstance = instance;
	return instance;
}

async function loadNaga(): Promise<NagaModule> {
	if (nagaInstance) {
		return nagaInstance;
	}
	const module = await import('../../naga/pkg');
	if (typeof module.default === 'function') {
		await module.default();
	}
	nagaInstance = module;
	return nagaInstance;
}

type GlslDefineValue = string | number | boolean;

function applyDefinesToGlsl(source: string, defines?: Record<string, GlslDefineValue>): string {
	if (!defines || Object.keys(defines).length === 0) {
		return source;
	}
	const defineLines = Object.entries(defines).map(([key, value]) => {
		if (value === true) {
			return `#define ${key} 1`;
		}
		if (value === false) {
			return `#define ${key} 0`;
		}
		return `#define ${key} ${value}`;
	});
	const lines = source.split('\n');
	let insertIndex = 0;
	if (lines.length > 0 && lines[0].startsWith('#version')) {
		insertIndex = 1;
	}
	lines.splice(insertIndex, 0, ...defineLines);
	return lines.join('\n');
}

export async function compileGlslStageToWgsl(
	source: string,
	stage: ShaderStage,
	defines?: Record<string, GlslDefineValue>,
): Promise<string> {
	const glslang = await loadGlslang();
	const decoratedSource = applyDefinesToGlsl(source, defines);
	let spirv: Uint32Array;
	try {
		spirv = glslang.compileGLSL(decoratedSource, stage, false);
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		throw new Error(`[WebRTX][Fallback] GLSL compilation failed for ${stage} stage: ${reason}`);
	}
	const naga = await loadNaga();
	let moduleIndex: number;
	try {
		moduleIndex = naga.spv_in(new Uint8Array(spirv.buffer));
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		throw new Error(`[WebRTX][Fallback] Failed to import SPIR-V for ${stage} stage: ${reason}`);
	}
	try {
		return naga.wgsl_out(moduleIndex);
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		throw new Error(`[WebRTX][Fallback] Failed to translate SPIR-V to WGSL for ${stage} stage: ${reason}`);
	}
}

export interface FallbackShaderCompileResult {
	readonly success: false;
	readonly reason: string;
}

export function compileFallbackShaders(): FallbackShaderCompileResult {
	const reason = 'Fallback shader compiler is not bundled in this build.';
	if (typeof console !== 'undefined') {
		console.warn(`[WebRTX][Fallback] ${reason}`);
	}
	return { success: false, reason };
}
