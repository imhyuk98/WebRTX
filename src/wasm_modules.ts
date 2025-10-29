import loadGlslangModule, { Glslang } from '@webgpu/glslang/dist/web-devel-onefile/glslang';
let _cachedGlslang: Glslang | undefined;
export async function glslangModule(): Promise<Glslang> {
  if (_cachedGlslang) {
    return _cachedGlslang;
  }
  _cachedGlslang = await loadGlslangModule();
  return _cachedGlslang;
}

type NagaWasmModule = typeof import('../naga/pkg');
let _cachedNaga: NagaWasmModule | undefined;
export async function nagaModule(): Promise<NagaWasmModule> {
  if (_cachedNaga) {
    return _cachedNaga;
  }
  _cachedNaga = await import('../naga/pkg');
  // Ensure wasm module is initialized
  if (typeof (_cachedNaga as any).default === 'function') {
    await (_cachedNaga as any).default();
  }
  return _cachedNaga;
}

type GlslTranspilerWasmModule = typeof import('../glsl/pkg');
let _cachedTranspiler: GlslTranspilerWasmModule | undefined;
export async function glslTranspilerModule(): Promise<GlslTranspilerWasmModule> {
  if (_cachedTranspiler) {
    return _cachedTranspiler;
  }
  _cachedTranspiler = await import('../glsl/pkg');
  // Ensure wasm module is initialized
  if (typeof (_cachedTranspiler as any).default === 'function') {
    await (_cachedTranspiler as any).default();
  }
  return _cachedTranspiler;
}
