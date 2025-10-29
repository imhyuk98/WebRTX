import * as fallbackVertexModule from './glsl/fallback.vert';
import * as fallbackFragmentModule from './glsl/fallback.frag';
import * as fallbackComputeModule from './glsl/fallback.comp';
import { compileGlslStageToWgsl } from './shader_compiler';
import { alignTo, debugInfo, debugLog, debugWarn, isWebrtxDebugLoggingEnabled } from '../util';
import { buildAabbBlasNodes } from '../wasm_bvh_builder';
import { summarizeSceneDescriptor } from './scene_descriptor';
import { createFpsOverlay, disposeFpsOverlay, updateFpsOverlay } from '../fps';
import type {
	FallbackSceneDescriptor,
	FallbackSphereBvh,
	SphereInstance,
	CylinderInstance,
	CircleInstance,
	EllipseInstance,
	ConeInstance,
	LineInstance,
	TorusInstance,
	PlaneInstance,
	BezierPatchInstance,
	Vec3,
} from './scene_descriptor';

interface CompiledFallbackShaders {
	vertex: string;
	fragment: string;
	compute: string;
}

interface PrimitiveCounts {
	sphere: number;
	cylinder: number;
	circle: number;
	ellipse: number;
	cone: number;
	line: number;
	torus: number;
	plane: number;
	bezier: number;
}

interface PrimitiveAccelerationData {
	primitiveCount: number;
	counts: PrimitiveCounts;
	primitiveRefs: Uint32Array;
	indices: Uint32Array;
	nodeBytes: Uint8Array;
	nodeCount: number;
	sphereData: Float32Array;
	analyticData: Float32Array;
	bezierData: Float32Array;
}

type ScenePrimitiveState = {
	spheres: SphereInstance[];
	cylinders: CylinderInstance[];
	circles: CircleInstance[];
	ellipses: EllipseInstance[];
	cones: ConeInstance[];
	lines: LineInstance[];
	tori: TorusInstance[];
	planes: PlaneInstance[];
	bezierPatches: BezierPatchInstance[];
};

interface ConvertedWebRtxBlas {
	nodeBytes: Uint8Array;
	nodeCount: number;
	indexData: Uint32Array;
}

export interface FallbackPipelineOptions {
	canvasOrId?: HTMLCanvasElement | string;
	width: number;
	height: number;
	adapter: GPUAdapter;
	scene: FallbackSceneDescriptor;
	device?: GPUDevice;
}

export interface FallbackPipelineHandle {
	renderer: 'fallback';
	updateCamera(pos: Vec3, look: Vec3, up: Vec3): Promise<void>;
	render(): Promise<void>;
	device: GPUDevice;
	setCameraFovY(deg: number): void;
	setCameraNearFar(near: number, far: number): void;
	resize(width: number, height: number, dpr?: number): void;
	setCylinder(instance: CylinderInstance): Promise<void>;
	setTorus(instance: TorusInstance): Promise<void>;
	setTori(instances: TorusInstance[]): Promise<void>;
	setCylinders(instances: CylinderInstance[]): Promise<void>;
	setSpheres(instances: SphereInstance[]): Promise<void>;
	setSpheresWithBvh(instances: SphereInstance[], bvh: FallbackSphereBvh): Promise<void>;
	setCircles(instances: CircleInstance[]): Promise<void>;
	setEllipses(instances: EllipseInstance[]): Promise<void>;
	setLines(instances: LineInstance[]): Promise<void>;
	setCones(instances: ConeInstance[]): Promise<void>;
	setLine(instance: LineInstance): Promise<void>;
	setBezierPatch(instance: BezierPatchInstance): Promise<void>;
	setBezierPatches(instances: BezierPatchInstance[]): Promise<void>;
	setScene(scene: FallbackSceneDescriptor): Promise<void>;
	stop(): void;
}

const BVH_INVALID_INDEX = 0xffffffff;
const BVH_NODE_FLOATS = 12;
const BVH_NODE_BYTES = BVH_NODE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const SPHERE_FLOATS = 4;
const ANALYTIC_FLOATS = 12;
const BEZIER_CONTROL_VEC4 = 18;
const BEZIER_FLOATS = BEZIER_CONTROL_VEC4 * 4;
const BEZIER_MAX_DEPTH_DEFAULT = 10;
const BEZIER_PIXEL_EPSILON_DEFAULT = 3;
const PRIMITIVE_REF_UINTS = 4;
const WORKGROUP_SIZE_X = 8;
const WORKGROUP_SIZE_Y = 8;
const PRIM_TYPE_SPHERE = 0;
const PRIM_TYPE_CYLINDER = 1;
const PRIM_TYPE_CIRCLE = 2;
const PRIM_TYPE_ELLIPSE = 3;
const PRIM_TYPE_CONE = 4;
const PRIM_TYPE_LINE = 5;
const PRIM_TYPE_TORUS = 6;
const PRIM_TYPE_PLANE = 7;
const PRIM_TYPE_BEZIER_PATCH = 8;

const asyncNoop = async () => {};

let debugPrimitiveSnapshotLogged = false;
let debugBvhSnapshotLogged = false;

let cachedFallbackShaderPromise: Promise<CompiledFallbackShaders> | null = null;

function resolveGlslModule(module: unknown, label: string): string {
	if (typeof module === 'string') {
		return module;
	}
	const fallback = (module as { default?: unknown })?.default;
	if (typeof fallback === 'string') {
		return fallback;
	}
	throw new Error(`Fallback GLSL module "${label}" missing string export`);
}

const fallbackVertexGlsl = resolveGlslModule(fallbackVertexModule, 'vertex');
const fallbackFragmentGlsl = resolveGlslModule(fallbackFragmentModule, 'fragment');
const fallbackComputeGlsl = resolveGlslModule(fallbackComputeModule, 'compute');

async function getFallbackShaderWgsl(): Promise<CompiledFallbackShaders> {
	if (!cachedFallbackShaderPromise) {
		if (isWebrtxDebugLoggingEnabled()) {
			debugInfo('[WebRTX] fallback shader sources', {
				vertex: typeof fallbackVertexGlsl,
				fragment: typeof fallbackFragmentGlsl,
				compute: typeof fallbackComputeGlsl,
				vertexLength: fallbackVertexGlsl?.length,
				fragmentLength: fallbackFragmentGlsl?.length,
				computeLength: fallbackComputeGlsl?.length,
			});
		}
		cachedFallbackShaderPromise = (async () => {
			const [vertex, fragment, compute] = await Promise.all([
				compileGlslStageToWgsl(fallbackVertexGlsl, 'vertex'),
				compileGlslStageToWgsl(fallbackFragmentGlsl, 'fragment'),
				compileGlslStageToWgsl(fallbackComputeGlsl, 'compute'),
			]);
			return { vertex, fragment, compute };
		})();
	}
	return cachedFallbackShaderPromise;
}

function isArrayBufferLikeValue(value: unknown): value is ArrayBufferLike {
	if (value instanceof ArrayBuffer) {
		return true;
	}
	if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
		return true;
	}
	return false;
}

function toUint8Array(source: ArrayBufferLike | ArrayBufferView): Uint8Array {
	if (source instanceof Uint8Array) {
		return source.byteLength === source.buffer.byteLength
			? source
			: new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
	}
	if (ArrayBuffer.isView(source)) {
		return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
	}
	if (isArrayBufferLikeValue(source)) {
		return new Uint8Array(source);
	}
	throw new Error('Unsupported BVH node data source');
}

function toUint8ArrayView(source: ArrayBufferLike | ArrayBufferView, expectedByteLength: number): Uint8Array {
	if (source instanceof Uint8Array) {
		if (source.byteLength < expectedByteLength) {
			throw new Error('Prebuilt BVH node data too small');
		}
		return source.byteLength === expectedByteLength
			? source
			: new Uint8Array(source.buffer, source.byteOffset, expectedByteLength);
	}
	if (ArrayBuffer.isView(source)) {
		if (source.byteLength < expectedByteLength) {
			throw new Error('Prebuilt BVH node view too small');
		}
		return new Uint8Array(source.buffer, source.byteOffset, expectedByteLength);
	}
	if (isArrayBufferLikeValue(source)) {
		if (source.byteLength < expectedByteLength) {
			throw new Error('Prebuilt BVH node buffer too small');
		}
		return new Uint8Array(source, 0, expectedByteLength);
	}
	throw new Error('Unsupported BVH node data source');
}

function toFloat32ArrayView(source: ArrayBufferLike | ArrayBufferView | ArrayLike<number>, expectedLength: number): Float32Array {
	if (source instanceof Float32Array) {
		if (source.length < expectedLength) {
			throw new Error('Prebuilt sphere data shorter than expected');
		}
		return source.length === expectedLength
			? source
			: new Float32Array(source.buffer, source.byteOffset, expectedLength);
	}
	if (ArrayBuffer.isView(source)) {
		if (source.byteLength < expectedLength * Float32Array.BYTES_PER_ELEMENT) {
			throw new Error('Prebuilt sphere view shorter than expected');
		}
		return new Float32Array(source.buffer, source.byteOffset, expectedLength);
	}
	if (isArrayBufferLikeValue(source)) {
		if (source.byteLength < expectedLength * Float32Array.BYTES_PER_ELEMENT) {
			throw new Error('Prebuilt sphere buffer shorter than expected');
		}
		return new Float32Array(source, 0, expectedLength);
	}
	const arrayLike = source as ArrayLike<number> | undefined;
	if (arrayLike && typeof arrayLike.length === 'number') {
		if (arrayLike.length < expectedLength) {
			throw new Error('Prebuilt sphere array shorter than expected');
		}
		const result = new Float32Array(expectedLength);
		for (let i = 0; i < expectedLength; i++) {
			result[i] = Number(arrayLike[i]) || 0;
		}
		return result;
	}
	throw new Error('Unsupported sphere data source');
}

function toUint32ArrayView(source: ArrayBufferLike | ArrayBufferView | ArrayLike<number> | undefined, expectedLength: number): Uint32Array {
	if (source instanceof Uint32Array) {
		if (source.length < expectedLength) {
			throw new Error('Prebuilt index data shorter than expected');
		}
		return source.length === expectedLength
			? source
			: new Uint32Array(source.buffer, source.byteOffset, expectedLength);
	}
	if (ArrayBuffer.isView(source)) {
		if (source.byteLength < expectedLength * Uint32Array.BYTES_PER_ELEMENT) {
			throw new Error('Prebuilt index view shorter than expected');
		}
		return new Uint32Array(source.buffer, source.byteOffset, expectedLength);
	}
	if (isArrayBufferLikeValue(source)) {
		if (source.byteLength < expectedLength * Uint32Array.BYTES_PER_ELEMENT) {
			throw new Error('Prebuilt index buffer shorter than expected');
		}
		return new Uint32Array(source, 0, expectedLength);
	}
	if (source && typeof source.length === 'number') {
		if (source.length < expectedLength) {
			throw new Error('Prebuilt index array shorter than expected');
		}
		const result = new Uint32Array(expectedLength);
		for (let i = 0; i < expectedLength; i++) {
			result[i] = (source[i] as number) >>> 0;
		}
		return result;
	}
	throw new Error('Unsupported index data source');
}

function validatePrimitiveRefTypes(refs: Uint32Array, counts: PrimitiveCounts): void {
	let slot = 0;
	const check = (count: number, expectedType: number, label: string) => {
		for (let i = 0; i < count; i++, slot++) {
			const base = slot * PRIMITIVE_REF_UINTS;
			if (base >= refs.length) {
				throw new Error(`[WebRTX] PrimitiveRef buffer too small when validating ${label}`);
			}
			const actualType = refs[base];
			if (actualType !== expectedType) {
				throw new Error(`[WebRTX] PrimitiveRef type mismatch for ${label} at slot ${slot}: expected ${expectedType}, got ${actualType}`);
			}
		}
	};
	check(counts.sphere, PRIM_TYPE_SPHERE, 'sphere');
	check(counts.cylinder, PRIM_TYPE_CYLINDER, 'cylinder');
	check(counts.circle, PRIM_TYPE_CIRCLE, 'circle');
	check(counts.ellipse, PRIM_TYPE_ELLIPSE, 'ellipse');
	check(counts.cone, PRIM_TYPE_CONE, 'cone');
	check(counts.line, PRIM_TYPE_LINE, 'line');
	check(counts.torus, PRIM_TYPE_TORUS, 'torus');
	check(counts.plane, PRIM_TYPE_PLANE, 'plane');
	check(counts.bezier, PRIM_TYPE_BEZIER_PATCH, 'bezier');
	if (slot * PRIMITIVE_REF_UINTS > refs.length) {
		throw new Error('[WebRTX] PrimitiveRef validation scanned past buffer length');
	}
}

function convertWebRtxBlasNodesToFallback(rawNodeBytes: Uint8Array, nodeCount: number, primitiveCount: number): ConvertedWebRtxBlas {
	if (nodeCount <= 0) {
		throw new Error('WebRTX BVH provided no nodes');
	}
	if (primitiveCount <= 0) {
		return {
			nodeBytes: new Uint8Array(0),
			nodeCount: 0,
			indexData: new Uint32Array(0),
		};
	}
	const nodeStride = rawNodeBytes.byteLength / nodeCount;
	if (!Number.isFinite(nodeStride) || nodeStride < 44 || !Number.isInteger(nodeStride)) {
		throw new Error('WebRTX BLAS node stride invalid');
	}
	const dataView = new DataView(rawNodeBytes.buffer, rawNodeBytes.byteOffset, rawNodeBytes.byteLength);
	type WebNode = {
		min: Vec3;
		max: Vec3;
		entry: number;
		exit: number;
		geometryId: number;
	};
	const webNodes: WebNode[] = new Array(nodeCount);
	const readFloat = (offset: number) => dataView.getFloat32(offset, true);
	const readUint = (offset: number) => dataView.getUint32(offset, true);
	const readInt = (offset: number) => dataView.getInt32(offset, true);
	for (let i = 0; i < nodeCount; i++) {
		const base = i * nodeStride;
		webNodes[i] = {
			min: [readFloat(base + 0), readFloat(base + 4), readFloat(base + 8)],
			max: [readFloat(base + 16), readFloat(base + 20), readFloat(base + 24)],
			entry: readUint(base + 32),
			exit: readUint(base + 36),
			geometryId: readInt(base + 40),
		};
	}
	const fallbackBuffer = new ArrayBuffer(nodeCount * BVH_NODE_BYTES);
	const fallbackBytes = new Uint8Array(fallbackBuffer);
	const fallbackF32 = new Float32Array(fallbackBuffer);
	const fallbackU32 = new Uint32Array(fallbackBuffer);
	const indexData = new Uint32Array(primitiveCount);
	for (let i = 0; i < primitiveCount; i++) {
		indexData[i] = i >>> 0;
	}

	if (isWebrtxDebugLoggingEnabled() && !debugBvhSnapshotLogged) {
		debugBvhSnapshotLogged = true;
		const simplifiedNodes = webNodes.map((node) => ({
			min: Array.from(node.min),
			max: Array.from(node.max),
			entry: node.entry,
			exit: node.exit,
			geometryId: node.geometryId,
		}));
		debugLog('[Fallback] WebRTX nodes snapshot', JSON.stringify(simplifiedNodes));
		debugLog('[Fallback] Primitive index LUT snapshot', JSON.stringify(Array.from(indexData)));
	}
	const SENTINEL = BVH_INVALID_INDEX >>> 0;
	const toValidIndex = (value: number): number => {
		if (value === SENTINEL) {
			return SENTINEL;
		}
		return value >= 0 && value < nodeCount ? value : SENTINEL;
	};
	for (let i = 0; i < nodeCount; i++) {
		const node = webNodes[i];
		const floatBase = i * BVH_NODE_FLOATS;
		fallbackF32[floatBase + 0] = node.min[0];
		fallbackF32[floatBase + 1] = node.min[1];
		fallbackF32[floatBase + 2] = node.min[2];
		fallbackF32[floatBase + 3] = 0.0;
		fallbackF32[floatBase + 4] = node.max[0];
		fallbackF32[floatBase + 5] = node.max[1];
		fallbackF32[floatBase + 6] = node.max[2];
		fallbackF32[floatBase + 7] = 0.0;
			const entryIndex = node.entry >>> 0;
			const exitIndex = node.exit >>> 0;
			if (node.geometryId >= 0) {
				let primitiveIndex = entryIndex;
				const geometryIndex = node.geometryId >>> 0;
				if (geometryIndex < primitiveCount) {
					primitiveIndex = geometryIndex;
				}
				if (primitiveIndex >= primitiveCount) {
					throw new Error('WebRTX BVH primitive index out of range');
				}
				fallbackU32[floatBase + 8] = SENTINEL;
				fallbackU32[floatBase + 9] = toValidIndex(exitIndex);
				fallbackU32[floatBase + 10] = primitiveIndex;
				fallbackU32[floatBase + 11] = 1;
		} else {
			fallbackU32[floatBase + 8] = toValidIndex(entryIndex);
			fallbackU32[floatBase + 9] = toValidIndex(exitIndex);
			fallbackU32[floatBase + 10] = 0;
			fallbackU32[floatBase + 11] = 0;
		}
	}
	return {
		nodeBytes: fallbackBytes,
		nodeCount,
		indexData,
	};
}

export async function runFallbackPipeline(options: FallbackPipelineOptions): Promise<FallbackPipelineHandle> {
	const { canvasOrId, width, height, adapter, scene } = options;
	if (isWebrtxDebugLoggingEnabled()) {
		debugInfo('[WebRTX] Fallback pipeline invoked', summarizeSceneDescriptor(scene));
	}
	const cloneVec3 = (v: Vec3): Vec3 => [v[0], v[1], v[2]];
	const cloneBezierPatch = (patch: BezierPatchInstance): BezierPatchInstance => ({
		p00: cloneVec3(patch.p00),
		p01: cloneVec3(patch.p01),
		p10: cloneVec3(patch.p10),
		p11: cloneVec3(patch.p11),
		du00: cloneVec3(patch.du00),
		du01: cloneVec3(patch.du01),
		du10: cloneVec3(patch.du10),
		du11: cloneVec3(patch.du11),
		dv00: cloneVec3(patch.dv00),
		dv01: cloneVec3(patch.dv01),
		dv10: cloneVec3(patch.dv10),
		dv11: cloneVec3(patch.dv11),
		duv00: cloneVec3(patch.duv00),
		duv01: cloneVec3(patch.duv01),
		duv10: cloneVec3(patch.duv10),
		duv11: cloneVec3(patch.duv11),
		maxDepth: patch.maxDepth,
		pixelEpsilon: patch.pixelEpsilon,
	});
	const cloneSceneToState = (desc: FallbackSceneDescriptor): ScenePrimitiveState => ({
		spheres: desc.spheres.map((s) => ({ center: cloneVec3(s.center), radius: s.radius })),
		cylinders: desc.cylinders.map((c) => ({
			center: cloneVec3(c.center),
			xdir: cloneVec3(c.xdir),
			ydir: cloneVec3(c.ydir),
			radius: c.radius,
			height: c.height,
			angleDeg: c.angleDeg,
		})),
		circles: desc.circles.map((c) => ({
			center: cloneVec3(c.center),
			xdir: cloneVec3(c.xdir),
			ydir: cloneVec3(c.ydir),
			radius: c.radius,
		})),
		ellipses: desc.ellipses.map((e) => ({
			center: cloneVec3(e.center),
			xdir: cloneVec3(e.xdir),
			ydir: cloneVec3(e.ydir),
			radiusX: e.radiusX,
			radiusY: e.radiusY,
		})),
		cones: desc.cones.map((c) => ({
			center: cloneVec3(c.center),
			xdir: cloneVec3(c.xdir),
			ydir: cloneVec3(c.ydir),
			radius: c.radius,
			height: c.height,
		})),
		lines: desc.lines.map((l) => ({
			p0: cloneVec3(l.p0),
			p1: cloneVec3(l.p1),
			radius: l.radius,
		})),
		tori: desc.tori.map((t) => ({
			center: cloneVec3(t.center),
			xdir: cloneVec3(t.xdir),
			ydir: cloneVec3(t.ydir),
			majorRadius: t.majorRadius,
			minorRadius: t.minorRadius,
			angleDeg: t.angleDeg,
		})),
		planes: desc.planes.map((p) => ({
			center: cloneVec3(p.center),
			xdir: cloneVec3(p.xdir),
			ydir: cloneVec3(p.ydir),
			halfWidth: p.halfWidth,
			halfHeight: p.halfHeight,
		})),
		bezierPatches: desc.bezierPatches.map(cloneBezierPatch),
	});
	const canvasLookup = typeof canvasOrId === 'string'
		? document.getElementById(canvasOrId)
		: canvasOrId ?? document.getElementById('gfx');
		if (!canvasLookup) {
			throw new Error('Fallback pipeline requires a canvas element');
		}
		if (!(canvasLookup instanceof HTMLCanvasElement)) {
			throw new Error('Fallback pipeline requires an HTMLCanvasElement');
		}
		const canvas = canvasLookup;
	const device = options.device ?? await (async () => {
		const limits = adapter.limits;
		return adapter.requestDevice({
			requiredLimits: {
				maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage,
			},
		});
	})();
		const context = canvas.getContext('webgpu');
		if (!context) {
		throw new Error('WebGPU context unavailable');
	}
		const gpuContext = context;
	const format = navigator.gpu.getPreferredCanvasFormat();
	const storageBufferLimit = (() => {
		const raw = device.limits?.maxStorageBufferBindingSize;
		if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
			return raw;
		}
		return 256 * 1024 * 1024;
	})();
	let logicalWidth = Math.max(1, Math.floor(width));
	let logicalHeight = Math.max(1, Math.floor(height));
	let dpr = globalThis.devicePixelRatio || 1;
	let internalWidth = 1;
	let internalHeight = 1;
	const cameraUniformData = new Float32Array(32);
	const cameraUniformBuffer = device.createBuffer({
		size: cameraUniformData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	const blitSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
	let cameraPosition: Vec3 = [0, 0, 5];
	let cameraForward: Vec3 = [0, 0, -1];
	let cameraRight: Vec3 = [1, 0, 0];
	let cameraUp: Vec3 = [0, 1, 0];
	let cameraFovYRadians = (60 * Math.PI) / 180;
	let cameraNear = 0.1;
	let cameraFar = 10000;
	let cameraAspect = Math.max(1, logicalWidth) / Math.max(1, logicalHeight);
	const zeroSphereWrite = new Float32Array(SPHERE_FLOATS);
	const zeroAnalyticWrite = new Float32Array(ANALYTIC_FLOATS);
	const zeroBezierWrite = new Float32Array(BEZIER_FLOATS);
	const zeroPrimitiveRefWrite = new Uint32Array(PRIMITIVE_REF_UINTS);
	const zeroIndexWrite = new Uint32Array(1);
	const zeroBvhWrite = new Uint32Array(BVH_NODE_FLOATS);
	const sphereStorage = { buffer: null as GPUBuffer | null, size: 0 };
	const analyticStorage = { buffer: null as GPUBuffer | null, size: 0 };
	const bezierStorage = { buffer: null as GPUBuffer | null, size: 0 };
	const primitiveRefStorage = { buffer: null as GPUBuffer | null, size: 0 };
	const primitiveIndexStorage = { buffer: null as GPUBuffer | null, size: 0 };
	const primitiveBvhStorage = { buffer: null as GPUBuffer | null, size: 0 };
	let primitiveCountValue = 0;
	let primitiveIndexCountValue = 0;
	let primitiveBvhNodeCountValue = 0;
	let currentAnalyticPrimitiveCount = 0;
	let currentBezierPatchCount = 0;
	let currentPrimitiveCounts: PrimitiveCounts = {
		sphere: 0,
		cylinder: 0,
		circle: 0,
		ellipse: 0,
		cone: 0,
		line: 0,
		torus: 0,
		plane: 0,
		bezier: 0,
	};
	let currentScenePrimitives: ScenePrimitiveState = {
		spheres: scene.spheres.map((s) => ({ ...s })),
		cylinders: scene.cylinders.map((c) => ({ ...c })),
		circles: scene.circles.map((c) => ({ ...c })),
		ellipses: scene.ellipses.map((e) => ({ ...e })),
		cones: scene.cones.map((c) => ({ ...c })),
		lines: scene.lines.map((l) => ({ ...l })),
		tori: scene.tori.map((t) => ({ ...t })),
		planes: scene.planes.map((p) => ({ ...p })),
		bezierPatches: scene.bezierPatches?.map(cloneBezierPatch) ?? [],
	};
	let computePipeline: GPUComputePipeline | null = null;
	let renderPipeline: GPURenderPipeline | null = null;
	let computeBindGroup: GPUBindGroup | null = null;
	let renderBindGroup: GPUBindGroup | null = null;
	let outputTexture: GPUTexture | null = null;
	let outputTextureView: GPUTextureView | null = null;
	const OUTPUT_TEXTURE_FORMAT = 'rgba8unorm';
	let cameraDebugLogged = false;

	function isZeroVec(v: Vec3): boolean {
		return Math.abs(v[0]) < 1e-5 && Math.abs(v[1]) < 1e-5 && Math.abs(v[2]) < 1e-5;
	}

	function normalizeVec3(v: Vec3): Vec3 {
		const len = Math.hypot(v[0], v[1], v[2]);
		if (!Number.isFinite(len) || len < 1e-5) {
			return [0, 0, 0];
		}
		const inv = 1 / len;
		return [v[0] * inv, v[1] * inv, v[2] * inv];
	}

	function crossVec3(a: Vec3, b: Vec3): Vec3 {
		return [
			a[1] * b[2] - a[2] * b[1],
			a[2] * b[0] - a[0] * b[2],
			a[0] * b[1] - a[1] * b[0],
		];
	}

	function subtractVec3(a: Vec3, b: Vec3): Vec3 {
		return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
	}

	function dotVec3(a: Vec3, b: Vec3): number {
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	function addVec3(a: Vec3, b: Vec3): Vec3 {
		return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
	}

	function scaleVec3(v: Vec3, scalar: number): Vec3 {
		return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
	}

	function absVec3(v: Vec3): Vec3 {
		return [Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])];
	}

	function orthonormalizePair(xdirIn: Vec3, ydirIn: Vec3): [Vec3, Vec3] {
		const xdir = normalizeVec3(xdirIn);
		let ydir = subtractVec3(ydirIn, scaleVec3(xdir, dotVec3(ydirIn, xdir)));
		ydir = normalizeVec3(ydir);
		return [xdir, ydir];
	}

	function toVec3(value: Vec3 | undefined, fallback: Vec3 = [0, 0, 0]): Vec3 {
		return [
			value?.[0] ?? fallback[0],
			value?.[1] ?? fallback[1],
			value?.[2] ?? fallback[2],
		];
	}

	const HERMITE_TO_BEZIER: number[][] = [
		[1, 0, 0, 0],
		[1, 0, 1 / 3, 0],
		[0, 1, 0, -1 / 3],
		[0, 1, 0, 0],
	];

	function addScaledVec3(acc: Vec3, v: Vec3, scalar: number): Vec3 {
		return [acc[0] + v[0] * scalar, acc[1] + v[1] * scalar, acc[2] + v[2] * scalar];
	}

	function sanitizeVec3(v: Vec3): Vec3 {
		return [
			Number.isFinite(v[0]) ? v[0] : 0,
			Number.isFinite(v[1]) ? v[1] : 0,
			Number.isFinite(v[2]) ? v[2] : 0,
		];
	}

	function multiplyHermiteMatrixLeft(matrix: readonly number[][], data: Vec3[][]): Vec3[][] {
		const result: Vec3[][] = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => [0, 0, 0] as Vec3));
		for (let i = 0; i < 4; i++) {
			for (let j = 0; j < 4; j++) {
				let sum: Vec3 = [0, 0, 0];
				for (let k = 0; k < 4; k++) {
					sum = addScaledVec3(sum, data[k][j], matrix[i][k]);
				}
				result[i][j] = sum;
			}
		}
		return result;
	}

	function multiplyHermiteMatrixRight(data: Vec3[][], matrix: readonly number[][]): Vec3[][] {
		const result: Vec3[][] = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => [0, 0, 0] as Vec3));
		for (let i = 0; i < 4; i++) {
			for (let j = 0; j < 4; j++) {
				let sum: Vec3 = [0, 0, 0];
				for (let k = 0; k < 4; k++) {
					sum = addScaledVec3(sum, data[i][k], matrix[j][k]);
				}
				result[i][j] = sum;
			}
		}
		return result;
	}

	interface BezierPatchConversion {
		controlPoints: Vec3[];
		boundsMin: Vec3;
		boundsMax: Vec3;
		maxDepth: number;
		pixelEpsilon: number;
	}

	function convertHermitePatchToBezier(patch: BezierPatchInstance): BezierPatchConversion {
		const hermite: Vec3[][] = [
			[toVec3(patch.p00), toVec3(patch.p01), toVec3(patch.dv00), toVec3(patch.dv01)],
			[toVec3(patch.p10), toVec3(patch.p11), toVec3(patch.dv10), toVec3(patch.dv11)],
			[toVec3(patch.du00), toVec3(patch.du01), toVec3(patch.duv00), toVec3(patch.duv01)],
			[toVec3(patch.du10), toVec3(patch.du11), toVec3(patch.duv10), toVec3(patch.duv11)],
		];
		const temp = multiplyHermiteMatrixLeft(HERMITE_TO_BEZIER, hermite);
		const bezier = multiplyHermiteMatrixRight(temp, HERMITE_TO_BEZIER);
		const controlPoints: Vec3[] = [];
		let boundsMin = sanitizeVec3(bezier[0][0]);
		let boundsMax = boundsMin.slice() as Vec3;
		for (let u = 0; u < 4; u++) {
			for (let vIdx = 0; vIdx < 4; vIdx++) {
				const point = sanitizeVec3(bezier[u][vIdx]);
				controlPoints.push(point);
				if (point[0] < boundsMin[0]) boundsMin[0] = point[0];
				if (point[1] < boundsMin[1]) boundsMin[1] = point[1];
				if (point[2] < boundsMin[2]) boundsMin[2] = point[2];
				if (point[0] > boundsMax[0]) boundsMax[0] = point[0];
				if (point[1] > boundsMax[1]) boundsMax[1] = point[1];
				if (point[2] > boundsMax[2]) boundsMax[2] = point[2];
			}
		}
		const depthSource = patch.maxDepth;
		let maxDepth = BEZIER_MAX_DEPTH_DEFAULT;
		if (typeof depthSource === 'number' && Number.isFinite(depthSource)) {
			maxDepth = Math.max(1, Math.floor(depthSource));
		}
		const epsilonSource = patch.pixelEpsilon;
		let pixelEpsilon = BEZIER_PIXEL_EPSILON_DEFAULT;
		if (typeof epsilonSource === 'number' && Number.isFinite(epsilonSource)) {
			pixelEpsilon = Math.max(1e-5, epsilonSource);
		}
		return { controlPoints, boundsMin, boundsMax, maxDepth, pixelEpsilon };
	}

	function flushCameraUniform(): void {
		const tanHalfFov = Math.tan(cameraFovYRadians * 0.5);
		cameraUniformData.set([
			cameraPosition[0], cameraPosition[1], cameraPosition[2], cameraNear,
			cameraForward[0], cameraForward[1], cameraForward[2], tanHalfFov,
			cameraRight[0], cameraRight[1], cameraRight[2], cameraAspect,
			cameraUp[0], cameraUp[1], cameraUp[2], cameraFar,
		]);
		cameraUniformData[16] = primitiveCountValue;
		cameraUniformData[17] = internalWidth;
		cameraUniformData[18] = internalHeight;
		cameraUniformData[19] = primitiveBvhNodeCountValue;
		cameraUniformData[20] = currentPrimitiveCounts.sphere;
		cameraUniformData[21] = currentPrimitiveCounts.cylinder;
		cameraUniformData[22] = currentPrimitiveCounts.circle;
		cameraUniformData[23] = currentPrimitiveCounts.ellipse;
		cameraUniformData[24] = currentPrimitiveCounts.cone;
		cameraUniformData[25] = currentPrimitiveCounts.line;
		cameraUniformData[26] = currentPrimitiveCounts.torus;
		cameraUniformData[27] = currentPrimitiveCounts.plane;
		cameraUniformData[28] = currentPrimitiveCounts.bezier;
		cameraUniformData[29] = currentBezierPatchCount;
		cameraUniformData[30] = 0;
		cameraUniformData[31] = 0;
		device.queue.writeBuffer(cameraUniformBuffer, 0, cameraUniformData);
	}

	function updateCameraOrientation(pos: Vec3, look: Vec3, up: Vec3): void {
		cameraPosition = [pos[0], pos[1], pos[2]];
		let forwardDir = normalizeVec3(subtractVec3(look, pos));
		if (isZeroVec(forwardDir)) {
			forwardDir = [0, 0, -1];
		}
		let upDir = normalizeVec3(up);
		if (isZeroVec(upDir)) {
			upDir = [0, 1, 0];
		}
		let rightDir = crossVec3(forwardDir, upDir);
		if (isZeroVec(rightDir)) {
			const fallbackUp: Vec3 = Math.abs(forwardDir[1]) > 0.9 ? [0, 0, 1] : [0, 1, 0];
			rightDir = crossVec3(forwardDir, fallbackUp);
		}
		rightDir = normalizeVec3(rightDir);
		let orthoUp = crossVec3(rightDir, forwardDir);
		if (isZeroVec(orthoUp)) {
			orthoUp = [0, 1, 0];
		} else {
			orthoUp = normalizeVec3(orthoUp);
		}
		cameraForward = forwardDir;
		cameraRight = rightDir;
		cameraUp = orthoUp;
		flushCameraUniform();
		if (isWebrtxDebugLoggingEnabled() && !cameraDebugLogged) {
			cameraDebugLogged = true;
			debugLog('[Fallback] Camera basis', {
				position: cameraPosition.slice(),
				forward: cameraForward.slice(),
				right: cameraRight.slice(),
				up: cameraUp.slice(),
				aspect: cameraAspect,
				logicalSize: [logicalWidth, logicalHeight],
				internalSize: [internalWidth, internalHeight],
				dpr,
			});
		}
	}

	function recreateOutputTexture(): void {
		outputTexture?.destroy();
		if (internalWidth <= 0 || internalHeight <= 0) {
			outputTexture = null;
			outputTextureView = null;
			return;
		}
		outputTexture = device.createTexture({
			size: { width: internalWidth, height: internalHeight, depthOrArrayLayers: 1 },
			format: OUTPUT_TEXTURE_FORMAT,
			usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
		});
		outputTextureView = outputTexture.createView();
		rebuildComputeBindGroup();
		rebuildRenderBindGroup();
	}

	function clampCountForFloatStorage(label: string, requestedCount: number, floatsPerPrimitive: number): number {
		const request = Math.floor(Math.max(0, requestedCount));
		if (request <= 0) {
			return 0;
		}
		const maxCount = Math.floor(storageBufferLimit / (floatsPerPrimitive * Float32Array.BYTES_PER_ELEMENT));
		if (maxCount <= 0) {
			debugWarn(`[WebRTX] Device storage limits prevent uploading any ${label} to fallback path`);
			return 0;
		}
		if (request > maxCount) {
			debugWarn(`[WebRTX] Truncating fallback ${label} upload to fit device limits`, maxCount, '/', request);
			return maxCount;
		}
		return request;
	}

	function ensureFloatStorageBuffer(storage: { buffer: GPUBuffer | null; size: number }, floatsPerPrimitive: number, count: number): boolean {
		const minimumCount = Math.max(1, count);
		const requiredBytes = alignTo(minimumCount * floatsPerPrimitive * Float32Array.BYTES_PER_ELEMENT, 16);
		if (!storage.buffer || storage.size < requiredBytes) {
			storage.buffer?.destroy();
			storage.buffer = device.createBuffer({
				size: requiredBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			storage.size = requiredBytes;
			return true;
		}
		return false;
	}

	function ensureUintStorageBuffer(storage: { buffer: GPUBuffer | null; size: number }, uintsPerEntry: number, count: number): boolean {
		const minimumCount = Math.max(1, count);
		const requiredBytes = alignTo(minimumCount * uintsPerEntry * Uint32Array.BYTES_PER_ELEMENT, 16);
		if (!storage.buffer || storage.size < requiredBytes) {
			storage.buffer?.destroy();
			storage.buffer = device.createBuffer({
				size: requiredBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			storage.size = requiredBytes;
			return true;
		}
		return false;
	}

	function ensureBvhStorageBuffer(storage: { buffer: GPUBuffer | null; size: number }, nodeCount: number): boolean {
		const minimumNodes = Math.max(1, nodeCount);
		const requiredBytes = alignTo(minimumNodes * BVH_NODE_BYTES, 16);
		if (!storage.buffer || storage.size < requiredBytes) {
			storage.buffer?.destroy();
			storage.buffer = device.createBuffer({
				size: requiredBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			storage.size = requiredBytes;
			return true;
		}
		return false;
	}

	function writeFloatStorageBuffer(storage: { buffer: GPUBuffer | null }, data: Float32Array, zeroFallback: Float32Array): void {
		if (!storage.buffer) {
			return;
		}
		if (data.length > 0) {		
			device.queue.writeBuffer(
				storage.buffer,
				0,
				data.buffer,
				data.byteOffset,
				data.byteLength,
			);
		} else {
			device.queue.writeBuffer(
				storage.buffer,
				0,
				zeroFallback.buffer,
				zeroFallback.byteOffset,
				zeroFallback.byteLength,
			);
		}
	}

	function writeUintStorageBuffer(storage: { buffer: GPUBuffer | null }, data: Uint32Array, zeroFallback: Uint32Array): void {
		if (!storage.buffer) {
			return;
		}
		if (data.length > 0) {
			device.queue.writeBuffer(
				storage.buffer,
				0,
				data.buffer,
				data.byteOffset,
				data.byteLength,
			);
		} else {
			device.queue.writeBuffer(
				storage.buffer,
				0,
				zeroFallback.buffer,
				zeroFallback.byteOffset,
				zeroFallback.byteLength,
			);
		}
	}

	function writeBvhStorageBuffer(storage: { buffer: GPUBuffer | null }, data: Uint8Array): void {
		if (!storage.buffer) {
			return;
		}
		if (data.byteLength > 0) {
			device.queue.writeBuffer(
				storage.buffer,
				0,
				data.buffer,
				data.byteOffset,
				data.byteLength,
			);
		} else {
			device.queue.writeBuffer(
				storage.buffer,
				0,
				zeroBvhWrite.buffer,
				zeroBvhWrite.byteOffset,
				zeroBvhWrite.byteLength,
			);
		}
	}

	async function buildPrimitiveAcceleration(prims: ScenePrimitiveState): Promise<PrimitiveAccelerationData> {
		const counts: PrimitiveCounts = {
			sphere: clampCountForFloatStorage('spheres', prims.spheres.length, SPHERE_FLOATS),
			cylinder: clampCountForFloatStorage('cylinders', prims.cylinders.length, ANALYTIC_FLOATS),
			circle: clampCountForFloatStorage('circles', prims.circles.length, ANALYTIC_FLOATS),
			ellipse: clampCountForFloatStorage('ellipses', prims.ellipses.length, ANALYTIC_FLOATS),
			cone: clampCountForFloatStorage('cones', prims.cones.length, ANALYTIC_FLOATS),
			line: clampCountForFloatStorage('lines', prims.lines.length, ANALYTIC_FLOATS),
			torus: clampCountForFloatStorage('tori', prims.tori.length, ANALYTIC_FLOATS),
			plane: clampCountForFloatStorage('planes', prims.planes.length, ANALYTIC_FLOATS),
			bezier: clampCountForFloatStorage('bezier patches', prims.bezierPatches.length, BEZIER_FLOATS),
		};

		const analyticReduceOrder: Array<keyof PrimitiveCounts> = ['plane', 'torus', 'line', 'cone', 'ellipse', 'circle', 'cylinder'];
		const maxAnalyticEntries = Math.floor(storageBufferLimit / (ANALYTIC_FLOATS * Float32Array.BYTES_PER_ELEMENT));
		let analyticCount = counts.cylinder + counts.circle + counts.ellipse + counts.cone + counts.line + counts.torus + counts.plane;
		if (analyticCount > maxAnalyticEntries) {
			if (maxAnalyticEntries <= 0) {
				debugWarn('[WebRTX] Device storage limits prevent uploading analytic primitives to fallback path');
				counts.cylinder = 0;
				counts.circle = 0;
				counts.ellipse = 0;
				counts.cone = 0;
				counts.line = 0;
				counts.torus = 0;
				counts.plane = 0;
				analyticCount = 0;
			} else {
				debugWarn('[WebRTX] Truncating analytic primitive upload to fit device limits', maxAnalyticEntries, '/', analyticCount);
				let excessAnalytic = analyticCount - maxAnalyticEntries;
				for (const key of analyticReduceOrder) {
					if (excessAnalytic <= 0) {
						break;
					}
					const available = counts[key];
					if (available <= 0) {
						continue;
					}
					const reduceAmount = Math.min(available, excessAnalytic);
					counts[key] -= reduceAmount;
					excessAnalytic -= reduceAmount;
				}
				analyticCount = counts.cylinder + counts.circle + counts.ellipse + counts.cone + counts.line + counts.torus + counts.plane;
			}
		}

		let primitiveCount =
			counts.sphere +
			counts.cylinder +
			counts.circle +
			counts.ellipse +
			counts.cone +
			counts.line +
			counts.torus +
			counts.plane +
			counts.bezier;
		const maxPrimitiveRefs = Math.floor(storageBufferLimit / (PRIMITIVE_REF_UINTS * Uint32Array.BYTES_PER_ELEMENT));
		if (primitiveCount > maxPrimitiveRefs) {
			if (maxPrimitiveRefs <= 0) {
				debugWarn('[WebRTX] Device storage limits prevent uploading primitives to fallback path');
				counts.sphere = 0;
				counts.cylinder = 0;
				counts.circle = 0;
				counts.ellipse = 0;
				counts.cone = 0;
				counts.line = 0;
				counts.torus = 0;
				counts.plane = 0;
				primitiveCount = 0;
			} else {
				debugWarn('[WebRTX] Truncating fallback primitive upload to fit device limits', maxPrimitiveRefs, '/', primitiveCount);
				let excess = primitiveCount - maxPrimitiveRefs;
				const reduceOrder: Array<keyof PrimitiveCounts> = ['bezier', 'plane', 'torus', 'line', 'cone', 'ellipse', 'circle', 'cylinder', 'sphere'];
				for (const key of reduceOrder) {
					if (excess <= 0) {
						break;
					}
					const available = counts[key];
					if (available <= 0) {
						continue;
					}
					const reduceAmount = Math.min(available, excess);
					counts[key] -= reduceAmount;
					excess -= reduceAmount;
				}
				primitiveCount =
					counts.sphere +
					counts.cylinder +
					counts.circle +
					counts.ellipse +
					counts.cone +
					counts.line +
					counts.torus +
					counts.plane;
			}
		}
		const spheres = prims.spheres.slice(0, counts.sphere);
		const cylinders = prims.cylinders.slice(0, counts.cylinder);
		const circles = prims.circles.slice(0, counts.circle);
		const ellipses = prims.ellipses.slice(0, counts.ellipse);
		const cones = prims.cones.slice(0, counts.cone);
		const lines = prims.lines.slice(0, counts.line);
		const tori = prims.tori.slice(0, counts.torus);
		const planes = prims.planes.slice(0, counts.plane);
		const bezierPatches = prims.bezierPatches.slice(0, counts.bezier);

		let aabbData = new Float32Array(Math.max(primitiveCount * 6, 0));
		let primitiveRefs = new Uint32Array(Math.max(primitiveCount * PRIMITIVE_REF_UINTS, 0));
		let indices = new Uint32Array(Math.max(primitiveCount, 0));
		for (let i = 0; i < indices.length; i++) {
			indices[i] = i;
		}

		const sphereData = new Float32Array(counts.sphere * SPHERE_FLOATS);
		const analyticStartCylinder = 0;
		const analyticStartCircle = analyticStartCylinder + counts.cylinder;
		const analyticStartEllipse = analyticStartCircle + counts.circle;
		const analyticStartCone = analyticStartEllipse + counts.ellipse;
		const analyticStartLine = analyticStartCone + counts.cone;
		const analyticStartTorus = analyticStartLine + counts.line;
		const analyticStartPlane = analyticStartTorus + counts.torus;
		const analyticTotal = analyticStartPlane + counts.plane;
		const analyticData = new Float32Array(Math.max(analyticTotal * ANALYTIC_FLOATS, 0));
		const bezierData = new Float32Array(Math.max(counts.bezier * BEZIER_FLOATS, 0));

		let cursor = 0;
		const pushPrimitive = (type: number, indexInType: number, min: Vec3, max: Vec3, analyticBaseOffset: number): void => {
			if (cursor >= primitiveCount) {
				return;
			}
			const base = cursor * 6;
			aabbData[base + 0] = min[0];
			aabbData[base + 1] = min[1];
			aabbData[base + 2] = min[2];
			aabbData[base + 3] = max[0];
			aabbData[base + 4] = max[1];
			aabbData[base + 5] = max[2];
			const refBase = cursor * PRIMITIVE_REF_UINTS;
			primitiveRefs[refBase + 0] = type >>> 0;
			primitiveRefs[refBase + 1] = indexInType >>> 0;
			primitiveRefs[refBase + 2] = analyticBaseOffset >>> 0;
			primitiveRefs[refBase + 3] = 0;
			cursor++;
		};

		for (let i = 0; i < spheres.length; i++) {
			const sphere = spheres[i];
			const center = toVec3(sphere.center);
			const radius = Math.max(0, sphere.radius ?? 0);
			const base = i * SPHERE_FLOATS;
			sphereData[base + 0] = center[0];
			sphereData[base + 1] = center[1];
			sphereData[base + 2] = center[2];
			sphereData[base + 3] = radius;
			const min: Vec3 = [center[0] - radius, center[1] - radius, center[2] - radius];
			const max: Vec3 = [center[0] + radius, center[1] + radius, center[2] + radius];
			pushPrimitive(PRIM_TYPE_SPHERE, i, min, max, 0);
		}

		for (let i = 0; i < cylinders.length; i++) {
			const cylinder = cylinders[i];
			const center = toVec3(cylinder.center);
			const radius = Math.max(0, cylinder.radius ?? 0);
			const height = Math.max(0, cylinder.height ?? 0);
			const angleDeg = Number.isFinite(cylinder.angleDeg) ? cylinder.angleDeg ?? 360 : 360;
			const [xdir, ydir] = orthonormalizePair(toVec3(cylinder.xdir, [1, 0, 0]), toVec3(cylinder.ydir, [0, 1, 0]));
			const axis = normalizeVec3(crossVec3(xdir, ydir));
			const halfH = height * 0.5;
			const ext = addVec3(
				addVec3(scaleVec3(absVec3(xdir), radius), scaleVec3(absVec3(ydir), radius)),
				scaleVec3(absVec3(axis), halfH),
			);
			const min: Vec3 = [center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]];
			const max: Vec3 = [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]];
			const globalIndex = analyticStartCylinder + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = center[0];
			analyticData[base + 1] = center[1];
			analyticData[base + 2] = center[2];
			analyticData[base + 3] = radius;
			analyticData[base + 4] = xdir[0];
			analyticData[base + 5] = xdir[1];
			analyticData[base + 6] = xdir[2];
			analyticData[base + 7] = height;
			analyticData[base + 8] = ydir[0];
			analyticData[base + 9] = ydir[1];
			analyticData[base + 10] = ydir[2];
			analyticData[base + 11] = angleDeg ?? 360;
			pushPrimitive(PRIM_TYPE_CYLINDER, i, min, max, analyticStartCylinder);
		}

		for (let i = 0; i < circles.length; i++) {
			const circle = circles[i];
			const center = toVec3(circle.center);
			const radius = Math.max(0, circle.radius ?? 0);
			const [xdir, ydir] = orthonormalizePair(toVec3(circle.xdir, [1, 0, 0]), toVec3(circle.ydir, [0, 1, 0]));
			const ext = scaleVec3(addVec3(absVec3(xdir), absVec3(ydir)), radius);
			const normal = normalizeVec3(crossVec3(xdir, ydir));
			const normalPadAmount = Math.max(5e-3, 0.02 * radius);
			const normalPad = scaleVec3(absVec3(normal), normalPadAmount);
			const min: Vec3 = [
				center[0] - ext[0] - normalPad[0],
				center[1] - ext[1] - normalPad[1],
				center[2] - ext[2] - normalPad[2],
			];
			const max: Vec3 = [
				center[0] + ext[0] + normalPad[0],
				center[1] + ext[1] + normalPad[1],
				center[2] + ext[2] + normalPad[2],
			];
			const globalIndex = analyticStartCircle + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = center[0];
			analyticData[base + 1] = center[1];
			analyticData[base + 2] = center[2];
			analyticData[base + 3] = radius;
			analyticData[base + 4] = xdir[0];
			analyticData[base + 5] = xdir[1];
			analyticData[base + 6] = xdir[2];
			analyticData[base + 7] = 0;
			analyticData[base + 8] = ydir[0];
			analyticData[base + 9] = ydir[1];
			analyticData[base + 10] = ydir[2];
			analyticData[base + 11] = 0;
			pushPrimitive(PRIM_TYPE_CIRCLE, i, min, max, analyticStartCircle);
		}

		for (let i = 0; i < ellipses.length; i++) {
			const ellipse = ellipses[i];
			const center = toVec3(ellipse.center);
			const radiusX = Math.max(0, ellipse.radiusX ?? 0);
			const radiusY = Math.max(0, ellipse.radiusY ?? 0);
			const [xdir, ydir] = orthonormalizePair(toVec3(ellipse.xdir, [1, 0, 0]), toVec3(ellipse.ydir, [0, 1, 0]));
			const normal = normalizeVec3(crossVec3(xdir, ydir));
			const thickness = Math.max(5e-3, 0.5 * Math.max(radiusX, radiusY));
			const ext = addVec3(
				addVec3(scaleVec3(absVec3(xdir), radiusX), scaleVec3(absVec3(ydir), radiusY)),
				scaleVec3(absVec3(normal), thickness),
			);
			const min: Vec3 = [center[0] - ext[0], center[1] - ext[1], center[2] - ext[2]];
			const max: Vec3 = [center[0] + ext[0], center[1] + ext[1], center[2] + ext[2]];
			const globalIndex = analyticStartEllipse + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = center[0];
			analyticData[base + 1] = center[1];
			analyticData[base + 2] = center[2];
			analyticData[base + 3] = radiusX;
			analyticData[base + 4] = xdir[0];
			analyticData[base + 5] = xdir[1];
			analyticData[base + 6] = xdir[2];
			analyticData[base + 7] = radiusY;
			analyticData[base + 8] = ydir[0];
			analyticData[base + 9] = ydir[1];
			analyticData[base + 10] = ydir[2];
			analyticData[base + 11] = 0;
			pushPrimitive(PRIM_TYPE_ELLIPSE, i, min, max, analyticStartEllipse);
		}

		for (let i = 0; i < cones.length; i++) {
			const cone = cones[i];
			const center = toVec3(cone.center);
			const radius = Math.max(0, cone.radius ?? 0);
			const height = Math.max(0, cone.height ?? 0);
			const [xdir, ydir] = orthonormalizePair(toVec3(cone.xdir, [1, 0, 0]), toVec3(cone.ydir, [0, 1, 0]));
			const axis = normalizeVec3(crossVec3(xdir, ydir));
			const halfH = height * 0.5;
			const baseCenter = addVec3(center, scaleVec3(axis, halfH));
			const apex = subtractVec3(center, scaleVec3(axis, halfH));
			const baseExt = addVec3(scaleVec3(absVec3(xdir), radius), scaleVec3(absVec3(ydir), radius));
			const pad = Math.max(1e-4, 1e-3 * Math.max(radius, height));
			const min: Vec3 = [
				Math.min(baseCenter[0] - baseExt[0], apex[0]) - pad,
				Math.min(baseCenter[1] - baseExt[1], apex[1]) - pad,
				Math.min(baseCenter[2] - baseExt[2], apex[2]) - pad,
			];
			const max: Vec3 = [
				Math.max(baseCenter[0] + baseExt[0], apex[0]) + pad,
				Math.max(baseCenter[1] + baseExt[1], apex[1]) + pad,
				Math.max(baseCenter[2] + baseExt[2], apex[2]) + pad,
			];
			const globalIndex = analyticStartCone + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = center[0];
			analyticData[base + 1] = center[1];
			analyticData[base + 2] = center[2];
			analyticData[base + 3] = radius;
			analyticData[base + 4] = xdir[0];
			analyticData[base + 5] = xdir[1];
			analyticData[base + 6] = xdir[2];
			analyticData[base + 7] = height;
			analyticData[base + 8] = ydir[0];
			analyticData[base + 9] = ydir[1];
			analyticData[base + 10] = ydir[2];
			analyticData[base + 11] = 0;
			pushPrimitive(PRIM_TYPE_CONE, i, min, max, analyticStartCone);
		}

		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			const p0 = toVec3(line.p0);
			const p1 = toVec3(line.p1);
			const radius = Math.max(0, line.radius ?? 0);
			const min: Vec3 = [
				Math.min(p0[0], p1[0]) - radius,
				Math.min(p0[1], p1[1]) - radius,
				Math.min(p0[2], p1[2]) - radius,
			];
			const max: Vec3 = [
				Math.max(p0[0], p1[0]) + radius,
				Math.max(p0[1], p1[1]) + radius,
				Math.max(p0[2], p1[2]) + radius,
			];
			const pad = Math.max(5e-4, 2e-2 * radius);
			min[0] -= pad; min[1] -= pad; min[2] -= pad;
			max[0] += pad; max[1] += pad; max[2] += pad;
			const globalIndex = analyticStartLine + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = p0[0];
			analyticData[base + 1] = p0[1];
			analyticData[base + 2] = p0[2];
			analyticData[base + 3] = radius;
			analyticData[base + 4] = p1[0];
			analyticData[base + 5] = p1[1];
			analyticData[base + 6] = p1[2];
			analyticData[base + 7] = radius;
			analyticData[base + 8] = 0;
			analyticData[base + 9] = 0;
			analyticData[base + 10] = 0;
			analyticData[base + 11] = 0;
			pushPrimitive(PRIM_TYPE_LINE, i, min, max, analyticStartLine);
		}

		for (let i = 0; i < tori.length; i++) {
			const torus = tori[i];
			const center = toVec3(torus.center);
			const majorRadius = Math.max(0, torus.majorRadius ?? 0);
			const minorRadius = Math.max(0, torus.minorRadius ?? 0);
			const angleDeg = Number.isFinite(torus.angleDeg) ? torus.angleDeg ?? 360 : 360;
			const [xdir, ydir] = orthonormalizePair(toVec3(torus.xdir, [1, 0, 0]), toVec3(torus.ydir, [0, 1, 0]));
			const axis = normalizeVec3(crossVec3(xdir, ydir));
			const ringRadius = majorRadius + minorRadius;
			const ext = addVec3(
				addVec3(scaleVec3(absVec3(xdir), ringRadius), scaleVec3(absVec3(axis), ringRadius)),
				[minorRadius, minorRadius, minorRadius],
			);
			const pad = Math.max(1e-2 * Math.max(majorRadius, 1.0), 2e-2 * minorRadius, 5e-3);
			const min: Vec3 = [center[0] - ext[0] - pad, center[1] - ext[1] - pad, center[2] - ext[2] - pad];
			const max: Vec3 = [center[0] + ext[0] + pad, center[1] + ext[1] + pad, center[2] + ext[2] + pad];
			const globalIndex = analyticStartTorus + i;
			const base = globalIndex * ANALYTIC_FLOATS;
			analyticData[base + 0] = center[0];
			analyticData[base + 1] = center[1];
			analyticData[base + 2] = center[2];
			analyticData[base + 3] = majorRadius;
			analyticData[base + 4] = xdir[0];
			analyticData[base + 5] = xdir[1];
			analyticData[base + 6] = xdir[2];
			analyticData[base + 7] = minorRadius;
			analyticData[base + 8] = ydir[0];
			analyticData[base + 9] = ydir[1];
			analyticData[base + 10] = ydir[2];
			analyticData[base + 11] = angleDeg ?? 360;
			pushPrimitive(PRIM_TYPE_TORUS, i, min, max, analyticStartTorus);
		}

		for (let i = 0; i < planes.length; i++) {
			const plane = planes[i];
			const center = toVec3(plane.center);
			const halfWidth = Math.max(plane.halfWidth ?? 0, 1e-4);
			const halfHeight = Math.max(plane.halfHeight ?? 0, 1e-4);
			const [xdirRaw, ydirRaw] = orthonormalizePair(
				toVec3(plane.xdir, [1, 0, 0]),
				toVec3(plane.ydir, [0, 0, 1]),
			);
			const normal = normalizeVec3(crossVec3(xdirRaw, ydirRaw));
			const pad = Math.max(5e-3, 0.02 * Math.max(halfWidth, halfHeight));
			const extent = addVec3(
				addVec3(
					scaleVec3(absVec3(xdirRaw), halfWidth),
					scaleVec3(absVec3(ydirRaw), halfHeight),
				),
				scaleVec3(absVec3(normal), pad),
			);
			const min: Vec3 = [
				center[0] - extent[0],
				center[1] - extent[1],
				center[2] - extent[2],
			];
			const max: Vec3 = [
				center[0] + extent[0],
				center[1] + extent[1],
				center[2] + extent[2],
			];
			const globalIndex = analyticStartPlane + i;
			const analyticBase = globalIndex * ANALYTIC_FLOATS;
			analyticData[analyticBase + 0] = center[0];
			analyticData[analyticBase + 1] = center[1];
			analyticData[analyticBase + 2] = center[2];
			analyticData[analyticBase + 3] = halfWidth;
			analyticData[analyticBase + 4] = xdirRaw[0];
			analyticData[analyticBase + 5] = xdirRaw[1];
			analyticData[analyticBase + 6] = xdirRaw[2];
			analyticData[analyticBase + 7] = halfHeight;
			analyticData[analyticBase + 8] = ydirRaw[0];
			analyticData[analyticBase + 9] = ydirRaw[1];
			analyticData[analyticBase + 10] = ydirRaw[2];
			analyticData[analyticBase + 11] = 0;
			pushPrimitive(PRIM_TYPE_PLANE, i, min, max, analyticStartPlane);
		}

		for (let i = 0; i < bezierPatches.length; i++) {
			const converted = convertHermitePatchToBezier(bezierPatches[i]);
			const base = i * BEZIER_FLOATS;
			for (let cp = 0; cp < converted.controlPoints.length; cp++) {
				const point = converted.controlPoints[cp];
				const offset = base + cp * 4;
				bezierData[offset + 0] = point[0];
				bezierData[offset + 1] = point[1];
				bezierData[offset + 2] = point[2];
				bezierData[offset + 3] = 0;
			}
			const boundsMinOffset = base + 16 * 4;
			bezierData[boundsMinOffset + 0] = converted.boundsMin[0];
			bezierData[boundsMinOffset + 1] = converted.boundsMin[1];
			bezierData[boundsMinOffset + 2] = converted.boundsMin[2];
			bezierData[boundsMinOffset + 3] = converted.maxDepth;
			const boundsMaxOffset = base + 17 * 4;
			bezierData[boundsMaxOffset + 0] = converted.boundsMax[0];
			bezierData[boundsMaxOffset + 1] = converted.boundsMax[1];
			bezierData[boundsMaxOffset + 2] = converted.boundsMax[2];
			bezierData[boundsMaxOffset + 3] = converted.pixelEpsilon;
			const spanX = converted.boundsMax[0] - converted.boundsMin[0];
			const spanY = converted.boundsMax[1] - converted.boundsMin[1];
			const spanZ = converted.boundsMax[2] - converted.boundsMin[2];
			const spanMax = Math.max(spanX, Math.max(spanY, spanZ));
			const pad = Math.max(1e-4, spanMax * 1e-3);
			const min: Vec3 = [
				converted.boundsMin[0] - pad,
				converted.boundsMin[1] - pad,
				converted.boundsMin[2] - pad,
			];
			const max: Vec3 = [
				converted.boundsMax[0] + pad,
				converted.boundsMax[1] + pad,
				converted.boundsMax[2] + pad,
			];
			pushPrimitive(PRIM_TYPE_BEZIER_PATCH, i, min, max, 0);
		}

		if (cursor < primitiveCount) {
			aabbData = aabbData.slice(0, cursor * 6);
			primitiveRefs = primitiveRefs.slice(0, cursor * PRIMITIVE_REF_UINTS);
			indices = indices.slice(0, cursor);
			primitiveCount = cursor;
		}

		validatePrimitiveRefTypes(primitiveRefs, counts);

		if (isWebrtxDebugLoggingEnabled() && !debugPrimitiveSnapshotLogged) {
			debugPrimitiveSnapshotLogged = true;
			debugLog('[Fallback] PrimitiveRefs snapshot', Array.from(primitiveRefs));
			debugLog('[Fallback] AABB snapshot', Array.from(aabbData));
			const analyticDump: Record<string, number[]> = {};
			let analyticCursor = 0;
			const dumpSegment = (label: keyof PrimitiveCounts, count: number) => {
				if (count <= 0) {
					return;
				}
				const start = analyticCursor * ANALYTIC_FLOATS;
				const end = start + count * ANALYTIC_FLOATS;
				analyticDump[label] = Array.from(analyticData.slice(start, end));
				analyticCursor += count;
			};
			dumpSegment('cylinder', counts.cylinder);
			dumpSegment('circle', counts.circle);
			dumpSegment('ellipse', counts.ellipse);
			dumpSegment('cone', counts.cone);
			dumpSegment('line', counts.line);
			dumpSegment('torus', counts.torus);
			dumpSegment('plane', counts.plane);
			debugLog('[Fallback] Analytic segments', JSON.stringify(analyticDump));
			if (counts.circle > 0) {
				const diagnostics: Array<{ index: number; lenX: number; lenY: number; dotXY: number; xdir: Vec3; ydir: Vec3; }> = [];
				for (let i = 0; i < counts.circle; i++) {
					const base = (analyticStartCircle + i) * ANALYTIC_FLOATS;
					const xdir: Vec3 = [analyticData[base + 4], analyticData[base + 5], analyticData[base + 6]];
					const ydir: Vec3 = [analyticData[base + 8], analyticData[base + 9], analyticData[base + 10]];
					const lenX = Math.hypot(xdir[0], xdir[1], xdir[2]);
					const lenY = Math.hypot(ydir[0], ydir[1], ydir[2]);
					const dotXY = xdir[0] * ydir[0] + xdir[1] * ydir[1] + xdir[2] * ydir[2];
					diagnostics.push({ index: i, lenX, lenY, dotXY, xdir, ydir });
				}
				debugLog('[Fallback] Circle basis diagnostics', JSON.stringify(diagnostics));
			}
		}

		let nodeBytes = new Uint8Array(0);
		let nodeCount = 0;
		if (primitiveCount > 0) {
			try {
				const webRtxBlas = await buildAabbBlasNodes(aabbData);
				if (webRtxBlas.nodeCount > 0 && webRtxBlas.nodeData.byteLength > 0) {
					const converted = convertWebRtxBlasNodesToFallback(webRtxBlas.nodeData, webRtxBlas.nodeCount, primitiveCount);
					if (converted.nodeBytes.byteLength <= storageBufferLimit) {
						nodeBytes = new Uint8Array(converted.nodeBytes);
						nodeCount = converted.nodeCount;
						indices = new Uint32Array(converted.indexData);
					} else {
						debugWarn('[WebRTX] BVH buffer would exceed device limits; falling back to linear traversal');
					}
				}
			} catch (error) {
				debugWarn('[WebRTX] Failed to build BVH for fallback primitives; falling back to linear traversal', error);
			}
		}

		return {
			primitiveCount,
			counts,
			primitiveRefs,
			indices,
			nodeBytes,
			nodeCount,
			sphereData,
			analyticData,
			bezierData,
		};
	}

	async function updateAllPrimitives(prims: ScenePrimitiveState): Promise<void> {
		const accel = await buildPrimitiveAcceleration(prims);
		primitiveCountValue = accel.primitiveCount;
		primitiveIndexCountValue = accel.indices.length;
		primitiveBvhNodeCountValue = accel.nodeCount;
		currentPrimitiveCounts = accel.counts;

		const analyticCount =
			accel.counts.cylinder +
			accel.counts.circle +
			accel.counts.ellipse +
			accel.counts.cone +
			accel.counts.line +
			accel.counts.torus +
			accel.counts.plane;
		currentAnalyticPrimitiveCount = analyticCount;
		currentBezierPatchCount = accel.counts.bezier;

		let resized = false;
		resized = ensureFloatStorageBuffer(sphereStorage, SPHERE_FLOATS, accel.counts.sphere) || resized;
		resized = ensureFloatStorageBuffer(analyticStorage, ANALYTIC_FLOATS, analyticCount) || resized;
		resized = ensureFloatStorageBuffer(bezierStorage, BEZIER_FLOATS, accel.counts.bezier) || resized;
		resized = ensureUintStorageBuffer(primitiveRefStorage, PRIMITIVE_REF_UINTS, Math.max(accel.primitiveCount, 1)) || resized;
		resized = ensureUintStorageBuffer(primitiveIndexStorage, 1, Math.max(accel.indices.length, 1)) || resized;
		resized = ensureBvhStorageBuffer(primitiveBvhStorage, accel.nodeCount) || resized;

		writeFloatStorageBuffer(sphereStorage, accel.sphereData, zeroSphereWrite);
		writeFloatStorageBuffer(analyticStorage, accel.analyticData, zeroAnalyticWrite);
		writeFloatStorageBuffer(bezierStorage, accel.bezierData, zeroBezierWrite);
		writeUintStorageBuffer(primitiveRefStorage, accel.primitiveRefs, zeroPrimitiveRefWrite);
		writeUintStorageBuffer(primitiveIndexStorage, accel.indices, zeroIndexWrite);
		writeBvhStorageBuffer(primitiveBvhStorage, accel.nodeBytes);

		flushCameraUniform();
		if (resized || !computeBindGroup) {
			rebuildComputeBindGroup();
		}
	}

	function rebuildComputeBindGroup(): void {
		if (!computePipeline) {
			return;
		}
		ensureFloatStorageBuffer(sphereStorage, SPHERE_FLOATS, currentPrimitiveCounts.sphere);
		ensureFloatStorageBuffer(analyticStorage, ANALYTIC_FLOATS, currentAnalyticPrimitiveCount);
		ensureFloatStorageBuffer(bezierStorage, BEZIER_FLOATS, currentBezierPatchCount);
		ensureUintStorageBuffer(primitiveRefStorage, PRIMITIVE_REF_UINTS, Math.max(primitiveCountValue, 1));
		ensureUintStorageBuffer(primitiveIndexStorage, 1, Math.max(primitiveIndexCountValue, 1));
		ensureBvhStorageBuffer(primitiveBvhStorage, primitiveBvhNodeCountValue);

		if (!sphereStorage.buffer ||
			!primitiveBvhStorage.buffer ||
			!primitiveIndexStorage.buffer ||
			!primitiveRefStorage.buffer ||
			!analyticStorage.buffer ||
			!bezierStorage.buffer ||
			!outputTextureView) {
			return;
		}

		computeBindGroup = device.createBindGroup({
			layout: computePipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: cameraUniformBuffer } },
				{ binding: 1, resource: { buffer: sphereStorage.buffer } },
				{ binding: 2, resource: outputTextureView },
				{ binding: 3, resource: { buffer: primitiveBvhStorage.buffer } },
				{ binding: 4, resource: { buffer: primitiveIndexStorage.buffer } },
				{ binding: 5, resource: { buffer: primitiveRefStorage.buffer } },
				{ binding: 6, resource: { buffer: analyticStorage.buffer } },
				{ binding: 7, resource: { buffer: bezierStorage.buffer } },
			],
		});
	}

	function rebuildRenderBindGroup(): void {
		if (!renderPipeline || !outputTextureView) {
			return;
		}
		renderBindGroup = device.createBindGroup({
			layout: renderPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: blitSampler },
				{ binding: 1, resource: outputTextureView },
			],
		});
	}

	let canvasDebugLogged = false;
	function configureCanvas(newLogicalWidth = logicalWidth, newLogicalHeight = logicalHeight, newDpr = dpr): void {
		logicalWidth = Math.max(1, Math.floor(newLogicalWidth));
		logicalHeight = Math.max(1, Math.floor(newLogicalHeight));
		dpr = Math.max(0.5, newDpr);
		internalWidth = alignTo(Math.floor(logicalWidth * dpr), 8);
		internalHeight = alignTo(Math.floor(logicalHeight * dpr), 8);
		const cssWidth = internalWidth / dpr;
		const cssHeight = internalHeight / dpr;
		canvas.style.width = `${cssWidth}px`;
		canvas.style.height = `${cssHeight}px`;
		canvas.width = internalWidth;
		canvas.height = internalHeight;
		gpuContext.configure({ device, format, alphaMode: 'opaque' });
		cameraAspect = logicalHeight > 0 ? logicalWidth / logicalHeight : 1;
		flushCameraUniform();
		recreateOutputTexture();
		if (isWebrtxDebugLoggingEnabled() && !canvasDebugLogged) {
			canvasDebugLogged = true;
			debugLog('[Fallback] Canvas metrics', {
				logicalSize: [logicalWidth, logicalHeight],
				internalSize: [internalWidth, internalHeight],
				cssSize: [cssWidth, cssHeight],
				clientSize: [canvas.clientWidth, canvas.clientHeight],
				dpr,
				cameraAspect,
			});
		}
	}

	configureCanvas(logicalWidth, logicalHeight, dpr);

	const fpsOverlay = createFpsOverlay(canvas);

	const fallbackShaders = await getFallbackShaderWgsl();
	const computeModule = device.createShaderModule({ code: fallbackShaders.compute });
	const vertexModule = device.createShaderModule({ code: fallbackShaders.vertex });
	const fragmentModule = device.createShaderModule({ code: fallbackShaders.fragment });

	computePipeline = device.createComputePipeline({
		layout: 'auto',
		compute: { module: computeModule, entryPoint: 'main' },
	});

	renderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: { module: vertexModule, entryPoint: 'main' },
		fragment: { module: fragmentModule, entryPoint: 'main', targets: [{ format }] },
		primitive: { topology: 'triangle-list' },
	});

	if (scene.sphereBvh) {
		debugWarn('[WebRTX] Prebuilt sphere BVH ignored by fallback renderer; rebuilding with WebRTX BVH');
	}

	await updateAllPrimitives(currentScenePrimitives);
	rebuildComputeBindGroup();
	rebuildRenderBindGroup();

	updateCameraOrientation([0, 0, 5], [0, 0, 0], [0, 1, 0]);

	let stopped = false;

	const handle: FallbackPipelineHandle = {
		renderer: 'fallback',
		updateCamera: async (pos, look, up) => {
			updateCameraOrientation(pos, look, up);
		},
		render: async () => {
			if (stopped) {
				return;
			}
			const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
			updateFpsOverlay(fpsOverlay, now);
			if (!computeBindGroup) {
				rebuildComputeBindGroup();
			}
			if (!renderBindGroup) {
				rebuildRenderBindGroup();
			}
			if (!computePipeline || !computeBindGroup || !renderPipeline || !renderBindGroup || !outputTextureView) {
				return;
			}
			const dispatchX = Math.ceil(internalWidth / WORKGROUP_SIZE_X);
			const dispatchY = Math.ceil(internalHeight / WORKGROUP_SIZE_Y);
			if (dispatchX <= 0 || dispatchY <= 0) {
				return;
			}
			const encoder = device.createCommandEncoder();
			const computePass = encoder.beginComputePass();
			computePass.setPipeline(computePipeline);
			computePass.setBindGroup(0, computeBindGroup);
			computePass.dispatchWorkgroups(dispatchX, dispatchY);
			computePass.end();
			const textureView = gpuContext.getCurrentTexture().createView();
			const pass = encoder.beginRenderPass({
				colorAttachments: [{
					view: textureView,
					loadOp: 'clear',
					clearValue: { r: 0.03, g: 0.05, b: 0.08, a: 1.0 },
					storeOp: 'store',
				}],
			});
			pass.setPipeline(renderPipeline);
			pass.setBindGroup(0, renderBindGroup);
			pass.draw(3, 1, 0, 0);
			pass.end();
			device.queue.submit([encoder.finish()]);
		},
		device,
		setCameraFovY: (deg: number) => {
			const clamped = Math.max(1, Math.min(179, deg));
			cameraFovYRadians = (clamped * Math.PI) / 180;
			flushCameraUniform();
		},
		setCameraNearFar: (near: number, far: number) => {
			const eps = 1e-3;
			cameraNear = Math.max(eps, Math.min(near, far - eps));
			cameraFar = Math.max(cameraNear + eps, far);
			flushCameraUniform();
		},
		resize: (newWidth: number, newHeight: number, newDpr?: number) => {
			configureCanvas(newWidth, newHeight, newDpr ?? dpr);
		},
		setCylinder: async (instance: CylinderInstance) => {
			currentScenePrimitives.cylinders = [{ ...instance }];
			await updateAllPrimitives(currentScenePrimitives);
		},
		setTorus: async (instance: TorusInstance) => {
			currentScenePrimitives.tori = [{ ...instance }];
			await updateAllPrimitives(currentScenePrimitives);
		},
		setTori: async (instances: TorusInstance[]) => {
			currentScenePrimitives.tori = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setCylinders: async (instances: CylinderInstance[]) => {
			currentScenePrimitives.cylinders = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setSpheres: async (instances: SphereInstance[]) => {
			currentScenePrimitives.spheres = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setSpheresWithBvh: async (instances: SphereInstance[], _bvh: FallbackSphereBvh) => {
			debugWarn('[WebRTX] Prebuilt sphere BVH ignored by fallback renderer; rebuilding with WebRTX BVH');
			currentScenePrimitives.spheres = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setCircles: async (instances: CircleInstance[]) => {
			currentScenePrimitives.circles = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setEllipses: async (instances: EllipseInstance[]) => {
			currentScenePrimitives.ellipses = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setLines: async (instances: LineInstance[]) => {
			currentScenePrimitives.lines = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setCones: async (instances: ConeInstance[]) => {
			currentScenePrimitives.cones = instances.map((inst) => ({ ...inst }));
			await updateAllPrimitives(currentScenePrimitives);
		},
		setLine: async (instance: LineInstance) => {
			currentScenePrimitives.lines = [{ ...instance }];
			await updateAllPrimitives(currentScenePrimitives);
		},
		setBezierPatch: async (instance: BezierPatchInstance) => {
			currentScenePrimitives.bezierPatches = [cloneBezierPatch(instance)];
			await updateAllPrimitives(currentScenePrimitives);
		},
		setBezierPatches: async (instances: BezierPatchInstance[]) => {
			currentScenePrimitives.bezierPatches = instances.map(cloneBezierPatch);
			await updateAllPrimitives(currentScenePrimitives);
		},
		setScene: async (desc: FallbackSceneDescriptor) => {
			currentScenePrimitives = cloneSceneToState(desc);
			await updateAllPrimitives(currentScenePrimitives);
		},
		stop: () => {
			stopped = true;
			disposeFpsOverlay(fpsOverlay);
		},
	};

	return handle;
}

