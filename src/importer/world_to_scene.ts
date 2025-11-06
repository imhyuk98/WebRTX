import { attachDTDSLoader } from './dtds_importer';
import type { WorldPrimitives, WPCircle, WPCone, WPCylinder, WPEllipse, WPLine, WPPlane, WPSphere, WPTorus } from './dtds_importer';
import type {
	BezierPatchInstance,
	CircleInstance,
	ConeInstance,
	CylinderInstance,
	EllipseInstance,
	FallbackSceneDescriptor,
	LineInstance,
	PlaneInstance,
	SphereInstance,
	TorusInstance,
	Vec3,
} from '../fallback/scene_descriptor';

export interface ImporterConversionOptions {
	/** Default square size to use if plane geometry does not provide explicit width/height. */
	planeSizeFallback?: number;
	/** Default radius to assign when line thickness is unspecified. */
	lineRadiusFallback?: number;
	/** Clamp the number of primitives per type to avoid pathological scenes. */
	maxPerType?: number;
}

const DEFAULT_PLANE_SIZE = 10;
const DEFAULT_LINE_RADIUS = 1e-3;
const SMALL_EPS = 1e-9;

const identityScene: FallbackSceneDescriptor = {
	spheres: [],
	cylinders: [],
	circles: [],
	ellipses: [],
	cones: [],
	lines: [],
	tori: [],
	planes: [],
	bezierPatches: [],
};

function clampCount<T>(items: T[], limit?: number): T[] {
	if (!limit || limit <= 0 || items.length <= limit) {
		return items;
	}
	return items.slice(0, limit);
}

function length(v: Vec3): number {
	return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v: Vec3 | undefined, fallback: Vec3): Vec3 {
	if (!v) return fallback.slice() as Vec3;
	const m = length(v as Vec3);
	if (!Number.isFinite(m) || m < SMALL_EPS) {
		return fallback.slice() as Vec3;
	}
	return [v[0] / m, v[1] / m, v[2] / m];
}

function dot(a: Vec3, b: Vec3): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
	return [
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: Vec3, b: Vec3): Vec3 {
	return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(v: Vec3, scalar: number): Vec3 {
	return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
}

function ensureVector(v: Vec3 | undefined, fallback: Vec3): Vec3 {
	if (!v) return fallback.slice() as Vec3;
	if (!Number.isFinite(v[0]) || !Number.isFinite(v[1]) || !Number.isFinite(v[2])) {
		return fallback.slice() as Vec3;
	}
	return [v[0], v[1], v[2]];
}

function buildBasisFromAxis(axis: Vec3 | undefined): { xdir: Vec3; ydir: Vec3; axis: Vec3 } {
	const normAxis = normalize(axis ?? [0, 1, 0], [0, 1, 0]);
	const ref: Vec3 = Math.abs(normAxis[0]) < 0.8 ? [1, 0, 0] : [0, 0, 1];
	let xdir = cross(ref, normAxis);
	let lx = length(xdir);
	if (!Number.isFinite(lx) || lx < SMALL_EPS) {
		xdir = cross([0, 1, 0], normAxis);
		lx = length(xdir);
	}
	if (!Number.isFinite(lx) || lx < SMALL_EPS) {
		xdir = [1, 0, 0];
		lx = 1;
	}
	xdir = [xdir[0] / lx, xdir[1] / lx, xdir[2] / lx];
	let ydir = cross(normAxis, xdir);
	let ly = length(ydir);
	if (!Number.isFinite(ly) || ly < SMALL_EPS) {
		ydir = cross(xdir, normAxis);
		ly = length(ydir);
	}
	if (!Number.isFinite(ly) || ly < SMALL_EPS) {
		ydir = cross(normAxis, [1, 0, 0]);
		ly = length(ydir);
	}
	if (!Number.isFinite(ly) || ly < SMALL_EPS) {
		ydir = [0, 0, 1];
		ly = 1;
	}
	ydir = [ydir[0] / ly, ydir[1] / ly, ydir[2] / ly];
	return { xdir, ydir, axis: normAxis };
}

function buildBasisFromNormal(normal: Vec3 | undefined, hintX?: Vec3 | undefined, hintY?: Vec3 | undefined): { xdir: Vec3; ydir: Vec3; normal: Vec3 } {
	let n = normalize(normal ?? [0, 1, 0], [0, 1, 0]);
	let xdir: Vec3 | undefined = hintX ? normalize(hintX, [1, 0, 0]) : undefined;
	if (xdir) {
		const proj = dot(xdir, n);
		xdir = [xdir[0] - proj * n[0], xdir[1] - proj * n[1], xdir[2] - proj * n[2]];
		if (length(xdir) < SMALL_EPS) xdir = undefined;
	}
	if (!xdir) {
		const ref: Vec3 = Math.abs(n[0]) < 0.8 ? [1, 0, 0] : [0, 0, 1];
		xdir = cross(ref, n);
		if (length(xdir) < SMALL_EPS) {
			xdir = cross([0, 1, 0], n);
		}
		if (length(xdir) < SMALL_EPS) {
			xdir = [1, 0, 0];
		}
	}
	xdir = normalize(xdir, [1, 0, 0]);
	let ydir = hintY ? normalize(hintY, [0, 0, 1]) : undefined;
	if (ydir) {
		const projY = dot(ydir, n);
		ydir = [ydir[0] - projY * n[0], ydir[1] - projY * n[1], ydir[2] - projY * n[2]];
		const projX = dot(ydir, xdir);
		ydir = [ydir[0] - projX * xdir[0], ydir[1] - projX * xdir[1], ydir[2] - projX * xdir[2]];
	}
	if (!ydir || length(ydir) < SMALL_EPS) {
		ydir = cross(n, xdir);
	}
	ydir = normalize(ydir, [0, 0, 1]);
	return { xdir, ydir, normal: n };
}

function convertSphere(s: WPSphere): SphereInstance | null {
	if (!Number.isFinite(s.radius) || s.radius <= 0) return null;
	const center = ensureVector(s.center, [0, 0, 0]);
	return { center, radius: s.radius };
}

function convertCylinder(c: WPCylinder): CylinderInstance | null {
	if (!Number.isFinite(c.radius) || c.radius <= 0 || !Number.isFinite(c.height) || c.height <= 0) return null;
	const center = ensureVector(c.center, [0, 0, 0]);
	const { xdir, ydir } = buildBasisFromAxis(c.axis);
	return {
		center,
		xdir,
		ydir,
		radius: c.radius,
		height: c.height,
		angleDeg: 360,
	};
}

function convertCircle(c: WPCircle): CircleInstance | null {
	if (!Number.isFinite(c.radius) || c.radius <= 0) return null;
	const center = ensureVector(c.center, [0, 0, 0]);
	const { xdir, ydir } = buildBasisFromNormal(c.normal);
	return {
		center,
		xdir,
		ydir,
		radius: c.radius,
	};
}

function convertEllipse(e: WPEllipse): EllipseInstance | null {
	if (!Number.isFinite(e.radiusA) || e.radiusA <= 0 || !Number.isFinite(e.radiusB) || e.radiusB <= 0) return null;
	const center = ensureVector(e.center, [0, 0, 0]);
	const { xdir, ydir } = buildBasisFromNormal(e.normal, e.xdir);
	return {
		center,
		xdir,
		ydir,
		radiusX: e.radiusA,
		radiusY: e.radiusB,
	};
}

function convertLine(l: WPLine, fallbackRadius: number): LineInstance | null {
	const start = ensureVector(l.start, [0, 0, 0]);
	const end = ensureVector(l.end, [0, 0, 0]);
	return {
		p0: start,
		p1: end,
		radius: Number.isFinite(l.thickness) && (l.thickness as number) > 0 ? (l.thickness as number) * 0.5 : fallbackRadius,
	};
}

function convertCone(c: WPCone): ConeInstance | null {
	if (!Number.isFinite(c.radius) || c.radius <= 0 || !Number.isFinite(c.height) || c.height <= 0) return null;
	const center = ensureVector(c.center, [0, 0, 0]);
	const { xdir, ydir } = buildBasisFromAxis(c.axis);
	return {
		center,
		xdir,
		ydir,
		radius: c.radius,
		height: c.height,
	};
}

function convertTorus(t: WPTorus): TorusInstance | null {
	if (!Number.isFinite(t.majorRadius) || t.majorRadius <= 0 || !Number.isFinite(t.minorRadius) || t.minorRadius <= 0) return null;
	const center = ensureVector(t.center, [0, 0, 0]);
	const xdir = normalize(ensureVector(t.xdir, [1, 0, 0]), [1, 0, 0]);
	const ydir = normalize(ensureVector(t.ydir, [0, 1, 0]), [0, 1, 0]);
	const angleDeg = Number.isFinite(t.angleDeg) ? (t.angleDeg as number) : 360;
	return {
		center,
		xdir,
		ydir,
		majorRadius: t.majorRadius,
		minorRadius: t.minorRadius,
		angleDeg,
	};
}

function convertPlane(p: WPPlane, fallbackSize: number): PlaneInstance | null {
	const center = ensureVector(p.center, [0, 0, 0]);
	const width = p.size && Number.isFinite(p.size[0]) && p.size[0] > 0 ? Math.abs(p.size[0]) : fallbackSize;
	const height = p.size && Number.isFinite(p.size[1]) && p.size[1] > 0 ? Math.abs(p.size[1]) : fallbackSize;
	const halfWidth = width * 0.5;
	const halfHeight = height * 0.5;
	if (halfWidth <= SMALL_EPS || halfHeight <= SMALL_EPS) {
		return null;
	}
	const normal = ensureVector(p.normal ?? [0, 1, 0], [0, 1, 0]);
	const hintX = p.xdir ? ensureVector(p.xdir, [1, 0, 0]) : undefined;
	const hintY = p.ydir ? ensureVector(p.ydir, [0, 0, 1]) : undefined;
	const { xdir, ydir } = buildBasisFromNormal(normal, hintX, hintY);
	return {
		center,
		xdir,
		ydir,
		halfWidth,
		halfHeight,
	};
}

function convertBezierPatches(): BezierPatchInstance[] {
	// Placeholder: DTDS Hermite faces currently only expose corner control points.
	// To avoid producing incorrect intersections we skip automatic conversion here.
	return [];
}

export function worldPrimitivesToFallbackScene(
	world: WorldPrimitives,
	options: ImporterConversionOptions = {}
): FallbackSceneDescriptor {
	if (!world) return identityScene;
	const planeSizeFallback = options.planeSizeFallback ?? DEFAULT_PLANE_SIZE;
	const lineRadiusFallback = options.lineRadiusFallback ?? DEFAULT_LINE_RADIUS;
	const limit = options.maxPerType;

	const spheres = clampCount(
		world.spheres
			.map(convertSphere)
			.filter((v): v is SphereInstance => !!v),
		limit
	);

	const cylinders = clampCount(
		world.cylinders
			.map(convertCylinder)
			.filter((v): v is CylinderInstance => !!v),
		limit
	);

	const circles = clampCount(
		world.circles
			.map(convertCircle)
			.filter((v): v is CircleInstance => !!v),
		limit
	);

	const ellipses = clampCount(
		world.ellipses
			.map(convertEllipse)
			.filter((v): v is EllipseInstance => !!v),
		limit
	);

	const cones = clampCount(
		world.cones
			.map(convertCone)
			.filter((v): v is ConeInstance => !!v),
		limit
	);

	const lines = clampCount(
		world.lines
			.map((line) => convertLine(line, lineRadiusFallback))
			.filter((v): v is LineInstance => !!v),
		limit
	);

	const tori = clampCount(
		(world.toruses ?? [])
			.map(convertTorus)
			.filter((v): v is TorusInstance => !!v),
		limit
	);

	const planes = clampCount(
		world.planes
			.map((plane) => convertPlane(plane, planeSizeFallback))
			.filter((v): v is PlaneInstance => !!v),
		limit
	);

	const bezierPatches = clampCount(convertBezierPatches(), limit).filter(
		(v): v is BezierPatchInstance => !!v
	);

	return {
		spheres,
		cylinders,
		circles,
		ellipses,
		cones,
		lines,
		tori,
		planes,
		bezierPatches,
	};
}

export interface SceneSettable {
	setScene(scene: FallbackSceneDescriptor): Promise<void>;
}

export interface SceneApplyContext {
	world: WorldPrimitives;
	fallbackScene: FallbackSceneDescriptor;
	file?: File;
}

export interface SceneApplyOptions extends ImporterConversionOptions {
	beforeSetScene?: (context: SceneApplyContext) => void | Promise<void>;
	afterSetScene?: (context: SceneApplyContext) => void | Promise<void>;
}

export interface SceneLoaderErrorContext {
	world: WorldPrimitives;
	file: File;
	fallbackScene?: FallbackSceneDescriptor;
}

export interface AttachDTDSSceneLoaderOptions extends SceneApplyOptions {
	loader?: {
		log?: boolean;
		samplePerKind?: number;
		collapsed?: boolean;
	};
	onError?: (error: unknown, context: SceneLoaderErrorContext) => void | Promise<void>;
}

export async function applyWorldPrimitivesToScene(
	world: WorldPrimitives,
	sceneHandle: SceneSettable,
	options: SceneApplyOptions = {},
	contextOverrides: Partial<SceneApplyContext> = {}
): Promise<FallbackSceneDescriptor> {
	const { beforeSetScene, afterSetScene } = options;
	const conversionOptions: ImporterConversionOptions = {
		planeSizeFallback: options.planeSizeFallback,
		lineRadiusFallback: options.lineRadiusFallback,
		maxPerType: options.maxPerType,
	};
	const fallbackScene =
		contextOverrides.fallbackScene ?? worldPrimitivesToFallbackScene(world, conversionOptions);
	const context: SceneApplyContext = {
		world,
		fallbackScene,
		file: contextOverrides.file ?? undefined,
	};
	if (beforeSetScene) {
		await beforeSetScene(context);
	}
	await sceneHandle.setScene(fallbackScene);
	if (afterSetScene) {
		await afterSetScene(context);
	}
	return fallbackScene;
}

export function attachDTDSSceneLoader(
	input: string | HTMLInputElement,
	sceneHandle: SceneSettable,
	options: AttachDTDSSceneLoaderOptions = {}
): void {
	const { loader, onError, ...applyOptions } = options;
	attachDTDSLoader(
		input,
		(world, file) => {
			void (async () => {
				let fallbackScene: FallbackSceneDescriptor | undefined;
				try {
					const conversionOptions: ImporterConversionOptions = {
						planeSizeFallback: applyOptions.planeSizeFallback,
						lineRadiusFallback: applyOptions.lineRadiusFallback,
						maxPerType: applyOptions.maxPerType,
					};
					fallbackScene = worldPrimitivesToFallbackScene(world, conversionOptions);
				} catch (error) {
					if (onError) {
						await onError(error, { world, file, fallbackScene });
					} else {
						console.error('[WebRTX] Failed to convert DTDS world to fallback scene.', error);
					}
					return;
				}
				try {
					await applyWorldPrimitivesToScene(world, sceneHandle, applyOptions, { file, fallbackScene });
				} catch (error) {
					if (onError) {
						await onError(error, { world, file, fallbackScene });
					} else {
						console.error('[WebRTX] Failed to apply DTDS scene to renderer.', error);
					}
				}
			})();
		},
		loader
	);
}

export type { WorldPrimitives } from './dtds_importer';
