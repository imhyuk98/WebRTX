export type Vec3 = [number, number, number];

export interface SphereInstance {
	center: Vec3;
	radius: number;
}

export interface CylinderInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	radius: number;
	height: number;
	angleDeg: number;
}

export interface CircleInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	radius: number;
}

export interface EllipseInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	radiusX: number;
	radiusY: number;
}

export interface LineInstance {
	p0: Vec3;
	p1: Vec3;
	radius: number;
}

export interface ConeInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	radius: number;
	height: number;
}

export interface TorusInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	majorRadius: number;
	minorRadius: number;
	angleDeg: number;
}

export interface PlaneInstance {
	center: Vec3;
	xdir: Vec3;
	ydir: Vec3;
	halfWidth: number;
	halfHeight: number;
}

export interface BezierPatchInstance {
	p00: Vec3;
	p01: Vec3;
	p10: Vec3;
	p11: Vec3;
	du00: Vec3;
	du01: Vec3;
	du10: Vec3;
	du11: Vec3;
	dv00: Vec3;
	dv01: Vec3;
	dv10: Vec3;
	dv11: Vec3;
	duv00: Vec3;
	duv01: Vec3;
	duv10: Vec3;
	duv11: Vec3;
	maxDepth?: number;
	pixelEpsilon?: number;
}

export type NumericArrayLike = ArrayLike<number>;

export type FallbackSphereBvhFormat = 'fallback' | 'webrtx-blas';

export interface FallbackSphereBvh {
	nodeData: ArrayBufferLike | ArrayBufferView;
	nodeCount: number;
	indexData?: ArrayBufferLike | ArrayBufferView | NumericArrayLike;
	sphereCount: number;
	sphereData?: ArrayBufferLike | ArrayBufferView | NumericArrayLike;
	nodeFormat?: FallbackSphereBvhFormat;
}

export interface FallbackSceneDescriptor {
	spheres: SphereInstance[];
	cylinders: CylinderInstance[];
	circles: CircleInstance[];
	ellipses: EllipseInstance[];
	cones: ConeInstance[];
	lines: LineInstance[];
	tori: TorusInstance[];
	planes: PlaneInstance[];
	bezierPatches: BezierPatchInstance[];
	sphereBvh?: FallbackSphereBvh;
}

export interface SceneSummary {
	spheres: number;
	cylinders: number;
	circles: number;
	ellipses: number;
	cones: number;
	lines: number;
	tori: number;
	planes: number;
	bezierPatches: number;
}

export function summarizeSceneDescriptor(desc: FallbackSceneDescriptor): SceneSummary {
	return {
		spheres: desc.spheres.length,
		cylinders: desc.cylinders.length,
		circles: desc.circles.length,
		ellipses: desc.ellipses.length,
		cones: desc.cones.length,
		lines: desc.lines.length,
		tori: desc.tori.length,
		planes: desc.planes.length,
		bezierPatches: desc.bezierPatches.length,
	};
}
