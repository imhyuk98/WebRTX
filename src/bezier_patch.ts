import type { BezierPatchInstance, Vec3 } from './fallback/scene_descriptor';

export interface BezierPatchConversion {
  controlPoints: Vec3[];
  boundsMin: Vec3;
  boundsMax: Vec3;
  maxDepth: number;
  pixelEpsilon: number;
}

export const BEZIER_CONTROL_COUNT = 16;
export const BEZIER_VEC4_COUNT = 18;
export const BEZIER_FLOATS = BEZIER_VEC4_COUNT * 4;
export const BEZIER_MAX_DEPTH_DEFAULT = 10;
export const BEZIER_PIXEL_EPSILON_DEFAULT = 3;

const HERMITE_TO_BEZIER: number[][] = [
  [1, 0, 0, 0],
  [1, 0, 1 / 3, 0],
  [0, 1, 0, -1 / 3],
  [0, 1, 0, 0],
];

function toVec3(value: Vec3 | undefined, fallback: Vec3 = [0, 0, 0]): Vec3 {
  return [
    value?.[0] ?? fallback[0],
    value?.[1] ?? fallback[1],
    value?.[2] ?? fallback[2],
  ];
}

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

export function convertHermitePatchToBezier(patch: BezierPatchInstance): BezierPatchConversion {
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

export function packBezierPatchFloats(
  converted: BezierPatchConversion,
  target?: Float32Array,
  offset = 0,
): Float32Array {
  const out = target ?? new Float32Array(BEZIER_FLOATS + offset);
  let cursor = offset;
  for (let i = 0; i < converted.controlPoints.length; i++) {
    const point = converted.controlPoints[i];
    out[cursor++] = point[0];
    out[cursor++] = point[1];
    out[cursor++] = point[2];
    out[cursor++] = 0;
  }
  out[cursor++] = converted.boundsMin[0];
  out[cursor++] = converted.boundsMin[1];
  out[cursor++] = converted.boundsMin[2];
  out[cursor++] = converted.maxDepth;
  out[cursor++] = converted.boundsMax[0];
  out[cursor++] = converted.boundsMax[1];
  out[cursor++] = converted.boundsMax[2];
  out[cursor++] = converted.pixelEpsilon;
  return out;
}
