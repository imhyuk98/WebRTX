// Minimal analytic renderer using plain compute (WGSL), removing ray tracing pipeline & acceleration structure dependency.

import { initBvhWasm } from './wasm_bvh_builder';
import { buildAnalyticPrimitivesTLAS } from './analytic_primitives_bvh';
import {
  getRayGenShader,
  getMissShader,
  getIntersectionShader,
  getClosestHitShader,
} from './analytic_shaders';
import {
  getRayGenShaderNoBezier,
  getMissShaderNoBezier,
  getIntersectionShaderNoBezier,
  getClosestHitShaderNoBezier,
} from './analytic_shaders_no_bezier';
import { runFallbackPipeline } from './fallback/pipeline';
import { BEZIER_FLOATS, convertHermitePatchToBezier, packBezierPatchFloats } from './bezier_patch';
import type {
  FallbackPipelineHandle,
} from './fallback/pipeline';
import type {
  FallbackSceneDescriptor,
  Vec3,
  SphereInstance,
  CylinderInstance,
  CircleInstance,
  EllipseInstance,
  ConeInstance,
  LineInstance,
  TorusInstance,
  PlaneInstance,
  BezierPatchInstance,
  FallbackSphereBvh,
} from './fallback/scene_descriptor';

const DEFAULT_WIDTH = 1920;
const DEFAULT_HEIGHT = 1080;
const DEFAULT_SPHERES_COUNT = 100;
const LINE_RADIUS = 0.001;
const DEFAULT_BEZIER_PATCHES: BezierPatchInstance[] = [{
  p00: [0.6, -1.15, -2.6],
  p10: [1.1, -1.05, -2.45],
  p01: [0.65, -1.0, -2.1],
  p11: [1.15, -0.95, -1.95],
  du00: [0.35, 0.05, 0.18],
  du10: [0.32, 0.05, 0.15],
  du01: [0.38, 0.04, 0.16],
  du11: [0.34, 0.04, 0.14],
  dv00: [0.04, 0.18, 0.42],
  dv10: [-0.03, 0.18, 0.38],
  dv01: [0.05, 0.16, 0.36],
  dv11: [-0.02, 0.16, 0.33],
  duv00: [0.02, -0.03, -0.04],
  duv10: [-0.015, -0.03, -0.035],
  duv01: [0.025, -0.025, -0.04],
  duv11: [-0.02, -0.025, -0.035],
  maxDepth: 5,
  pixelEpsilon: 1.5,
}];

interface MinimalSceneConfig {
  sphereCenter: Vec3;
  sphereRadius: number;
  cylinderCenter: Vec3;
  cylinderXDir: Vec3;
  cylinderYDir: Vec3;
  cylinderRadius: number;
  cylinderHeight: number;
  cylinderAngleDeg: number;
  circleCenter: Vec3;
  circleXDir: Vec3;
  circleYDir: Vec3;
  circleRadius: number;
  lineP0: Vec3;
  lineP1: Vec3;
  ellipseCenter: Vec3;
  ellipseXDir: Vec3;
  ellipseYDir: Vec3;
  ellipseRadiusX: number;
  ellipseRadiusY: number;
  torusCenter: Vec3;
  torusXDir: Vec3;
  torusYDir: Vec3;
  torusMajorRadius: number;
  torusMinorRadius: number;
  torusAngleDeg: number;
  coneCenter: Vec3;
  coneXDir: Vec3;
  coneYDir: Vec3;
  coneRadius: number;
  coneHeight: number;
  planeCenter: Vec3;
  planeXDir: Vec3;
  planeYDir: Vec3;
  planeHalfWidth: number;
  planeHalfHeight: number;
  bezierPatches?: BezierPatchInstance[];
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

function normalizeVec3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (!Number.isFinite(len) || len <= 1e-6) {
    return [0, 1, 0];
  }
  const inv = 1 / len;
  return [v[0] * inv, v[1] * inv, v[2] * inv];
}

function orthonormalizePair(xdir: Vec3, ydir: Vec3): [Vec3, Vec3] {
  const x = normalizeVec3(xdir);
  let y = ydir;
  const dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  y = [y[0] - dot * x[0], y[1] - dot * x[1], y[2] - dot * x[2]];
  y = normalizeVec3(y);
  if (!Number.isFinite(y[0]) || Math.hypot(y[0], y[1], y[2]) <= 1e-6) {
    const fallbackBasis: Vec3 = Math.abs(x[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const fbDot = x[0] * fallbackBasis[0] + x[1] * fallbackBasis[1] + x[2] * fallbackBasis[2];
    const adjusted: Vec3 = [
      fallbackBasis[0] - fbDot * x[0],
      fallbackBasis[1] - fbDot * x[1],
      fallbackBasis[2] - fbDot * x[2],
    ];
    y = normalizeVec3(adjusted);
  }
  return [x, y];
}

function buildMinimalSceneState(config: MinimalSceneConfig): MinimalSceneConfig {
  return {
    sphereCenter: cloneVec3(config.sphereCenter),
    sphereRadius: config.sphereRadius,
    cylinderCenter: cloneVec3(config.cylinderCenter),
    cylinderXDir: cloneVec3(config.cylinderXDir),
    cylinderYDir: cloneVec3(config.cylinderYDir),
    cylinderRadius: config.cylinderRadius,
    cylinderHeight: config.cylinderHeight,
    cylinderAngleDeg: config.cylinderAngleDeg,
    circleCenter: cloneVec3(config.circleCenter),
    circleXDir: cloneVec3(config.circleXDir),
    circleYDir: cloneVec3(config.circleYDir),
    circleRadius: config.circleRadius,
    lineP0: cloneVec3(config.lineP0),
    lineP1: cloneVec3(config.lineP1),
    ellipseCenter: cloneVec3(config.ellipseCenter),
    ellipseXDir: cloneVec3(config.ellipseXDir),
    ellipseYDir: cloneVec3(config.ellipseYDir),
    ellipseRadiusX: config.ellipseRadiusX,
    ellipseRadiusY: config.ellipseRadiusY,
    torusCenter: cloneVec3(config.torusCenter),
    torusXDir: cloneVec3(config.torusXDir),
    torusYDir: cloneVec3(config.torusYDir),
    torusMajorRadius: config.torusMajorRadius,
    torusMinorRadius: config.torusMinorRadius,
    torusAngleDeg: config.torusAngleDeg,
    coneCenter: cloneVec3(config.coneCenter),
    coneXDir: cloneVec3(config.coneXDir),
    coneYDir: cloneVec3(config.coneYDir),
    coneRadius: config.coneRadius,
    coneHeight: config.coneHeight,
    planeCenter: cloneVec3(config.planeCenter),
    planeXDir: cloneVec3(config.planeXDir),
    planeYDir: cloneVec3(config.planeYDir),
    planeHalfWidth: config.planeHalfWidth,
    planeHalfHeight: config.planeHalfHeight,
    bezierPatches: config.bezierPatches ? config.bezierPatches.map(cloneBezierPatch) : [],
  };
}

function buildFallbackSceneDescriptor(config: MinimalSceneConfig): FallbackSceneDescriptor {
  const [cylinderX, cylinderY] = orthonormalizePair(config.cylinderXDir, config.cylinderYDir);
  const [circleX, circleY] = orthonormalizePair(config.circleXDir, config.circleYDir);
  const [ellipseX, ellipseY] = orthonormalizePair(config.ellipseXDir, config.ellipseYDir);
  const [torusX, torusY] = orthonormalizePair(config.torusXDir, config.torusYDir);
  const [coneX, coneY] = orthonormalizePair(config.coneXDir, config.coneYDir);
  const overrides = (globalThis as any).__webrtxOverrides || {};
  const lineStart = (overrides.startPoint ?? config.lineP0) as Vec3;
  const lineEnd = (overrides.endPoint ?? config.lineP1) as Vec3;
  const [planeX, planeY] = orthonormalizePair(config.planeXDir, config.planeYDir);
  return {
    spheres: [{ center: cloneVec3(config.sphereCenter), radius: config.sphereRadius }],
    cylinders: [{
      center: cloneVec3(config.cylinderCenter),
      xdir: cylinderX,
      ydir: cylinderY,
      radius: config.cylinderRadius,
      height: config.cylinderHeight,
      angleDeg: config.cylinderAngleDeg,
    }],
    circles: [{
      center: cloneVec3(config.circleCenter),
      xdir: circleX,
      ydir: circleY,
      radius: config.circleRadius,
    }],
    ellipses: [{
      center: cloneVec3(config.ellipseCenter),
      xdir: ellipseX,
      ydir: ellipseY,
      radiusX: config.ellipseRadiusX,
      radiusY: config.ellipseRadiusY,
    }],
    cones: [{
      center: cloneVec3(config.coneCenter),
      xdir: coneX,
      ydir: coneY,
      radius: config.coneRadius,
      height: config.coneHeight,
    }],
    lines: [{
      p0: cloneVec3(lineStart),
      p1: cloneVec3(lineEnd),
      radius: LINE_RADIUS,
    }],
    tori: [{
      center: cloneVec3(config.torusCenter),
      xdir: torusX,
      ydir: torusY,
      majorRadius: config.torusMajorRadius,
      minorRadius: config.torusMinorRadius,
      angleDeg: config.torusAngleDeg,
    }],
    planes: [{
      center: cloneVec3(config.planeCenter),
      xdir: planeX,
      ydir: planeY,
      halfWidth: config.planeHalfWidth,
      halfHeight: config.planeHalfHeight,
    }],
    bezierPatches: config.bezierPatches && config.bezierPatches.length > 0
      ? config.bezierPatches.map(cloneBezierPatch)
      : [],
  };
}

function computeOverrideSphereInstances(): SphereInstance[] | null {
  const overrides = (globalThis as any).__webrtxOverrides || {};
  try {
    const hash = (globalThis as any).location?.hash || '';
    const match = /spheres=(\d+)/.exec(hash);
    if (match && !overrides.spheresCount) {
      overrides.spheresCount = parseInt(match[1], 10);
    }
  } catch {}
  const hasGrid = Array.isArray(overrides.spheresGrid) && overrides.spheresGrid.length === 3;
  const explicitCount = typeof overrides.spheresCount === 'number' ? overrides.spheresCount : undefined;
  const spheresCount = explicitCount ?? 0;
  if (!hasGrid && !(spheresCount > 0)) {
    return null;
  }
  let nx = 0; let ny = 0; let nz = 0;
  const targetCount = hasGrid ? 0 : Math.max(1, spheresCount);
  if (hasGrid) {
    nx = Math.max(1, overrides.spheresGrid[0] | 0);
    ny = Math.max(1, overrides.spheresGrid[1] | 0);
    nz = Math.max(1, overrides.spheresGrid[2] | 0);
  } else {
    const maxDim = Math.max(1, Math.ceil(Math.cbrt(targetCount)));
    const remaining = Math.max(1, Math.ceil(targetCount / maxDim));
    const midDim = Math.max(1, Math.ceil(Math.sqrt(remaining)));
    const minDim = Math.max(1, Math.ceil(remaining / midDim));
    nx = maxDim;
    ny = midDim;
    nz = minDim;
  }
  const radius = (typeof overrides.spheresRadius === 'number' && overrides.spheresRadius > 0)
    ? overrides.spheresRadius
    : 0.25;
  const spacing = (typeof overrides.spheresSpacing === 'number' && overrides.spheresSpacing > 0)
    ? overrides.spheresSpacing
    : (radius * 2.4);
  const ox = (nx - 1) * 0.5 * spacing;
  const oy = (ny - 1) * 0.5 * spacing;
  const oz = (nz - 1) * 0.5 * spacing;
  const list: SphereInstance[] = [];
  const maxCount = hasGrid ? nx * ny * nz : targetCount;
outer:
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        list.push({
          center: [x * spacing - ox, y * spacing - oy, z * spacing - oz],
          radius,
        });
        if (!hasGrid && list.length >= maxCount) {
          break outer;
        }
      }
    }
  }
  return hasGrid ? list : list.slice(0, maxCount);
}

async function launchFallbackRenderer(
  canvasOrId: HTMLCanvasElement | string | undefined,
  width: number,
  height: number,
  config: MinimalSceneConfig,
  adapter?: GPUAdapter,
  device?: GPUDevice,
) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is not supported; fallback renderer unavailable.');
  }
  const resolvedAdapter = adapter ?? await navigator.gpu.requestAdapter();
  if (!resolvedAdapter) {
    throw new Error('No GPU adapter available for fallback renderer.');
  }
  const overrides = (globalThis as any).__webrtxOverrides || {};
  const allowBezierUploads = !(overrides.disableBezier === true || overrides.disableHermite === true);
  const handle = await runFallbackPipeline({
    canvasOrId,
    width,
    height,
    adapter: resolvedAdapter,
    device,
    scene: buildFallbackSceneDescriptor(config),
    enableBezierUploads: allowBezierUploads,
  });
  const overrideSpheres = computeOverrideSphereInstances();
  if (overrideSpheres && overrideSpheres.length > 0) {
    try {
      await handle.setSpheres(overrideSpheres);
    } catch (err) {
      console.warn('[WebRTX] Failed to apply sphere overrides on fallback renderer.', err);
    }
  }
  const disableBezier = overrides.disableBezier === true || overrides.disableHermite === true;
  if (disableBezier) {
    try {
      await handle.setBezierUploadsEnabled(false);
    } catch (err) {
      console.warn('[WebRTX] Failed to disable Bezier uploads on fallback renderer.', err);
    }
  }
  if (disableBezier || overrides.keepBezier !== true) {
    try {
      await handle.setBezierPatches([]);
    } catch (err) {
      console.warn('[WebRTX] Failed to clear default Bezier patches on fallback renderer.', err);
    }
  }
  return handle;
}

async function runMinimalSceneImpl(
  canvasOrId?: HTMLCanvasElement | string,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  sphereCenter: [number, number, number] = [0, 0, 0],
  sphereRadius = 1.2,
  cylinderCenter: [number, number, number] = [1.8, 0.0, 0.0],
  cylinderXDir: [number, number, number] = [1.0, 0.0, 0.0],
  cylinderYDir: [number, number, number] = [0.0, 0.0, -1.0],
  cylinderRadius = 0.4,
  cylinderHeight = 2.0,
  cylinderAngleDeg = 360.0,
  circleCenter: [number, number, number] = [-1.8, 0.0, 0.0],
  circleXDir: [number, number, number] = [1.0, 0.0, 0.0],
  circleYDir: [number, number, number] = [0.0, 1.0, 0.0],
  circleRadius = 0.9,
  // Line segment (optional demo values)
  lineP0: [number, number, number] = [-2.8, 0.8, -0.5],
  lineP1: [number, number, number] = [-1.6, 1.6, -0.2],
  // radius is fixed in scene; ignore external thickness to avoid artifacts
  // Ellipse (optional)
  // Moved aside so it doesn't overlap the circle
  ellipseCenter: [number, number, number] = [-4.0, 0.0, 0.0],
  ellipseXDir: [number, number, number] = [1.0, 0.0, 0.0],
  ellipseYDir: [number, number, number] = [0.0, 1.0, 0.0],
  ellipseRadiusX = 0.9,
  ellipseRadiusY = 0.5,
  // Torus (new)
  torusCenter: [number, number, number] = [4.0, 0.2, 0.0],
  torusXDir: [number, number, number] = [1.0, 0.0, 0.0],
  torusYDir: [number, number, number] = [0.0, 1.0, 0.0],
  torusMajorRadius = 0.9,
  torusMinorRadius = 0.25,
  torusAngleDeg = 360.0,
  // New cone defaults
  coneCenter: [number, number, number] = [0.0, -1.0, -1.2],
  coneXDir: [number, number, number] = [1.0, 0.0, 0.0],
  coneYDir: [number, number, number] = [0.0, 0.0, 1.0],
  coneRadius = 0.7,
  coneHeight = 1.6,
  planeCenter: [number, number, number] = [0.0, -2.0, 0.0],
  planeXDir: [number, number, number] = [1.0, 0.0, 0.0],
  planeYDir: [number, number, number] = [0.0, 0.0, 1.0],
  planeHalfWidth = 10.0,
  planeHalfHeight = 10.0,
  baseSceneState?: MinimalSceneConfig,
){
  const __ovr = (globalThis as any).__webrtxOverrides || {};
  const fallbackSceneState = baseSceneState ?? buildMinimalSceneState({
    sphereCenter,
    sphereRadius,
    cylinderCenter,
    cylinderXDir,
    cylinderYDir,
    cylinderRadius,
    cylinderHeight,
    cylinderAngleDeg,
    circleCenter,
    circleXDir,
    circleYDir,
    circleRadius,
    lineP0,
    lineP1,
    ellipseCenter,
    ellipseXDir,
    ellipseYDir,
    ellipseRadiusX,
    ellipseRadiusY,
    torusCenter,
    torusXDir,
    torusYDir,
    torusMajorRadius,
    torusMinorRadius,
    torusAngleDeg,
    coneCenter,
    coneXDir,
    coneYDir,
    coneRadius,
    coneHeight,
    planeCenter,
    planeXDir,
    planeYDir,
    planeHalfWidth,
    planeHalfHeight,
    bezierPatches: DEFAULT_BEZIER_PATCHES,
  });
  const disableBezier = __ovr.disableBezier === true || __ovr.disableHermite === true;
  if (disableBezier) {
    fallbackSceneState.bezierPatches = [];
  }
  // Allow interactive options to override line as startPoint/endPoint with default thickness
  const lp0: [number,number,number] = (__ovr.startPoint ?? lineP0) as [number,number,number];
  const lp1: [number,number,number] = (__ovr.endPoint ?? lineP1) as [number,number,number];
  const lrad: number = LINE_RADIUS;
  // This refactored function now uses the ray tracing pipeline (no compute fallback)
  const canvas: HTMLCanvasElement | null = typeof canvasOrId === 'string'
    ? document.getElementById(canvasOrId) as HTMLCanvasElement
    : (canvasOrId || document.getElementById('gfx')) as HTMLCanvasElement;
  if(!canvas) throw new Error('Canvas not found');
  if(!navigator.gpu) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  if(!adapter) throw new Error('No GPU adapter');
  let device: GPUDevice;
  try {
    const lim = adapter.limits as any;
    device = await adapter.requestDevice({
      requiredFeatures: ['ray_tracing' as GPUFeatureName],
      requiredLimits: { maxStorageBuffersPerShaderStage: lim.maxStorageBuffersPerShaderStage }
    } as any);
  } catch {
    const lim = adapter?.limits as any;
    device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: lim?.maxStorageBuffersPerShaderStage } } as any);
  }
  const supportsRayTracing = Boolean(((device?.features as any)?.has?.('ray_tracing')));
  if (!supportsRayTracing) {
    console.warn('[WebRTX] Ray tracing feature unavailable; switching to fallback renderer.');
    return launchFallbackRenderer(canvasOrId, width, height, fallbackSceneState, adapter, device);
  }
  if((globalThis as any).WebRTX?.initBvhWasm) { await (globalThis as any).WebRTX.initBvhWasm(); } else { await initBvhWasm(); }
  const context = canvas.getContext('webgpu')!;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  // DPR-aware internal size, align to 8 (workgroup size)
  let logicalWidth = width; let logicalHeight = height; let dpr = (globalThis as any).devicePixelRatio || 1;
  // Optional global overrides to scale down rendering cost without changing API
  const __ovrRes = (globalThis as any).__webrtxOverrides || {};
  const lockDpr: boolean = __ovrRes.lockDpr === true;
  const renderScale: number = (typeof __ovrRes.renderScale === 'number' && __ovrRes.renderScale > 0) ? __ovrRes.renderScale : 1.0;
  if (lockDpr) dpr = 1;
  function align8(v:number){ return Math.max(8, (v+7)&~7); }
  let internalWidth = align8(Math.floor(logicalWidth * dpr * renderScale));
  let internalHeight = align8(Math.floor(logicalHeight * dpr * renderScale));
  function configureCanvas(){
  const c = canvas as HTMLCanvasElement; // already validated
  c.style.width = logicalWidth + 'px';
  c.style.height = logicalHeight + 'px';
  c.width = internalWidth; c.height = internalHeight;
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
  }
  configureCanvas();
  // Camera uniform
  const cameraBuf = device.createBuffer({ size: 16*4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let currentFov = 60*Math.PI/180, currentNear=0.0001, currentFar=1000.0;
  function writeCamera(pos:[number,number,number], look:[number,number,number], up:[number,number,number]){
    const d = new Float32Array([
      pos[0],pos[1],pos[2],0,
      look[0],look[1],look[2],0,
      up[0],up[1],up[2],0,
      currentFov,0,currentNear,currentFar
    ]); device.queue.writeBuffer(cameraBuf,0,d);
  }
  writeCamera([0,0,3],[0,0,0],[0,1,0]);
  // Color buffer
  let colorBuffer = device.createBuffer({ size: internalWidth*internalHeight*4*Float32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  // Primitive uniform buffers
  const sphereBuf = device.createBuffer({ size:16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sphereBuf,0,new Float32Array([sphereCenter[0],sphereCenter[1],sphereCenter[2],sphereRadius]));
  const xdn = (()=>{ const v=new Float32Array(cylinderXDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const ydn0 = (()=>{ const v=new Float32Array(cylinderYDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const dotXY = xdn[0]*ydn0[0]+xdn[1]*ydn0[1]+xdn[2]*ydn0[2];
  let ydn = [ ydn0[0]-dotXY*xdn[0], ydn0[1]-dotXY*xdn[1], ydn0[2]-dotXY*xdn[2] ];
  const lY = Math.hypot(ydn[0],ydn[1],ydn[2])||1; ydn=[ydn[0]/lY, ydn[1]/lY, ydn[2]/lY];
  const cylBuf = device.createBuffer({ size: 16*3, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const cylData = new Float32Array([
    cylinderCenter[0],cylinderCenter[1],cylinderCenter[2],cylinderRadius,
    xdn[0],xdn[1],xdn[2],cylinderHeight,
    ydn[0],ydn[1],ydn[2],cylinderAngleDeg
  ]); device.queue.writeBuffer(cylBuf,0,cylData);
  // Cylinder SSBO array (binding 12): mirror of uniform layout per instance
  let cylArrayBuf = device.createBuffer({ size: 16*3, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let cylArrayCapacityBytes = 16*3;
  function writeCylinderArrayFromCurr(){
    const list = ((curr as any).cylinders && (curr as any).cylinders.length>0)
      ? (curr as any).cylinders
      : [{ center: curr.cylinderCenter, xdir: xdn as [number,number,number], ydir: ydn as [number,number,number], radius: curr.cylinderRadius, height: curr.cylinderHeight, angleDeg: curr.cylinderAngleDeg }];
    const n = list.length; const bytes = n*16*3; if (cylArrayCapacityBytes < bytes) {
      cylArrayCapacityBytes = Math.max(bytes, 16*3);
      cylArrayBuf = device.createBuffer({ size: cylArrayCapacityBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*12);
    for(let i=0;i<n;i++){
      const c = list[i]; const base=i*12;
      arr[base+0]=c.center[0]; arr[base+1]=c.center[1]; arr[base+2]=c.center[2]; arr[base+3]=c.radius;
      arr[base+4]=c.xdir[0]; arr[base+5]=c.xdir[1]; arr[base+6]=c.xdir[2]; arr[base+7]=c.height;
      arr[base+8]=c.ydir[0]; arr[base+9]=c.ydir[1]; arr[base+10]=c.ydir[2]; arr[base+11]=(c.angleDeg ?? 360.0);
    }
    device.queue.writeBuffer(cylArrayBuf, 0, arr);
  }
  // Defer initial write until after 'curr' is initialized below to avoid TDZ
  const cxd = (()=>{ const v=new Float32Array(circleXDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const cyd0 = (()=>{ const v=new Float32Array(circleYDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const dotCX = cxd[0]*cyd0[0]+cxd[1]*cyd0[1]+cxd[2]*cyd0[2];
  let cyd = [cyd0[0]-dotCX*cxd[0],cyd0[1]-dotCX*cxd[1],cyd0[2]-dotCX*cxd[2]]; const lCY=Math.hypot(cyd[0],cyd[1],cyd[2])||1; cyd=[cyd[0]/lCY,cyd[1]/lCY,cyd[2]/lCY];
  const circleBuf = device.createBuffer({ size:16*3, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const circleData = new Float32Array([
    circleCenter[0],circleCenter[1],circleCenter[2],circleRadius,
    cxd[0],cxd[1],cxd[2],0,
    cyd[0],cyd[1],cyd[2],0,
  ]); device.queue.writeBuffer(circleBuf,0,circleData);
  // Circle SSBO array (binding 13)
  // Replaced by packed buffer
  let circleArrayBuf: GPUBuffer | null = null; let circleArrayCapBytes = 0;
  function writeCircleArrayFromCurr(){
    if(!circleArrayBuf) return; // packed path in use; keep for TLAS-only callers
    const list = ((curr as any).circles && (curr as any).circles.length>0)
      ? (curr as any).circles
      : [{ center: curr.circleCenter, xdir: cxd as [number,number,number], ydir: cyd as [number,number,number], radius: curr.circleRadius }];
    const n = list.length; const bytes = n*16*3; if (circleArrayCapBytes < bytes) {
      circleArrayCapBytes = Math.max(bytes, 16*3);
      circleArrayBuf = device.createBuffer({ size: circleArrayCapBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*12);
    for(let i=0;i<n;i++){
      const d = list[i]; const base=i*12;
      arr[base+0]=d.center[0]; arr[base+1]=d.center[1]; arr[base+2]=d.center[2]; arr[base+3]=d.radius;
      arr[base+4]=d.xdir[0]; arr[base+5]=d.xdir[1]; arr[base+6]=d.xdir[2]; arr[base+7]=0;
      arr[base+8]=d.ydir[0]; arr[base+9]=d.ydir[1]; arr[base+10]=d.ydir[2]; arr[base+11]=0;
    }
    device.queue.writeBuffer(circleArrayBuf, 0, arr);
  }
  // Line uniform (binding 8): l0: p0.xyz, radius; l1: p1.xyz, 0
  const lineBuf = device.createBuffer({ size:16*2, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lineData = new Float32Array([
    lp0[0], lp0[1], lp0[2], lrad,
    lp1[0], lp1[1], lp1[2], 0,
  ]); device.queue.writeBuffer(lineBuf,0,lineData);
  // Line SSBO array (binding 15)
  // Replaced by packed buffer
  let lineArrayBuf: GPUBuffer | null = null; let lineArrayCapBytes = 0;
  function writeLineArrayFromCurr(){
    if(!lineArrayBuf) return; // packed path in use; keep for TLAS-only callers
    const list = ((curr as any).lines && (curr as any).lines.length>0)
      ? (curr as any).lines
      : [{ p0: curr.lineP0 as [number,number,number], p1: curr.lineP1 as [number,number,number], radius: curr.lineRadius as number }];
    const n = list.length; const bytes = n*16*2; if (lineArrayCapBytes < bytes) {
      lineArrayCapBytes = Math.max(bytes, 16*2);
      lineArrayBuf = device.createBuffer({ size: lineArrayCapBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*8);
    for(let i=0;i<n;i++){
      const L = list[i]; const base=i*8;
      arr[base+0]=L.p0[0]; arr[base+1]=L.p0[1]; arr[base+2]=L.p0[2]; arr[base+3]=L.radius;
      arr[base+4]=L.p1[0]; arr[base+5]=L.p1[1]; arr[base+6]=L.p1[2]; arr[base+7]=0;
    }
    device.queue.writeBuffer(lineArrayBuf, 0, arr);
  }
  // Ellipse uniform (binding 7): e0: center.xyz, rx; e1: xdir.xyz, ry; e2: ydir.xyz, 0
  const exd = (()=>{ const v=new Float32Array(ellipseXDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const eyd0 = (()=>{ const v=new Float32Array(ellipseYDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const dE = exd[0]*eyd0[0]+exd[1]*eyd0[1]+exd[2]*eyd0[2];
  let eyd = [eyd0[0]-dE*exd[0], eyd0[1]-dE*exd[1], eyd0[2]-dE*exd[2]]; { const L=Math.hypot(eyd[0],eyd[1],eyd[2])||1; eyd=[eyd[0]/L,eyd[1]/L,eyd[2]/L]; }
  const ellipseBuf = device.createBuffer({ size:16*3, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ellipseData = new Float32Array([
    ellipseCenter[0],ellipseCenter[1],ellipseCenter[2],ellipseRadiusX,
    exd[0],exd[1],exd[2],ellipseRadiusY,
    eyd[0],eyd[1],eyd[2],0,
  ]); device.queue.writeBuffer(ellipseBuf,0,ellipseData);
  // Ellipse SSBO array (binding 14)
  // Replaced by packed buffer
  let ellipseArrayBuf: GPUBuffer | null = null; let ellipseArrayCapBytes = 0;
  function writeEllipseArrayFromCurr(){
    if(!ellipseArrayBuf) return; // packed path in use; keep for TLAS-only callers
    const list = ((curr as any).ellipses && (curr as any).ellipses.length>0)
      ? (curr as any).ellipses
      : [{ center: curr.ellipseCenter, xdir: exd as [number,number,number], ydir: eyd as [number,number,number], rx: curr.ellipseRadiusX, ry: curr.ellipseRadiusY }];
    const n = list.length; const bytes = n*16*3; if (ellipseArrayCapBytes < bytes) {
      ellipseArrayCapBytes = Math.max(bytes, 16*3);
      ellipseArrayBuf = device.createBuffer({ size: ellipseArrayCapBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*12);
    for(let i=0;i<n;i++){
      const e = list[i]; const base=i*12;
      arr[base+0]=e.center[0]; arr[base+1]=e.center[1]; arr[base+2]=e.center[2]; arr[base+3]=e.rx;
      arr[base+4]=e.xdir[0]; arr[base+5]=e.xdir[1]; arr[base+6]=e.xdir[2]; arr[base+7]=e.ry;
      arr[base+8]=e.ydir[0]; arr[base+9]=e.ydir[1]; arr[base+10]=e.ydir[2]; arr[base+11]=0;
    }
    device.queue.writeBuffer(ellipseArrayBuf, 0, arr);
  }
  // Torus SSBO array (binding 10) will be used instead of single uniform
  // Torus array SSBO (binding 10): struct of 3 vec4s per torus
  let torusArrayBuf = device.createBuffer({ size: 16*3, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let torusArrayCapacityBytes = 16*3;
  // Cone uniform (binding 6): e0: center.xyz, radius; e1: axis.xyz (from xdirxydir), height
  const kx = (()=>{ const v=new Float32Array(coneXDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const ky0 = (()=>{ const v=new Float32Array(coneYDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const dK = kx[0]*ky0[0]+kx[1]*ky0[1]+kx[2]*ky0[2];
  let ky = [ky0[0]-dK*kx[0], ky0[1]-dK*kx[1], ky0[2]-dK*kx[2]]; const lKY=Math.hypot(ky[0],ky[1],ky[2])||1; ky=[ky[0]/lKY,ky[1]/lKY,ky[2]/lKY];
  const kz = [ kx[1]*ky[2]-kx[2]*ky[1], kx[2]*ky[0]-kx[0]*ky[2], kx[0]*ky[1]-kx[1]*ky[0] ];
  const coneBuf = device.createBuffer({ size:16*2, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const coneData = new Float32Array([
    coneCenter[0],coneCenter[1],coneCenter[2],coneRadius,
    kz[0],kz[1],kz[2],coneHeight,
  ]); device.queue.writeBuffer(coneBuf,0,coneData);
  // Replaced by packed buffer
  let coneArrayBuf: GPUBuffer | null = null; let coneArrayCapBytes = 0;
  function writeConeArrayFromCurr(){
    if(!coneArrayBuf) return; // packed path in use; keep for TLAS-only callers
    const list = ((curr as any).cones && (curr as any).cones.length>0)
      ? (curr as any).cones
      : [{ center: curr.coneCenter as [number,number,number], xdir: kx as [number,number,number], ydir: ky as [number,number,number], radius: curr.coneRadius as number, height: curr.coneHeight as number }];
    const n = list.length; const bytes = n*16*2; if (coneArrayCapBytes < bytes) {
      coneArrayCapBytes = Math.max(bytes, 16*2);
      coneArrayBuf = device.createBuffer({ size: coneArrayCapBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*8);
    for(let i=0;i<n;i++){
      const co = list[i]; const base=i*8;
      const x = (()=>{ const v=co.xdir; const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; })();
      const y0 = (()=>{ const v=co.ydir; const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; })();
      const dotK = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
      let y:[number,number,number] = [y0[0]-dotK*x[0], y0[1]-dotK*x[1], y0[2]-dotK*x[2]]; { const L=Math.hypot(y[0],y[1],y[2])||1; y=[y[0]/L,y[1]/L,y[2]/L]; }
      const axis:[number,number,number] = [ x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0] ];
      arr[base+0]=co.center[0]; arr[base+1]=co.center[1]; arr[base+2]=co.center[2]; arr[base+3]=co.radius;
      arr[base+4]=axis[0]; arr[base+5]=axis[1]; arr[base+6]=axis[2]; arr[base+7]=co.height;
    }
    device.queue.writeBuffer(coneArrayBuf, 0, arr);
  }
  // Remove sentinel triangle; build BLAS with AABB geometries for analytic primitives
  // Pack all AABBs into a single buffer; each geometry uses an offset and stride=12 bytes (3 floats)
  const packedAabbs: number[] = [];
  function packAabb(min: [number,number,number], max: [number,number,number]): number {
    const offsetBytes = packedAabbs.length * 4; // bytes before pushing
    packedAabbs.push(min[0],min[1],min[2], max[0],max[1],max[2]);
    return offsetBytes;
  }
  // Recompute conservative AABBs with guaranteed min<=max per axis
  function compMin(a:number,b:number){return a<b?a:b;} function compMax(a:number,b:number){return a>b?a:b;}
  const sphereMin: [number,number,number] = [sphereCenter[0]-sphereRadius, sphereCenter[1]-sphereRadius, sphereCenter[2]-sphereRadius];
  const sphereMax: [number,number,number] = [sphereCenter[0]+sphereRadius, sphereCenter[1]+sphereRadius, sphereCenter[2]+sphereRadius];
  // Cylinder extents: compute axis from xdn, ydn, then extents = radius along xdn & ydn, halfH along axis
  const axis = [
    xdn[1]*ydn[2]-xdn[2]*ydn[1],
    xdn[2]*ydn[0]-xdn[0]*ydn[2],
    xdn[0]*ydn[1]-xdn[1]*ydn[0]
  ];
  const cylHalfH = cylinderHeight*0.5;
  const cx = cylinderCenter[0], cy = cylinderCenter[1], cz = cylinderCenter[2];
  function absVecScale(v:number[], s:number){ return [Math.abs(v[0])*s, Math.abs(v[1])*s, Math.abs(v[2])*s]; }
  const exR = absVecScale(xdn as any, cylinderRadius);
  const eyR = absVecScale(ydn as any, cylinderRadius);
  const ezH = absVecScale(axis as any, cylHalfH);
  const cylExtent = [ exR[0]+eyR[0]+ezH[0], exR[1]+eyR[1]+ezH[1], exR[2]+eyR[2]+ezH[2] ];
  const cylMin: [number,number,number] = [cx-cylExtent[0], cy-cylExtent[1], cz-cylExtent[2]];
  const cylMax: [number,number,number] = [cx+cylExtent[0], cy+cylExtent[1], cz+cylExtent[2]];
  // Disk: oriented circle: conservative box center±(radius * (|xdir|+|ydir|))
  const cxc = circleCenter[0], cyc = circleCenter[1], czc = circleCenter[2];
  const cxdAbs = [Math.abs(cxd[0]),Math.abs(cxd[1]),Math.abs(cxd[2])];
  const cydAbs = [Math.abs(cyd[0]),Math.abs(cyd[1]),Math.abs(cyd[2])];
  const diskExt = [ (cxdAbs[0]+cydAbs[0])*circleRadius, (cxdAbs[1]+cydAbs[1])*circleRadius, (cxdAbs[2]+cydAbs[2])*circleRadius ];
  const diskMin: [number,number,number] = [cxc-diskExt[0], cyc-diskExt[1], czc-diskExt[2]];
  const diskMax: [number,number,number] = [cxc+diskExt[0], cyc+diskExt[1], czc+diskExt[2]];
  const [planeXNorm, planeYNorm] = orthonormalizePair(planeXDir as Vec3, planeYDir as Vec3);
  const planeCenterVec: [number,number,number] = [planeCenter[0], planeCenter[1], planeCenter[2]];
  const planeMin: [number,number,number] = [-10, -2.05, -10];
  const planeMax: [number,number,number] = [ 10, -1.95,  10];
  const offSphere = packAabb(sphereMin, sphereMax);
  const offCyl = packAabb(cylMin, cylMax);
  const offDisk = packAabb(diskMin, diskMax);
  const offPlane = packAabb(planeMin, planeMax);
  // Line AABB (compute before BLAS array)
  const r=LINE_RADIUS; const minX=Math.min(lp0[0],lp1[0])-r, minY=Math.min(lp0[1],lp1[1])-r, minZ=Math.min(lp0[2],lp1[2])-r;
  const maxX=Math.max(lp0[0],lp1[0])+r, maxY=Math.max(lp0[1],lp1[1])+r, maxZ=Math.max(lp0[2],lp1[2])+r;
  const pad=Math.max(5e-4, 2e-2*r);
  const offLine = packAabb([minX-pad,minY-pad,minZ-pad],[maxX+pad,maxY+pad,maxZ+pad]);
  // Ellipse AABB: oriented with rx, ry along xdir, ydir
  const exdAbs = [Math.abs(exd[0]),Math.abs(exd[1]),Math.abs(exd[2])];
  const eydAbs = [Math.abs(eyd[0]),Math.abs(eyd[1]),Math.abs(eyd[2])];
  const ellRMax = Math.max(ellipseRadiusX, ellipseRadiusY);
  const nDir = [ exd[1]*eyd[2]-exd[2]*eyd[1], exd[2]*eyd[0]-exd[0]*eyd[2], exd[0]*eyd[1]-exd[1]*eyd[0] ];
  const nLen = Math.hypot(nDir[0],nDir[1],nDir[2])||1; const nUnit = [ nDir[0]/nLen, nDir[1]/nLen, nDir[2]/nLen ];
  const nAbs = [ Math.abs(nUnit[0]), Math.abs(nUnit[1]), Math.abs(nUnit[2]) ];
  const normalThick = Math.max(5e-2, 0.5 * ellRMax);
  const safetyScale = 2.0;
  const ellExt = [ safetyScale*((exdAbs[0]+eydAbs[0])*ellRMax + nAbs[0]*normalThick), safetyScale*((exdAbs[1]+eydAbs[1])*ellRMax + nAbs[1]*normalThick), safetyScale*((exdAbs[2]+eydAbs[2])*ellRMax + nAbs[2]*normalThick) ];
  const ellPad = Math.max(5e-3, 5e-2 * Math.max(ellipseRadiusX, ellipseRadiusY));
  const ellMin: [number,number,number] = [ ellipseCenter[0]-ellExt[0]-ellPad, ellipseCenter[1]-ellExt[1]-ellPad, ellipseCenter[2]-ellExt[2]-ellPad ];
  const ellMax: [number,number,number] = [ ellipseCenter[0]+ellExt[0]+ellPad, ellipseCenter[1]+ellExt[1]+ellPad, ellipseCenter[2]+ellExt[2]+ellPad ];
  const offEllipse = packAabb(ellMin, ellMax);
  // Torus AABB: conservative using bounding sphere or oriented extents.
  // A safe bound is a sphere of radius (R + r), but we tighten using basis.
  // Build orthonormal frame from initial function params
  const _txd0 = (()=>{ const v=new Float32Array(torusXDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const _tyd00 = (()=>{ const v=new Float32Array(torusYDir); const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; })();
  const _dotT = _txd0[0]*_tyd00[0]+_txd0[1]*_tyd00[1]+_txd0[2]*_tyd00[2];
  let _tyd = [ _tyd00[0]-_dotT*_txd0[0], _tyd00[1]-_dotT*_txd0[1], _tyd00[2]-_dotT*_txd0[2] ]; { const L=Math.hypot(_tyd[0],_tyd[1],_tyd[2])||1; _tyd=[_tyd[0]/L,_tyd[1]/L,_tyd[2]/L]; }
  const tz = [ _txd0[1]*_tyd[2]-_txd0[2]*_tyd[1], _txd0[2]*_tyd[0]-_txd0[0]*_tyd[2], _txd0[0]*_tyd[1]-_txd0[1]*_tyd[0] ];
  const txAbs = [Math.abs(_txd0[0]),Math.abs(_txd0[1]),Math.abs(_txd0[2])];
  const tzAbs = [Math.abs(tz[0]),Math.abs(tz[1]),Math.abs(tz[2])];
  const R = torusMajorRadius, rMinor = torusMinorRadius;
  // Use major-ring plane (xdir–zdir); tube adds r in any axis
  const sRing = (R + rMinor);
  const safety = 1.3;
  let torExt = [ (txAbs[0]+tzAbs[0])*sRing + rMinor,
                 (txAbs[1]+tzAbs[1])*sRing + rMinor,
                 (txAbs[2]+tzAbs[2])*sRing + rMinor ];
  torExt = [ torExt[0]*safety, torExt[1]*safety, torExt[2]*safety ];
  const tPad = Math.max(1e-2*sRing, 2e-2*rMinor, 5e-3);
  const torMin: [number,number,number] = [ torusCenter[0]-torExt[0]-tPad, torusCenter[1]-torExt[1]-tPad, torusCenter[2]-torExt[2]-tPad ];
  const torMax: [number,number,number] = [ torusCenter[0]+torExt[0]+tPad, torusCenter[1]+torExt[1]+tPad, torusCenter[2]+torExt[2]+tPad ];
  const offTorus = packAabb(torMin, torMax);
  // Create the single packed AABB buffer
  const aabbPackData = new Float32Array(packedAabbs);
  const aabbPackBuf = device.createBuffer({ size: aabbPackData.byteLength, usage: (globalThis as any).GPUBufferUsageRTX?.ACCELERATION_STRUCTURE_BUILD_INPUT_READONLY ?? GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Float32Array(aabbPackBuf.getMappedRange()).set(aabbPackData); aabbPackBuf.unmap();
  const bottom: GPURayTracingAccelerationContainerDescriptor_bottom = {
    usage: (globalThis as any).GPURayTracingAccelerationContainerUsage?.NONE ?? (0 as any),
    level:'bottom',
    geometries:[
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offSphere, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offCyl, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offDisk, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offPlane, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offLine, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offEllipse, format:'float32x2', stride: 12, size: 24 } },
  { usage:(GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs', aabb:{ buffer: aabbPackBuf, offset: offTorus,   format:'float32x2', stride: 12, size: 24 } },
    ]
  };
  const topDesc: GPURayTracingAccelerationContainerDescriptor_top = {
    usage: (globalThis as any).GPURayTracingAccelerationContainerUsage?.NONE ?? (0 as any),
    level:'top',
    instances:[{ usage:(GPURayTracingAccelerationInstanceUsage as any).NONE, mask:0xff, instanceSBTRecordOffset:0, blas: bottom }]
  };
  // Keep current primitive params for TLAS rebuilds on-the-fly
  const currPlane = {
    center: planeCenterVec as [number,number,number],
    xdir: planeXNorm as [number,number,number],
    ydir: planeYNorm as [number,number,number],
    halfWidth: planeHalfWidth,
    halfHeight: planeHalfHeight,
  };
  let curr: any = {
    sphereCenter, sphereRadius,
    cylinderCenter, cylinderXDir, cylinderYDir, cylinderRadius, cylinderHeight, cylinderAngleDeg,
    circleCenter, circleXDir, circleYDir, circleRadius,
    ellipseCenter, ellipseXDir, ellipseYDir, ellipseRadiusX, ellipseRadiusY,
    coneCenter, coneXDir, coneYDir, coneRadius, coneHeight,
    torusCenter, torusXDir, torusYDir, torusMajorRadius, torusMinorRadius, torusAngleDeg,
    planeCenter: planeCenterVec,
    planeXDir: planeXNorm as [number,number,number],
    planeYDir: planeYNorm as [number,number,number],
    planeHalfWidth,
    planeHalfHeight,
    planes: [currPlane],
    lineP0: lp0, lineP1: lp1, lineRadius: LINE_RADIUS,
    bezierPatches: fallbackSceneState.bezierPatches && fallbackSceneState.bezierPatches.length > 0
      ? fallbackSceneState.bezierPatches.map(cloneBezierPatch)
      : [],
  };
  // No default mass torus instances; user can call setTori([...]) to add many.
  const { tlas: tlas0, meta: meta0 } = await buildAnalyticPrimitivesTLAS(device, curr);
  let tlas = tlas0;
  // New: scene meta (gid bases/counts) UBO
  const sceneMetaBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  function writeMeta(m:any){
    // Pack ranges for all known primitives into SceneMeta uvec4s
    // m0: [torusBase, torusCount, ellipseBase, ellipseCount]
    // m1: [sphereBase, sphereCount, cylinderBase, cylinderCount]
    // m2: [circleBase, circleCount, planeBase, planeCount]
    // m3: [coneBase, coneCount, lineBase, lineCount]
    // m4: [bezierBase, bezierCount, 0, 0]
    const d = new Uint32Array([
      m.torusBase ?? 0, m.torusCount ?? 0, m.ellipseBase ?? 0, m.ellipseCount ?? 0,
      m.sphereBase ?? 0, m.sphereCount ?? 1, m.cylinderBase ?? 1, m.cylinderCount ?? 1,
      m.circleBase ?? 0, m.circleCount ?? 0, m.planeBase ?? 0, m.planeCount ?? 0,
      m.coneBase ?? 0, m.coneCount ?? 0, m.lineBase ?? 0, m.lineCount ?? 0,
      m.bezierBase ?? 0, m.bezierCount ?? 0, 0, 0,
    ]); // total 20 u32 = 80 bytes
    device.queue.writeBuffer(sceneMetaBuf, 0, d);
  }
  writeMeta(meta0 as any);
  function writeTorusArrayFromCurr(){
    const list = ((curr as any).tori && (curr as any).tori.length>0)
      ? (curr as any).tori
      : [{ center: curr.torusCenter, xdir: curr.torusXDir, ydir: curr.torusYDir, majorR: curr.torusMajorRadius, minorR: curr.torusMinorRadius, angleDeg: curr.torusAngleDeg }];
    const n = list.length; const bytes = n * 16 * 3; if (torusArrayCapacityBytes < bytes) {
      torusArrayCapacityBytes = Math.max(bytes, 16*3);
      torusArrayBuf = device.createBuffer({ size: torusArrayCapacityBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*12);
    for (let i=0;i<n;i++){
      const t = list[i];
      const base = i*12;
      arr[base+0]=t.center[0]; arr[base+1]=t.center[1]; arr[base+2]=t.center[2]; arr[base+3]=t.majorR;
      arr[base+4]=t.xdir[0]; arr[base+5]=t.xdir[1]; arr[base+6]=t.xdir[2]; arr[base+7]=t.minorR;
      arr[base+8]=t.ydir[0]; arr[base+9]=t.ydir[1]; arr[base+10]=t.ydir[2]; arr[base+11]=(t.angleDeg ?? 360.0);
    }
    device.queue.writeBuffer(torusArrayBuf, 0, arr);
  }
  writeTorusArrayFromCurr();
  // Sphere SSBO (binding 11): vec4(center.xyz, radius)
  let sphereArrayBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let sphereArrayCapBytes = 16;
  function writeSphereArrayFromCurr(){
    const list = ((curr as any).spheres && (curr as any).spheres.length>0)
      ? (curr as any).spheres
      : [{ center: curr.sphereCenter, radius: curr.sphereRadius }];
    const n = list.length; const bytes = n*16; if (sphereArrayCapBytes < bytes) {
      sphereArrayCapBytes = Math.max(bytes, 16);
      sphereArrayBuf = device.createBuffer({ size: sphereArrayCapBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    const arr = new Float32Array(n*4);
    for(let i=0;i<n;i++){
      const s = list[i]; const base = i*4;
      arr[base+0]=s.center[0]; arr[base+1]=s.center[1]; arr[base+2]=s.center[2]; arr[base+3]=s.radius;
    }
    device.queue.writeBuffer(sphereArrayBuf, 0, arr);
  }
  writeSphereArrayFromCurr();
  // Now safe to initialize cylinder array from 'curr'
  writeCylinderArrayFromCurr();
  writeCircleArrayFromCurr();
  writeEllipseArrayFromCurr();
  writeLineArrayFromCurr();
  writeConeArrayFromCurr();
  // Packed buffer for Circle/Ellipse/Line/Cone: binding(13) and offsets uniform binding(9)
  const packedOffsetsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let packedBuf = device.createBuffer({ size: 16*8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let packedCap = 16*8;
  function writePackedFromCurr(){
    const circles = ((curr as any).circles && (curr as any).circles.length>0) ? (curr as any).circles : [{ center: curr.circleCenter, xdir: cxd as [number,number,number], ydir: cyd as [number,number,number], radius: curr.circleRadius }];
    const ellipses = ((curr as any).ellipses && (curr as any).ellipses.length>0) ? (curr as any).ellipses : (curr.ellipseCenter ? [{ center: curr.ellipseCenter, xdir: exd as [number,number,number], ydir: eyd as [number,number,number], rx: curr.ellipseRadiusX, ry: curr.ellipseRadiusY }] : []);
    const lines = ((curr as any).lines && (curr as any).lines.length>0) ? (curr as any).lines : [{ p0: curr.lineP0 as [number,number,number], p1: curr.lineP1 as [number,number,number], radius: curr.lineRadius as number }];
    const cones = ((curr as any).cones && (curr as any).cones.length>0) ? (curr as any).cones : [{ center: curr.coneCenter as [number,number,number], xdir: kx as [number,number,number], ydir: ky as [number,number,number], radius: curr.coneRadius as number, height: curr.coneHeight as number }];
    const strideCircle = 3; const strideEllipse = 3; const strideLine = 2; const strideCone = 2; // in vec4 units
    const baseCircle = 0;
    const baseEllipse = baseCircle + circles.length * strideCircle;
    const baseLine = baseEllipse + ellipses.length * strideEllipse;
    const baseCone = baseLine + lines.length * strideLine;
    const totalVec4 = baseCone + cones.length * strideCone;
    const totalBytes = totalVec4 * 16;
    if(packedCap < totalBytes){ packedCap = Math.max(totalBytes, 16*8); packedBuf = device.createBuffer({ size: packedCap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); }
    const arr = new Float32Array(totalVec4*4);
    let w = 0;
    // Circles: d0(center, radius), d1(xdir,0), d2(ydir,0)
    for(const d of circles){ arr[w+0]=d.center[0]; arr[w+1]=d.center[1]; arr[w+2]=d.center[2]; arr[w+3]=d.radius; w+=4; arr[w+0]=d.xdir[0]; arr[w+1]=d.xdir[1]; arr[w+2]=d.xdir[2]; arr[w+3]=0; w+=4; arr[w+0]=d.ydir[0]; arr[w+1]=d.ydir[1]; arr[w+2]=d.ydir[2]; arr[w+3]=0; w+=4; }
    // Ellipses: e0(center, rx), e1(xdir, ry), e2(ydir,0)
    for(const e of ellipses){ arr[w+0]=e.center[0]; arr[w+1]=e.center[1]; arr[w+2]=e.center[2]; arr[w+3]=e.rx; w+=4; arr[w+0]=e.xdir[0]; arr[w+1]=e.xdir[1]; arr[w+2]=e.xdir[2]; arr[w+3]=e.ry; w+=4; arr[w+0]=e.ydir[0]; arr[w+1]=e.ydir[1]; arr[w+2]=e.ydir[2]; arr[w+3]=0; w+=4; }
    // Lines: l0(p0, r), l1(p1,0)
    for(const L of lines){ arr[w+0]=L.p0[0]; arr[w+1]=L.p0[1]; arr[w+2]=L.p0[2]; arr[w+3]=L.radius; w+=4; arr[w+0]=L.p1[0]; arr[w+1]=L.p1[1]; arr[w+2]=L.p1[2]; arr[w+3]=0; w+=4; }
    // Cones: e0(center, radius), e1(axis, height) where axis = normalize(x cross y)
    for(const co of cones){
      const x = (()=>{ const v=co.xdir; const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; })();
      const y0 = (()=>{ const v=co.ydir; const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; })();
      const d = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
      let y:[number,number,number] = [y0[0]-d*x[0], y0[1]-d*x[1], y0[2]-d*x[2]]; { const L=Math.hypot(y[0],y[1],y[2])||1; y=[y[0]/L,y[1]/L,y[2]/L]; }
      const axis:[number,number,number] = [ x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0] ];
      arr[w+0]=co.center[0]; arr[w+1]=co.center[1]; arr[w+2]=co.center[2]; arr[w+3]=co.radius; w+=4;
      arr[w+0]=axis[0]; arr[w+1]=axis[1]; arr[w+2]=axis[2]; arr[w+3]=co.height; w+=4;
    }
    device.queue.writeBuffer(packedBuf, 0, arr);
    const baseU32 = new Uint32Array([baseCircle, baseEllipse, baseLine, baseCone]);
    const strideU32 = new Uint32Array([strideCircle, strideEllipse, strideLine, strideCone]);
    device.queue.writeBuffer(packedOffsetsBuf, 0, baseU32);
    device.queue.writeBuffer(packedOffsetsBuf, 16, strideU32);
  }
  writePackedFromCurr();
  let bezierArrayBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let bezierArrayCapFloats = 4;
  function writeBezierArrayFromCurr(){
    const patches: BezierPatchInstance[] = ((curr as any).bezierPatches && (curr as any).bezierPatches.length>0)
      ? (curr as any).bezierPatches
      : [];
    const count = patches.length;
    const requiredFloats = count * BEZIER_FLOATS;
    if (requiredFloats > bezierArrayCapFloats) {
      bezierArrayCapFloats = Math.max(requiredFloats, BEZIER_FLOATS);
      bezierArrayBuf = device.createBuffer({ size: bezierArrayCapFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    if (count === 0) {
      return;
    }
    const arr = new Float32Array(requiredFloats);
    let cursor = 0;
    for (let i = 0; i < count; i++) {
      const converted = convertHermitePatchToBezier(patches[i]);
      packBezierPatchFloats(converted, arr, cursor);
      cursor += BEZIER_FLOATS;
    }
    device.queue.writeBuffer(bezierArrayBuf, 0, arr);
  }
  writeBezierArrayFromCurr();
  // Shader sources (moved to dedicated module)
  function hasBezierPatches(): boolean {
    return (curr.bezierPatches?.length ?? 0) > 0;
  }
  const groups: GPURayTracingShaderGroupDescriptor[] = [
    { type: 'general', generalIndex: 0 },
    { type: 'general', generalIndex: 1 },
    { type: 'procedural-hit-group', intersectionIndex: 2, closestHitIndex: 3 },
  ];
  let pipeline: any;
  const handleSize = (device as any).ShaderGroupHandleSize as number;
  const baseAlign = (device as any).ShaderGroupBaseAlignment as number;
  const handleAlign = (device as any).ShaderGroupHandleAlignment as number;
  function align(v:number,a:number){ return Math.ceil(v/a)*a; }
  let sbtBuffer: GPUBuffer;
  let sbt: GPUShaderBindingTable;
  async function recreatePipeline(){
    // Recreate pipeline with shader set chosen by current Bezier usage
    const useBezier = hasBezierPatches();
    const stageSources = useBezier
      ? {
        rgen: getRayGenShader(),
        rmiss: getMissShader(),
        rint: getIntersectionShader(),
        rchit: getClosestHitShader(),
      }
      : {
        rgen: getRayGenShaderNoBezier(),
        rmiss: getMissShaderNoBezier(),
        rint: getIntersectionShaderNoBezier(),
        rchit: getClosestHitShaderNoBezier(),
      };
  console.info('[WebRTX] Ray tracing pipeline rebuild – active shader set:', useBezier ? 'hermite' : 'no-bezier');
  const stages: GPURayTracingShaderStageDescriptor[] = [
      { stage: (globalThis as any).GPUShaderStageRTX?.RAY_GENERATION, glslCode: stageSources.rgen, entryPoint: 'main' },
      { stage: (globalThis as any).GPUShaderStageRTX?.RAY_MISS, glslCode: stageSources.rmiss, entryPoint: 'main' },
      { stage: (globalThis as any).GPUShaderStageRTX?.RAY_INTERSECTION, glslCode: stageSources.rint, entryPoint: 'main' },
      { stage: (globalThis as any).GPUShaderStageRTX?.RAY_CLOSEST_HIT, glslCode: stageSources.rchit, entryPoint: 'main' },
    ];
    pipeline = await (device as any).createRayTracingPipeline({ stages, groups }, tlas);
    const handles = pipeline.getShaderGroupHandles(0, groups.length);
    const sbtRayGenStart = 0; const sbtRayMissStart = align(sbtRayGenStart + baseAlign, baseAlign); const sbtHitStart = align(sbtRayMissStart + handleAlign, baseAlign); const sbtTotal = sbtHitStart + handleAlign;
    sbtBuffer = device.createBuffer({ size: sbtTotal, usage: ((globalThis as any).GPUBufferUsageRTX?.SHADER_BINDING_TABLE ?? GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE, mappedAtCreation: true });
    const sbtU32 = new Uint32Array(sbtBuffer.getMappedRange()); sbtU32[sbtRayGenStart>>2]=handles[0]; sbtU32[sbtRayMissStart>>2]=handles[1]; sbtU32[sbtHitStart>>2]=handles[2]; sbtBuffer.unmap();
    sbt = { buffer: sbtBuffer, rayGen:{ start:sbtRayGenStart, stride: handleAlign, size: baseAlign }, rayMiss:{ start:sbtRayMissStart, stride: handleAlign, size: handleAlign }, rayHit:{ start:sbtHitStart, stride: handleAlign, size: handleAlign }, callable:{ start:0, stride: handleAlign, size:0 } } as any;
  }
  await recreatePipeline();
  // Cached blit resources (reuse across frames; recreate on resize)
  const quadVS = `@vertex fn vs_main(@builtin(vertex_index) vi:u32)->@builtin(position) vec4<f32>{ var p=array<vec2<f32>,3>(vec2<f32>(-1.,-3.),vec2<f32>(3.,1.),vec2<f32>(-1.,1.)); return vec4<f32>(p[vi],0.,1.); }`;
  const quadFS = `struct P{w:u32,h:u32}; @group(0)@binding(0) var<uniform> p:P; struct C{data:array<vec4<f32>>}; @group(0)@binding(1) var<storage,read> c:C; @fragment fn fs_main(@builtin(position) pos:vec4<f32>)->@location(0) vec4<f32>{ let x=i32(pos.x); let y=i32(pos.y); if(x<0||y<0||x>=i32(p.w)||y>=i32(p.h)){ discard;} return c.data[u32(y)*p.w+u32(x)]; }`;
  let blitPipeline: GPURenderPipeline | null = null;
  let blitParams: GPUBuffer | null = null;
  let blitBG: GPUBindGroup | null = null;
  function ensureBlitResources(){
    if(!blitPipeline){
      blitPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: device.createShaderModule({ code: quadVS }), entryPoint: 'vs_main' },
        fragment: { module: device.createShaderModule({ code: quadFS }), entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
        primitive: { topology: 'triangle-list' }
      });
    }
    if(!blitParams){ blitParams = device.createBuffer({ size:8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); }
    // Always update dimensions
    device.queue.writeBuffer(blitParams, 0, new Uint32Array([internalWidth, internalHeight]));
    // Recreate bind group whenever colorBuffer changes or not yet created
    if(!blitBG){
      blitBG = device.createBindGroup({ layout: blitPipeline.getBindGroupLayout(0), entries:[ {binding:0, resource:{ buffer: blitParams }}, {binding:1, resource:{ buffer: colorBuffer }} ] });
    }
  }
  function createUserBG(){
  const list = [
      { binding: 0, resource: { buffer: colorBuffer } },
      { binding: 1, resource: { buffer: cameraBuf } },
      { binding: 2, resource: { buffer: sceneMetaBuf } },
      { binding: 9, resource: { buffer: packedOffsetsBuf } },
      { binding: 4, resource: { buffer: cylBuf } },
      { binding: 13, resource: { buffer: packedBuf } },
      { binding: 10, resource: { buffer: torusArrayBuf } },
      { binding: 11, resource: { buffer: sphereArrayBuf } },
      { binding: 12, resource: { buffer: cylArrayBuf } },
    ];
    if (hasBezierPatches()) {
      list.push({ binding: 14, resource: { buffer: bezierArrayBuf } });
    }
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: list }) as any;
    // Attach acceleration structure metadata for the polyfill pass encoder
    bg.__accel_container = tlas;
    return bg;
  }
  let userBG = createUserBG();
  // Optional: seed instances via overrides for quick demos (e.g., window.__webrtxOverrides={spheresCount:1000})
  async function seedFromOverrides(){
    const sphereOverrides = computeOverrideSphereInstances();
    if (sphereOverrides && sphereOverrides.length > 0) {
      await setSpheres(sphereOverrides);
      if (disableBezier || (__ovr.keepBezier !== true)) {
        try {
          await setBezierPatches([]);
        } catch (err) {
          console.warn('[WebRTX] Failed to clear default Bezier patches for overrides.', err);
        }
      }
    }
  }
  // Rebuild TLAS when primitives move/resize so BVH AABBs match uniforms
  async function rebuildTLAS(){
    const res = await buildAnalyticPrimitivesTLAS(device, curr);
    tlas = res.tlas;
    if ((res as any).meta) writeMeta((res as any).meta);
    writeTorusArrayFromCurr();
    writeSphereArrayFromCurr();
    writeCylinderArrayFromCurr();
    writeCircleArrayFromCurr(); // still used for AABB packing helpers
    writeEllipseArrayFromCurr();
    writeLineArrayFromCurr();
    writeConeArrayFromCurr();
    writePackedFromCurr();
    writeBezierArrayFromCurr();
    // Recreate pipeline + SBT bound to the new TLAS and then rebind
    await recreatePipeline();
    userBG = createUserBG();
    // Optional: debug
    console.log('[WebRTX] TLAS rebuilt with params', curr);
  }
  function render(){
    ensureBlitResources();
    const encoder = device.createCommandEncoder();
    const pass = (encoder as any).beginRayTracingPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0,userBG);
    pass.traceRays(device, sbt, internalWidth, internalHeight, 1);
    pass.end();
    // blit
    const rpass = encoder.beginRenderPass({ colorAttachments:[{ view: context.getCurrentTexture().createView(), loadOp:'clear', clearValue:{r:0,g:0,b:0,a:1}, storeOp:'store' }] });
    rpass.setPipeline(blitPipeline!);
    rpass.setBindGroup(0, blitBG!);
    rpass.draw(3);
    rpass.end();
    device.queue.submit([encoder.finish()]);
  }
  // FPS overlay and frame loop (glass card style)
  // Remove old overlay if present to avoid duplicates
  const oldHud = document.getElementById('webrtx-fps') as HTMLDivElement | null; if (oldHud) oldHud.remove();
  const fpsEl = document.createElement('div');
  fpsEl.id = 'webrtx-fps';
  fpsEl.style.position = 'absolute';
  fpsEl.style.top = '10px';
  fpsEl.style.left = '10px';
  fpsEl.style.padding = '8px 10px';
  fpsEl.style.borderRadius = '10px';
  fpsEl.style.background = 'linear-gradient(135deg, rgba(18,18,20,0.65), rgba(38,38,44,0.55))';
  fpsEl.style.border = '1px solid rgba(255,255,255,0.08)';
  fpsEl.style.boxShadow = '0 6px 18px rgba(0,0,0,0.25)';
  fpsEl.style.color = 'rgba(240,240,244,0.95)';
  fpsEl.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  fpsEl.style.fontSize = '12px';
  fpsEl.style.lineHeight = '1.2';
  fpsEl.style.zIndex = '1000';
  fpsEl.style.pointerEvents = 'none';
  fpsEl.style.userSelect = 'none';
  fpsEl.style.transform = 'translateZ(0)';
  fpsEl.style.willChange = 'transform, opacity';
  fpsEl.style.letterSpacing = '0.2px';
  // Backdrop blur for supported browsers
  fpsEl.style.setProperty('-webkit-backdrop-filter', 'blur(8px)');
  fpsEl.style.setProperty('backdrop-filter', 'blur(8px)');
  // Build inner layout: [dot] FPS [value] | [ms]
  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.alignItems = 'baseline';
  row.style.gap = '8px';
  const dot = document.createElement('span');
  dot.style.display = 'inline-block';
  dot.style.width = '8px';
  dot.style.height = '8px';
  dot.style.borderRadius = '50%';
  dot.style.boxShadow = '0 0 0 2px rgba(255,255,255,0.06) inset, 0 0 10px currentColor';
  dot.style.marginTop = '2px';
  const label = document.createElement('span');
  label.textContent = 'FPS';
  label.style.opacity = '0.9';
  label.style.fontWeight = '600';
  const value = document.createElement('span');
  value.textContent = '--';
  value.style.fontSize = '16px';
  value.style.fontWeight = '700';
  value.style.minWidth = '28px';
  value.style.textAlign = 'right';
  const sep = document.createElement('span');
  sep.textContent = '·';
  sep.style.opacity = '0.5';
  const ms = document.createElement('span');
  ms.textContent = '-- ms';
  ms.style.opacity = '0.9';
  row.appendChild(dot); row.appendChild(label); row.appendChild(value); row.appendChild(sep); row.appendChild(ms);
  fpsEl.appendChild(row);
  // Place overlay relative to the canvas parent
  const parent = canvas.parentElement || document.body;
  parent.style.position = parent === document.body ? 'relative' : (getComputedStyle(parent).position || 'relative');
  parent.appendChild(fpsEl);
  let rafId: number | null = null;
  let lastFpsTime = performance.now();
  let fpsCount = 0;
  function frame(ts: number){
    // FPS update every ~250ms to avoid DOM spam
    fpsCount++;
    const dt = ts - lastFpsTime;
    if(dt >= 250){
      const fps = Math.round((fpsCount * 1000) / dt);
      const avgMs = dt / Math.max(1, fpsCount);
      value.textContent = String(fps);
      ms.textContent = `${avgMs.toFixed(1)} ms`;
      // Accent color based on tier
      let accent = '#00e676'; // high
      if (fps < 30) accent = '#ff5252'; else if (fps < 55) accent = '#ffca28';
      dot.style.color = accent;
      value.style.color = accent;
      // subtle border color hint
      fpsEl.style.border = `1px solid ${accent}20`;
      fpsCount = 0;
      lastFpsTime = ts;
    }
    render();
    rafId = requestAnimationFrame(frame);
  }
  function startLoop(){ if(rafId === null){ rafId = requestAnimationFrame(frame); } }
  function stop(){ if(rafId !== null){ cancelAnimationFrame(rafId); rafId = null; } }
  // Kick off loop
  startLoop();
  // Fire-and-forget seeding
  seedFromOverrides();
  function resize(newLogicalW:number, newLogicalH:number, newDpr?:number){
    logicalWidth=Math.max(1,Math.floor(newLogicalW)); logicalHeight=Math.max(1,Math.floor(newLogicalH));
    dpr=newDpr?newDpr:((globalThis as any).devicePixelRatio||1);
    // Re-read overrides at resize time to allow dynamic tuning from console
    const ovr = (globalThis as any).__webrtxOverrides || {};
    const lock = ovr.lockDpr === true; const scale = (typeof ovr.renderScale === 'number' && ovr.renderScale>0)? ovr.renderScale : 1.0;
    if (lock) dpr = 1;
    const newIntW=align8(Math.floor(logicalWidth*dpr*scale));
    const newIntH=align8(Math.floor(logicalHeight*dpr*scale));
    if(newIntW===internalWidth && newIntH===internalHeight){ configureCanvas(); return; }
    internalWidth=newIntW; internalHeight=newIntH;
    configureCanvas();
    colorBuffer = device.createBuffer({ size: internalWidth*internalHeight*4*Float32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
    userBG = createUserBG(); blitBG = null;
  }
  window.addEventListener('resize',()=>resize(logicalWidth,logicalHeight));
  async function updateCamera(pos:[number,number,number], look:[number,number,number], up:[number,number,number]){ writeCamera(pos,look,up); }
  function setCameraFovY(deg:number){ currentFov = deg*Math.PI/180; writeCamera([0,0,3],[0,0,0],[0,1,0]); }
  function setCameraNearFar(n:number,f:number){ currentNear=n; currentFar=f; writeCamera([0,0,3],[0,0,0],[0,1,0]); }
  // Runtime setters for cylinder and torus; they rewrite uniforms and rebuild TLAS
  function _norm3(v:[number,number,number]): [number,number,number]{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }
  async function setLine(params: { startPoint: [number,number,number]; endPoint: [number,number,number]; }){
    const p0 = params.startPoint; const p1 = params.endPoint;
    // update uniform
    const data = new Float32Array([
      p0[0], p0[1], p0[2], LINE_RADIUS,
      p1[0], p1[1], p1[2], 0,
    ]);
    device.queue.writeBuffer(lineBuf, 0, data);
    // update curr and rebuild TLAS (AABB depends on endpoints)
    curr.lineP0 = p0; curr.lineP1 = p1; curr.lineRadius = LINE_RADIUS;
    await rebuildTLAS();
  }
  async function setCylinder(params: { center: [number,number,number]; xdir: [number,number,number]; ydir: [number,number,number]; radius: number; height: number; angleDeg?: number; }){
    const x = _norm3(params.xdir);
    const y0 = _norm3(params.ydir);
    const dotXY = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
    let y: [number,number,number] = [y0[0]-dotXY*x[0], y0[1]-dotXY*x[1], y0[2]-dotXY*x[2]]; y = _norm3(y);
    const data = new Float32Array([
      params.center[0], params.center[1], params.center[2], params.radius,
      x[0], x[1], x[2], params.height,
      y[0], y[1], y[2], (params.angleDeg ?? 360.0),
    ]);
    device.queue.writeBuffer(cylBuf, 0, data);
    // update curr and rebuild TLAS
    curr.cylinderCenter = params.center; curr.cylinderXDir = x as any; curr.cylinderYDir = y as any; curr.cylinderRadius = params.radius; curr.cylinderHeight = params.height; curr.cylinderAngleDeg = params.angleDeg ?? 360.0;
    await rebuildTLAS();
  }
  async function setTorus(params: { center: [number,number,number]; xdir: [number,number,number]; ydir: [number,number,number]; majorR: number; minorR: number; angleDeg?: number; }){
    const x = _norm3(params.xdir);
    const y0 = _norm3(params.ydir);
    const dotXY = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
    let y: [number,number,number] = [y0[0]-dotXY*x[0], y0[1]-dotXY*x[1], y0[2]-dotXY*x[2]]; y = _norm3(y);
    // update curr and rebuild TLAS
    curr.torusCenter = params.center; curr.torusXDir = x as any; curr.torusYDir = y as any; curr.torusMajorRadius = params.majorR; curr.torusMinorRadius = params.minorR; curr.torusAngleDeg = (params.angleDeg ?? curr.torusAngleDeg ?? 360.0);
    // refresh SSBO contents and TLAS
    writeTorusArrayFromCurr();
    await rebuildTLAS();
  }
  // New: multi-instance spheres
  async function setSpheres(instances: Array<{ center:[number,number,number]; radius:number; }>){
    (curr as any).spheres = instances;
    await rebuildTLAS();
  }
  // Multiple cylinders API for TLAS + SSBO
  async function setCylinders(instances: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; height:number; angleDeg?:number; }>){
    const norm = (v:[number,number,number])=>{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; };
    const safe = instances.map(c=>{
      const x = norm(c.xdir);
      const y0 = norm(c.ydir);
      const d = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
      let y:[number,number,number] = [y0[0]-d*x[0], y0[1]-d*x[1], y0[2]-d*x[2]]; y = norm(y);
      return { center: c.center, xdir: x, ydir: y, radius: c.radius, height: c.height, angleDeg: c.angleDeg ?? 360.0 };
    });
    (curr as any).cylinders = safe;
    await rebuildTLAS();
  }
  // Multiple torus instances for TLAS (shaders still read single torus uniform)
  async function setTori(instances: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; majorR:number; minorR:number; angleDeg?:number; }>){
    const norm = (v:[number,number,number])=>{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; };
    const safe = instances.map(t=>{
      const x = norm(t.xdir);
      const y0 = norm(t.ydir);
      const d = x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2];
      let y:[number,number,number] = [y0[0]-d*x[0], y0[1]-d*x[1], y0[2]-d*x[2]]; y = norm(y);
      return { center: t.center, xdir: x, ydir: y, majorR: t.majorR, minorR: t.minorR, angleDeg: t.angleDeg ?? (curr.torusAngleDeg ?? 360.0) };
    });
    (curr as any).tori = safe;
    await rebuildTLAS();
  }
  // New: multi-instance setters for all remaining primitives (packed buffer)
  async function setCircles(instances: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; }>){
    const norm = (v:[number,number,number])=>{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; };
    const ortho = (x:[number,number,number], y0:[number,number,number])=>{ const d=x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2]; let y:[number,number,number]=[y0[0]-d*x[0],y0[1]-d*x[1],y0[2]-d*x[2]]; return norm(y); };
    const safe = instances.map(d=>{ const x=norm(d.xdir); const y=ortho(x, norm(d.ydir)); return { center:d.center, xdir:x, ydir:y, radius:d.radius }; });
    (curr as any).circles = safe; await rebuildTLAS();
  }
  async function setEllipses(instances: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; rx:number; ry:number; }>){
    const norm = (v:[number,number,number])=>{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; };
    const ortho = (x:[number,number,number], y0:[number,number,number])=>{ const d=x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2]; let y:[number,number,number]=[y0[0]-d*x[0],y0[1]-d*x[1],y0[2]-d*x[2]]; return norm(y); };
    const safe = instances.map(e=>{ const x=norm(e.xdir); const y=ortho(x, norm(e.ydir)); return { center:e.center, xdir:x, ydir:y, rx:e.rx, ry:e.ry }; });
    (curr as any).ellipses = safe; await rebuildTLAS();
  }
  async function setLines(instances: Array<{ p0:[number,number,number]; p1:[number,number,number]; radius:number; }>){
    const safe = instances.map(L=>({ p0:L.p0, p1:L.p1, radius: Math.max(1e-4, L.radius) }));
    (curr as any).lines = safe; await rebuildTLAS();
  }
  async function setCones(instances: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; height:number; }>){
    const norm = (v:[number,number,number])=>{ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l] as [number,number,number]; };
    const ortho = (x:[number,number,number], y0:[number,number,number])=>{ const d=x[0]*y0[0]+x[1]*y0[1]+x[2]*y0[2]; let y:[number,number,number]=[y0[0]-d*x[0],y0[1]-d*x[1],y0[2]-d*x[2]]; return norm(y); };
    const safe = instances.map(co=>{ const x=norm(co.xdir); const y=ortho(x, norm(co.ydir)); return { center:co.center, xdir:x, ydir:y, radius:co.radius, height:co.height }; });
    (curr as any).cones = safe; await rebuildTLAS();
  }
  async function setBezierPatches(instances: BezierPatchInstance[]){
    (curr as any).bezierPatches = instances.map(cloneBezierPatch);
    await rebuildTLAS();
  }
  async function setBezierPatch(instance: BezierPatchInstance){
    await setBezierPatches([instance]);
  }
  return {
    renderer: 'raytracing' as const,
    updateCamera,
    render,
    device,
    setCameraFovY,
    setCameraNearFar,
    resize,
    setCylinder,
    setTorus,
    setTori,
    setCylinders,
    setSpheres,
    setCircles,
    setEllipses,
    setLines,
    setCones,
    setLine,
    setBezierPatch,
    setBezierPatches,
    setScene: async (scene: FallbackSceneDescriptor) => {
      curr = {
        sphereCenter: scene.spheres[0]?.center ?? curr.sphereCenter,
        sphereRadius: scene.spheres[0]?.radius ?? curr.sphereRadius,
        cylinderCenter: scene.cylinders[0]?.center ?? curr.cylinderCenter,
        cylinderXDir: scene.cylinders[0]?.xdir ?? curr.cylinderXDir,
        cylinderYDir: scene.cylinders[0]?.ydir ?? curr.cylinderYDir,
        cylinderRadius: scene.cylinders[0]?.radius ?? curr.cylinderRadius,
        cylinderHeight: scene.cylinders[0]?.height ?? curr.cylinderHeight,
        cylinderAngleDeg: scene.cylinders[0]?.angleDeg ?? curr.cylinderAngleDeg,
        circleCenter: scene.circles[0]?.center ?? curr.circleCenter,
        circleXDir: scene.circles[0]?.xdir ?? curr.circleXDir,
        circleYDir: scene.circles[0]?.ydir ?? curr.circleYDir,
        circleRadius: scene.circles[0]?.radius ?? curr.circleRadius,
        ellipseCenter: scene.ellipses[0]?.center ?? curr.ellipseCenter,
        ellipseXDir: scene.ellipses[0]?.xdir ?? curr.ellipseXDir,
        ellipseYDir: scene.ellipses[0]?.ydir ?? curr.ellipseYDir,
        ellipseRadiusX: scene.ellipses[0]?.radiusX ?? curr.ellipseRadiusX,
        ellipseRadiusY: scene.ellipses[0]?.radiusY ?? curr.ellipseRadiusY,
        coneCenter: scene.cones[0]?.center ?? curr.coneCenter,
        coneXDir: scene.cones[0]?.xdir ?? curr.coneXDir,
        coneYDir: scene.cones[0]?.ydir ?? curr.coneYDir,
        coneRadius: scene.cones[0]?.radius ?? curr.coneRadius,
        coneHeight: scene.cones[0]?.height ?? curr.coneHeight,
        torusCenter: scene.tori[0]?.center ?? curr.torusCenter,
        torusXDir: scene.tori[0]?.xdir ?? curr.torusXDir,
        torusYDir: scene.tori[0]?.ydir ?? curr.torusYDir,
        torusMajorRadius: scene.tori[0]?.majorRadius ?? curr.torusMajorRadius,
        torusMinorRadius: scene.tori[0]?.minorRadius ?? curr.torusMinorRadius,
        torusAngleDeg: scene.tori[0]?.angleDeg ?? curr.torusAngleDeg,
        lineP0: scene.lines[0]?.p0 ?? curr.lineP0,
        lineP1: scene.lines[0]?.p1 ?? curr.lineP1,
        lineRadius: scene.lines[0]?.radius ?? curr.lineRadius,
        planeCenter: scene.planes[0]?.center ?? curr.planeCenter,
        planeXDir: scene.planes[0]?.xdir ?? curr.planeXDir,
        planeYDir: scene.planes[0]?.ydir ?? curr.planeYDir,
        planeHalfWidth: scene.planes[0]?.halfWidth ?? curr.planeHalfWidth,
        planeHalfHeight: scene.planes[0]?.halfHeight ?? curr.planeHalfHeight,
      };
      (curr as any).spheres = scene.spheres.map((s) => ({ center: s.center, radius: s.radius }));
      (curr as any).cylinders = scene.cylinders.map((c) => ({
        center: c.center,
        xdir: c.xdir,
        ydir: c.ydir,
        radius: c.radius,
        height: c.height,
        angleDeg: c.angleDeg,
      }));
      (curr as any).circles = scene.circles.map((c) => ({
        center: c.center,
        xdir: c.xdir,
        ydir: c.ydir,
        radius: c.radius,
      }));
      (curr as any).ellipses = scene.ellipses.map((e) => ({
        center: e.center,
        xdir: e.xdir,
        ydir: e.ydir,
        rx: e.radiusX,
        ry: e.radiusY,
      }));
      (curr as any).cones = scene.cones.map((c) => ({
        center: c.center,
        xdir: c.xdir,
        ydir: c.ydir,
        radius: c.radius,
        height: c.height,
      }));
      (curr as any).lines = scene.lines.map((l) => ({ p0: l.p0, p1: l.p1, radius: l.radius }));
      (curr as any).tori = scene.tori.map((t) => ({
        center: t.center,
        xdir: t.xdir,
        ydir: t.ydir,
        majorR: t.majorRadius,
        minorR: t.minorRadius,
        angleDeg: t.angleDeg,
      }));
      (curr as any).planes = scene.planes.map((p) => ({
        center: p.center,
        xdir: p.xdir,
        ydir: p.ydir,
        halfWidth: p.halfWidth,
        halfHeight: p.halfHeight,
      }));
      (curr as any).bezierPatches = scene.bezierPatches.map(cloneBezierPatch);
      await rebuildTLAS();
    },
    stop,
  };
}

export async function runMinimalScene(
  canvasOrId?: HTMLCanvasElement | string,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  sphereCenter: [number, number, number] = [0, 0, 0],
  sphereRadius = 1.2,
  cylinderCenter: [number, number, number] = [1.8, 0.0, 0.0],
  cylinderXDir: [number, number, number] = [1.0, 0.0, 0.0],
  cylinderYDir: [number, number, number] = [0.0, 0.0, -1.0],
  cylinderRadius = 0.4,
  cylinderHeight = 2.0,
  cylinderAngleDeg = 360.0,
  circleCenter: [number, number, number] = [-1.8, 0.0, 0.0],
  circleXDir: [number, number, number] = [1.0, 0.0, 0.0],
  circleYDir: [number, number, number] = [0.0, 1.0, 0.0],
  circleRadius = 0.9,
  lineP0: [number, number, number] = [-2.8, 0.8, -0.5],
  lineP1: [number, number, number] = [-1.6, 1.6, -0.2],
  ellipseCenter: [number, number, number] = [-4.0, 0.0, 0.0],
  ellipseXDir: [number, number, number] = [1.0, 0.0, 0.0],
  ellipseYDir: [number, number, number] = [0.0, 1.0, 0.0],
  ellipseRadiusX = 0.9,
  ellipseRadiusY = 0.5,
  torusCenter: [number, number, number] = [4.0, 0.2, 0.0],
  torusXDir: [number, number, number] = [1.0, 0.0, 0.0],
  torusYDir: [number, number, number] = [0.0, 1.0, 0.0],
  torusMajorRadius = 0.9,
  torusMinorRadius = 0.25,
  torusAngleDeg = 360.0,
  coneCenter: [number, number, number] = [0.0, -1.0, -1.2],
  coneXDir: [number, number, number] = [1.0, 0.0, 0.0],
  coneYDir: [number, number, number] = [0.0, 0.0, 1.0],
  coneRadius = 0.7,
  coneHeight = 1.6,
  planeCenter: [number, number, number] = [0.0, -2.0, 0.0],
  planeXDir: [number, number, number] = [1.0, 0.0, 0.0],
  planeYDir: [number, number, number] = [0.0, 0.0, 1.0],
  planeHalfWidth = 10.0,
  planeHalfHeight = 10.0,
) {
  const baseSceneState = buildMinimalSceneState({
    sphereCenter,
    sphereRadius,
    cylinderCenter,
    cylinderXDir,
    cylinderYDir,
    cylinderRadius,
    cylinderHeight,
    cylinderAngleDeg,
    circleCenter,
    circleXDir,
    circleYDir,
    circleRadius,
    lineP0,
    lineP1,
    ellipseCenter,
    ellipseXDir,
    ellipseYDir,
    ellipseRadiusX,
    ellipseRadiusY,
    torusCenter,
    torusXDir,
    torusYDir,
    torusMajorRadius,
    torusMinorRadius,
    torusAngleDeg,
    coneCenter,
    coneXDir,
    coneYDir,
    coneRadius,
    coneHeight,
    planeCenter,
    planeXDir,
    planeYDir,
    planeHalfWidth,
    planeHalfHeight,
    bezierPatches: DEFAULT_BEZIER_PATCHES,
  });
  try {
    return await runMinimalSceneImpl(
      canvasOrId,
      width,
      height,
      sphereCenter,
      sphereRadius,
      cylinderCenter,
      cylinderXDir,
      cylinderYDir,
      cylinderRadius,
      cylinderHeight,
      cylinderAngleDeg,
      circleCenter,
      circleXDir,
      circleYDir,
      circleRadius,
      lineP0,
      lineP1,
      ellipseCenter,
      ellipseXDir,
      ellipseYDir,
      ellipseRadiusX,
      ellipseRadiusY,
      torusCenter,
      torusXDir,
      torusYDir,
      torusMajorRadius,
      torusMinorRadius,
      torusAngleDeg,
      coneCenter,
      coneXDir,
      coneYDir,
      coneRadius,
      coneHeight,
      planeCenter,
      planeXDir,
      planeYDir,
      planeHalfWidth,
      planeHalfHeight,
      baseSceneState,
    );
  } catch (error) {
    console.warn('[WebRTX] Ray tracing pipeline failed; attempting fallback renderer.', error);
    return launchFallbackRenderer(canvasOrId, width, height, baseSceneState);
  }
}

export async function runSphereScene(
  sphereCenter: [number, number, number],
  sphereRadius: number,
  canvasOrId?: HTMLCanvasElement | string,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
) {
  return runMinimalScene(canvasOrId, width, height);
}

export async function runPlaneScene(){ throw new Error('Plane scene not implemented in ray tracing refactor'); }

// Backwards-compatible alias: legacy users can still call runRayTracingScene
export const runRayTracingScene = runMinimalScene;

type RayTracingHandle = Awaited<ReturnType<typeof runMinimalSceneImpl>>;
type FallbackHandle = Awaited<ReturnType<typeof launchFallbackRenderer>>;

export type SharedSceneHandle = RayTracingHandle | FallbackHandle;

export interface SharedSceneResult {
  active: SharedSceneHandle;
  fallback: FallbackHandle | null;
  raytracing: RayTracingHandle | null;
  setScene(scene: FallbackSceneDescriptor): Promise<void>;
  switchActive(renderer: 'fallback' | 'raytracing'): SharedSceneHandle | null;
  updateCamera(pos: Vec3, look: Vec3, up: Vec3): Promise<void>;
  render(): Promise<void>;
  readonly device: GPUDevice;
  readonly renderer: 'fallback' | 'raytracing';
  setCameraFovY(deg: number): void;
  setCameraNearFar(near: number, far: number): void;
  resize(width: number, height: number, dpr?: number): void;
  setCylinder(instance: CylinderInstance): Promise<void>;
  setTorus(instance: TorusInstance): Promise<void>;
  setTori(instances: TorusInstance[]): Promise<void>;
  setCylinders(instances: CylinderInstance[]): Promise<void>;
  setSpheres(instances: SphereInstance[]): Promise<void>;
  setCircles(instances: CircleInstance[]): Promise<void>;
  setEllipses(instances: EllipseInstance[]): Promise<void>;
  setLines(instances: LineInstance[]): Promise<void>;
  setCones(instances: ConeInstance[]): Promise<void>;
  setLine(instance: LineInstance): Promise<void>;
  stop(): void;
}

export async function runSharedScene(
  canvasOrId?: HTMLCanvasElement | string,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  sceneConfig?: MinimalSceneConfig,
): Promise<SharedSceneResult> {
  const config = sceneConfig ?? buildMinimalSceneState({
    sphereCenter: [0, 0, 0],
    sphereRadius: 1.2,
    cylinderCenter: [1.8, 0.0, 0.0],
    cylinderXDir: [1.0, 0.0, 0.0],
    cylinderYDir: [0.0, 0.0, -1.0],
    cylinderRadius: 0.4,
    cylinderHeight: 2.0,
    cylinderAngleDeg: 360.0,
    circleCenter: [-1.8, 0.0, 0.0],
    circleXDir: [1.0, 0.0, 0.0],
    circleYDir: [0.0, 1.0, 0.0],
    circleRadius: 0.9,
    lineP0: [-2.8, 0.8, -0.5],
    lineP1: [-1.6, 1.6, -0.2],
    ellipseCenter: [-4.0, 0.0, 0.0],
    ellipseXDir: [1.0, 0.0, 0.0],
    ellipseYDir: [0.0, 1.0, 0.0],
    ellipseRadiusX: 0.9,
    ellipseRadiusY: 0.5,
    torusCenter: [4.0, 0.2, 0.0],
    torusXDir: [1.0, 0.0, 0.0],
    torusYDir: [0.0, 1.0, 0.0],
    torusMajorRadius: 0.9,
    torusMinorRadius: 0.25,
    torusAngleDeg: 360.0,
    coneCenter: [0.0, -1.0, -1.2],
    coneXDir: [1.0, 0.0, 0.0],
    coneYDir: [0.0, 0.0, 1.0],
    coneRadius: 0.7,
    coneHeight: 1.6,
    planeCenter: [0.0, -2.0, 0.0],
    planeXDir: [1.0, 0.0, 0.0],
    planeYDir: [0.0, 0.0, 1.0],
    planeHalfWidth: 10.0,
    planeHalfHeight: 10.0,
    bezierPatches: DEFAULT_BEZIER_PATCHES,
  });
  const fallbackPromise = launchFallbackRenderer(canvasOrId, width, height, config).catch(() => null);
  const rtPromise = runMinimalSceneImpl(
    canvasOrId,
    width,
    height,
    config.sphereCenter,
    config.sphereRadius,
    config.cylinderCenter,
    config.cylinderXDir,
    config.cylinderYDir,
    config.cylinderRadius,
    config.cylinderHeight,
    config.cylinderAngleDeg,
    config.circleCenter,
    config.circleXDir,
    config.circleYDir,
    config.circleRadius,
    config.lineP0,
    config.lineP1,
    config.ellipseCenter,
    config.ellipseXDir,
    config.ellipseYDir,
    config.ellipseRadiusX,
    config.ellipseRadiusY,
    config.torusCenter,
    config.torusXDir,
    config.torusYDir,
    config.torusMajorRadius,
    config.torusMinorRadius,
    config.torusAngleDeg,
    config.coneCenter,
    config.coneXDir,
    config.coneYDir,
    config.coneRadius,
    config.coneHeight,
    config.planeCenter,
    config.planeXDir,
    config.planeYDir,
    config.planeHalfWidth,
    config.planeHalfHeight,
    config,
  ).catch(() => null);
  const [fallbackHandle, rtHandle] = await Promise.all([fallbackPromise, rtPromise]);
  const fallback = fallbackHandle;
  const raytracing = rtHandle;
  const activeInitial = raytracing ?? fallback;
  if (!activeInitial) {
    throw new Error('Failed to initialize both ray tracing and fallback renderers.');
  }
  let current: SharedSceneHandle = activeInitial;
  const setSceneAll = async (scene: FallbackSceneDescriptor) => {
    const tasks: Promise<void>[] = [];
    if (fallback) {
      tasks.push(fallback.setScene(scene));
    }
    if (raytracing) {
      tasks.push(raytracing.setScene(scene));
    }
    await Promise.all(tasks);
  };
  const switchActive = (renderer: 'fallback' | 'raytracing'): SharedSceneHandle | null => {
    if (renderer === 'raytracing' && raytracing) {
      current = raytracing;
      return current;
    }
    if (renderer === 'fallback' && fallback) {
      current = fallback;
      return current;
    }
    return null;
  };
  const callCurrent = (method: string, args: unknown[]): any => {
    const fn = (current as any)[method];
    if (typeof fn !== 'function') {
      throw new Error(`Active renderer does not implement method: ${method}`);
    }
    return fn.apply(current, args);
  };
  const callCurrentAsync = (method: string, args: unknown[]): Promise<void> => {
    const result = callCurrent(method, args);
    return Promise.resolve(result);
  };
  const stopAll = () => {
    if (fallback) {
      try { fallback.stop(); } catch {}
    }
    if (raytracing && raytracing !== fallback) {
      try { raytracing.stop(); } catch {}
    }
  };
  return {
    get active() {
      return current;
    },
    get device() {
      return current.device;
    },
    get renderer() {
      return current.renderer;
    },
    fallback,
    raytracing,
    setScene: setSceneAll,
    switchActive,
    updateCamera: (pos: Vec3, look: Vec3, up: Vec3) => callCurrentAsync('updateCamera', [pos, look, up]),
    render: () => callCurrentAsync('render', []),
    setCameraFovY: (deg: number) => { callCurrent('setCameraFovY', [deg]); },
    setCameraNearFar: (near: number, far: number) => { callCurrent('setCameraNearFar', [near, far]); },
    resize: (w: number, h: number, dpr?: number) => { callCurrent('resize', [w, h, dpr]); },
    setCylinder: (inst: CylinderInstance) => callCurrentAsync('setCylinder', [inst]),
    setTorus: (inst: TorusInstance) => callCurrentAsync('setTorus', [inst]),
    setTori: (insts: TorusInstance[]) => callCurrentAsync('setTori', [insts]),
    setCylinders: (insts: CylinderInstance[]) => callCurrentAsync('setCylinders', [insts]),
    setSpheres: (insts: SphereInstance[]) => callCurrentAsync('setSpheres', [insts]),
    setCircles: (insts: CircleInstance[]) => callCurrentAsync('setCircles', [insts]),
    setEllipses: (insts: EllipseInstance[]) => callCurrentAsync('setEllipses', [insts]),
    setLines: (insts: LineInstance[]) => callCurrentAsync('setLines', [insts]),
    setCones: (insts: ConeInstance[]) => callCurrentAsync('setCones', [insts]),
    setLine: (inst: LineInstance) => callCurrentAsync('setLine', [inst]),
    stop: stopAll,
  };
}
