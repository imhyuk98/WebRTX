import { initBvhWasm } from './wasm_bvh_builder';
import { convertHermitePatchToBezier } from './bezier_patch';
import type { BezierPatchInstance } from './fallback/scene_descriptor';

export interface AnalyticPrimitiveParams {
  sphereCenter: [number,number,number];
  sphereRadius: number;
  cylinderCenter: [number,number,number];
  cylinderXDir: [number,number,number];
  cylinderYDir: [number,number,number];
  cylinderRadius: number;
  cylinderHeight: number;
  cylinderAngleDeg: number;
  circleCenter: [number,number,number];
  circleXDir: [number,number,number];
  circleYDir: [number,number,number];
  circleRadius: number;
  // New: ellipse (oriented, non-uniform radii)
  ellipseCenter?: [number,number,number];
  ellipseXDir?: [number,number,number];
  ellipseYDir?: [number,number,number];
  ellipseRadiusX?: number;
  ellipseRadiusY?: number;
  // New: torus (oriented by x/y basis)
  torusCenter?: [number,number,number];
  torusXDir?: [number,number,number];
  torusYDir?: [number,number,number];
  torusMajorRadius?: number;
  torusMinorRadius?: number;
  torusAngleDeg?: number;
  // Multi-instance torus support (preferred): if provided, overrides single torus above
  tori?: Array<{
    center: [number,number,number];
    xdir: [number,number,number];
    ydir: [number,number,number];
    majorR: number;
    minorR: number;
    angleDeg?: number;
  }>;
  // Optional multi-instance arrays for other primitives (accepted now; shaders unchanged yet)
  spheres?: Array<{ center:[number,number,number]; radius:number; }>;
  cylinders?: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; height:number; angleDeg?:number; }>;
  circles?: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; }>;
  ellipses?: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; rx:number; ry:number; }>;
  cones?: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; radius:number; height:number; }>;
  lines?: Array<{ p0:[number,number,number]; p1:[number,number,number]; radius:number; }>;
  // New: cone
  coneCenter: [number,number,number];
  coneXDir: [number,number,number]; // local x at base
  coneYDir: [number,number,number]; // local y at base
  coneRadius: number; // base radius
  coneHeight: number; // along axis (xdir x ydir)
  // New: line segment (optional)
  // startPoint -> lineP0, endPoint -> lineP1, thickness is radius (not length)
  lineP0?: [number,number,number]; // startPoint
  lineP1?: [number,number,number]; // endPoint
  lineRadius?: number; // thickness as radius
  bezierPatches?: BezierPatchInstance[];
}

function normalize(v: [number,number,number]): [number,number,number] {
  const l = Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l];
}

// We'll pack all AABBs into a single buffer and use per-geometry offsets.
function packAabb(acc: number[], min: [number,number,number], max: [number,number,number]) {
  const offsetBytes = acc.length * 4; // current bytes before pushing
  acc.push(min[0],min[1],min[2], max[0],max[1],max[2]);
  return offsetBytes;
}

export async function buildAnalyticPrimitivesTLAS(device: GPUDevice, p: AnalyticPrimitiveParams) {
  if ((globalThis as any).WebRTX?.initBvhWasm) { await (globalThis as any).WebRTX.initBvhWasm(); } else { await initBvhWasm(); }
  // Lightweight performance overrides from a global knob for quick profiling
  const __ovr = (globalThis as any).__webrtxOverrides || {};
  const spheresOnly: boolean = __ovr.spheresOnly === true;
  const includePlane: boolean = __ovr.includePlane !== false; // default: include plane
  // Build arrays for all primitives (fallback to single values if arrays absent)
  const packed: number[] = [];
  // Spheres
  const spheres = (p.spheres && p.spheres.length>0) ? p.spheres : [{ center: p.sphereCenter, radius: p.sphereRadius }];
  const offSpheres: number[] = [];
  for(const s of spheres){
    offSpheres.push(packAabb(packed, [s.center[0]-s.radius, s.center[1]-s.radius, s.center[2]-s.radius], [s.center[0]+s.radius, s.center[1]+s.radius, s.center[2]+s.radius]));
  }
  // Cylinders
  const cylinders = spheresOnly ? [] : ((p.cylinders && p.cylinders.length>0) ? p.cylinders : [{ center: p.cylinderCenter, xdir: p.cylinderXDir, ydir: p.cylinderYDir, radius: p.cylinderRadius, height: p.cylinderHeight, angleDeg: p.cylinderAngleDeg }]);
  const offCylinders: number[] = [];
  for(const c of cylinders){
    const xdn = normalize(c.xdir as any);
    const ydn0 = normalize(c.ydir as any);
    const dotXY = xdn[0]*ydn0[0]+xdn[1]*ydn0[1]+xdn[2]*ydn0[2];
    let ydn: [number,number,number] = [ydn0[0]-dotXY*xdn[0], ydn0[1]-dotXY*xdn[1], ydn0[2]-dotXY*xdn[2]]; ydn = normalize(ydn);
    const axis: [number,number,number] = [ xdn[1]*ydn[2]-xdn[2]*ydn[1], xdn[2]*ydn[0]-xdn[0]*ydn[2], xdn[0]*ydn[1]-xdn[1]*ydn[0] ];
    const halfH = c.height*0.5;
    const absScale = (v:[number,number,number], s:number)=>[Math.abs(v[0])*s,Math.abs(v[1])*s,Math.abs(v[2])*s] as [number,number,number];
    const ex = absScale(xdn, c.radius), ey = absScale(ydn, c.radius), ez = absScale(axis, halfH);
    const ext: [number,number,number] = [ex[0]+ey[0]+ez[0], ex[1]+ey[1]+ez[1], ex[2]+ey[2]+ez[2]];
    offCylinders.push(packAabb(packed, [c.center[0]-ext[0], c.center[1]-ext[1], c.center[2]-ext[2]], [c.center[0]+ext[0], c.center[1]+ext[1], c.center[2]+ext[2]]));
  }
  // Circles
  const circles = spheresOnly ? [] : ((p.circles && p.circles.length>0) ? p.circles : [{ center: p.circleCenter, xdir: p.circleXDir, ydir: p.circleYDir, radius: p.circleRadius }]);
  const offCircles: number[] = [];
  for(const d of circles){
    const cxd = normalize(d.xdir as any); let cyd = normalize(d.ydir as any); const dotCX = cxd[0]*cyd[0]+cxd[1]*cyd[1]+cxd[2]*cyd[2];
    cyd = normalize([cyd[0]-dotCX*cxd[0], cyd[1]-dotCX*cxd[1], cyd[2]-dotCX*cxd[2]]);
    const cxdAbs:[number,number,number] = [Math.abs(cxd[0]),Math.abs(cxd[1]),Math.abs(cxd[2])];
    const cydAbs:[number,number,number] = [Math.abs(cyd[0]),Math.abs(cyd[1]),Math.abs(cyd[2])];
    const ext:[number,number,number] = [(cxdAbs[0]+cydAbs[0])*d.radius, (cxdAbs[1]+cydAbs[1])*d.radius, (cxdAbs[2]+cydAbs[2])*d.radius];
    offCircles.push(packAabb(packed, [d.center[0]-ext[0], d.center[1]-ext[1], d.center[2]-ext[2]], [d.center[0]+ext[0], d.center[1]+ext[1], d.center[2]+ext[2]]));
  }
  // Plane single (optional via override)
  let offPlane: number | null = null;
  if (includePlane) {
    const planeMin: [number,number,number] = [-10, -2.05, -10];
    const planeMax: [number,number,number] = [ 10, -1.95,  10];
    offPlane = packAabb(packed, planeMin, planeMax);
  }
  // Cones
  const cones = spheresOnly ? [] : ((p.cones && p.cones.length>0) ? p.cones : [{ center:p.coneCenter, xdir:p.coneXDir, ydir:p.coneYDir, radius:p.coneRadius, height:p.coneHeight }]);
  const offCones: number[] = [];
  for(const co of cones){
    const kx = normalize(co.xdir as any); let ky0 = normalize(co.ydir as any); const dotK = kx[0]*ky0[0]+kx[1]*ky0[1]+kx[2]*ky0[2];
    let ky:[number,number,number] = [ky0[0]-dotK*kx[0], ky0[1]-dotK*kx[1], ky0[2]-dotK*kx[2]]; ky = normalize(ky);
    const kz:[number,number,number] = [ kx[1]*ky[2]-kx[2]*ky[1], kx[2]*ky[0]-kx[0]*ky[2], kx[0]*ky[1]-kx[1]*ky[0] ];
    const coneHalfH = co.height * 0.5;
    const baseCenter:[number,number,number] = [ co.center[0] + kz[0]*coneHalfH, co.center[1] + kz[1]*coneHalfH, co.center[2] + kz[2]*coneHalfH ];
    const apex:[number,number,number] = [ co.center[0] - kz[0]*coneHalfH, co.center[1] - kz[1]*coneHalfH, co.center[2] - kz[2]*coneHalfH ];
    const baseExt:[number,number,number] = [ Math.abs(kx[0])*co.radius + Math.abs(ky[0])*co.radius, Math.abs(kx[1])*co.radius + Math.abs(ky[1])*co.radius, Math.abs(kx[2])*co.radius + Math.abs(ky[2])*co.radius ];
    const minX = Math.min(baseCenter[0]-baseExt[0], apex[0]);
    const minY = Math.min(baseCenter[1]-baseExt[1], apex[1]);
    const minZ = Math.min(baseCenter[2]-baseExt[2], apex[2]);
    const maxX = Math.max(baseCenter[0]+baseExt[0], apex[0]);
    const maxY = Math.max(baseCenter[1]+baseExt[1], apex[1]);
    const maxZ = Math.max(baseCenter[2]+baseExt[2], apex[2]);
    const pad = Math.max(1e-4, 1e-3 * Math.max(co.radius, co.height));
    offCones.push(packAabb(packed, [minX - pad, minY - pad, minZ - pad], [maxX + pad, maxY + pad, maxZ + pad]));
  }
  // Ellipses
  const ellipses = spheresOnly ? [] : ((p.ellipses && p.ellipses.length>0)
    ? p.ellipses
    : (p.ellipseCenter && p.ellipseXDir && p.ellipseYDir && p.ellipseRadiusX && p.ellipseRadiusY
    ? [{ center:p.ellipseCenter, xdir:p.ellipseXDir, ydir:p.ellipseYDir, rx:p.ellipseRadiusX, ry:p.ellipseRadiusY }]
    : []));
  const offEllipses: number[] = [];
  for(const e of ellipses){
    const exd = normalize(e.xdir as any); let eyd = normalize(e.ydir as any); const dE = exd[0]*eyd[0]+exd[1]*eyd[1]+exd[2]*eyd[2];
    eyd = normalize([eyd[0]-dE*exd[0], eyd[1]-dE*exd[1], eyd[2]-dE*exd[2]]);
    const exdAbs:[number,number,number] = [Math.abs(exd[0]),Math.abs(exd[1]),Math.abs(exd[2])];
    const eydAbs:[number,number,number] = [Math.abs(eyd[0]),Math.abs(eyd[1]),Math.abs(eyd[2])];
    const n:[number,number,number] = [ exd[1]*eyd[2]-exd[2]*eyd[1], exd[2]*eyd[0]-exd[0]*eyd[2], exd[0]*eyd[1]-exd[1]*eyd[0] ];
    const nLen = Math.hypot(n[0],n[1],n[2])||1; const nUnit:[number,number,number] = [n[0]/nLen, n[1]/nLen, n[2]/nLen];
    const nAbs:[number,number,number] = [Math.abs(nUnit[0]),Math.abs(nUnit[1]),Math.abs(nUnit[2])];
    const rx = e.rx; const ry = e.ry; const rMax = Math.max(rx, ry);
    const normalThick = Math.max(5e-2, 0.5 * rMax); const safetyScale = 2.0;
    const ellExt: [number,number,number] = [ safetyScale * ((exdAbs[0]+eydAbs[0])*rMax + nAbs[0]*normalThick), safetyScale * ((exdAbs[1]+eydAbs[1])*rMax + nAbs[1]*normalThick), safetyScale * ((exdAbs[2]+eydAbs[2])*rMax + nAbs[2]*normalThick) ];
    const epad = Math.max(5e-3, 5e-2 * Math.max(rx, ry));
    offEllipses.push(packAabb(packed, [ e.center[0]-ellExt[0]-epad, e.center[1]-ellExt[1]-epad, e.center[2]-ellExt[2]-epad ], [ e.center[0]+ellExt[0]+epad, e.center[1]+ellExt[1]+epad, e.center[2]+ellExt[2]+epad ]));
  }
  // Lines
  const offLines: number[] = [];
  if (!spheresOnly && p.lines && p.lines.length>0){
    for(const L of p.lines){
      const p0 = L.p0, p1 = L.p1, r = L.radius;
      const minX = Math.min(p0[0], p1[0]) - r, minY = Math.min(p0[1], p1[1]) - r, minZ = Math.min(p0[2], p1[2]) - r;
      const maxX = Math.max(p0[0], p1[0]) + r, maxY = Math.max(p0[1], p1[1]) + r, maxZ = Math.max(p0[2], p1[2]) + r;
      const pad = Math.max(5e-4, 2e-2 * r);
      offLines.push(packAabb(packed, [minX-pad,minY-pad,minZ-pad], [maxX+pad,maxY+pad,maxZ+pad]));
    }
  } else if (!spheresOnly && p.lineP0 && p.lineP1 && (p.lineRadius ?? 0) > 0){
    const p0 = p.lineP0, p1 = p.lineP1, r = p.lineRadius!;
    const minX = Math.min(p0[0], p1[0]) - r, minY = Math.min(p0[1], p1[1]) - r, minZ = Math.min(p0[2], p1[2]) - r;
    const maxX = Math.max(p0[0], p1[0]) + r, maxY = Math.max(p0[1], p1[1]) + r, maxZ = Math.max(p0[2], p1[2]) + r;
    const pad = Math.max(5e-4, 2e-2 * r);
    offLines.push(packAabb(packed, [minX-pad,minY-pad,minZ-pad], [maxX+pad,maxY+pad,maxZ+pad]));
  }
  // Torus AABB if provided
  // Multi- or single-torus packing
  const offTorusList: number[] = [];
  const torusItems = spheresOnly ? [] : ((p.tori && p.tori.length>0)
    ? p.tori.map(t=>({
        center: t.center,
        xdir: t.xdir,
        ydir: t.ydir,
        R: t.majorR,
        r: t.minorR,
      }))
    : ((p.torusCenter && p.torusXDir && p.torusYDir && p.torusMajorRadius && p.torusMinorRadius)
        ? [{ center: p.torusCenter, xdir: p.torusXDir, ydir: p.torusYDir, R: p.torusMajorRadius, r: p.torusMinorRadius }]
        : []));
  for (const t of torusItems){
    const txd = normalize(t.xdir);
    let tyd = normalize(t.ydir);
    const dt = txd[0]*tyd[0]+txd[1]*tyd[1]+txd[2]*tyd[2];
    tyd = normalize([tyd[0]-dt*txd[0], tyd[1]-dt*txd[1], tyd[2]-dt*txd[2]]);
    const tz:[number,number,number] = [ txd[1]*tyd[2]-txd[2]*tyd[1], txd[2]*tyd[0]-txd[0]*tyd[2], txd[0]*tyd[1]-txd[1]*tyd[0] ];
    const txAbs:[number,number,number] = [Math.abs(txd[0]),Math.abs(txd[1]),Math.abs(txd[2])];
    const tzAbs:[number,number,number] = [Math.abs(tz[0]),Math.abs(tz[1]),Math.abs(tz[2])];
    const sRing = (t.R + t.r);
    const safety = 1.3;
    let ext:[number,number,number] = [ (txAbs[0]+tzAbs[0])*sRing + t.r, (txAbs[1]+tzAbs[1])*sRing + t.r, (txAbs[2]+tzAbs[2])*sRing + t.r ];
    ext = [ ext[0]*safety, ext[1]*safety, ext[2]*safety ];
    const pad = Math.max(1e-2*sRing, 2e-2*t.r, 5e-3);
    const tmin:[number,number,number] = [ t.center[0]-ext[0]-pad, t.center[1]-ext[1]-pad, t.center[2]-ext[2]-pad ];
    const tmax:[number,number,number] = [ t.center[0]+ext[0]+pad, t.center[1]+ext[1]+pad, t.center[2]+ext[2]+pad ];
    offTorusList.push(packAabb(packed, tmin, tmax));
  }
  // Bezier patch bounds
  const bezierPatches = spheresOnly ? [] : (p.bezierPatches ?? []);
  const offBezier: number[] = [];
  for (const patch of bezierPatches) {
    const converted = convertHermitePatchToBezier(patch);
    offBezier.push(packAabb(packed, converted.boundsMin, converted.boundsMax));
  }

  const aabbPackData = new Float32Array(packed);
  const aabbPackBuf = device.createBuffer({ size: aabbPackData.byteLength, usage: (GPUBufferUsageRTX as any).ACCELERATION_STRUCTURE_BUILD_INPUT_READONLY, mappedAtCreation: true });
  new Float32Array(aabbPackBuf.getMappedRange()).set(aabbPackData); aabbPackBuf.unmap();

  // Compute gid bases dynamically so shaders can map gid→type safely (current order preserved)
  let gid = 0;
  const gidSphereBase = gid; gid += offSpheres.length;
  const gidCylBase = gid; gid += offCylinders.length;
  const gidCircleBase = gid; gid += offCircles.length;
  const gidPlaneBase = gid; gid += includePlane ? 1 : 0;
  const gidConeBase = gid; gid += offCones.length;
  const gidEllipseBase = gid; gid += offEllipses.length;
  const gidLineBase = gid; gid += offLines.length;
  const gidTorusBase = gid; gid += offTorusList.length;
  const gidBezierBase = gid; gid += offBezier.length;

  const bottom: GPURayTracingAccelerationContainerDescriptor_bottom = {
    usage: (GPURayTracingAccelerationContainerUsage as any).NONE,
    level: 'bottom',
    geometries: [
  ...offSpheres.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offCylinders.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offCircles.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...(includePlane && offPlane!==null ? [{ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: offPlane!, format:'float32x2' as const, stride:12, size:24 } }] : []),
  ...offCones.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offEllipses.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offLines.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offTorusList.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
  ...offBezier.map(off => ({ usage: (GPURayTracingAccelerationGeometryUsage as any).NONE, type:'aabbs' as const, aabb:{ buffer: aabbPackBuf, offset: off, format:'float32x2' as const, stride:12, size:24 } })),
    ]
  };
  const top: GPURayTracingAccelerationContainerDescriptor_top = {
    usage:(GPURayTracingAccelerationContainerUsage as any).NONE,
    level:'top',
    instances:[{ usage:(GPURayTracingAccelerationInstanceUsage as any).NONE, mask:0xff, instanceSBTRecordOffset:0, blas: bottom }]
  };
  const tlas = device.createRayTracingAccelerationContainer(top);
  (tlas as any).hostBuild?.(device);
  const meta = {
    sphereBase: gidSphereBase, sphereCount: offSpheres.length,
    cylinderBase: gidCylBase, cylinderCount: offCylinders.length,
    circleBase: gidCircleBase, circleCount: offCircles.length,
    planeBase: gidPlaneBase, planeCount: includePlane ? 1 : 0,
    coneBase: gidConeBase, coneCount: offCones.length,
    ellipseBase: gidEllipseBase, ellipseCount: offEllipses.length,
    lineBase: gidLineBase, lineCount: offLines.length,
    torusBase: gidTorusBase, torusCount: offTorusList.length,
    bezierBase: gidBezierBase, bezierCount: offBezier.length,
  } as const;
  return { tlas, meta };
}
