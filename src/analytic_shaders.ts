export function getRayGenShader(){return `#version 460 core\n#extension GL_EXT_ray_tracing : require\n#ifndef WEBRTX_DECL_RT_OUT\n#define WEBRTX_DECL_RT_OUT\nlayout(set=0,binding=0,std430) buffer RtOut { vec4 data[]; } rtOut;\n#endif\n#ifndef WEBRTX_DECL_CAM\n#define WEBRTX_DECL_CAM\nlayout(set=0,binding=1) uniform Cam { vec4 c0; vec4 c1; vec4 c2; vec4 c3; } cam;\n#endif\n#ifndef WEBRTX_DECL_META\n#define WEBRTX_DECL_META\nlayout(set=0,binding=2) uniform SceneMeta { uvec4 m0; uvec4 m1; uvec4 m2; uvec4 m3; uvec4 m4; } meta;\n#endif\nvoid main(){ uvec2 pix=gl_LaunchIDEXT.xy; uvec2 dim=gl_LaunchSizeEXT.xy; vec2 uv=(vec2(pix)+0.5)/vec2(dim); vec2 ndc=vec2(uv.x*2.0-1.0,1.0-uv.y*2.0); float aspect=float(dim.x)/float(dim.y); vec3 pos=cam.c0.xyz; vec3 look=cam.c1.xyz; vec3 up=normalize(cam.c2.xyz); vec3 f=normalize(look-pos); vec3 r=normalize(cross(f,up)); vec3 u=normalize(cross(r,f)); float fovY=cam.c3.x; float tanH=tan(fovY*0.5); vec3 rd=normalize(f + ndc.x*aspect*tanH*r + ndc.y*tanH*u); traceRayEXT(uvec2(0,0),0,0xff,0,2,0,pos,0.001,rd,1e38,0); }`;}
export function getMissShader(){return `#version 460 core\n#extension GL_EXT_ray_tracing : require\n#ifndef WEBRTX_DECL_RT_OUT\n#define WEBRTX_DECL_RT_OUT\nlayout(set=0,binding=0,std430) buffer RtOut { vec4 data[]; } rtOut;\n#endif\nvoid main(){ uvec2 pix=gl_LaunchIDEXT.xy; uvec2 dim=gl_LaunchSizeEXT.xy; vec2 uv=(vec2(pix)+0.5)/vec2(dim); vec3 col=mix(vec3(0.2,0.3,0.5),vec3(0.6,0.8,1.0),uv.y); rtOut.data[pix.y*dim.x+pix.x]=vec4(col,1.0); }`;}
export function getIntersectionShader(){return `#version 460 core
#extension GL_EXT_ray_tracing : require
#ifndef WEBRTX_DECL_CAM
#define WEBRTX_DECL_CAM
layout(set=0,binding=1) uniform Cam { vec4 c0; vec4 c1; vec4 c2; vec4 c3; } cam;
#endif
#ifndef WEBRTX_DECL_PACKED_OFFSETS
#define WEBRTX_DECL_PACKED_OFFSETS
// base: x=circle, y=ellipse, z=line, w=cone; values measured in vec4 units
// stride: number of vec4s per instance for each type
layout(set=0,binding=9) uniform PackedOffsets { uvec4 base; uvec4 stride; } packOfs;
#endif
#ifndef WEBRTX_DECL_PACKED_BUFFER
#define WEBRTX_DECL_PACKED_BUFFER
layout(set=0,binding=13,std430) buffer PackedPrims { vec4 data[]; } packed;
#endif
#ifndef WEBRTX_DECL_META
#define WEBRTX_DECL_META
layout(set=0,binding=2) uniform SceneMeta { uvec4 m0; uvec4 m1; uvec4 m2; uvec4 m3; uvec4 m4; } meta;
#endif
#ifndef WEBRTX_DECL_CYL_ARRAY
#define WEBRTX_DECL_CYL_ARRAY
struct CylParam { vec4 c0; vec4 c1; vec4 c2; };
layout(set=0,binding=12,std430) buffer CylParams { CylParam items[]; } sCyl;
#endif
#ifndef WEBRTX_DECL_SPHERE_ARRAY
#define WEBRTX_DECL_SPHERE_ARRAY
struct SphereParam { vec4 s0; };
layout(set=0,binding=11,std430) buffer SphereParams { SphereParam items[]; } sSphere;
#endif
#define WEBRTX_NO_CIRCLE_ELLIPSE_LINE_CONE_ARRAYS 1
#ifndef WEBRTX_DECL_TORUS_ARRAY
#define WEBRTX_DECL_TORUS_ARRAY
struct TorusParam { vec4 t0; vec4 t1; vec4 t2; };
layout(set=0,binding=10,std430) buffer TorusParams { TorusParam items[]; } sTorus;
#endif
#ifndef WEBRTX_DECL_CYL
#define WEBRTX_DECL_CYL
layout(set=0,binding=4) uniform Cylinder { vec4 c0; vec4 c1; vec4 c2; } uCyl;
#endif
#ifndef WEBRTX_DECL_CIRCLE
#define WEBRTX_DECL_CIRCLE
layout(set=0,binding=5) uniform Circle { vec4 d0; vec4 d1; vec4 d2; } uCircle;
#endif
#ifndef WEBRTX_DECL_ELLIPSE
#define WEBRTX_DECL_ELLIPSE
// Ellipse packed like Circle but with radii in d0.w (rx) and d1.w (ry)
layout(set=0,binding=7) uniform Ellipse { vec4 e0; vec4 e1; vec4 e2; } uEllipse;
#endif
#ifndef WEBRTX_DECL_LINE
#define WEBRTX_DECL_LINE
// Line segment: l0 = p0.xyz, radius; l1 = p1.xyz, unused
layout(set=0,binding=8) uniform Line { vec4 l0; vec4 l1; } uLine;
#endif
#ifndef WEBRTX_DECL_CONE
#define WEBRTX_DECL_CONE
layout(set=0,binding=6) uniform Cone { vec4 e0; vec4 e1; } uCone; // e0: center.xyz, radius; e1: axis.xyz, height
#endif
#ifndef WEBRTX_DECL_BEZIER_BUFFER
#define WEBRTX_DECL_BEZIER_BUFFER
layout(set=0,binding=14,std430) buffer BezierPatches { vec4 data[]; } sBezier;
#endif
hitAttributeEXT vec2 attrUv;

const uint BEZIER_VEC4_PER_PATCH = 18u;
const int BEZIER_MAX_QUEUE = 48;

struct BezierPatchData {
  vec4 cp[16];
  vec3 boundsMin;
  vec3 boundsMax;
  float maxDepth;
  float pixelEpsilon;
};

struct BezierNode {
  vec4 cp[16];
  vec3 boundsMin;
  vec3 boundsMax;
  float u0;
  float v0;
  float u1;
  float v1;
  float tEnter;
  int depth;
};

vec3 makeSafeInverse(vec3 dir){
  const float EPS = 1e-6;
  vec3 safe = vec3(
    abs(dir.x) > EPS ? dir.x : (dir.x >= 0.0 ? EPS : -EPS),
    abs(dir.y) > EPS ? dir.y : (dir.y >= 0.0 ? EPS : -EPS),
    abs(dir.z) > EPS ? dir.z : (dir.z >= 0.0 ? EPS : -EPS)
  );
  return 1.0 / safe;
}

bool intersectAabbExt(vec3 bmin, vec3 bmax, vec3 origin, vec3 invDir, float tMin, float tMax, out float tEnter, out float tExit){
  vec3 t0 = (bmin - origin) * invDir;
  vec3 t1 = (bmax - origin) * invDir;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  tEnter = max(max(max(tmin.x, tmin.y), tmin.z), tMin);
  tExit = min(min(min(tmax.x, tmax.y), tmax.z), tMax);
  return tExit >= tEnter;
}

int bezierIdx(int u, int v){
  return u * 4 + v;
}

vec3 bezierGet(in vec4 cp[16], int u, int v){
  return cp[bezierIdx(u, v)].xyz;
}

void bezierSet(inout vec4 cp[16], int u, int v, vec3 value){
  cp[bezierIdx(u, v)] = vec4(value, 0.0);
}

BezierPatchData loadBezierPatch(uint index){
  uint base = index * BEZIER_VEC4_PER_PATCH;
  BezierPatchData result;
  for(uint i=0u;i<16u;++i){
    result.cp[i] = sBezier.data[base + i];
  }
  vec4 bounds0 = sBezier.data[base + 16u];
  vec4 bounds1 = sBezier.data[base + 17u];
  result.boundsMin = bounds0.xyz;
  result.boundsMax = bounds1.xyz;
  result.maxDepth = bounds0.w;
  result.pixelEpsilon = bounds1.w;
  return result;
}

void bezierEval(in vec4 cp[16], float u, float v, out vec3 P, out vec3 dPu, out vec3 dPv){
  vec3 Cv[4];
  for(int i=0;i<4;++i){
    vec3 p0 = bezierGet(cp, i, 0);
    vec3 p1 = bezierGet(cp, i, 1);
    vec3 p2 = bezierGet(cp, i, 2);
    vec3 p3 = bezierGet(cp, i, 3);
    vec3 a = mix(p0, p1, v);
    vec3 b = mix(p1, p2, v);
    vec3 c = mix(p2, p3, v);
    vec3 d = mix(a, b, v);
    vec3 e = mix(b, c, v);
    Cv[i] = mix(d, e, v);
  }
  vec3 A = mix(Cv[0], Cv[1], u);
  vec3 B = mix(Cv[1], Cv[2], u);
  vec3 C = mix(Cv[2], Cv[3], u);
  vec3 D = mix(A, B, u);
  vec3 E = mix(B, C, u);
  P = mix(D, E, u);
  dPu = 3.0 * (E - D);

  vec3 Ru[4];
  for(int j=0;j<4;++j){
    vec3 p0 = bezierGet(cp, 0, j);
    vec3 p1 = bezierGet(cp, 1, j);
    vec3 p2 = bezierGet(cp, 2, j);
    vec3 p3 = bezierGet(cp, 3, j);
    vec3 a = mix(p0, p1, u);
    vec3 b = mix(p1, p2, u);
    vec3 c = mix(p2, p3, u);
    vec3 d = mix(a, b, u);
    vec3 e = mix(b, c, u);
    Ru[j] = mix(d, e, u);
  }
  vec3 A2 = mix(Ru[0], Ru[1], v);
  vec3 B2 = mix(Ru[1], Ru[2], v);
  vec3 C2 = mix(Ru[2], Ru[3], v);
  vec3 D2 = mix(A2, B2, v);
  vec3 E2 = mix(B2, C2, v);
  dPv = 3.0 * (E2 - D2);
}

void splitAlongV(in vec4 input[16], out vec4 lo[16], out vec4 hi[16]){
  for(int i=0;i<4;++i){
    vec3 a = bezierGet(input, i, 0);
    vec3 b = bezierGet(input, i, 1);
    vec3 c = bezierGet(input, i, 2);
    vec3 d = bezierGet(input, i, 3);
    vec3 ab = mix(a, b, 0.5);
    vec3 bc = mix(b, c, 0.5);
    vec3 cd = mix(c, d, 0.5);
    vec3 abc = mix(ab, bc, 0.5);
    vec3 bcd = mix(bc, cd, 0.5);
    vec3 abcd = mix(abc, bcd, 0.5);
    bezierSet(lo, i, 0, a);
    bezierSet(lo, i, 1, ab);
    bezierSet(lo, i, 2, abc);
    bezierSet(lo, i, 3, abcd);
    bezierSet(hi, i, 0, abcd);
    bezierSet(hi, i, 1, bcd);
    bezierSet(hi, i, 2, cd);
    bezierSet(hi, i, 3, d);
  }
}

void subdivideBezierPatch(in vec4 src[16], out vec4 c0[16], out vec4 c1[16], out vec4 c2[16], out vec4 c3[16]){
  vec4 left[16];
  vec4 right[16];
  for(int j=0;j<4;++j){
    vec3 a = bezierGet(src, 0, j);
    vec3 b = bezierGet(src, 1, j);
    vec3 c = bezierGet(src, 2, j);
    vec3 d = bezierGet(src, 3, j);
    vec3 ab = mix(a, b, 0.5);
    vec3 bc = mix(b, c, 0.5);
    vec3 cd = mix(c, d, 0.5);
    vec3 abc = mix(ab, bc, 0.5);
    vec3 bcd = mix(bc, cd, 0.5);
    vec3 abcd = mix(abc, bcd, 0.5);
    bezierSet(left, 0, j, a);
    bezierSet(left, 1, j, ab);
    bezierSet(left, 2, j, abc);
    bezierSet(left, 3, j, abcd);
    bezierSet(right, 0, j, abcd);
    bezierSet(right, 1, j, bcd);
    bezierSet(right, 2, j, cd);
    bezierSet(right, 3, j, d);
  }
  splitAlongV(left, c0, c2);
  splitAlongV(right, c1, c3);
}

void computeBezierBounds(in vec4 cp[16], out vec3 bmin, out vec3 bmax){
  bmin = cp[0].xyz;
  bmax = cp[0].xyz;
  for(int i=1;i<16;++i){
    vec3 p = cp[i].xyz;
    bmin = min(bmin, p);
    bmax = max(bmax, p);
  }
}

bool newtonRefine(in vec4 cp[16], vec3 rayOrigin, vec3 rayDir, float tMin, float tMax, inout float u, inout float v, inout float t){
  const int MAX_IT = 8;
  const float EPS_F = 1e-4;
  const float EPS_P = 1e-5;
  for(int iter=0; iter<MAX_IT; ++iter){
    vec3 P;
    vec3 dPu;
    vec3 dPv;
    bezierEval(cp, u, v, P, dPu, dPv);
    vec3 F = P - (rayOrigin + rayDir * t);
    float err = length(F);
    if(err < EPS_F){
      bool inside = u >= -0.01 && u <= 1.01 && v >= -0.01 && v <= 1.01 && t >= tMin && t <= tMax;
      return inside;
    }
    vec3 c0 = dPu;
    vec3 c1 = dPv;
    vec3 c2 = -rayDir;
    vec3 r0 = cross(c1, c2);
    vec3 r1 = cross(c2, c0);
    vec3 r2 = cross(c0, c1);
    float det = dot(c0, r0);
    if(abs(det) < 1e-10){
      return false;
    }
    float invDet = -1.0 / det;
    float du = dot(r0, F) * invDet;
    float dv = dot(r1, F) * invDet;
    float dt = dot(r2, F) * invDet;
    u += du;
    v += dv;
    t += dt;
    if(abs(du) < EPS_P && abs(dv) < EPS_P && abs(dt) < EPS_P){
      vec3 P2;
      vec3 tmpDu;
      vec3 tmpDv;
      bezierEval(cp, u, v, P2, tmpDu, tmpDv);
      vec3 F2 = P2 - (rayOrigin + rayDir * t);
      float err2 = length(F2);
      bool inside = u >= -0.01 && u <= 1.01 && v >= -0.01 && v <= 1.01 && t >= tMin && t <= tMax;
      if(inside && err2 < EPS_F){
        return true;
      }
    }
  }
  return false;
}

bool intersectBezierPatch(vec3 rayOrigin, float rayTMin, vec3 rayDir, float rayTMax, BezierPatchData patch, float pixelWorldSlope, out float hitT, out vec2 hitUV){
  vec3 invDir = makeSafeInverse(rayDir);
  float enterT;
  float exitT;
  float farLimit = rayTMax;
  if(!intersectAabbExt(patch.boundsMin, patch.boundsMax, rayOrigin, invDir, rayTMin, farLimit, enterT, exitT)){
    return false;
  }
  float bestT = min(rayTMax, exitT);
  vec2 bestUV = vec2(0.0);
  float pixelEps = max(patch.pixelEpsilon, 1e-4);
  int maxDepth = clamp(int(floor(patch.maxDepth + 0.5)), 1, 16);
  vec3 rootExtents = patch.boundsMax - patch.boundsMin;
  float rootMaxEdge = max(max(rootExtents.x, rootExtents.y), rootExtents.z);
  bool hasHit = false;
  float depthEdgeTarget = rootMaxEdge * exp2(-float(maxDepth));
  float baseLeafEdge = max(pixelEps, depthEdgeTarget);
  BezierNode queue[BEZIER_MAX_QUEUE];
  int size = 0;
  BezierNode root;
  for(int i=0;i<16;++i){ root.cp[i] = patch.cp[i]; }
  root.boundsMin = patch.boundsMin;
  root.boundsMax = patch.boundsMax;
  root.u0 = 0.0;
  root.v0 = 0.0;
  root.u1 = 1.0;
  root.v1 = 1.0;
  root.tEnter = enterT;
  root.depth = 0;
  queue[size++] = root;
  while(size > 0){
    int bestIdx = 0;
    float bestEnter = queue[0].tEnter;
    for(int i=1;i<size;++i){
      if(queue[i].tEnter < bestEnter){
        bestEnter = queue[i].tEnter;
        bestIdx = i;
      }
    }
    BezierNode node = queue[bestIdx];
    queue[bestIdx] = queue[--size];
    if(node.tEnter >= bestT){
      continue;
    }
    vec3 extents = node.boundsMax - node.boundsMin;
    float maxEdge = max(max(extents.x, extents.y), extents.z);
    float distanceEstimate = max(max(node.tEnter, rayTMin), 0.0);
    float screenLeafEdge = max(distanceEstimate * pixelWorldSlope, 1e-4);
    float nodeLeafEdge = max(baseLeafEdge, screenLeafEdge);
  bool leaf = (maxEdge <= nodeLeafEdge) || (node.depth >= maxDepth);
    if(leaf){
      float uLocal = 0.5;
      float vLocal = 0.5;
      float tCandidate = clamp(node.tEnter, rayTMin, bestT);
      float tUpper = min(bestT, rayTMax);
      if(newtonRefine(node.cp, rayOrigin, rayDir, rayTMin, tUpper, uLocal, vLocal, tCandidate)){
        if(tCandidate >= rayTMin && tCandidate < bestT && tCandidate <= rayTMax){
          float uGlobal = mix(node.u0, node.u1, clamp(uLocal, 0.0, 1.0));
          float vGlobal = mix(node.v0, node.v1, clamp(vLocal, 0.0, 1.0));
          bestT = tCandidate;
          bestUV = vec2(uGlobal, vGlobal);
          hasHit = true;
        }
      }
      continue;
    }
    vec4 child0[16];
    vec4 child1[16];
    vec4 child2[16];
    vec4 child3[16];
    subdivideBezierPatch(node.cp, child0, child1, child2, child3);
    float uMid = 0.5 * (node.u0 + node.u1);
    float vMid = 0.5 * (node.v0 + node.v1);
    float farBound = min(bestT, rayTMax);

  vec3 bmin0; vec3 bmax0; computeBezierBounds(child0, bmin0, bmax0);
  float enter0; float exit0;
  if(size < BEZIER_MAX_QUEUE && intersectAabbExt(bmin0, bmax0, rayOrigin, invDir, rayTMin, farBound, enter0, exit0)){
      BezierNode child;
      for(int k=0;k<16;++k){ child.cp[k] = child0[k]; }
      child.boundsMin = bmin0 - vec3(1e-6);
      child.boundsMax = bmax0 + vec3(1e-6);
      child.depth = node.depth + 1;
      child.u0 = node.u0; child.u1 = uMid; child.v0 = node.v0; child.v1 = vMid;
      child.tEnter = enter0;
      queue[size++] = child;
    }

  vec3 bmin1; vec3 bmax1; computeBezierBounds(child1, bmin1, bmax1);
  float enter1; float exit1;
  if(size < BEZIER_MAX_QUEUE && intersectAabbExt(bmin1, bmax1, rayOrigin, invDir, rayTMin, farBound, enter1, exit1)){
      BezierNode child;
      for(int k=0;k<16;++k){ child.cp[k] = child1[k]; }
      child.boundsMin = bmin1 - vec3(1e-6);
      child.boundsMax = bmax1 + vec3(1e-6);
      child.depth = node.depth + 1;
      child.u0 = uMid; child.u1 = node.u1; child.v0 = node.v0; child.v1 = vMid;
      child.tEnter = enter1;
      queue[size++] = child;
    }

  vec3 bmin2; vec3 bmax2; computeBezierBounds(child2, bmin2, bmax2);
  float enter2; float exit2;
  if(size < BEZIER_MAX_QUEUE && intersectAabbExt(bmin2, bmax2, rayOrigin, invDir, rayTMin, farBound, enter2, exit2)){
      BezierNode child;
      for(int k=0;k<16;++k){ child.cp[k] = child2[k]; }
      child.boundsMin = bmin2 - vec3(1e-6);
      child.boundsMax = bmax2 + vec3(1e-6);
      child.depth = node.depth + 1;
      child.u0 = node.u0; child.u1 = uMid; child.v0 = vMid; child.v1 = node.v1;
      child.tEnter = enter2;
      queue[size++] = child;
    }

  vec3 bmin3; vec3 bmax3; computeBezierBounds(child3, bmin3, bmax3);
  float enter3; float exit3;
  if(size < BEZIER_MAX_QUEUE && intersectAabbExt(bmin3, bmax3, rayOrigin, invDir, rayTMin, farBound, enter3, exit3)){
      BezierNode child;
      for(int k=0;k<16;++k){ child.cp[k] = child3[k]; }
      child.boundsMin = bmin3 - vec3(1e-6);
      child.boundsMax = bmax3 + vec3(1e-6);
      child.depth = node.depth + 1;
      child.u0 = uMid; child.u1 = node.u1; child.v0 = vMid; child.v1 = node.v1;
      child.tEnter = enter3;
      queue[size++] = child;
    }
  }
  if(hasHit && bestT < min(rayTMax, 1e37)){
    hitT = bestT;
    hitUV = bestUV;
    return true;
  }
  return false;
}

void main(){
  // Use built-in ray state to avoid recomputing camera rays per intersection
  vec3 ro = gl_WorldRayOriginEXT;
  vec3 rd = gl_WorldRayDirectionEXT;
  float nearT = cam.c3.z; // keep near from camera UBO
  float INF = 1e38; const float EPS = 1e-5; uint gid = gl_GeometryIndexEXT;
  if(gid >= meta.m1.x && gid < meta.m1.x + meta.m1.y){
    uint i = gid - meta.m1.x; SphereParam sp = sSphere.items[i];
    vec3 c=sp.s0.xyz; float R=sp.s0.w; vec3 oc=ro-c; float b=dot(oc,rd); float c2=dot(oc,oc)-R*R; float disc=b*b-c2; if(disc>=0.0){ float s=sqrt(disc); float t0=-b-s; float t1=-b+s; float tt=t0; if(tt<nearT) tt=t1; if(tt>=nearT && tt<INF) reportIntersectionEXT(tt, gl_HitKindFrontFacingTriangleEXT); }
  } else if(gid >= meta.m1.z && gid < meta.m1.z + meta.m1.w){
    // cylinder via SSBO array
    uint i = gid - meta.m1.z;
    CylParam cylp = sCyl.items[i];
    vec3 center=cylp.c0.xyz; vec3 xdir=normalize(cylp.c1.xyz);
    vec3 yIn=normalize(cylp.c2.xyz - xdir*dot(cylp.c2.xyz,xdir));
    vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float radius=cylp.c0.w; float height=cylp.c1.w; float angleDeg=cylp.c2.w; float halfH=height*0.5;
    vec3 roW=ro-center; float roZ=dot(roW,axis); float rdZ=dot(rd,axis); vec3 roXY=roW-axis*roZ; vec3 rdXY=rd-axis*rdZ;
    float bestT=INF; int bestKind=-1; // 0 side,1 top,2 bottom
    float a=dot(rdXY,rdXY);
    if(a>0.0){
      float b=2.0*dot(roXY,rdXY); float cc=dot(roXY,roXY)-radius*radius; float disc=b*b-4.0*a*cc;
      if(disc>=0.0){
        float sdisc=sqrt(disc); float inv2a=0.5/a; float cand[2]; cand[0]=(-b - sdisc)*inv2a; cand[1]=(-b + sdisc)*inv2a;
        for(int i2=0;i2<2;i2++){
          float tt=cand[i2]; if(tt<nearT || tt>=bestT) continue; float z=roZ+tt*rdZ; if(z<-halfH-EPS || z>halfH+EPS) continue;
          vec3 pLocal=roXY+rdXY*tt; float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(angleDeg<360.0 && ang>angleDeg) continue; bestT=tt; bestKind=0; }
      }
    }
    if(abs(rdZ)>1e-7){
      float tTop=(halfH-roZ)/rdZ; if(tTop>=nearT && tTop<bestT){ vec3 pLocal=roXY+rdXY*tTop; if(dot(pLocal,pLocal)<=radius*radius-1e-6){ float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(!(angleDeg<360.0 && ang>angleDeg)) { bestT=tTop; bestKind=1; } } }
      float tBot=(-halfH-roZ)/rdZ; if(tBot>=nearT && tBot<bestT){ vec3 pLocal=roXY+rdXY*tBot; if(dot(pLocal,pLocal)<=radius*radius-1e-6){ float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(!(angleDeg<360.0 && ang>angleDeg)) { bestT=tBot; bestKind=2; } } }
    }
    if(bestKind>=0 && bestT<INF) reportIntersectionEXT(bestT, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid==1u){
    // cylinder
    vec3 center=uCyl.c0.xyz; vec3 xdir=normalize(uCyl.c1.xyz); vec3 yIn=normalize(uCyl.c2.xyz - xdir*dot(uCyl.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float radius=uCyl.c0.w; float height=uCyl.c1.w; float angleDeg=uCyl.c2.w; float halfH=height*0.5;
    vec3 roW=ro-center; float roZ=dot(roW,axis); float rdZ=dot(rd,axis); vec3 roXY=roW-axis*roZ; vec3 rdXY=rd-axis*rdZ;
    float bestT=INF; int bestKind=-1; // 0 side,1 top,2 bottom
    float a=dot(rdXY,rdXY);
    if(a>0.0){
      float b=2.0*dot(roXY,rdXY); float cc=dot(roXY,roXY)-radius*radius; float disc=b*b-4.0*a*cc;
      if(disc>=0.0){
        float sdisc=sqrt(disc); float inv2a=0.5/a; float cand[2]; cand[0]=(-b - sdisc)*inv2a; cand[1]=(-b + sdisc)*inv2a;
        for(int i=0;i<2;i++){
          float tt=cand[i]; if(tt<nearT || tt>=bestT) continue; float z=roZ+tt*rdZ; if(z<-halfH-EPS || z>halfH+EPS) continue;
          vec3 pLocal=roXY+rdXY*tt; float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(angleDeg<360.0 && ang>angleDeg) continue; bestT=tt; bestKind=0; }
      }
    }
    if(abs(rdZ)>1e-7){
      float tTop=(halfH-roZ)/rdZ; if(tTop>=nearT && tTop<bestT){ vec3 pLocal=roXY+rdXY*tTop; if(dot(pLocal,pLocal)<=radius*radius-1e-6){ float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(!(angleDeg<360.0 && ang>angleDeg)) { bestT=tTop; bestKind=1; } } }
      float tBot=(-halfH-roZ)/rdZ; if(tBot>=nearT && tBot<bestT){ vec3 pLocal=roXY+rdXY*tBot; if(dot(pLocal,pLocal)<=radius*radius-1e-6){ float ang=degrees(atan(pLocal.y,pLocal.x)); if(ang<0.0) ang+=360.0; if(!(angleDeg<360.0 && ang>angleDeg)) { bestT=tBot; bestKind=2; } } }
    }
    if(bestKind>=0 && bestT<INF) reportIntersectionEXT(bestT, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m2.x && gid < meta.m2.x + meta.m2.y){
    // disk (circle) from packed buffer
    uint i = gid - meta.m2.x; uint idx = packOfs.base.x + i*packOfs.stride.x;
    vec4 d0 = packed.data[idx+0]; vec4 d1 = packed.data[idx+1]; vec4 d2 = packed.data[idx+2];
    vec3 center = d0.xyz; float radius = d0.w;
    vec3 xdir = normalize(d1.xyz);
    vec3 yIn = normalize(d2.xyz - xdir*dot(d2.xyz, xdir));
    vec3 normal = normalize(cross(xdir, yIn));
    float denom = dot(rd, normal);
    if (abs(denom) > 1e-6) {
      float tt = dot(center - ro, normal) / denom;
      if (tt >= nearT) {
        vec3 hp = ro + rd * tt;
        vec3 rel = hp - center;
        float px = dot(rel, xdir);
        float py = dot(rel, yIn);
        if (px*px + py*py <= radius*radius + EPS) {
          reportIntersectionEXT(tt, gl_HitKindFrontFacingTriangleEXT);
        }
      }
    }
  } else if(gid >= meta.m2.z && gid < meta.m2.z + meta.m2.w){
    // plane
    float planeY=-2.0; if(abs(rd.y)>1e-6){ float t=(planeY-ro.y)/rd.y; if(t>=nearT){ vec3 hp=ro+rd*t; if(abs(hp.x)<10.0 && abs(hp.z)<10.0) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT); } }
  } else if(gid >= meta.m3.x && gid < meta.m3.x + meta.m3.y){
    // Open cone from packed buffer (no base cap)
    uint i = gid - meta.m3.x; uint idx = packOfs.base.w + i*packOfs.stride.w;
    vec4 e0 = packed.data[idx+0]; vec4 e1 = packed.data[idx+1];
    vec3 center=e0.xyz; float radius=e0.w; vec3 axis=normalize(e1.xyz); float h=e1.w;
    vec3 nDummy; float t = intersect_cone_open(ro, nearT, rd, INF, center, axis, h, radius, nDummy);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m0.z && gid < meta.m0.z + meta.m0.w){
    // ellipse from packed buffer
    uint i = gid - meta.m0.z; uint idx = packOfs.base.y + i*packOfs.stride.y;
    vec4 e0 = packed.data[idx+0]; vec4 e1 = packed.data[idx+1]; vec4 e2 = packed.data[idx+2];
    vec3 center=e0.xyz; float rx=e0.w; vec3 xdir=normalize(e1.xyz); float ry=e1.w; vec3 yIn=normalize(e2.xyz - xdir*dot(e2.xyz,xdir));
    vec3 n; float t = intersect_ellipse(ro, nearT, rd, INF, center, xdir, yIn, rx, ry, n);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m3.z && gid < meta.m3.z + meta.m3.w){
    // line segment from packed buffer (thin cylinder without hemispherical caps)
    uint i = gid - meta.m3.z; uint idx = packOfs.base.z + i*packOfs.stride.z; vec4 l0 = packed.data[idx+0]; vec4 l1 = packed.data[idx+1];
    vec3 p0=l0.xyz; vec3 p1=l1.xyz; float r=l0.w; vec3 n; float tHit;
    bool hit = intersect_line_segment(ro, nearT, rd, INF, p0, p1, r, tHit, n);
    if(hit && tHit>=nearT && tHit<INF) reportIntersectionEXT(tHit, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m4.x && gid < meta.m4.x + meta.m4.y){
    uint patchIndex = gid - meta.m4.x;
    BezierPatchData patch = loadBezierPatch(patchIndex);
    float tHit;
    vec2 uvHit;
  if(intersectBezierPatch(ro, nearT, rd, INF, patch, 0.0, tHit, uvHit)){
      attrUv = clamp(uvHit, vec2(0.0), vec2(1.0));
      reportIntersectionEXT(tHit, gl_HitKindFrontFacingTriangleEXT);
    }
  } else if(gid >= meta.m0.x && gid < meta.m0.x + meta.m0.y) {
    // torus array via SSBO: use SceneMeta.m0.x (base) and m0.y (count)
    uint i = gid - meta.m0.x;
    TorusParam tp = sTorus.items[i];
    vec3 center = tp.t0.xyz; float R = tp.t0.w;
    vec3 xdir = normalize(tp.t1.xyz); float r = tp.t1.w;
    vec3 ydir = normalize(tp.t2.xyz - xdir*dot(tp.t2.xyz,xdir)); float angleDeg = tp.t2.w;
    vec3 n; float t = intersect_torus_wedge(ro, nearT, rd, INF, center, xdir, ydir, R, r, angleDeg, n);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  }
}`;}
export function getClosestHitShader(){return `#version 460 core
#extension GL_EXT_ray_tracing : require
#ifndef WEBRTX_DECL_RT_OUT
#define WEBRTX_DECL_RT_OUT
layout(set=0,binding=0,std430) buffer RtOut { vec4 data[]; } rtOut;
#endif
#ifndef WEBRTX_DECL_PACKED_OFFSETS
#define WEBRTX_DECL_PACKED_OFFSETS
layout(set=0,binding=9) uniform PackedOffsets { uvec4 base; uvec4 stride; } packOfs;
#endif
#ifndef WEBRTX_DECL_PACKED_BUFFER
#define WEBRTX_DECL_PACKED_BUFFER
layout(set=0,binding=13,std430) buffer PackedPrims { vec4 data[]; } packed;
#endif
#ifndef WEBRTX_DECL_CAM
#define WEBRTX_DECL_CAM
layout(set=0,binding=1) uniform Cam { vec4 c0; vec4 c1; vec4 c2; vec4 c3; } cam;
#endif
#ifndef WEBRTX_DECL_META
#define WEBRTX_DECL_META
layout(set=0,binding=2) uniform SceneMeta { uvec4 m0; uvec4 m1; uvec4 m2; uvec4 m3; uvec4 m4; } meta;
#endif
#ifndef WEBRTX_DECL_CYL_ARRAY
#define WEBRTX_DECL_CYL_ARRAY
struct CylParam { vec4 c0; vec4 c1; vec4 c2; };
layout(set=0,binding=12,std430) buffer CylParams { CylParam items[]; } sCyl;
#endif
#ifndef WEBRTX_DECL_SPHERE_ARRAY
#define WEBRTX_DECL_SPHERE_ARRAY
struct SphereParam { vec4 s0; };
layout(set=0,binding=11,std430) buffer SphereParams { SphereParam items[]; } sSphere;
#endif
#define WEBRTX_NO_CIRCLE_ELLIPSE_LINE_CONE_ARRAYS 1
#ifndef WEBRTX_DECL_TORUS_ARRAY
#define WEBRTX_DECL_TORUS_ARRAY
struct TorusParam { vec4 t0; vec4 t1; vec4 t2; };
layout(set=0,binding=10,std430) buffer TorusParams { TorusParam items[]; } sTorus;
#endif
#ifndef WEBRTX_DECL_CYL
#define WEBRTX_DECL_CYL
layout(set=0,binding=4) uniform Cylinder { vec4 c0; vec4 c1; vec4 c2; } uCyl;
#endif
#ifndef WEBRTX_DECL_CIRCLE
#define WEBRTX_DECL_CIRCLE
layout(set=0,binding=5) uniform Circle { vec4 d0; vec4 d1; vec4 d2; } uCircle;
#endif
#ifndef WEBRTX_DECL_ELLIPSE
#define WEBRTX_DECL_ELLIPSE
layout(set=0,binding=7) uniform Ellipse { vec4 e0; vec4 e1; vec4 e2; } uEllipse;
#endif
#ifndef WEBRTX_DECL_LINE
#define WEBRTX_DECL_LINE
layout(set=0,binding=8) uniform Line { vec4 l0; vec4 l1; } uLine;
#endif
#ifndef WEBRTX_DECL_CONE
#define WEBRTX_DECL_CONE
layout(set=0,binding=6) uniform Cone { vec4 e0; vec4 e1; } uCone;
#endif
#ifndef WEBRTX_DECL_BEZIER_BUFFER
#define WEBRTX_DECL_BEZIER_BUFFER
layout(set=0,binding=14,std430) buffer BezierPatches { vec4 data[]; } sBezier;
#endif
hitAttributeEXT vec2 attrUv;

const uint BEZIER_VEC4_PER_PATCH = 18u;

struct BezierPatchData {
  vec4 cp[16];
  vec3 boundsMin;
  vec3 boundsMax;
  float maxDepth;
  float pixelEpsilon;
};

int bezierIdx(int u, int v){ return u * 4 + v; }

vec3 bezierGet(in vec4 cp[16], int u, int v){ return cp[bezierIdx(u, v)].xyz; }

BezierPatchData loadBezierPatch(uint index){
  uint base = index * BEZIER_VEC4_PER_PATCH;
  BezierPatchData result;
  for(uint i=0u;i<16u;++i){ result.cp[i] = sBezier.data[base + i]; }
  vec4 bounds0 = sBezier.data[base + 16u];
  vec4 bounds1 = sBezier.data[base + 17u];
  result.boundsMin = bounds0.xyz;
  result.boundsMax = bounds1.xyz;
  result.maxDepth = bounds0.w;
  result.pixelEpsilon = bounds1.w;
  return result;
}

void bezierEval(in vec4 cp[16], float u, float v, out vec3 P, out vec3 dPu, out vec3 dPv){
  vec3 Cv[4];
  for(int i=0;i<4;++i){
    vec3 p0 = bezierGet(cp, i, 0);
    vec3 p1 = bezierGet(cp, i, 1);
    vec3 p2 = bezierGet(cp, i, 2);
    vec3 p3 = bezierGet(cp, i, 3);
    vec3 a = mix(p0, p1, v);
    vec3 b = mix(p1, p2, v);
    vec3 c = mix(p2, p3, v);
    vec3 d = mix(a, b, v);
    vec3 e = mix(b, c, v);
    Cv[i] = mix(d, e, v);
  }
  vec3 A = mix(Cv[0], Cv[1], u);
  vec3 B = mix(Cv[1], Cv[2], u);
  vec3 C = mix(Cv[2], Cv[3], u);
  vec3 D = mix(A, B, u);
  vec3 E = mix(B, C, u);
  P = mix(D, E, u);
  dPu = 3.0 * (E - D);

  vec3 Ru[4];
  for(int j=0;j<4;++j){
    vec3 p0 = bezierGet(cp, 0, j);
    vec3 p1 = bezierGet(cp, 1, j);
    vec3 p2 = bezierGet(cp, 2, j);
    vec3 p3 = bezierGet(cp, 3, j);
    vec3 a = mix(p0, p1, u);
    vec3 b = mix(p1, p2, u);
    vec3 c = mix(p2, p3, u);
    vec3 d = mix(a, b, u);
    vec3 e = mix(b, c, u);
    Ru[j] = mix(d, e, u);
  }
  vec3 A2 = mix(Ru[0], Ru[1], v);
  vec3 B2 = mix(Ru[1], Ru[2], v);
  vec3 C2 = mix(Ru[2], Ru[3], v);
  vec3 D2 = mix(A2, B2, v);
  vec3 E2 = mix(B2, C2, v);
  dPv = 3.0 * (E2 - D2);
}

void main(){
  const float EPS=1e-5;
  uvec2 pix=gl_LaunchIDEXT.xy; uvec2 dim=gl_LaunchSizeEXT.xy; uint gid=gl_GeometryIndexEXT; float t=gl_HitTEXT; vec3 ro=gl_WorldRayOriginEXT; vec3 rd=gl_WorldRayDirectionEXT; vec3 hp=ro+rd*t; vec3 col=vec3(0.6); vec3 light=normalize(vec3(0.3,0.7,0.5));
  if(gid >= meta.m1.x && gid < meta.m1.x + meta.m1.y){
    SphereParam sp = sSphere.items[gid - meta.m1.x];
    vec3 n=normalize(hp-sp.s0.xyz); float diff=max(dot(n,light),0.0); col=vec3(1.0,0.6,0.2)*(0.2+0.8*diff);
  } else if(gid >= meta.m1.z && gid < meta.m1.z + meta.m1.w){
    // cylinder shading via SSBO params
    CylParam cylp = sCyl.items[gid - meta.m1.z];
    vec3 center=cylp.c0.xyz; vec3 xdir=normalize(cylp.c1.xyz); vec3 yIn=normalize(cylp.c2.xyz - xdir*dot(cylp.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
  float height=cylp.c1.w; float halfH=height*0.5; vec3 lpc=hp-center; float z=dot(lpc,axis); vec3 n; if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpc-axis*z; n=normalize(radial); } float diff=max(dot(n,light),0.0); col=mix(vec3(0.07,0.12,0.1), vec3(0.2,0.85,0.55), diff);
  } else if(gid==1u){
    // cylinder
  vec3 center=uCyl.c0.xyz; vec3 xdir=normalize(uCyl.c1.xyz); vec3 yIn=normalize(uCyl.c2.xyz - xdir*dot(uCyl.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0); float height=uCyl.c1.w; float halfH=height*0.5; vec3 lpu=hp-center; float z=dot(lpu,axis); vec3 n; if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpu-axis*z; n=normalize(radial); } float diff=max(dot(n,light),0.0); col=mix(vec3(0.07,0.12,0.1), vec3(0.2,0.85,0.55), diff);
  } else if(gid >= meta.m2.x && gid < meta.m2.x + meta.m2.y){
    // disk (circle)
    uint i = gid - meta.m2.x; uint idx = packOfs.base.x + i*packOfs.stride.x; vec4 d1 = packed.data[idx+1]; vec4 d2 = packed.data[idx+2];
    vec3 xdir = normalize(d1.xyz);
    vec3 yIn = normalize(d2.xyz - xdir*dot(d2.xyz, xdir));
    vec3 n = normalize(cross(xdir, yIn));
    float diff = max(dot(n, light), 0.0);
    col = vec3(0.4,0.4,1.0) * (0.2 + 0.8*diff);
  } else if(gid >= meta.m2.z && gid < meta.m2.z + meta.m2.w){
    // plane
    float scale=1.5; float cx=floor(hp.x/scale); float cz=floor(hp.z/scale); float checker=mod(cx+cz,2.0); col=mix(vec3(0.9), vec3(0.2), checker);
  } else if(gid >= meta.m3.x && gid < meta.m3.x + meta.m3.y){
    // Open cone side normal only (no base cap)
    uint i = gid - meta.m3.x; uint idx = packOfs.base.w + i*packOfs.stride.w; vec4 e0 = packed.data[idx+0]; vec4 e1 = packed.data[idx+1];
    vec3 center=e0.xyz; float radius=e0.w; vec3 axis=normalize(e1.xyz); float h=e1.w;
    vec3 apex = center - axis*(0.5*h);
    vec3 to_apex = hp - apex; float proj = dot(to_apex, axis);
    vec3 radial = to_apex - axis*proj; float rlen = length(radial);
    vec3 n;
    if(rlen < 1e-6){ n = axis; } else {
      vec3 rdir = radial / rlen; float hyp = length(vec2(h, radius)); float cos_a = (hyp>1e-8)? (h/hyp) : 0.9999; float sin_a = (hyp>1e-8)? (radius/hyp) : 0.012;
      n = normalize(rdir * cos_a + axis * sin_a);
    }
    float diff=max(dot(n,light),0.0); col=mix(vec3(0.12,0.10,0.08), vec3(0.9,0.7,0.3), diff);
  } else if(gid >= meta.m0.z && gid < meta.m0.z + meta.m0.w){
    // ellipse shading similar to disk, normal is plane normal
    uint i = gid - meta.m0.z; uint idx = packOfs.base.y + i*packOfs.stride.y; vec4 e1 = packed.data[idx+1]; vec4 e2 = packed.data[idx+2];
    vec3 xdir=normalize(e1.xyz); vec3 ydir=normalize(e2.xyz - xdir*dot(e2.xyz,xdir));
    vec3 n=normalize(cross(xdir, ydir)); float diff=max(dot(n,light),0.0); col=vec3(0.8,0.3,0.9)*(0.2+0.8*diff);
  } else if(gid >= meta.m3.z && gid < meta.m3.z + meta.m3.w){
    // line segment shading similar to cylinder
    uint i = gid - meta.m3.z; uint idx = packOfs.base.z + i*packOfs.stride.z; vec4 l0 = packed.data[idx+0]; vec4 l1 = packed.data[idx+1];
    vec3 p0=l0.xyz; vec3 p1=l1.xyz; float r=l0.w;
    vec3 center=0.5*(p0+p1); vec3 axis=normalize(p1-p0); float h=length(p1-p0); float halfH=0.5*h;
  vec3 lpl=hp-center; float z=dot(lpl,axis); vec3 n;
    const float EPS=1e-5;
  if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpl-axis*z; n=normalize(radial); }
    float diff=max(dot(n,light),0.0); col=mix(vec3(0.1,0.1,0.12), vec3(0.7,0.7,0.9), diff);
  } else if(gid >= meta.m4.x && gid < meta.m4.x + meta.m4.y){
    uint patchIndex = gid - meta.m4.x;
    BezierPatchData patch = loadBezierPatch(patchIndex);
    vec2 uv = clamp(attrUv, vec2(0.0), vec2(1.0));
    vec3 dPu;
    vec3 dPv;
    bezierEval(patch.cp, uv.x, uv.y, hp, dPu, dPv);
    vec3 n = normalize(cross(dPu, dPv));
    if(dot(n, rd) > 0.0) n = -n;
    float diff = max(dot(n, light), 0.0);
    col = mix(vec3(0.05,0.07,0.1), vec3(0.95,0.55,0.25), diff);
  } else if(gid >= meta.m0.x && gid < meta.m0.x + meta.m0.y) {
    // torus shading from SSBO params
    uint i = gid - meta.m0.x; TorusParam tp = sTorus.items[i];
    vec3 xdir=normalize(tp.t1.xyz); vec3 ydir=normalize(tp.t2.xyz - xdir*dot(tp.t2.xyz,xdir)); vec3 zdir=normalize(cross(xdir, ydir)); vec3 c=tp.t0.xyz; float R=tp.t0.w;
  vec3 lpt = vec3(dot(hp-c,xdir), dot(hp-c,ydir), dot(hp-c,zdir)); float s=max(length(lpt.xz),1e-12); float aa=1.0-(R/s); vec3 nloc=vec3(aa*lpt.x, lpt.y, aa*lpt.z); vec3 n=normalize(nloc.x*xdir + nloc.y*ydir + nloc.z*zdir);
    float diff=max(dot(n,light),0.0); col=mix(vec3(0.08,0.08,0.1), vec3(0.9,0.5,0.2), diff);
  }
  rtOut.data[pix.y*dim.x+pix.x]=vec4(col,1.0);
}`;}
