import { getRayGenShader, getMissShader } from './analytic_shaders';

export function getRayGenShaderNoBezier(){
  return getRayGenShader();
}

export function getMissShaderNoBezier(){
  return getMissShader();
}

export function getIntersectionShaderNoBezier(){return `#version 460 core
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

void main(){
  vec3 ro = gl_WorldRayOriginEXT;
  vec3 rd = gl_WorldRayDirectionEXT;
  float nearT = cam.c3.z;
  float INF = 1e38; const float EPS = 1e-5; uint gid = gl_GeometryIndexEXT;
  if(gid >= meta.m1.x && gid < meta.m1.x + meta.m1.y){
    uint i = gid - meta.m1.x; SphereParam sp = sSphere.items[i];
    vec3 c=sp.s0.xyz; float R=sp.s0.w; vec3 oc=ro-c; float b=dot(oc,rd); float c2=dot(oc,oc)-R*R; float disc=b*b-c2; if(disc>=0.0){ float s=sqrt(disc); float t0=-b-s; float t1=-b+s; float tt=t0; if(tt<nearT) tt=t1; if(tt>=nearT && tt<INF) reportIntersectionEXT(tt, gl_HitKindFrontFacingTriangleEXT); }
  } else if(gid >= meta.m1.z && gid < meta.m1.z + meta.m1.w){
    uint i = gid - meta.m1.z;
    CylParam cylp = sCyl.items[i];
    vec3 center=cylp.c0.xyz; vec3 xdir=normalize(cylp.c1.xyz);
    vec3 yIn=normalize(cylp.c2.xyz - xdir*dot(cylp.c2.xyz,xdir));
    vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float radius=cylp.c0.w; float height=cylp.c1.w; float angleDeg=cylp.c2.w; float halfH=height*0.5;
    vec3 roW=ro-center; float roZ=dot(roW,axis); float rdZ=dot(rd,axis); vec3 roXY=roW-axis*roZ; vec3 rdXY=rd-axis*rdZ;
    float bestT=INF; int bestKind=-1;
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
    vec3 center=uCyl.c0.xyz; vec3 xdir=normalize(uCyl.c1.xyz); vec3 yIn=normalize(uCyl.c2.xyz - xdir*dot(uCyl.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float radius=uCyl.c0.w; float height=uCyl.c1.w; float angleDeg=uCyl.c2.w; float halfH=height*0.5;
    vec3 roW=ro-center; float roZ=dot(roW,axis); float rdZ=dot(rd,axis); vec3 roXY=roW-axis*roZ; vec3 rdXY=rd-axis*rdZ;
    float bestT=INF; int bestKind=-1;
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
    float planeY=-2.0; if(abs(rd.y)>1e-6){ float t=(planeY-ro.y)/rd.y; if(t>=nearT){ vec3 hp=ro+rd*t; if(abs(hp.x)<10.0 && abs(hp.z)<10.0) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT); } }
  } else if(gid >= meta.m3.x && gid < meta.m3.x + meta.m3.y){
    uint i = gid - meta.m3.x; uint idx = packOfs.base.w + i*packOfs.stride.w;
    vec4 e0 = packed.data[idx+0]; vec4 e1 = packed.data[idx+1];
    vec3 center=e0.xyz; float radius=e0.w; vec3 axis=normalize(e1.xyz); float h=e1.w;
    vec3 nDummy; float t = intersect_cone_open(ro, nearT, rd, INF, center, axis, h, radius, nDummy);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m0.z && gid < meta.m0.z + meta.m0.w){
    uint i = gid - meta.m0.z; uint idx = packOfs.base.y + i*packOfs.stride.y;
    vec4 e0 = packed.data[idx+0]; vec4 e1 = packed.data[idx+1]; vec4 e2 = packed.data[idx+2];
    vec3 center=e0.xyz; float rx=e0.w; vec3 xdir=normalize(e1.xyz); float ry=e1.w; vec3 yIn=normalize(e2.xyz - xdir*dot(e2.xyz,xdir));
    vec3 n; float t = intersect_ellipse(ro, nearT, rd, INF, center, xdir, yIn, rx, ry, n);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m3.z && gid < meta.m3.z + meta.m3.w){
    uint i = gid - meta.m3.z; uint idx = packOfs.base.z + i*packOfs.stride.z; vec4 l0 = packed.data[idx+0]; vec4 l1 = packed.data[idx+1];
    vec3 p0=l0.xyz; vec3 p1=l1.xyz; float r=l0.w; vec3 n; float tHit;
    bool hit = intersect_line_segment(ro, nearT, rd, INF, p0, p1, r, tHit, n);
    if(hit && tHit>=nearT && tHit<INF) reportIntersectionEXT(tHit, gl_HitKindFrontFacingTriangleEXT);
  } else if(gid >= meta.m0.x && gid < meta.m0.x + meta.m0.y) {
    uint i = gid - meta.m0.x;
    TorusParam tp = sTorus.items[i];
    vec3 center = tp.t0.xyz; float R = tp.t0.w;
    vec3 xdir = normalize(tp.t1.xyz); float r = tp.t1.w;
    vec3 ydir = normalize(tp.t2.xyz - xdir*dot(tp.t2.xyz,xdir)); float angleDeg = tp.t2.w;
    vec3 n; float t = intersect_torus_wedge(ro, nearT, rd, INF, center, xdir, ydir, R, r, angleDeg, n);
    if(t>=nearT && t<INF) reportIntersectionEXT(t, gl_HitKindFrontFacingTriangleEXT);
  }
}`;}

export function getClosestHitShaderNoBezier(){return `#version 460 core
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

void main(){
  const float EPS=1e-5;
  uvec2 pix=gl_LaunchIDEXT.xy; uvec2 dim=gl_LaunchSizeEXT.xy; uint gid=gl_GeometryIndexEXT; float t=gl_HitTEXT; vec3 ro=gl_WorldRayOriginEXT; vec3 rd=gl_WorldRayDirectionEXT; vec3 hp=ro+rd*t; vec3 col=vec3(0.6); vec3 light=normalize(vec3(0.3,0.7,0.5));
  if(gid >= meta.m1.x && gid < meta.m1.x + meta.m1.y){
    SphereParam sp = sSphere.items[gid - meta.m1.x];
    vec3 n=normalize(hp-sp.s0.xyz); float diff=max(dot(n,light),0.0); col=vec3(1.0,0.6,0.2)*(0.2+0.8*diff);
  } else if(gid >= meta.m1.z && gid < meta.m1.z + meta.m1.w){
    CylParam cylp = sCyl.items[gid - meta.m1.z];
    vec3 center=cylp.c0.xyz; vec3 xdir=normalize(cylp.c1.xyz); vec3 yIn=normalize(cylp.c2.xyz - xdir*dot(cylp.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float height=cylp.c1.w; float halfH=height*0.5; vec3 lpc=hp-center; float z=dot(lpc,axis); vec3 n; if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpc-axis*z; n=normalize(radial); } float diff=max(dot(n,light),0.0); col=mix(vec3(0.07,0.12,0.1), vec3(0.2,0.85,0.55), diff);
  } else if(gid==1u){
    vec3 center=uCyl.c0.xyz; vec3 xdir=normalize(uCyl.c1.xyz); vec3 yIn=normalize(uCyl.c2.xyz - xdir*dot(uCyl.c2.xyz,xdir)); vec3 axis=normalize(cross(xdir,yIn)); if(length(axis)<1e-6) axis=vec3(0,1,0);
    float height=uCyl.c1.w; float halfH=height*0.5; vec3 lpu=hp-center; float z=dot(lpu,axis); vec3 n; if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpu-axis*z; n=normalize(radial); } float diff=max(dot(n,light),0.0); col=mix(vec3(0.07,0.12,0.1), vec3(0.2,0.85,0.55), diff);
  } else if(gid >= meta.m2.x && gid < meta.m2.x + meta.m2.y){
    uint i = gid - meta.m2.x; uint idx = packOfs.base.x + i*packOfs.stride.x; vec4 d1 = packed.data[idx+1]; vec4 d2 = packed.data[idx+2];
    vec3 xdir = normalize(d1.xyz);
    vec3 yIn = normalize(d2.xyz - xdir*dot(d2.xyz, xdir));
    vec3 n = normalize(cross(xdir, yIn));
    float diff = max(dot(n, light), 0.0);
    col = vec3(0.4,0.4,1.0) * (0.2 + 0.8*diff);
  } else if(gid >= meta.m2.z && gid < meta.m2.z + meta.m2.w){
    float scale=1.5; float cx=floor(hp.x/scale); float cz=floor(hp.z/scale); float checker=mod(cx+cz,2.0); col=mix(vec3(0.9), vec3(0.2), checker);
  } else if(gid >= meta.m3.x && gid < meta.m3.x + meta.m3.y){
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
    uint i = gid - meta.m0.z; uint idx = packOfs.base.y + i*packOfs.stride.y; vec4 e1 = packed.data[idx+1]; vec4 e2 = packed.data[idx+2];
    vec3 xdir=normalize(e1.xyz); vec3 ydir=normalize(e2.xyz - xdir*dot(e2.xyz,xdir));
    vec3 n=normalize(cross(xdir, ydir)); float diff=max(dot(n,light),0.0); col=vec3(0.8,0.3,0.9)*(0.2+0.8*diff);
  } else if(gid >= meta.m3.z && gid < meta.m3.z + meta.m3.w){
    uint i = gid - meta.m3.z; uint idx = packOfs.base.z + i*packOfs.stride.z; vec4 l0 = packed.data[idx+0]; vec4 l1 = packed.data[idx+1];
    vec3 p0=l0.xyz; vec3 p1=l1.xyz; float r=l0.w;
    vec3 center=0.5*(p0+p1); vec3 axis=normalize(p1-p0); float h=length(p1-p0); float halfH=0.5*h;
    vec3 lpl=hp-center; float z=dot(lpl,axis); vec3 n;
    if(z>halfH-EPS) n=axis; else if(z<-halfH+EPS) n=-axis; else { vec3 radial=lpl-axis*z; n=normalize(radial); }
    float diff=max(dot(n,light),0.0); col=mix(vec3(0.1,0.1,0.12), vec3(0.7,0.7,0.9), diff);
  } else if(gid >= meta.m0.x && gid < meta.m0.x + meta.m0.y) {
    uint i = gid - meta.m0.x; TorusParam tp = sTorus.items[i];
    vec3 xdir=normalize(tp.t1.xyz); vec3 ydir=normalize(tp.t2.xyz - xdir*dot(tp.t2.xyz,xdir)); vec3 zdir=normalize(cross(xdir, ydir)); vec3 c=tp.t0.xyz; float R=tp.t0.w;
    vec3 lpt = vec3(dot(hp-c,xdir), dot(hp-c,ydir), dot(hp-c,zdir)); float s=max(length(lpt.xz),1e-12); float aa=1.0-(R/s); vec3 nloc=vec3(aa*lpt.x, lpt.y, aa*lpt.z); vec3 n=normalize(nloc.x*xdir + nloc.y*ydir + nloc.z*zdir);
    float diff=max(dot(n,light),0.0); col=mix(vec3(0.08,0.08,0.1), vec3(0.9,0.5,0.2), diff);
  }
  rtOut.data[pix.y*dim.x+pix.x]=vec4(col,1.0);
}`;}
