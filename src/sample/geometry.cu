#include <optix.h>

#include <sutil/vec_math.h>

#include "scene.h"
#include "helpers.h"

#include "stdio.h"

#define float3_as_uints( u ) __float_as_uint( u.x ), __float_as_uint( u.y ), __float_as_uint( u.z )

extern "C" __global__ void __intersection__sphere()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 O      = ray_orig - hit_group_data->geometry_data.getSphere().center;
    const float  l      = 1.0f / length( ray_dir );
    const float3 D      = ray_dir * l;
    const float  radius = hit_group_data->geometry_data.getSphere().radius;

    float b    = dot( O, D );
    float c    = dot( O, O ) - radius * radius;
    float disc = b * b - c;
    if( disc > 0.0f )
    {
        float sdisc        = sqrtf( disc );
        float root1        = ( -b - sdisc );
        float root11       = 0.0f;
        bool  check_second = true;

        const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

        if( do_refine )
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b         = dot( O1, D );
            c         = dot( O1, O1 ) - radius * radius;
            disc      = b * b - c;

            if( disc > 0.0f )
            {
                sdisc  = sqrtf( disc );
                root11 = ( -b - sdisc );
            }
        }

        float  t;
        float3 normal;
        t = ( root1 + root11 ) * l;
        if( t > ray_tmin && t < ray_tmax )
        {
            normal = ( O + ( root1 + root11 ) * D ) / radius;
            if( optixReportIntersection( t, 0, float3_as_uints( normal ), __float_as_uint( radius ) ) )
                check_second = false;
        }

        if( check_second )
        {
            float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
            t           = root2 * l;
            normal      = ( O + root2 * D ) / radius;
            if( t > ray_tmin && t < ray_tmax )
                optixReportIntersection( t, 0, float3_as_uints( normal ), __float_as_uint( radius ) );
        }
        
    }
}

extern "C" __global__ void __intersection__torus()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    const float3 center   = hit_group_data->geometry_data.getTorus().center;
    const float  R1       = hit_group_data->geometry_data.getTorus().major_radius;
    const float  r1       = hit_group_data->geometry_data.getTorus().minor_radius;
	const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getTorus().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getTorus().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getTorus().rot_z) 
    );

    // Tranform Ray into Torus Local Space
    float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);

    // float3 margin_ray_orig = center + R1 + r1 * 2;
    const float3 ray_dir = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    // Optimized Ray-Bounding Ellipsoid Intersection Discriminant Calculation
    float scale_factor = 1.1f;
    float BBB = (R1 + r1) * sqrt(r1 / (2.0f * R1 + r1));
    float3 r2 = make_float3((R1 + r1) * (R1 + r1) * scale_factor, BBB * BBB * scale_factor, (R1 + r1) * (R1 + r1) * scale_factor);
    float aaa = dot(ray_dir, ray_dir / r2);
    float bbb = dot(ray_orig, ray_dir / r2);
    float ccc = dot(ray_orig, ray_orig / r2);
    float discriminant = bbb * bbb - aaa * (ccc - 1.0f);


    if (!(discriminant < 0.0f))
    {
        // Compute Ray-Bounding Ellipsoid Intersection
        float sqrt_disc = sqrtf(discriminant);
        float t0 = (-bbb - sqrt_disc) / aaa;
        float t1 = (-bbb + sqrt_disc) / aaa;
        float t_hit = FLT_MAX;
        if (t0 > ray_tmin && t0 < t_hit)
            t_hit = t0;
        if (t1 > ray_tmin && t1 < t_hit)
            t_hit = t1;
        if (t_hit < ray_tmin || t_hit > ray_tmax) {
            return;
        }

        // Ray Origin Translation for solving FPP
        float3 new_ray_orig = ray_orig + (t_hit - 0.5f) * ray_dir;
        float3 original_ray_orig = ray_orig;
        ray_orig = new_ray_orig;

        // Set up the coefficients of a quartic equation for the Ray-Torus intersection
        float A1 = dot(ray_dir, ray_dir);
        float B1 = 2.0f * dot(ray_orig, ray_dir);
        float C1 = dot(ray_orig, ray_orig) + R1 * R1 - r1 * r1;
        float D1 = 4.0f * R1 * R1 * (ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z);
        float E1 = 8.0f * R1 * R1 * (ray_orig.x * ray_dir.x + ray_orig.z * ray_dir.z);
        float F1 = 4.0f * R1 * R1 * (ray_orig.x * ray_orig.x + ray_orig.z * ray_orig.z);

        float roots[4] = { 0, 0, 0, 0 };
        float coeff[5]{ 0, 0, 0, 0, 0 };
        coeff[4] = A1 * A1;
        coeff[3] = 2.0f * A1 * B1;
        coeff[2] = 2.0f * A1 * C1 + B1 * B1 - D1;
        coeff[1] = 2.0f * B1 * C1 - E1;
        coeff[0] = C1 * C1 - F1;


        //------------------------------------------------------------------------------
        // Analytical (Ferrari's method)
        //------------------------------------------------------------------------------
        int     num_real_roots;
        int     i, num_roots;
        float  coeffs[4];
        float  z, u, v, sub;
        float  A, B, C, D;
        float  squared_A, p, q, r;

        // x^4 + Ax^3 + Bx^2 + Cx + D = 0 
        A = coeff[3] / coeff[4];			// b/a
        B = coeff[2] / coeff[4];			// c/a
        C = coeff[1] / coeff[4];			// d/a
        D = coeff[0] / coeff[4];			// e/a

        // Substitute x = y - A/4 to eliminate cubic term
        // x^4 + px^2 + qx + r = 0 
        squared_A = A * A;
        p = -3.0f / 8.0f * squared_A + B;
        q = 1.0f / 8.0f * squared_A * A - 1.0f / 2.0f * A * B + C;
        r = -3.0f / 256.0f * squared_A * squared_A + 1.0f / 16.0f * squared_A * B - 1.0f / 4.0f * A * C + D;

        if (is_zero_float(r)) {     // No absolute term when r = 0, which means y(y^3 + py + q) = 0 
            coeffs[0] = q;
            coeffs[1] = p;
            coeffs[2] = 0;
            coeffs[3] = 1;

            num_roots = solve_cubic(coeffs, roots);

            roots[num_roots++] = 0;
        }
        else {      // r != 0, thus solves resolvent cubic equation

            // z^3 - 1/2pz^2 - rz + 1/2rp - 1/8q^2 = 0
            coeffs[0] = 1.0f / 2.0f * r * p - 1.0f / 8.0f * q * q;
            coeffs[1] = -r;
            coeffs[2] = -1.0f / 2.0f * p;
            coeffs[3] = 1.0f;
            solve_cubic(coeffs, roots);		// Find 1 real root

            // Using found root z, divide into 2 quadratic equations
            // y^2 + uy + v = 0
            // y^2 - uy - v = 0
            // u = sqrt(z^2 - r)
            // v = sqrt(2z  - p)
            z = roots[0];
            u = z * z - r;
            v = 2.0f * z - p;

            //if (u < 0.0f && -0.1 < u) printf(" u : %f, roots[0] : %f, r : %f\n", u, roots[0], r);
            // Prevents sqrt(negative value)
            if (is_zero_float(u))		u = 0;
            else if (u > 0)		u = sqrt(u);
            else {
                num_real_roots = 0;
            }

            if (is_zero_float(v))		v = 0;
            else if (v > 0)		v = sqrt(v);
            else
            {
                num_real_roots = 0;
            }

            // First quadratic equation
            coeffs[0] = z - u;
            coeffs[1] = q < 0 ? -v : v;
            coeffs[2] = 1;
            num_roots = solve_quadratic(coeffs, roots);

            // Second quadratic equation
            coeffs[0] = z + u;
            coeffs[1] = q < 0 ? v : -v;
            coeffs[2] = 1;
            num_roots += solve_quadratic(coeffs, roots + num_roots);
        }

        // Resubstitute to compute x
        sub = 1.0f / 4.0f * A;
        for (i = 0; i < num_roots; ++i) roots[i] -= sub;

        num_real_roots = num_roots;

        //------------------------------------------------------------------------------
        // Intersection Calculation
        //------------------------------------------------------------------------------
        bool	intersected = false;
        float 	t = FLT_MAX;

        if (num_real_roots > 0)
        {
            // Find the smallest root greater than 0.001
            for (int j = 0; j < num_real_roots; j++)
            {
                // For 1/4 Torus
                float3 a = make_float3(ray_orig.x, ray_orig.y, ray_orig.z);
                float3 b = make_float3(ray_dir.x * roots[j], ray_dir.y * roots[j], ray_dir.z * roots[j]);
                float3 p = make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

                // Torus Angle Devision
                float angle_deg = hit_group_data->geometry_data.getTorus().angle; // degree 단위
                float angle = angle_deg * (3.141592f / 180.0f); // 라디안으로 변환

                // p의 (x,z) 평면상의 극각 계산 (atan2는 결과가 라디안임)
                float theta = atan2(p.z, p.x);

                // 결과가 음수라면 0~2π 범위로 보정
                if (theta < 0.0f) 
                {
                    theta += 2.0f * 3.14159f;
                }

                //if (roots[j] > 0.001f && roots[j] < FLT_MAX && theta >= 0.0f && theta <= angle)
                if (roots[j] < FLT_MAX && theta >= 0.0f && theta <= angle)
                {
                    intersected = true;
                    if (roots[j] < t)
                        t = roots[j];
                }
            }
            if (intersected) {
                // Ray Origin Inverse Translation
                float3 a = make_float3(ray_orig.x, ray_orig.y, ray_orig.z);
                float3 b = make_float3(ray_dir.x * t, ray_dir.y * t, ray_dir.z * t);
                float3 p = make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

                float aa = 1.0f - (R1 / sqrtf(p.x * p.x + p.z * p.z));
                float3 normal;
                normal.x = aa * p.x;
                normal.y = p.y;
                normal.z = aa * p.z;

                normal = rotate_point(normal, rotation);

                // Compute Our Real t
                t = (p.x - original_ray_orig.x) / ray_dir.x;

                float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                normal.x = normal.x / length;
                normal.y = normal.y / length;
                normal.z = normal.z / length;

                optixReportIntersection(t, 0, float3_as_uints(normal));
            }
        }
    }
}

extern "C" __global__ void __intersection__cylinder()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    const float3 center   = hit_group_data->geometry_data.getCylinder().center;
    const float  radius   = (float)hit_group_data->geometry_data.getCylinder().radius;
    const float  height   = (float)hit_group_data->geometry_data.getCylinder().height;
    const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getCylinder().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getCylinder().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getCylinder().rot_z)
    );

    // Tranform Ray into Cylinder Local Space
    float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);
    const float3 ray_dir = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    // Ray-Bounding Cube Intersection
    float c_width = radius + 0.5f;
    float c_depth = radius + 0.5f;
    float c_height = height + 0.5f;
    float3 half_extents = make_float3(c_width, c_height, c_depth);

    const float3 cube_min = make_float3(-half_extents.x, -half_extents.y, -half_extents.z);
    const float3 cube_max = make_float3(half_extents.x, half_extents.y, half_extents.z);

    float3 inv_dir = make_float3(1.0f) / ray_dir;
    float3 t0 = (cube_min - ray_orig) * inv_dir;
    float3 t1 = (cube_max - ray_orig) * inv_dir;

    float3 tsmaller = make_float3(fminf(t0.x, t1.x),
        fminf(t0.y, t1.y),
        fminf(t0.z, t1.z));
    float3 tbigger = make_float3(fmaxf(t0.x, t1.x),
        fmaxf(t0.y, t1.y),
        fmaxf(t0.z, t1.z));

    float t_enter = fmaxf(fmaxf(tsmaller.x, tsmaller.y), fmaxf(tsmaller.z, ray_tmin));
    float t_exit = fminf(fminf(tbigger.x, tbigger.y), tbigger.z);

    if (t_enter > t_exit || t_enter > ray_tmax)
        return;

    float t_hit = t_enter;
    if (t_hit < ray_tmin)
        t_hit = t_exit;
    if (t_hit < ray_tmin || t_hit > ray_tmax)
        return;

    // Ray Origin Translation for solving FPP
    float3 new_ray_orig = ray_orig + (t_hit - 0.5f) * ray_dir;
    float3 original_ray_orig = ray_orig;
    ray_orig = new_ray_orig;

    // Ray-Cylinder Intersection
    float a = (ray_dir.x * ray_dir.x) + (ray_dir.z * ray_dir.z);
    float b = 2 * (ray_dir.x * ray_orig.x + ray_dir.z * ray_orig.z);
    float c = (ray_orig.x * ray_orig.x + ray_orig.z * ray_orig.z - (radius * radius));

    bool  is_real_root = true;
    float discriminant = b * b - 4.0f * (a * c);

    if (discriminant < 0.0f) is_real_root = false;

    if (is_real_root)
    {
        float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
        float t2 = (-b + sqrt(discriminant)) / (2.0f * a);

        // Sort t1 <= t2
        if (t1 > t2)
        {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }

        float3 p1 = ray_orig + t1 * ray_dir;
        float3 p2 = ray_orig + t2 * ray_dir;

        if (t1 > ray_tmin && t1 < ray_tmax && (p1.y >= -0.5f * height) && (p1.y <= 0.5f * height))
        {
            float3 normal = make_float3(p1.x, 0.0f, p1.z);
            normal = rotate_point(normal, rotation);
            normal = normalize(normal);

            // Compute Our Real t
            float real_t = (p1.x - original_ray_orig.x) / ray_dir.x;
            real_t -= 0.0001f;
            optixReportIntersection(real_t, 0, float3_as_uints(normal));
        }
        else if (t2 > ray_tmin && t2 < ray_tmax && (p2.y >= -0.5f * height) && (p2.y <= 0.5f * height))
        {
            float3 normal = make_float3(p2.x, 0.0f, p2.z);
            normal = rotate_point(normal, rotation);
            normal = normalize(normal);

            // Compute Our Real t
            float real_t = (p2.x - original_ray_orig.x) / ray_dir.x;
            real_t -= 0.0001f;
            optixReportIntersection(real_t, 0, float3_as_uints(normal));
        }
    }
}

extern "C" __global__ void __intersection__plane()
{
    const scene::HitGroupData* hit_group_data = 
        reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    // 1) Plane Data
    //const float3 center   = half3_to_float3(hit_group_data->geometry_data.getPlane().center);
    const float3 center   = hit_group_data->geometry_data.getPlane().center;
    const float  width    = (float)hit_group_data->geometry_data.getPlane().width;     
    const float  depth    = (float)hit_group_data->geometry_data.getPlane().depth; 

    const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getPlane().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getPlane().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getPlane().rot_z)
    );

    // 2) Tranform Ray into Plane Local Space
    const float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);
    const float3 ray_dir  = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();
    
    // Plane locates where y = 0
    // If the ray is parallel to the plane
    if ( fabsf( ray_dir.y ) < 1e-6f ){ return; }

    float t = -ray_orig.y / ray_dir.y;
    if (t < ray_tmin || t > ray_tmax) return;
    float3 hit_local = ray_orig + t * ray_dir;

    float half_width = width * 0.5f;
    float half_depth = depth * 0.5f;
    if ( fabsf( hit_local.x ) > half_width || fabsf( hit_local.z ) > half_depth ) { return; }

    float3 local_normal = (ray_dir.y < 0.0f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(0.0f, -1.0f, 0.0f);
    float3 world_normal = rotate_point(local_normal, rotation);
    world_normal = normalize(world_normal);

    /*
    // 3) 평면 방정식 방식으로 t 계산
    //    로컬 공간에서의 평면 법선과 상수
    const float3 local_n = make_float3(0.0f, 1.0f, 0.0f);
    const float  d = 0.0f;  // 로컬 y=0 평면이므로 dot(local_n, X)=0

    // 분모(denom) 체크: 평행일 때 무시
    float denom = dot(local_n, ray_dir);
    if (fabsf(denom) < 1e-6f) {
        return;
    }

    // t = (d - dot(n,orig)) / dot(n,dir)
    float t = (d - dot(local_n, ray_orig)) / denom;
    if (t < ray_tmin || t > ray_tmax) {
        return;
    }

    // 4) 로컬 교차점
    float3 hit_local = ray_orig + t * ray_dir;

    // 5) 바운딩 박스 검사 (half_width, half_depth)
    float half_width = width * 0.5f;
    float half_depth = depth * 0.5f;
    if (fabsf(hit_local.x) > half_width ||
        fabsf(hit_local.z) > half_depth)
    {
        return;
    }

    // 6) 로컬 → 월드 법선 복원
    //    denom 부호로 앞/뒷면 구분
    float3 local_normal = (denom < 0.0f)
        ? make_float3(0.0f, 1.0f, 0.0f)
        : make_float3(0.0f, -1.0f, 0.0f);
    float3 world_normal = rotate_point(local_normal, rotation);
    world_normal = normalize(world_normal);
    */

	//printf("t : %f, width : %f, depth : %f, center : %f, %f, %f\n", t, width, depth, center.x, center.y, center.z);
    t -= 0.001f;
    optixReportIntersection(t, 0, float3_as_uints(world_normal), 0);
}

extern "C" __global__ void __intersection__cone()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    const float3 center = hit_group_data->geometry_data.getCone().center;
    const float  radius = (float)hit_group_data->geometry_data.getCone().radius;
    const float  height = (float)hit_group_data->geometry_data.getCone().height;
    const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getCone().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getCone().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getCone().rot_z)
    );

    // Tranform Ray into Plane Local Space
    const float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);
    const float3 ray_dir  = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float r_over_h = radius / height;
    const float r_over_h2 = r_over_h * r_over_h;

    const float a = ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z
        - r_over_h2 * ray_dir.y * ray_dir.y;
    const float b = 2.0f * (ray_orig.x * ray_dir.x + ray_orig.z * ray_dir.z)
        + 2.0f * r_over_h2 * ray_dir.y * (height - ray_orig.y);
    const float c = ray_orig.x * ray_orig.x + ray_orig.z * ray_orig.z
        - r_over_h2 * (height - ray_orig.y) * (height - ray_orig.y);

    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return; 

    const float sqrt_disc = sqrtf(discriminant);
    const float t0 = (-b - sqrt_disc) / (2.0f * a);
    const float t1 = (-b + sqrt_disc) / (2.0f * a);

    float t_candidate = 1e20f;  

    if (t0 > ray_tmin && t0 < ray_tmax)
    {
        float y0 = ray_orig.y + t0 * ray_dir.y;
        if (y0 >= 0.0f && y0 <= height)
            t_candidate = t0;
    }
    if (t1 > ray_tmin && t1 < ray_tmax)
    {
        float y1 = ray_orig.y + t1 * ray_dir.y;
        if (y1 >= 0.0f && y1 <= height)
            t_candidate = fmin(t_candidate, t1);
    }
    if (t_candidate < ray_tmin || t_candidate > ray_tmax || t_candidate == 1e20f)
        return; 

    const float3 hit_local = ray_orig + t_candidate * ray_dir;

    float3 normal_local;
    normal_local.x = hit_local.x;
    normal_local.y = r_over_h2 * (height - hit_local.y);
    normal_local.z = hit_local.z;
    normal_local = normalize(normal_local);

    const float3 world_normal = rotate_point(normal_local, rotation);

    optixReportIntersection(t_candidate, 0, float3_as_uints(world_normal), 0);
}

extern "C" __global__ void __intersection__circle()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    const float3 center   = hit_group_data->geometry_data.getCircle().center;
    const float  radius   = (float)hit_group_data->geometry_data.getCircle().radius;
    const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getCircle().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getCircle().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getCircle().rot_z)
    );

    // Tranform Ray into Circle Local Space
    const float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);
    const float3 ray_dir = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    // Intersection test
    if (fabs(ray_dir.y) > 0.0001f)
    {
        // Compute the intersection parameter 't' with the plane at y = 0
        float t = -ray_orig.y / ray_dir.y;

        // Verify that 't' is within the valid ray segment
        if (t >= ray_tmin && t <= ray_tmax)
        {
            // Calculate the intersection point in local space
            float3 p = ray_orig + t * ray_dir;

            // Check if the point lies within the circle's radius
            float dist2 = p.x * p.x + p.z * p.z;
            if (dist2 <= radius * radius)
            {
                // Define a local normal vector for the plane (y=0)
                float3 local_normal = make_float3(0.0f, 1.0f, 0.0f);

                // Rotate the local normal into world space
                float3 world_normal = rotate_point(local_normal, rotation);

                // Normalize the transformed normal
                world_normal = normalize(world_normal);

                // Report the intersection to OptiX
                optixReportIntersection(t, 0, float3_as_uints(world_normal));
            }
        }
    }
}

extern "C" __global__ void __intersection__ellipse()
{
    const scene::HitGroupData* hit_group_data = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    // Tranform Ray into Ellipse Local Space
    const float3 center = hit_group_data->geometry_data.getEllipse().center;
    const float3 rotation = make_float3(
        dequantizeAngle240(hit_group_data->geometry_data.getEllipse().rot_x),
        dequantizeAngle240(hit_group_data->geometry_data.getEllipse().rot_y),
        dequantizeAngle240(hit_group_data->geometry_data.getEllipse().rot_z)
    );
    const float  radiusX = (float)hit_group_data->geometry_data.getEllipse().radius_x;
    const float  radiusY = (float)hit_group_data->geometry_data.getEllipse().radius_y;

    float3 ray_orig_world = optixGetWorldRayOrigin();
    float3 ray_dir_world = optixGetWorldRayDirection();
    float3 ray_orig_local = inverse_rotate_point(ray_orig_world - center, rotation);
    float3 ray_dir_local = inverse_rotate_point(ray_dir_world, rotation);

    float ray_tmin = optixGetRayTmin();
    float ray_tmax = optixGetRayTmax();

    if (fabs(ray_dir_local.y) > 1e-6f)
    {
        // t = -ray_orig_local.y / ray_dir_local.y
        float t = -ray_orig_local.y / ray_dir_local.y;

        if (t >= ray_tmin && t <= ray_tmax)
        {
            // p = ray_orig_local + t*ray_dir_local
            float3 p = ray_orig_local + t * ray_dir_local;

            // if (x^2 / rX^2 + z^2 / rY^2) <= 1, then its inside
            float ellipse_eq = (p.x * p.x) / (radiusX * radiusX) +
                (p.z * p.z) / (radiusY * radiusY);

            if (ellipse_eq <= 1.0f)
            {
                // Normal vector
                float3 local_normal = make_float3(0.0f, 1.0f, 0.0f);
                float3 world_normal = rotate_point(local_normal, rotation);
                world_normal = normalize(world_normal);

                optixReportIntersection(t, 0, float3_as_uints(world_normal));
            }
        }
    }
}


//------------------------------------------------------------------------
// Helper Funcfion for Iaan Demo (Need to Refactor)
//------------------------------------------------------------------------
__device__ void buildLineBasis_yUp(
    const float3 dir_w,  // 선분 방향(정규화 전)
    float3& x_w,         // [out] 로컬 x축 (normalized)
    float3& y_w,         // [out] 로컬 y축 (기본은 (0,1,0)에 가깝게)
    float3& z_w          // [out] 로컬 z축
)
{
    // 1) x_w = normalize(dir_w)
    x_w = normalize(dir_w);

    // 2) 우선적으로 up = (0,1,0)을 사용하되,
    //    만약 x_w와 평행에 가깝다면 다른 up(예: (0,0,1))을 임시로 사용
    float3 up = make_float3(0.f, 1.f, 0.f);
    float dotVal = fabsf(dot(x_w, up));
    if (dotVal > 0.9999f)
    {
        // x와 y축이 거의 평행 -> (0,1,0) 대신 (0,0,1) 등 사용
        up = make_float3(0.f, 0.f, 1.f);
    }

    // 3) z_w = x_w × up  (RH rule)
    z_w = cross(x_w, up);
    z_w = normalize(z_w);

    // 4) y_w = z_w × x_w
    y_w = cross(z_w, x_w);
    y_w = normalize(y_w);
}

__device__ float3 lineInverseTransform_yUp(
    const float3 p_w,      // 월드 좌표 상의 점
    const float3 center_w, // 선분 중심(월드)
    const float3 x_w,      // buildLineBasis_yUp()에서 구한 x축
    const float3 y_w,      // y축
    const float3 z_w       // z축
)
{
    float3 tmp = p_w - center_w;  // 먼저 중심만큼 이동
    return make_float3(
        dot(tmp, x_w),
        dot(tmp, y_w),
        dot(tmp, z_w)
    );
}

__device__ float3 lineInverseRotate_yUp(
    const float3 v_w,
    const float3 x_w,
    const float3 y_w,
    const float3 z_w
)
{
    return make_float3(
        dot(v_w, x_w),
        dot(v_w, y_w),
        dot(v_w, z_w)
    );
}

__device__ float3 lineRotate_yUp(
    const float3 v_local,
    const float3 x_w,
    const float3 y_w,
    const float3 z_w
)
{
    return x_w * v_local.x
        + y_w * v_local.y
        + z_w * v_local.z;
}

__device__ __forceinline__ float length2(const float3& v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

extern "C" __global__ void __intersection__line()
{
    // -------------------------
    // (C) 광선 정보 (World)
    // -------------------------
    float3 ro_w = optixGetWorldRayOrigin();
    float3 rd_w = optixGetWorldRayDirection();
    float  tmin = optixGetRayTmin();
    float  tmax = optixGetRayTmax();

    // -------------------------
    // (A) 선분(Line) 기초 정보
    // -------------------------
    const scene::HitGroupData* hit_group_data =
        reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());

    float3 start = half3_to_float3(hit_group_data->geometry_data.getLine().start);
    float3 end = half3_to_float3(hit_group_data->geometry_data.getLine().end);

    float3 center = 0.5f * (start + end);
    float3 dir_w = end - start;
    float  line_length = length(dir_w);
    if (line_length < 1e-7f)
        return;  // 선분이 너무 짧아 "점"에 가까움

    // 예: 선분 폭/높이, 깊이
    //     line을 x축으로 놓고, y,z 방향으로 '두께'를 준다.
    float tmp_length = length(ro_w - center);
	float offset = tmp_length * 0.001f; 

    float halfLen = 0.5f * line_length;
    float halfY = offset;  
    float halfZ = offset;   

    // -------------------------
    // (B) 로컬 좌표계 구성
    // -------------------------
    //  y축은 (0,1,0) 방향 유지(단, dir_w와 평행하면 보정)
    //  x축 = normalize(dir_w)
    //  z축 = cross(x,y)
    //  y축 = cross(z,x)
    // => buildLineBasis_yUp()
    float3 x_w, y_w, z_w;
    buildLineBasis_yUp(dir_w, x_w, y_w, z_w);

    // 월드->로컬 변환
    float3 ro_l = lineInverseTransform_yUp(ro_w, center, x_w, y_w, z_w);
    float3 rd_l = lineInverseRotate_yUp(rd_w, x_w, y_w, z_w);

    // -------------------------
    // (D) "AABB" in local space
    // -------------------------
    // local box: x ∈ [-halfLen, +halfLen]
    //            y ∈ [-halfY, +halfY]
    //            z ∈ [-halfZ, +halfZ]
    float3 minB = make_float3(-halfLen, -halfY, -halfZ);
    float3 maxB = make_float3(+halfLen, +halfY, +halfZ);

    // -------------------------
    // (E) Slab 교차 알고리즘
    // -------------------------
    float tNear = -FLT_MAX;
    float tFar = FLT_MAX;
    float3 normalNear = make_float3(0.f); // 충돌면 법선

    // 축별 검사 (x,y,z)
    // macro or inline function can be used
    //  i = x(0), y(1), z(2)
    //  but let's do it dimension by dimension
    //  also track which axis gave us tNear
    {
        // x축
        if (fabsf(rd_l.x) < 1e-8f)
        {
            // 광선이 box x-face와 평행
            //  -> x 범위 안에 들어있는지?
            if (ro_l.x < minB.x || ro_l.x > maxB.x) return; // no intersect
        }
        else
        {
            float invDx = 1.f / rd_l.x;
            float t1 = (minB.x - ro_l.x) * invDx;
            float t2 = (maxB.x - ro_l.x) * invDx;
            float t1min = fminf(t1, t2);
            float t1max = fmaxf(t1, t2);

            // tNear, tFar 갱신
            if (t1min > tNear)
            {
                tNear = t1min;
                // 법선은 +/- x축
                // ray_dir.x > 0 => 교차면은 minB.x
                // ray_dir.x < 0 => 교차면은 maxB.x
                // => 부호에 따라 normalNear=(±1,0,0)
                float sign = (t1 == t1min) ? -copysignf(1.f, rd_l.x) : +copysignf(1.f, rd_l.x);
                // t1min이 minB.x를 통과하면 rd_l.x>0
                // 복잡하니 간단히 sign만으로...
                normalNear = make_float3(sign, 0, 0);
            }
            tFar = fminf(tFar, t1max);
            if (tNear > tFar) return;
        }

        // y축
        if (fabsf(rd_l.y) < 1e-8f)
        {
            if (ro_l.y < minB.y || ro_l.y > maxB.y) return;
        }
        else
        {
            float invDy = 1.f / rd_l.y;
            float t1 = (minB.y - ro_l.y) * invDy;
            float t2 = (maxB.y - ro_l.y) * invDy;
            float t1min = fminf(t1, t2);
            float t1max = fmaxf(t1, t2);

            if (t1min > tNear)
            {
                tNear = t1min;
                normalNear = make_float3(0.f, -copysignf(1.f, rd_l.y), 0.f);
            }
            tFar = fminf(tFar, t1max);
            if (tNear > tFar) return;
        }

        // z축
        if (fabsf(rd_l.z) < 1e-8f)
        {
            if (ro_l.z < minB.z || ro_l.z > maxB.z) return;
        }
        else
        {
            float invDz = 1.f / rd_l.z;
            float t1 = (minB.z - ro_l.z) * invDz;
            float t2 = (maxB.z - ro_l.z) * invDz;
            float t1min = fminf(t1, t2);
            float t1max = fmaxf(t1, t2);

            if (t1min > tNear)
            {
                tNear = t1min;
                normalNear = make_float3(0.f, 0.f, -copysignf(1.f, rd_l.z));
            }
            tFar = fminf(tFar, t1max);
            if (tNear > tFar) return;
        }
    }

    // -------------------------
    // (F) 유효 교차
    // -------------------------
    // tNear가 광선 범위 [tmin, tmax] 밖이면 무시
    if (tNear > tmax) return;
    if (tFar < tmin) return;  // box 뒤쪽
    float tHit = (tNear < tmin) ? tmin : tNear;  // tNear가 tmin보다 작으면 tmin이 더 앞

    if (tHit > tmax) return;

    // -------------------------
    // (G) 최종 교차점/법선
    // -------------------------
    // local normalNear는 ±(1,0,0), (0,±1,0), (0,0,±1)
    // => 월드로 회전


    float3 n_w = lineRotate_yUp(normalNear, x_w, y_w, z_w);
    n_w = normalize(n_w); // 정규화

    // report
    optixReportIntersection(tHit, /*hitKind=*/0, float3_as_uints(n_w), 0);
}

// Helper function to project a 3D point onto 2D plane coordinates.
__forceinline__ __device__ float2 project2D(float3 pt, float3 center, float3 tangent, float3 bitangent)
{
    float3 rel = pt - center;
    return make_float2(dot(rel, tangent), dot(rel, bitangent));
}

extern "C" __global__ void __intersection__mixed_surface()
{
    // 1) SBT 에서 직렬화 버퍼 디바이스 포인터 가져오기
    const scene::HitGroupData* hg =
        reinterpret_cast<const scene::HitGroupData*>(optixGetSbtDataPointer());
    const char* base = hg->geometry_data.getMixedSurfacePtr();

	// (2) segCount, pad1 
    const uint16_t segCount = *reinterpret_cast<const uint16_t*>(base);
    base += sizeof(uint16_t); // numSegments
    base += sizeof(uint16_t); // pad1

	// (3) SegmentAligned
    const SegmentAligned* segments = reinterpret_cast<const SegmentAligned*>(base);
    base += segCount * sizeof(SegmentAligned);

    // (4) center, normal, rotation, width, depth 
    // center
    const float* hc_x = reinterpret_cast<const float*>(base);
    float center_x = *hc_x;
    base += sizeof(float);

    const float* hc_y = reinterpret_cast<const float*>(base);
    float center_y = *hc_y;
    base += sizeof(float);

    const float* hc_z = reinterpret_cast<const float*>(base);
    float center_z = *hc_z;
    base += sizeof(float);

	float3 center = make_float3(center_x, center_y, center_z);

    // normal
    const __half2* hn = reinterpret_cast<const __half2*>(base);
    __half2 packed_normal = *hn;
	float3  normal = unpackNormal(packed_normal);
    base += sizeof(__half2);

    // rotation(1바이트*3 + pad1바이트=4)
    const uint8_t* hr_x = reinterpret_cast<const uint8_t*>(base);
    float rotX = dequantizeAngle240(*hr_x);
    base += sizeof(uint8_t);

    const uint8_t* hr_y = reinterpret_cast<const uint8_t*>(base);
    float rotY = dequantizeAngle240(*hr_y);
    base += sizeof(uint8_t);

    const uint8_t* hr_z = reinterpret_cast<const uint8_t*>(base);
    float rotZ = dequantizeAngle240(*hr_z);
    base += sizeof(uint8_t);
    base += sizeof(uint8_t); // pad

    float3 rotation = make_float3(rotX, rotY, rotZ);

    // width, depth
    const __half* pw = reinterpret_cast<const __half*>(base);
    float width = __half2float(*pw);
    base += sizeof(__half);

    const __half* pd = reinterpret_cast<const __half*>(base);
    float depth = __half2float(*pd);


    // --- 이하 기존 플레인 교차 계산 ---
    const float3 ray_orig = inverse_rotate_point(optixGetWorldRayOrigin() - center, rotation);
    const float3 ray_dir  = inverse_rotate_point(optixGetWorldRayDirection(), rotation);
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();
   
    // If the ray is parallel to the plane
    if (fabsf(ray_dir.y) < 1e-6f) { return; }
    float t = -ray_orig.y / ray_dir.y;
    if (t < ray_tmin || t > ray_tmax) return;

    float3 hit_local = ray_orig + t * ray_dir;
    float half_width = width * 0.5f;
    float half_depth = depth * 0.5f;
    if (fabsf(hit_local.x) > half_width || fabsf(hit_local.z) > half_depth) { return; }

    float3 local_normal = (ray_dir.y < 0.0f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(0.0f, -1.0f, 0.0f);
    float3 world_normal = rotate_point(local_normal, rotation);
    world_normal = normalize(world_normal);

    // 4) 점-다각형 내부 판정: 이제 lines 배열 사용
    bool inside = false;
    float2 p2D = make_float2(hit_local.x, hit_local.z);

    // 바퀴살 방식 세분화 시 필요한 상수
    const int ARC_SUBDIV = 8;
    const float OFFSET = 0.01f;   // Hermite 거리 체크용
    
	for (uint32_t i = 0; i < segCount; ++i)
    {
        // 세그먼트 타입을 먼저 확인
        SegmentType stype = segments[i].type;

        if (stype == SegmentType::Line)
        {
            // (A) Line 교차/와인딩 판정
            float3 s = dequantizeSNORM10(
                segments[i].start,
                center,
                width,
                depth,
                rotation
            );
            float3 e = dequantizeSNORM10(
                segments[i].end,
                center,
                width,
                depth,
                rotation
            );
           // printf("line %d: start %f, %f, %f, end %f, %f, %f\n",
			//	i, s.x, s.y, s.z, e.x, e.y, e.z);

            float3 si = inverse_rotate_point(s - center, rotation);
            float3 ei = inverse_rotate_point(e - center, rotation);

            float2 vi = make_float2(si.x, si.z);
            float2 vj = make_float2(ei.x, ei.z);

            float dy = vj.y - vi.y;
            if (fabsf(dy) < 1e-12f) continue;

            bool yCheck = ((vi.y > p2D.y) != (vj.y > p2D.y));
            if (yCheck)
            {
                float xInt = (vj.x - vi.x) * (p2D.y - vi.y) / dy + vi.x;
                if (p2D.x < xInt) inside = !inside;
            }
        }
        else if (stype == SegmentType::CircleCurve)
        {
            // (1) arc.center, arc.start, arc.end 불러오기
            float3 c = dequantizeSNORM10(
                segments[i].start_tangent_or_center,
                center,
                width,
                depth,
                rotation
            );

            float3 s = dequantizeSNORM10(
                segments[i].start,
                center,
                width,
                depth,
                rotation
            );

            float3 e = dequantizeSNORM10(
                segments[i].end,
                center,
                width,
                depth,
                rotation
            );

            // (1) **반드시** plane 중심과 회전을 보정
            float3 ci = inverse_rotate_point(c - center, rotation);
            float3 si = inverse_rotate_point(s - center, rotation);
            float3 ei = inverse_rotate_point(e - center, rotation);

            // (2) 2D 평면상( x, z )으로 투영
            float2 arcC = make_float2(ci.x, ci.z);
            float2 arcS = make_float2(si.x, si.z);
            float2 arcE = make_float2(ei.x, ei.z) ;
            
            // 검사점
            float2 Pxy = p2D;

            // 1) 교차 해 구하기 (t ≥ 0)
            float dx = Pxy.x - arcC.x, dy = Pxy.y - arcC.y;
			float r = (float)segments[i].end_tangent_or_radius / 1000000.f; // importer에서 1000000 곱한거 되돌리는 것
            float B = 2 * dx, C = dx * dx + dy * dy - r * r;
            float D = B * B - 4 * C;
            if (D < 1e-7f) continue;


            float sqrtD = sqrtf(D);
            float t0 = (-B + sqrtD) * 0.5f;
            float t1 = (-B - sqrtD) * 0.5f;
            float ts[2] = { t0, t1 };

            for (int k = 0; k < 2; ++k) {
                float t = ts[k];
                if (t < 1e-5f) continue; // ray 시작점 접촉 제거

                float2 P = { Pxy.x + t, Pxy.y };

                float2 v0 = arcS - arcC;
                float2 v1 = arcE - arcC;
                float2 vP = P - arcC;

                float a0 = atan2f(v0.y, v0.x);
                float a1 = atan2f(v1.y, v1.x);
                float aP = atan2f(vP.y, vP.x);

                if (a0 < 0) a0 += 2 * M_PI;
                if (a1 < 0) a1 += 2 * M_PI;
                if (aP < 0) aP += 2 * M_PI;

                float cross = v0.x * v1.y - v0.y * v1.x;
                bool isCCW = (cross > 0);

                float sweep = isCCW ?
                    (a1 >= a0 ? a1 - a0 : a1 - a0 + 2 * M_PI) :
                    (a0 >= a1 ? a0 - a1 : a0 - a1 + 2 * M_PI);

                float deltaP = isCCW ?
                    (aP >= a0 ? aP - a0 : aP - a0 + 2 * M_PI) :
                    (aP <= a0 ? a0 - aP : a0 - aP + 2 * M_PI);

                if (deltaP <= sweep + 1e-4f) {
                    inside = !inside;
                }
            }
        }
        else if (stype == SegmentType::HermiteSpline)
        {
            // (1) 로컬 2D로 Hermite 시작/끝/탄젠트
            float3 hs3  = dequantizeSNORM10(
                segments[i].start,
                center,
                width,
                depth,
                rotation
            );

            float3 he3  = dequantizeSNORM10(
                segments[i].end,
                center,
                width,
                depth,
                rotation
            );

            float3 hts3 = dequantizeTangentToRange11110(
                segments[i].start_tangent_or_center
            );

            float3 hte3 = dequantizeTangentToRange11110(
                segments[i].end_tangent_or_radius
            );

            float3 hs3i = inverse_rotate_point(hs3 - center, rotation);
            float3 he3i = inverse_rotate_point(he3 - center, rotation);
            float3 hts3i = inverse_rotate_point(hts3, rotation);
            float3 hte3i = inverse_rotate_point(hte3, rotation);

            // 2D 투영
            float2 P0 = make_float2(hs3i.x, hs3i.z);
            float2 P1 = make_float2(he3i.x, he3i.z);
            float2 R0 = make_float2(hts3i.x, hts3i.z);
            float2 R1 = make_float2(hte3i.x, hte3i.z);

            float2 Pxy = p2D;  // 검사점
			//printf("Hermite: P0=(%.3f,%.3f), P1=(%.3f,%.3f), R0=(%.3f,%.3f), R1=(%.3f,%.3f), Pxy=(%.3f,%.3f)\n",
			//	P0.x, P0.y, P1.x, P1.y, R0.x, R0.y, R1.x, R1.y, Pxy.x, Pxy.y);
            const float Y_EPS = 1e-2f;
            const float X_EPS = 1e-3f;

            // (5) 3차 방정식 구성:  y(t) - Pxy.y = 0
            // A, B, C, D는 아래처럼 전개해 구할 수 있음
            float A = 2.f * P0.y + R0.y - 2.f * P1.y + R1.y;
            float B = -3.f * P0.y - 2.f * R0.y + 3.f * P1.y - R1.y;
            float C = R0.y;
            float D = P0.y - Pxy.y;  // y(t) - Pxy.y = 0 형태

            float coeffs[4] = { D, C, B, A }; // solve_cubic(d, c, b, a)
            float roots[3];
            int nRoots = solve_cubic(coeffs, roots);

            // (6) 근들을 검사
            for (int r = 0; r < nRoots; ++r)
            {
                float t_range = roots[r];

                // (a) t 범위 체크
                //     오차를 감안해서 약간의 범위를 허용하고 싶다면
                //     if (t < -Y_EPS || t > 1.f + Y_EPS) 처럼 할 수도 있음
                if (t_range < -Y_EPS || t_range > 1.f + Y_EPS)
                    continue;

                // (c) x(t) 구하고, 레이 방향(1,0)이므로 x(t) >= Pxy.x - X_EPS 체크
                {
                    float t2 = t_range * t_range;
                    float t3 = t2 * t_range;

                    float h00 = 2.f * t3 - 3.f * t2 + 1.f;
                    float h10 = t3 - 2.f * t2 + t_range;
                    float h01 = -2.f * t3 + 3.f * t2;
                    float h11 = t3 - t2;
                    float xVal = h00 * P0.x + h10 * R0.x + h01 * P1.x + h11 * R1.x;
					float yVal = h00 * P0.y + h10 * R0.y + h01 * P1.y + h11 * R1.y;

                    // ===> 여기서 '재검증'을 수행 <===
                    if (fabs(yVal - Pxy.y) > Y_EPS) {
                        // 실제로는 y(t)가 Pxy.y와 꽤 차이나면 무효
                        continue;
                    }

                    // 약간의 오차 허용: xVal >= Pxy.x - X_EPS
                    if (xVal + X_EPS >= Pxy.x)
                    {
                        inside = !inside;
                        break; // 한 번이라도 만나면 교차
                    }
                }
            }
        }
    }
  
    // If inside, report the intersection
    if (inside)
    {
		//printf("Inside Mixed Surface\n");
        t -= 0.001f;
        optixReportIntersection(t, 0, float3_as_uints(world_normal), 0);
    }
}

// ────────────────────────────────────────────────────────────────
// Hermite 1‑D basis와 1차 미분값을 동시에 계산하는 보조 함수
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void evalHermite(float s,
    float& h1, float& h2, float& h3, float& h4,
    float& dh1, float& dh2, float& dh3, float& dh4) noexcept
{
    float s2 = s * s;
    float s3 = s2 * s;
    h1 = 2.f * s3 - 3.f * s2 + 1.f;
    h2 = -2.f * s3 + 3.f * s2;
    h3 = s3 - 2.f * s2 + s;
    h4 = s3 - s2;

    dh1 = 6.f * s2 - 6.f * s;
    dh2 = -6.f * s2 + 6.f * s;
    dh3 = 3.f * s2 - 4.f * s + 1.f;
    dh4 = 3.f * s2 - 2.f * s;
}

// ────────────────────────────────────────────────────────────────
// World space의 range인 [-무한대, +무한대]를 
// Parameter space의 range인 [0,1]로 변환하는 함수
// ────────────────────────────────────────────────────────────────
// 
// 점 → [0,1]
__device__ __forceinline__
float3 point_to_param_space(
    const float3& p,
    const float3& min,
    const float3& max)
{
    float3 range = max - min;
    return make_float3(
        (p.x - min.x) / range.x,
        (p.y - min.y) / range.y,
        (p.z - min.z) / range.z
    );
}

// 1차 도함수(접선) → 파라미터 공간
__device__ __forceinline__
float3 tangent_to_param_space(
    const float3& t,
    const float3& minP,
    const float3& maxP)
{
    float3 range = maxP - minP;
    return make_float3(
        t.x * range.x,
        t.y * range.y,
        t.z * range.z
    );
}

// 2차 혼합 도함수 → 파라미터 공간
__device__ __forceinline__
float3 mixed_to_param_space(
    const float3& m,
    const float3& minP,
    const float3& maxP)
{
    float3 r = maxP - minP;
    return make_float3(
        m.x * r.x * r.y,
        m.y * r.y * r.z,
        m.z * r.z * r.x
    );
}

// ────────────────────────────────────────────────────────────────
// Bezier 패치‑레이 교차 커널 
// ────────────────────────────────────────────────────────────────

struct Hit
{
    float   t;          // param along ray, FLT_MAX if miss
    float   u, v;       // barycentric in patch domain
};

struct Ray
{
    float3 ori;
    float3 dir;
    float  t_min;
    float  t_max;
};

// ─────────────────────────────────────────────────────────────────────────────
// 기본 float3 유틸 (CUDA 벡터 연산 사용)
// ─────────────────────────────────────────────────────────────────────────────

__host__ __device__ static inline float3 fmin3(const float3& a, const float3& b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ static inline float3 fmax3(const float3& a, const float3& b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

template<typename T> __host__ __device__ static inline void mySwap(T& x, T& y)
{
    T tmp = x;  x = y;  y = tmp;
}


// ─────────────────────────────────────────────────────────────────────────────
// AABB 슬랩 테스트
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__
bool aabb_intersect(const float3& bmin, const float3& bmax,
    const float3& ro, const float3& rdir_inv,
    float& t_enter, float& t_exit)
{
    // X축 ------------------------------------------------------------
    {
        float lo = (bmin.x - ro.x) * rdir_inv.x;
        float hi = (bmax.x - ro.x) * rdir_inv.x;
        if (lo > hi) mySwap(lo, hi);

        t_enter = fmaxf(t_enter, lo);
        t_exit = fminf(t_exit, hi);
        if (t_enter > t_exit) return false;  // early-out
    }

    // Y축 ------------------------------------------------------------
    {
        float lo = (bmin.y - ro.y) * rdir_inv.y;
        float hi = (bmax.y - ro.y) * rdir_inv.y;
        if (lo > hi) mySwap(lo, hi);

        t_enter = fmaxf(t_enter, lo);
        t_exit = fminf(t_exit, hi);
        if (t_enter > t_exit) return false;
    }

    // Z축 ------------------------------------------------------------
    {
        float lo = (bmin.z - ro.z) * rdir_inv.z;
        float hi = (bmax.z - ro.z) * rdir_inv.z;
        if (lo > hi) mySwap(lo, hi);

        t_enter = fmaxf(t_enter, lo);
        t_exit = fminf(t_exit, hi);
        if (t_enter > t_exit) return false;
    }

    return true;  
}


// ─────────────────────────────────────────────────────────────────────────────
// Bézier 평가 + 1차 편미분 (de Casteljau 2‑D)
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__
void bezier_eval(const BezierPatch& bp,
    float u, float v,
    float3& P,
    float3& dPu, float3& dPv)
{
    float3 Cv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const float3* cp = bp.P[i];
        float3 a = lerp(cp[0], cp[1], v);
        float3 b = lerp(cp[1], cp[2], v);
        float3 c = lerp(cp[2], cp[3], v);
        float3 d = lerp(a, b, v);
        float3 e = lerp(b, c, v);
        Cv[i] = lerp(d, e, v);
    }
    float3 A = lerp(Cv[0], Cv[1], u);
    float3 B = lerp(Cv[1], Cv[2], u);
    float3 C = lerp(Cv[2], Cv[3], u);
    float3 D = lerp(A, B, u);
    float3 E = lerp(B, C, u);
    P = lerp(D, E, u);
    dPu = make_float3(3.f * (E.x - D.x), 3.f * (E.y - D.y), 3.f * (E.z - D.z));

    float3 Ru[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float3 a = lerp(bp.P[0][j], bp.P[1][j], u);
        float3 b = lerp(bp.P[1][j], bp.P[2][j], u);
        float3 c = lerp(bp.P[2][j], bp.P[3][j], u);
        float3 d = lerp(a, b, u);
        float3 e = lerp(b, c, u);
        Ru[j] = lerp(d, e, u);
    }
    float3 A2 = lerp(Ru[0], Ru[1], v);
    float3 B2 = lerp(Ru[1], Ru[2], v);
    float3 C2 = lerp(Ru[2], Ru[3], v);
    float3 D2 = lerp(A2, B2, v);
    float3 E2 = lerp(B2, C2, v);
    dPv = make_float3(3.f * (E2.x - D2.x), 3.f * (E2.y - D2.y), 3.f * (E2.z - D2.z));
}


// ─────────────────────────────────────────────────────────────────────────────
// 패치 4등분 (LL, LR, UL, UR) + AABB 재계산
// ─────────────────────────────────────────────────────────────────────────────
// ────────────────────────────────────────────────────────────────
// 한 열(또는 행) 방향으로 4×4 제어점을 반으로 자르는 보조 함수
// in  : 입력 4×4 제어점 배열
// lo  : 아래(또는 왼쪽) 절반으로 분할된 패치
// hi  : 위(또는 오른쪽) 절반으로 분할된 패치
// ────────────────────────────────────────────────────────────────
__host__ __device__ inline
void splitV(float3 in[4][4], BezierPatch& lo, BezierPatch& hi)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float3 a = in[i][0], b = in[i][1], c = in[i][2], d = in[i][3];
        float3 ab = (a + b) * 0.5f, bc = (b + c) * 0.5f, cd = (c + d) * 0.5f;
        float3 abc = (ab + bc) * 0.5f, bcd = (bc + cd) * 0.5f, abcd = (abc + bcd) * 0.5f;

        lo.P[i][0] = a;     lo.P[i][1] = ab;   lo.P[i][2] = abc;  lo.P[i][3] = abcd;
        hi.P[i][0] = abcd;  hi.P[i][1] = bcd;  hi.P[i][2] = cd;   hi.P[i][3] = d;
    }
}

// ────────────────────────────────────────────────────────────────
// 메인 함수: 4×4 Bézier 패치를 네 개의 균일한 하위 패치로 분할
// child[0] : 좌하
// child[1] : 우하
// child[2] : 좌상
// child[3] : 우상
// ────────────────────────────────────────────────────────────────
__host__ __device__
void subdivide_patch(const BezierPatch& src, BezierPatch child[4])
{
    // 먼저 U 방향(수평)으로 절반 자르기  ───────────────
    float3 L[4][4], R[4][4];

#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float3 a = src.P[0][j], b = src.P[1][j],
            c = src.P[2][j], d = src.P[3][j];

        float3 ab = (a + b) * 0.5f,
            bc = (b + c) * 0.5f,
            cd = (c + d) * 0.5f;

        float3 abc = (ab + bc) * 0.5f,
            bcd = (bc + cd) * 0.5f,
            abcd = (abc + bcd) * 0.5f;

        L[0][j] = a;     L[1][j] = ab;    L[2][j] = abc;  L[3][j] = abcd;   // 좌측 절반
        R[0][j] = abcd;  R[1][j] = bcd;   R[2][j] = cd;   R[3][j] = d;      // 우측 절반
    }

    // 이어서 V 방향(수직)으로 절반 자르기  ───────────────
    splitV(L, child[0], child[2]);   // 좌측 →  아래·위
    splitV(R, child[1], child[3]);   // 우측 →  아래·위

    // 각 하위 패치의 AABB 갱신  ──────────────────────────
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        float3 mn = child[c].P[0][0];
        float3 mx = mn;

#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                mn = fmin3(mn, child[c].P[i][j]);
                mx = fmax3(mx, child[c].P[i][j]);
            }

        child[c].b_min = mn;
        child[c].b_max = mx;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// 뉴턴 보정
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__
bool newton_refine(const BezierPatch& bp, const Ray& ray,
    float& u, float& v, float& t)
{
    const int   MAX_IT = 8;
    const float EPS_F = 1e-4f;
    const float EPS_P = 1e-5f;

    for (int it = 0; it < MAX_IT; ++it)
    {
        float3 P, dPu, dPv; bezier_eval(bp, u, v, P, dPu, dPv);
        float3 F = make_float3(P.x - (ray.ori.x + t * ray.dir.x),
            P.y - (ray.ori.y + t * ray.dir.y),
            P.z - (ray.ori.z + t * ray.dir.z));
        if (length(F) < EPS_F) return (u >= 0.f && u <= 1.f && v >= 0.f && v <= 1.f);

        float3 c0 = dPu, c1 = dPv, c2 = make_float3(-ray.dir.x, -ray.dir.y, -ray.dir.z);
        float3 r0 = cross(c1, c2);
        float3 r1 = cross(c2, c0);
        float3 r2 = cross(c0, c1);
        float det = dot(c0, r0);
        if (fabsf(det) < 1e-8f) return false;

        float invD = 1.f / det;
        float du = dot(r0, F) * (-invD);
        float dv = dot(r1, F) * (-invD);
        float dt = dot(r2, F) * (-invD);
        u += du; v += dv; t += dt;
        if (fabsf(du) < EPS_P && fabsf(dv) < EPS_P && fabsf(dt) < EPS_P)
            return (u >= 0.f && u <= 1.f && v >= 0.f && v <= 1.f);
    }
    return false;
}


// ─────────────────────────────────────────────────────────────────────────────
// 미니 힙 (32) :  한 레이에 충분
// ─────────────────────────────────────────────────────────────────────────────
struct Node { BezierPatch patch; float u0, v0, u1, v1; float t_enter; };

template<int N = 32> struct MiniHeap
{
    Node buf[N]; int size = 0;

    __host__ __device__ void push(const Node& n)
    {
        if (size == N) return; buf[size] = n; int i = size++; while (i && buf[(i - 1) >> 1].t_enter > buf[i].t_enter)
        {
            mySwap(buf[(i - 1) >> 1], buf[i]); i = (i - 1) >> 1;
        }
    }
    __host__ __device__ bool pop(Node& out)
    {
        if (!size) return false; 

        out = buf[0]; buf[0] = buf[--size];

        int i = 0; 
        while (true) 
        {
            int l = (i << 1) + 1, r = l + 1, s = i;
            if (l < size && buf[l].t_enter < buf[s].t_enter) s = l;
            if (r < size && buf[r].t_enter < buf[s].t_enter) s = r;
            if (s == i) break; mySwap(buf[i], buf[s]); i = s;
        } return true;
    }
};

extern "C" __global__ void __intersection__bezier_patch()
{
    // Ray
    float3 ro = optixGetWorldRayOrigin();
    float3 rd = optixGetWorldRayDirection();
    float  tmin = optixGetRayTmin();
    float  tmax = optixGetRayTmax();

	Ray ray = { ro, rd, tmin, tmax };

    float3 invDir = make_float3(1.f / rd.x, 1.f / rd.y, 1.f / rd.z);

    // Patch
    const scene::HitGroupData* hg = reinterpret_cast<scene::HitGroupData*>(optixGetSbtDataPointer());
    const char* base = hg->geometry_data.getBezierPatchPtr();
    const BezierPatch* bp = reinterpret_cast<const BezierPatch*>(base);

    BezierPatch root = *bp;

	// Hit record initialization
    Hit best_hit;
    best_hit.t = tmax;
    best_hit.u = best_hit.v = 0.f;

    // t initialization
    float t_enter = tmin;
    float t_exit  = best_hit.t;

    // Mini-heap initialization
    MiniHeap<> heap;
    heap.push( Node{ root, 0.f, 0.f, 1.f, 1.f, t_enter } );

    //
    Node current;
    int depth = 0; 
    int max_depth = 10;
    int pixel_epsilon = 3;

    while (heap.pop(current) && depth < max_depth)
    {
        // End condition (AABB 크기가 pixel보다 작을 때)
        float3 ext = make_float3(
            current.patch.b_max.x - current.patch.b_min.x,
            current.patch.b_max.y - current.patch.b_min.y,
            current.patch.b_max.z - current.patch.b_min.z
        );
        float max_edge = fmaxf( fmaxf( ext.x, ext.y ), ext.z );

        //------------- 교차 루프 내부 수정 -------------//
        if (max_edge < pixel_epsilon)    // leaf 조건
        {
            // 1) leaf 패치의 (u,v) 초기값 ― 로컬 공간 == [0,1]^2
            float u_local = 0.5f;
            float v_local = 0.5f;
            float t = current.t_enter;

            // 2) 뉴턴 보정은 "로컬" 파라메터로 수행
            if (newton_refine(current.patch, ray, u_local, v_local, t) &&
                t < best_hit.t && t > ray.t_min)
            {
                // 3) 로컬 → 전역 파라메터 변환
                float u_global = current.u0 + u_local * (current.u1 - current.u0);
                float v_global = current.v0 + v_local * (current.v1 - current.v0);

                best_hit.t = t;
                best_hit.u = u_global;
                best_hit.v = v_global;
            }
            continue;    // 다음 heap pop
        }

        //Subdivision
        BezierPatch child[4];
        subdivide_patch(current.patch, child);

		float du = 0.5f * (current.u1 - current.u0);
		float dv = 0.5f * (current.v1 - current.v0);

		for (int c = 0; c < 4; ++c)
		{
            float t0 = ray.t_min;
			float t1 = best_hit.t;

			if (!aabb_intersect(
				child[c].b_min, child[c].b_max,
				ray.ori, invDir, t0, t1
			)) continue;

            Node n;
            n.patch = child[c];
            n.t_enter = t0;

            switch (c) 
            {
            case 0:n.u0 = current.u0;       n.u1 = current.u0 + du;  n.v0 = current.v0;       n.v1 = current.v0 + dv; break;
            case 1:n.u0 = current.u0 + du;  n.u1 = current.u1;       n.v0 = current.v0;       n.v1 = current.v0 + dv; break;
            case 2:n.u0 = current.u0;       n.u1 = current.u0 + du;  n.v0 = current.v0 + dv;  n.v1 = current.v1;      break;
            case 3:n.u0 = current.u0 + du;  n.u1 = current.u1;       n.v0 = current.v0 + dv;  n.v1 = current.v1;      break;
            }
            heap.push(n);
		}
		++depth;
    }
   
	if (best_hit.t < tmax && best_hit.t > ray.t_min)
	{
		// Patch에서 u, v 좌표를 world space로 변환
		float3 P;
		bezier_eval(root, best_hit.u, best_hit.v, P, make_float3(0), make_float3(0));

		// Normal 계산
		float3 dPu, dPv;
		bezier_eval(root, best_hit.u, best_hit.v, P, dPu, dPv);

		float3 normal = normalize(cross(dPu, dPv));

		// OptiX 교차 보고
		optixReportIntersection(best_hit.t, 0, float3_as_uints(normal), 0);
	}
   // else
		//printf("Bezier patch miss: t=%.3f, u=%.3f, v=%.3f\n", best_hit.t, best_hit.u, best_hit.v);
} 