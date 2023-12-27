


// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// see https://iquilezles.org/articles/distfunctions/
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// smooth minimum function to create gradual transitions between SDFs
// https://iquilezles.org/articles/smin/
float smin(float d0, float d1, float k)
{
    float h = max( k-abs(d0-d1), 0.0 ) / k;
    return min( d0, d1 ) - h*h*k*(1.0/4.0);
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;


    d = smin(sdSphere((p - vec3(0.0, 0.32, -0.7)), 0.35), 
    sdSphere((p - vec3(0.0, 0.75, -0.6)), 0.2), 0.3);

    float hand_bx = 0.27, hand_ex = 0.20;
    float hand_by = 0.52, hand_ey = -0.2; 
    float angle = 0.0;
    float hand_left = sdCapsule(p, 
                            vec3(-hand_bx,hand_by,-0.7), 
                            vec3(-hand_bx - hand_ex, hand_by + abs(lazycos(iTime * 10.)) * hand_ey,-0.7), 
                            0.04);
    d = min(hand_left, d);
    float hand_right = sdCapsule(p, vec3(hand_bx,hand_by,-0.7), vec3(hand_ex + hand_bx,hand_ey + hand_by,-0.7), 0.04);
    d = min(hand_right, d);
    

    float leg_x = 0.15;
    float leg_left = sdCapsule(p, vec3(-leg_x,-0.04,-0.7), vec3(-leg_x,0.2,-0.7), 0.05);
    d = min(leg_left, d);
    float leg_right = sdCapsule(p, vec3(leg_x,-0.04,-0.7), vec3(leg_x,0.2,-0.7), 0.05);
    d = min(leg_right, d);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}


vec4 sdEyeBall(vec3 p)
{

    float d0 = 0.2;
    float d = sdSphere(p - vec3(0, 0.62, -0.52), d0);
    
    return vec4(d, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p)
{
    float d0 = 0.05;
    float d = sdSphere(p - vec3(0, 0.62, -0.32), d0);
    

    return vec4(d, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p)
{
    float d0 = 0.1;
    float d = sdSphere(p - vec3(0, 0.62, -0.39), d0);

    return vec4(d, vec3(0.0, 1.0, 1.0));

}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);
    vec4 ba = sdEyeBall(p);
    vec4 pu = sdEyePupil(p);
    vec4 ir = sdEyeIris(p);
    float d = min(ba[0], pu[0]);
    d = min(d, ir[0]);
    if (d == ba[0]) {
        res = ba;
    } else if (d == pu[0]) {
        res = pu;
    } else {
        res = ir;
    }
    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = sdBody(p);
    
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);
    
    
    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 1.0, 1.0));
    }
    
    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    
    float EPS = 1e-3;
    
    
    // p = ray_origin + t * ray_direction;
    
    float t = 0.0;
    
    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{
    
    vec3 light_dir = normalize(light_source - p);
    
    float shading = dot(light_dir, normal);
    
    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);
    
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{
    
    vec3 light_dir = p - light_source;
    
    float target_dist = length(light_dir);
    
    
    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }
    
    return 1.0;
}

#iChannel0 "file://sdf_snow.glsl"
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;
    
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));
    

    vec4 res = raycast(ray_origin, ray_direction);
    
    
    
    vec3 col = res.yzw;
    
    
    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);
    
    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);
    
    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;
    
    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;
    
    col += texture(iChannel0, fragCoord/iResolution.xy).rgb;
    
    fragColor = vec4(col, 1.0);
}