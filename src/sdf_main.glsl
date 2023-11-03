
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

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    vec3 point = p - vec3(0.0, 0.45, -0.7);
    if (point.y > 0.0) {
        point = point * vec3(1.0, 0.55, 1.0);
    }

    // return distance and color
    return vec4(sdSphere(point, 0.35), vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 point = p - vec3(0.0, 0.7, -0.45);

    float dist = sdSphere(point, 0.15);

    float radius = length(vec3(point.x, point.y, 0.0));
    if (radius > 0.07) {
        return vec4(dist, 1.0, 1.0, 1.0);
    } else if (radius > 0.04) {
        return vec4(dist, 0.0, 0.0, 1.0);
    } else {
        return vec4(dist, 0.0, 0.0, 0.0);
    }
}

vec4 sdLegs(vec3 p, float x)
{
    vec3 point = (p - vec3(x, 0.0, -0.90)) * vec3(1.2, 0.95, 1.3);
    float dist = sdSphere(point, 0.15);

    return vec4(dist, 0.1, 0.9, 0.0);
}

vec4 sdArms(vec3 p, vec3 a, vec3 b, float r)
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  float res = length( pa - ba*h ) - r;
  return vec4(res, 0.5, 0.9, 0.3);
}

vec4 sdMonster(vec3 p)
{
    float time = iTime;
    float shift = 0.23 * sin(time);

    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = sdBody(p);
    
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    vec4 legR = sdLegs(p, min(-0.2, shift));
    if (legR.x < res.x) {
        res = legR;
    }

    vec4 legL = sdLegs(p, max(0.2, shift));
    if (legL.x < res.x) {
        res = legL;
    }

    vec3 armOut = vec3(0.05, 0.13 + 0.1 * lazycos(3.0 * time), 0.15);
    vec3 armIn = vec3(-0.15, 0.03, -0.07);
    vec4 armL = sdArms(p - vec3(0.35, 0.3, -0.55),
        armOut,
        armIn,
        0.06);
    if (armL.x < res.x) {
        res = armL;
    }

    vec4 armR = sdArms(p - vec3(-0.35, 0.3, -0.55),
        armOut * vec3(-1.0, 1.0 + 0.5 * sin(2.0 * time), 1.0),
        armIn * vec3(-1.0, 1.0, 1.0),
        0.06);
    if (armR.x < res.x) {
        res = armR;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);
    
    
    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
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
    
    
    
    // Output to screen
    fragColor = vec4(col, 1.0);
}
