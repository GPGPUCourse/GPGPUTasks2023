const float M_PI  = radians(180.0); // 3.14159265359;
const float M_2PI = radians(360.0); // 6.28318530718;
const float EPS = 1e-3;

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlaneXZ(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / M_2PI) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

vec4 sdMin(vec4 sda, vec4 sdb)
{
    return sda.w < sdb.w ? sda : sdb;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);

    // return distance and color
    return vec4(d, vec3(0, 1, 0));
}

vec4 sdEye(vec3 p)
{
    vec4 res = vec4(1e10, 0, 0, 0);
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
    vec4 res;
    res = sdMonster(p).yzwx;
    res = sdMin(res, vec4(1, 0, 0, sdPlaneXZ(p)));
    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal(vec3 p) // for function f(p)
{
    const vec2 h = vec2(EPS, 0);
    return normalize( vec3(
        sdTotal(p + h.xyy).w - sdTotal(p - h.xyy).w,
        sdTotal(p + h.yxy).w - sdTotal(p - h.yxy).w,
        sdTotal(p + h.yyx).w - sdTotal(p - h.yyx).w) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t * ray_direction);
        t += res.w;
        if (res.w < EPS) {
            return vec4(res.xyz, t);
        }
    }

    return vec4(vec3(0), 1e10);
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

    float light_dist = raycast(light_source, normalize(light_dir)).w;

    if (light_dist + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage(out vec4 fragColor, vec2 fragCoord)
{
    vec2 uv = fragCoord/iResolution.y;
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5 * wh, -1.0));

    vec4 res = raycast(ray_origin, ray_direction);
    vec3 col = res.xyz;

    vec3 surface_point = ray_origin + res.w * ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;

    // Output to screen
    fragColor = vec4(col, 1.0);
}
