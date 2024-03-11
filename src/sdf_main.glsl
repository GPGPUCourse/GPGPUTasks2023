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

vec3 rotateX(vec3 p, float a)
{
    return  p * mat3(
        1.0, 0.0, 0.0,
        0.0, cos(a), -sin(a),
        0.0, sin(a), cos(a)
    );
}

vec3 rotateY(vec3 p, float a)
{
    return p * mat3(
        cos(a), 0.0, sin(a),
        0.0, 1.0, 0.0,
        -sin(a), 0.0, cos(a)
    );
}

vec3 rotateZ(vec3 p, float a)
{
    return p * mat3(
        cos(a), -sin(a), 0.0,
        sin(a), cos(a), 0.0,
        0.0, 0.0, 1.0
    );
}

float smin(float a, float b, float k)
{
    float h = max(k-abs(a-b), 0.0) / k;
    return min(a, b) - h*h*h*k*(1.0 / 5.0);
}

float sdStretchedSphere(vec3 p, float k, float r)
{
    return sqrt(p.x * p.x + p.y * p.y / k + p.z * p.z)  - r;
}

float sdInclinedCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdVerticalCapsule( vec3 p, float h, float r)
{
    p.y -= clamp(p.y, 0.0, h);
    return length(p) - r;
}

float sdHand(vec3 p, bool isRight)
{
    return sdInclinedCapsule(p - vec3(isRight ? 0.3 : -0.3, 0.5, -0.7),
                     vec3(0.0, 0.0, 0.0),
                     rotateX(vec3(isRight ? 0.07 : -0.07, -0.25, 0.0), -1.0 + lazycos(iTime)),
                     0.05);
}

float sdLeg(vec3 p, bool isRight)
{
    return sdVerticalCapsule(p - vec3(isRight ? 0.1 : -0.1, 0.0, -0.7), 0.2, 0.05);
}

float sdCube(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    float r = 0.34;
    float bound = 0.4;

    if (p.y < bound)
    d = sdSphere((p - vec3(0.0, bound, -0.7)), r);
    else
    d = sdStretchedSphere((p - vec3(0.0, bound, -0.7)), 2.0, r);

    float leftHand = sdHand(p, false);
    float rightHand = sdHand(p, true);

    d = smin(d, leftHand, 0.03);
    d = smin(d, rightHand, 0.03);

    float leftLeg = sdLeg(p, false);
    float rightLeg = sdLeg(p, true);

    d = smin(d, leftLeg, 0.05);
    d = smin(d, rightLeg, 0.05);

    // return distance and color
    return vec4(d, vec3(0.5, 0.7, 0.5));
}

vec4 sdEye(vec3 p)
{
    vec3 black = vec3(0.2, 0.2, 0.1);
    vec3 blue = vec3(0.5, 0.7, 1.0);
    vec3 color = black;

    vec3 a = vec3(0.0, 0.5, -0.45);
    float d = sdCube(p - a, vec3(0.1, 0.09, 0.1), 0.07);

    if (sqrt((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y)) < clamp(0.05 * cos(iTime), 0.03, 0.08))
    color = blue;

    return vec4(d, color);
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