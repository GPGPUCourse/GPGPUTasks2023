const float M_PI  = radians(180.0); // 3.14159265359;
const float M_2PI = radians(360.0); // 6.28318530718;
const float EPS = 1e-3;
const float INF = 1e10;

float opSdSmoothUnion(float da, float db, float k)
{
    float h = clamp(0.5 + 0.5 * (db - da) / k, 0.0, 1.0);
    return mix(db, da, h) - k * h * (1.0 - h);
}

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

float sdCapsule(vec3 p, vec3 axis, float r)
{
    vec3 axisNorm = normalize(axis);
    float axisLen = length(axis);
    float toReduce = dot(p, axisNorm);
    p -= clamp(toReduce, 0.0, axisLen) * axisNorm;
    return length(p) - r;
}

float sdCapsuleLine(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 ap = p - a;
    vec3 ab = b - a;
    return sdCapsule(ap, ab, r);
}

float sdRoundCone(vec3 p, vec3 axis, float ra, float rb)
{
    vec3 axisNorm = normalize(axis);
    float axisLen = length(axis);
    float sinA = (rb - ra) / axisLen;
    float cosA = sqrt(1.0 - sinA * sinA);

    float y = dot(p, axisNorm);
    vec2 q = vec2(sqrt(dot(p, p) - y * y), y);
    float k = dot(q, vec2(sinA, cosA));
    if (k < 0.0) return length(q) - ra;
    if (k > cosA * axisLen) return length(q - vec2(0.0, axisLen)) - rb;
    return dot(q, vec2(cosA, -sinA)) - ra;
}

float sdRoundConeLine(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    vec3 ap = p - a;
    vec3 ab = b - a;
    return sdRoundCone(ap, ab, ra, rb);
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

// min() for signed distance + color
vec4 sdcCombine(vec4 a, vec4 b)
{
    return a.w < b.w ? a : b;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdcBody(vec3 p)
{
    float sdTorsoBot = sdSphere(p - vec3(0.0, 0.4, 0.0), 0.3);
    float sdTorsoTop = sdSphere(p - vec3(0.0, 0.7, 0.0), 0.2);
    float d = opSdSmoothUnion(sdTorsoBot, sdTorsoTop, 0.3);

    vec3 pMirrorX = vec3(abs(p.x), p.yz);
    float sdLegs = sdCapsuleLine(pMirrorX, vec3(0.1, 0.04, 0.0), vec3(0.1, 0.5, 0.0), 0.05);
    d = min(d, sdLegs);

    // return distance and color
    return vec4(vec3(0, 1, 0), d);
}

vec4 sdcEye(vec3 p)
{
    vec4 res = vec4(0, 0, 0, INF);
    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p += vec3(0.0, 0.0, 0.5);

    vec4 res = sdcBody(p);
    res = sdcCombine(res, sdcEye(p));

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res;
    res = sdMonster(p);
    res = sdcCombine(res, vec4(1, 0, 0, sdPlaneXZ(p)));
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


vec4 raycast(vec3 rayOrigin, vec3 rayDirection)
{
    // p = rayOrigin + t * rayDirection;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(rayOrigin + t * rayDirection);
        t += res.w;
        if (res.w < EPS) {
            return vec4(res.xyz, t);
        }
    }

    return vec4(vec3(0), INF);
}


float shading(vec3 p, vec3 lightSource, vec3 normal)
{
    vec3 lightDir = normalize(lightSource - p);
    float shading = dot(lightDir, normal);
    return clamp(shading, 0.5, 1.0);
}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 lightSource, vec3 N, vec3 cameraCenter, float shinyness)
{
    vec3 L = normalize(p - lightSource);
    vec3 R = reflect(L, N);
    vec3 V = normalize(cameraCenter - p);
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 lightSource)
{
    vec3 lightDir = p - lightSource;
    float targetDist = length(lightDir);

    float lightDist = raycast(lightSource, normalize(lightDir)).w;

    if (lightDist + 0.001 < targetDist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage(out vec4 fragColor, vec2 fragCoord)
{
    vec2 uv = fragCoord/iResolution.y;
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 rayOrigin = vec3(0.0, 0.5, 1.0);
    vec3 rayDirection = normalize(vec3(uv - 0.5 * wh, -1.0));

    vec4 res = raycast(rayOrigin, rayDirection);
    vec3 col = res.xyz;

    vec3 surfacePoint = rayOrigin + res.w * rayDirection;
    vec3 normal = calcNormal(surfacePoint);

    vec3 lightSource = vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);

    float shad = shading(surfacePoint, lightSource, normal);
    shad = min(shad, castShadow(surfacePoint, lightSource));
    col *= shad;

    float spec = specular(surfacePoint, lightSource, normal, rayOrigin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;

    // Output to screen
    fragColor = vec4(col, 1.0);
}
