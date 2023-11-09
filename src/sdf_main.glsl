vec3 ray_origin;


// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdEllipsoid(in vec3 p, in vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

// XZ plane
float sdPlane(vec3 p) {
    return p.y;
}


float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}


// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle) {
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdVesicaSegment(in vec3 p, in vec3 a, in vec3 b, in float w) {
    vec3 c = (a + b) * 0.5;
    float l = length(b - a);
    vec3 v = (b - a) / l;
    float y = dot(p - c, v);
    vec2 q = vec2(length(p - c - y * v), abs(y));

    float r = 0.5 * l;
    float d = 0.5 * (r * r - w * w) / w;
    vec3 h = (r * q.x < d * (q.y - r)) ? vec3(0.0, r, 0.0) : vec3(-d, 0.0, d + w);

    return length(q - h.xy) - h.z;
}


vec4 sdEyebrows(vec3 p, vec3 pos, float width, float height, float rot, float cornersZ, float thickness) {
    float d = sdCapsule(p, vec3(pos.x - width, pos.y, pos.z - rot + cornersZ), vec3(pos.x, pos.y + height, pos.z), thickness);
    d = min(d, sdCapsule(p, vec3(pos.x, pos.y + height, pos.z), vec3(pos.x + width, pos.y, pos.z + rot + cornersZ), thickness));

    return vec4(d, vec3(0, 0, 0));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
vec4 sdBody(vec3 p, float leftEyeDist, float rightEyeDist) {
    float d = 1e10;

    {
        //body shape
        d = sdEllipsoid((p - vec3(0.0, 0.35, -0.7)), vec3(0.38, 0.35, 0.35));
        d = opSmoothUnion(d, sdEllipsoid((p - vec3(0.0, 0.75, -0.7)), vec3(0.3, 0.25, 0.25)), 0.3);
    }

    {
        // nose
        d = opSmoothUnion(d, sdEllipsoid((p - vec3(0.0, 0.6, -0.38)), vec3(0.02, 0.05, 0.025)), 0.08);
        d = opSmoothUnion(d, sdEllipsoid((p - vec3(0.0, 0.5, -0.35)), vec3(0.03, 0.02, 0.03)), 0.1);
    }

    {
        // eye holes
        d = opSmoothSubtraction(leftEyeDist, d, 0.02);
        d = opSmoothSubtraction(rightEyeDist, d, 0.02);
    }

    {
        //mouth
        vec4 a = sdEyebrows(p, vec3(0, 0.38, -0.355), 0.1, -0.005, 0.0, -0.013, 0.015);
        float mouthD = sdCapsule(p, vec3(-0.1, 0.38, -0.36), vec3(0.1, 0.38, -0.36), 0.015);
        if (a.x < d) {
            return vec4(a.x, vec3(0.8823529411764706, 0.5843137254901961, 0.5450980392156862));
        }
    }

    return vec4(d, vec3(0.8352941176470589, 0.5647058823529412, 0.396078431372549));
}

vec4 sdEye(vec3 p, vec3 eyePos, float eyeRadius, float width, float rotate) {
    vec3 eyeDir = normalize(ray_origin - eyePos);
    vec3 color = vec3(1.0, 1.0, 1.0);

    //float d = sdSphere((p - eyePos), eyeRadius);
    float height = 0.25 * eyeRadius;
    float d = sdVesicaSegment(p, vec3(eyePos.x - width, eyePos.y, eyePos.z + rotate),
                              vec3(eyePos.x + width, eyePos.y, eyePos.z - rotate), height);
    if (length(normalize(ray_origin - p) - eyeDir) < 0.015)
        color = vec3(0.0, 0.0, 0.0);
    return vec4(d, color);
}

vec4 minDist(vec4 lhs, vec4 rhs) {
    if (lhs.x < rhs.x)
        return lhs;
    return rhs;
}

vec4 sdMonster(vec3 p) {
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    vec3 rightEyesPos = vec3(0.1, 0.68, -0.42);
    vec3 leftEyesPos = vec3(-rightEyesPos.x, rightEyesPos.yz);

    float animTime = min(1.0, 1.0 + cos(iTime));

    float eyeRotate = 0.01;
    float eyeHeight = 0.13;
    float eyeWidth = 0.08;
    vec4 leftEye = sdEye(p, leftEyesPos, eyeHeight * (1.0 - 0.5 * animTime), eyeWidth, -eyeRotate);
    vec4 rightEye = sdEye(p, rightEyesPos, eyeHeight * (1.0 + 1.0 * animTime), eyeWidth, eyeRotate);

    float eyebrowWidth = 0.09;
    float eyebrowHeight = 0.009;
    float eyebrowThickness = 0.01;
    vec4 leftEyebrow = sdEyebrows(p, vec3(leftEyesPos.x, leftEyesPos.y + 0.06 - 0.02 * animTime, leftEyesPos.z),
                                  eyebrowWidth, eyebrowHeight, eyeRotate, 0.0, eyebrowThickness);
    vec4 rightEyebrow = sdEyebrows(p, vec3(rightEyesPos.x, rightEyesPos.y + 0.06 + 0.02 * animTime, rightEyesPos.z),
                                   eyebrowWidth, eyebrowHeight + 0.03 * animTime, -eyeRotate, 0.0, eyebrowThickness);

    vec4 res = sdBody(p, leftEye.x, rightEye.x);


    res = minDist(res, leftEyebrow);
    res = minDist(res, rightEyebrow);
    res = minDist(res, leftEye);
    res = minDist(res, rightEye);


    return res;
}


vec4 sdTotal(vec3 p) {
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(0.6, 0.6, 0.6));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal(in vec3 p)// for function f(p)
{
    const float eps = 0.0001;// or some other value
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(sdTotal(p + h.xyy).x - sdTotal(p - h.xyy).x, sdTotal(p + h.yxy).x - sdTotal(p - h.yxy).x,
                          sdTotal(p + h.yyx).x - sdTotal(p - h.yyx).x));
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction) {

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t * ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }


    return vec4(1e10, vec3(0.0, 1.3, 2));
}


float shading(vec3 p, vec3 light_source, vec3 normal) {

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);
}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness) {
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source) {

    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);


    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = (cross(cu, cw));
    return mat3(cu, cv, cw);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec2 mo = iMouse.xy / iResolution.xy;
    float time = 0.0;//32.0 + iTime*1.5;

    float animTime = 1.4 - 0.8 * min(1.0, 1.0 + cos(iTime));

    // camera
    vec3 ta = vec3(0.0, 0.5, -0.7);
    ray_origin = ta + vec3(0.5 * cos(1.1 * animTime ), 0.5, 1.5 * sin(1.1 * animTime ));
    mat3 ca = setCamera(ray_origin, ta, 0.0);
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    const float fl = 2.0;
    vec3 ray_direction = ca * normalize(vec3(p, fl));


    vec4 res = raycast(ray_origin, ray_direction);


    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x * ray_direction;
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