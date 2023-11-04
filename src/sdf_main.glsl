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

// 2D Random
float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0 - 2.0*f);
    u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
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

float sdRoundCone( vec3 p, float r1, float r2, float h )
{
  // sampling independent computations (only depend on shape)
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);

  // sampling dependant computations
  vec2 q = vec2( length(p.xz), p.y );
  float k = dot(q,vec2(-b,a));
  if( k<0.0 ) return length(q) - r1;
  if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
  return dot(q, vec2(a,b) ) - r1;
}

mat3 rotmatX(float a)
{
    return mat3(
        1.0, 0.0, 0.0,
        0.0, cos(a), -sin(a),
        0.0, sin(a), cos(a)
    );
}

mat3 rotmatY(float a)
{
    return mat3(
        cos(a), 0.0, sin(a),
        0.0, 1.0, 0.0,
        -sin(a), 0.0, cos(a)
    );
}

mat3 rotmatZ(float a)
{
    return mat3(
        cos(a), -sin(a), 0.0,
        sin(a), cos(a), 0.0,
        0.0, 0.0, 1.0
    );
}

float sdArm(vec3 p, float s, bool anim)
{
    if (anim) {
        float t = mod(iTime, 3.0);
        if (t < 0.15) {
            p *= rotmatZ(t * 2.5);
            p *= rotmatY(t * 10.0);
        } else if (t < 1.0) {
            p *= rotmatZ(t * 2.5);
            p *= rotmatY(1.5);
        } else if (t < 1.8) {
            p *= rotmatZ(5.0 - t * 2.5);
            p *= rotmatY(1.5);
        } else if (t < 2.0) {
            p *= rotmatZ(5.0 - t * 2.5);
            p *= rotmatY(19.5 - t * 10.0);
        }
    }

    p.x *= 2.0;
    p *= rotmatZ(-s * 0.5);


    p.y += 0.12;
    return sdRoundCone(p, 0.09, 0.08, 0.12);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    vec4 res = vec4(1e10, vec3(0.0, 0.0, 0.0));

    res = vec4(
        sdRoundCone(p, 0.3, 0.26, 0.35),
        vec3(0.9, 0.8, 0.1)
    );

    float lArm = sdArm(p - vec3(-0.28, 0.16, 0.0), -1.0, true);
    float rArm = sdArm(p - vec3(0.28, 0.16, 0.0), 1.0, false);
    float arms = min(lArm, rArm);

    if (arms < res.x) {
        res = vec4(arms, vec3(0.9, 0.8, 0.1));
    }

    float lLeg = sdRoundCone(p - vec3(-0.15, -0.21, 0.0), 0.13, 0.12, 0.04);
    float rLeg = sdRoundCone(p - vec3(0.15, -0.21, 0.0), 0.13, 0.12, 0.04);
    float legs = min(lLeg, rLeg);

    if (legs < res.x) {
        res = vec4(legs, vec3(0.4, 0.3, 0.1));
    }

    return res;
}

vec4 sdEye(vec3 p, vec3 color)
{
    float radius = 0.1;

    float d = 1e10;
    d = sdSphere(p, radius);

    vec3 focus = vec3(1.0 + 2.5*sin(iTime), 5.0, 10.0);
    vec3 focus_dir = normalize(focus) * radius;
    float focus_dist = length(p - focus_dir);

    if (focus_dist < 0.01) {
        return vec4(d, vec3(0.0, 0.0, 0.0));
    }

    if (focus_dist < 0.05) {
        return vec4(d, color * min(abs(tanh((focus_dist / 0.05) - 0.5) / 0.5) + 0.2, 1.0));
    }

    return vec4(d, vec3(1.0, 1.0, 1.0));
}

vec4 sdEyes(vec3 p)
{
    vec4 res = vec4(1e10, vec3(0.0, 0.0, 0.0));

    vec4 l = sdEye(p + vec3(0.1, 0.0, 0.0), vec3(0.4, 0.7, 0.38));
    if (l.x < res.x) {
        res = l;
    }

    vec4 r = sdEye(p - vec3(0.1, 0.0, 0.0), vec3(0.3, 0.53, 0.48));
    if (r.x < res.x) {
        res = r;
    }

    return res;
}

vec4 sdMonster(vec3 p)
{
    vec4 res = sdBody(p);

    vec4 eye = sdEyes(p - vec3(0.0, 0.33, 0.2));
    if (eye.x < res.x) {
        res = eye;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.3, -0.65);

    // p *= rotmatY(-0.6);
    p *= rotmatY(iTime / 5.0);
    p *= rotmatZ(0.1);

    vec4 res = sdMonster(p);

    p += vec3(0.0, 0.3, -0.65);

    float dist = sdPlane(p);
    if (dist < res.x) {
        vec3 c = p + normalize(p) * dist;
        res = vec4(dist, vec3(0.2, 0.8, 0.2) - noise(c.xz * 215.0 + vec2(6.0 * sin(iTime), 2.0 * cos(iTime))) * 0.8);
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

    return vec4(1e10, vec3(0.1, 0.7, 1.0));
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

    if (res.x < 1e10) {
        vec3 surface_point = ray_origin + res.x*ray_direction;
        vec3 normal = calcNormal(surface_point);

        vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

        float shad = shading(surface_point, light_source, normal);
        shad = min(shad, castShadow(surface_point, light_source));
        col *= shad;

        float spec = specular(surface_point, light_source, normal, ray_origin, 40.0);
        col += vec3(1.0, 1.0, 1.0) * spec;
    }

    // Output to screen
    fragColor = vec4(col, 1.0);
}
