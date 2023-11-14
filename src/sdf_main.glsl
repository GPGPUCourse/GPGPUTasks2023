
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

float smin( float a, float b )
{
    float k = 0.01;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}

vec4 smin2 (vec4 d1, vec4 d2, float k)
{
    float h = clamp (.5 + .5*(d2.a - d1.a)/k, .0, 1.);
    return mix (d2, d1, h) - h*k*(1. - h);
}

// Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

// Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

#define PI     3.14159265

// Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    //float c = lazycos(theta);
    //float s = lazycos(PI / 2.0 - theta);
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;
    float offset = -0.18;
    float handR1 = 0.05;
    float handR2 = 0.03;
    float handHeight = 0.15;
    float zOffset = 0.08;

    float cone = sdRoundCone(p - vec3(0, 0.3, 0), 0.25, 0.2, 0.3);
    float zAngle = mix(4.0, 4.8, 1.0 - lazycos(iTime * 5.0));
    float xAngle = mix(0.0, 0.3, 1.0 - lazycos(iTime * 5.0));
    float leftHand = sdRoundCone(
        (p - vec3(offset, 0.4, zOffset)) * rotateZ(zAngle) * rotateX(xAngle),
        handR1, handR2, handHeight
    );
    float rightHand = sdRoundCone(
        (p - vec3(-offset, 0.4, zOffset)) * rotateZ(-4.0) * rotateX(0.0),
        handR1, handR2, handHeight
    );

    float legR = 0.05;
    float legOffset = 0.08;
    float leftLeg = sdRoundCone(
        (p - vec3(legOffset, 0, zOffset)),
        legR, legR, 0.1
    );
    float rightLeg = sdRoundCone(
        (p - vec3(-legOffset, 0, zOffset)),
        legR, legR, 0.1
    );

    float legMin = min(leftLeg, rightLeg);
    float handMin = min(leftHand, rightHand);

    float partMin = min(legMin, handMin);
    // return distance and color
    return vec4(smin(cone, partMin), vec3(0.0, 1.0, 0.0));
}

vec3 RED = vec3(1., 0., 0.0);
vec3 GREEN = vec3(0, 1, 0);
vec3 BLUE = vec3(0, 0, 1);
vec3 BLACK = vec3(0, 0, 0);
vec3 CYAN = vec3(0, 1, 1);
vec3 WHITE = vec3(1, 1, 1);

vec4 sdEye(vec3 p)
{
    p -= vec3(0, 0.57, 0.12);
    vec4 whiteEye = vec4(sdSphere(p, 0.15), WHITE);
    vec4 cyanEye = vec4(sdSphere(p - vec3(0, 0, 0.08), 0.09), CYAN);
    vec4 blackEye = vec4(sdSphere(p - vec3(0, 0, 0.15), 0.04), BLACK);

    if (whiteEye.x < cyanEye.x) {
        return whiteEye;
    }
    if (cyanEye.x < blackEye.x) {
        return cyanEye;
    }
    return blackEye;

    //   return min(whiteEye, cyanEye, 16.);
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
