float dot2(in vec3 v)
{
    return dot(v,v);
}

float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdCutSphere( vec3 p, float r, float h )
{
    // sampling independent computations (only depend on shape)
    float w = sqrt(r*r-h*h);

    // sampling dependant computations
    vec2 q = vec2( length(p.xz), p.y );
    float s = max( (h-r)*q.x*q.x+w*w*(h+r-2.0*q.y), h*q.x-w*q.y );
    return (s<0.0) ? length(q)-r :
    (q.x<w) ? h - q.y     :
    length(q-vec2(w,h));
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );

    return length( pa - ba*h ) - r;
}

float sdRoundCone( vec3 p, vec3 a, vec3 b, float r1, float r2 )
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;

    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
    return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}


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
float lazycos_close_eye(float angle)
{
    int nsleep = 7;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 2) {
        return cos(angle);
    }

    return 1.0;
}

float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 5) {
        return cos(angle);
    }

    return 0.0;
}


float lazysin(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 5) {
        return sin(angle);
    }

    return -1.0;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float body = sdRoundCone(p, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.2, 0.0), 0.2, 0.16);
    vec4 res = vec4(body, vec3(0.0, 1.0, 0.0));

    float right_hand = sdCapsule(p, vec3(0.23, 0.0, 0.0), vec3(0.15, 0.1, 0.0), 0.03);
    right_hand = smin(body, right_hand, 0.01);
    vec4 res_right_hand = vec4(right_hand, vec3(0.0, 1.0, 0.0));

    if (res_right_hand.x < res.x) {
        res = res_right_hand;
    }

    float x = -0.23;
    float y = (lazysin(10.0 * iTime) + 1.0) / 13.0;
    float z = abs(lazycos(10.0 * iTime) / 20.0);

    float left_hand = sdCapsule(p, vec3(x, y, z), vec3(-0.15, 0.1, 0.0), 0.03);
    left_hand = smin(body, left_hand, 0.01);
    vec4 res_left_hand = vec4(left_hand, vec3(0.0, 1.0, 0.0));

    if (res_left_hand.x < res.x) {
        res = res_left_hand;
    }

    float right_leg = sdCapsule(p, vec3(0.05, 0.0, 0.0), vec3(0.05, -0.22, 0.0), 0.03);
    right_leg = smin(body, right_leg, 0.01);
    vec4 res_right_leg = vec4(right_leg, vec3(0.0, 1.0, 0.0));

    if (res_right_leg.x < res.x) {
        res = res_right_leg;
    }

    float left_leg = sdCapsule(p, vec3(-0.05, 0.0, 0.0), vec3(-0.05, -0.22, 0.0), 0.03);
    left_leg = smin(body, left_leg, 0.01);
    vec4 res_left_leg = vec4(left_leg, vec3(0.0, 1.0, 0.0));

    if (res_left_leg.x < res.x) {
        res = res_left_leg;
    }

    float right_ear = sdCapsule(p, vec3(0.15, 0.37, 0.0), vec3(0.11, 0.31, 0.0), 0.01);
    right_ear = smin(body, right_ear, 0.032);
    vec4 res_right_ear = vec4(right_ear, vec3(0.0, 1.0, 0.0));

    if (res_right_ear.x < res.x) {
        res = res_right_ear;
    }

    float left_ear = sdCapsule(p, vec3(-0.15, 0.37, 0.0), vec3(-0.11, 0.31, 0.0), 0.01);
    left_ear = smin(body, left_ear, 0.03);
    vec4 res_left_ear = vec4(left_ear, vec3(0.0, 1.0, 0.0));

    if (res_left_ear.x < res.x) {
        res = res_left_ear;
    }

    return res;
}

vec4 sdEye(vec3 p)
{
    float d_white = sdSphere((p - vec3(0.0, 0.2, 0.11)), 0.1);
    vec4 eye_white = vec4(d_white, vec3(1.5, 1.4, 1.6));

    float d_blue = sdSphere((p - vec3(0.0, 0.2, 0.165)), 0.06);
    d_blue = smin(d_blue, d_white, 0.001);
    vec4 eye_blue = vec4(d_blue, vec3(0.0, 1.4, 23.0));

    if (eye_blue.x < eye_white.x) {
        eye_white = eye_blue;
    }

    float d_black = sdSphere((p - vec3(0.0, 0.2, 0.19)), 0.04);
    d_black = smin(d_blue, d_black, 0.0001);
    vec4 eye_black = vec4(d_black, vec3(0.0, 0.0, 0.0));

    if (eye_black.x < eye_white.x) {
        eye_white = eye_black;
    }

    return eye_white;
}

vec4 sdCloseEye(vec3 p)
{
    vec3 a = vec3(0.0, 0.2, 0.13);
    float d = sdCutSphere((p - a), 0.1, 0.1*lazycos_close_eye(iTime * 10.0));
    vec4 close_eye = vec4(d, vec3(0.0, 1.0, 0.0));

    return close_eye;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.45, 0.0);

    vec4 res = sdBody(p);
    vec4 eye = sdEye(p);

    if (eye.x < res.x) {
        res = eye;
    }

    vec4 close_eye = sdCloseEye(p);

    if (close_eye.x < res.x) {
        res = close_eye;
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
    float EPS = 1e-6;
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

    vec3 ray_origin = vec3(0.0, 0.55, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));

    vec4 res = raycast(ray_origin, ray_direction);

    vec3 col = res.yzw;

    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 1.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;

    // Output to screen
    fragColor = vec4(col, 1.0);
}
