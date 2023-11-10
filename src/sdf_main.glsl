
const vec3 RED = vec3(1.0, 0.0, 0.0);
const vec3 GREEN = vec3(0.0, 1.0, 0.0);
const vec3 BLUE = vec3(0.0, 0.0, 1.0);
const vec3 BLACK = vec3(0.0, 0.0, 0.0);
const vec3 WHITE = vec3(1.0, 1.0, 1.0);
const vec3 EYE_COLOR = vec3(24, 159, 237) / 255.0;

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

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}


float sdCutHollowSphere( vec3 p, float r, float h, float t )
{
  // sampling independent computations (only depend on shape)
  float w = sqrt(r*r-h*h);

  // sampling dependant computations
  vec2 q = vec2( length(p.xz), p.y );
  return ((h*q.x<w*q.y) ? length(q-vec2(w,h)) :
                          abs(length(q)-r) ) - t;
}

vec3 rotateY(vec3 pos, float theta) {
    float X = pos.x*cos(theta) + pos.z*sin(theta);
    float Y = pos.y;
    float Z = pos.z*cos(theta) - pos.x*sin(theta);
    return vec3(X, Y, Z);
}

vec3 rotateX(vec3 pos, float theta) {
    float X = pos.x;
    float Y = pos.y*cos(theta) - pos.z*sin(theta);
    float Z = pos.y*sin(theta) + pos.z*cos(theta);
    return vec3(X, Y, Z);
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

float smin( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}

vec4 upd(vec4 res, float new_dist, vec3 new_col) {
    if (new_dist < res.x) {
        return vec4(new_dist, new_col);
    }
    else {
        return res;
    }
}



// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    vec4 res = vec4(1e10, BLACK);

    // body
    const float body_depth = -0.7;

    res = upd(res,
            smin(sdSphere(p - vec3(0.0, 0.35, body_depth), 0.35),
                sdSphere(p - vec3(0.0, 0.7, body_depth + 0.05), 0.20),
                0.3),
            GREEN);

    // legs
    const float leg_rad = 0.07;
    const float leg_len = 0.20;
    const float leg_dist = 0.1;
    const float leg_h_pos = 0.1;

    res = upd(res,
            sdCapsule(p, vec3(-leg_dist, leg_h_pos, body_depth), vec3(-leg_dist, leg_h_pos - leg_len, body_depth), leg_rad),
            GREEN);

    res = upd(res,
            sdCapsule(p, vec3(leg_dist, leg_h_pos, body_depth), vec3(leg_dist, leg_h_pos - leg_len, body_depth), leg_rad),
            GREEN);

    // hands
    const float hand_dist = 0.30;
    const float hand_len = 0.1;
    const float hand_h_pos = 0.35;
    const float hand_depth_delta = 0.05;
    const float hand_rad = 0.12;
    const float hand_rotation_speed = 10.0;

    const float hand_depth = body_depth + hand_depth_delta;
    res = upd(res,
        sdCapsule(p, vec3(-hand_dist, hand_h_pos, hand_depth), vec3(-hand_dist - hand_len, hand_h_pos - hand_rad * lazycos(hand_rotation_speed * iTime), hand_depth + hand_rad * (1.0 - lazycos(hand_rotation_speed * iTime))), leg_rad),
        GREEN);

    res = upd(res,
        sdCapsule(p, vec3(hand_dist, hand_h_pos, hand_depth), vec3(hand_dist + hand_len, hand_h_pos - hand_rad, hand_depth), leg_rad),
        GREEN);



    return res;
}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, BLACK);

    vec3 pos = rotateX(p - vec3(0.0, 0.55, -0.5), 1.7);
    const float r = 0.2;
    const float t = 0.01;
    const float eps = 1e-5;

    res = upd(res,
        sdCutHollowSphere(pos, r, 0.0, t),
        WHITE);

    res = upd(res,
        sdCutHollowSphere(pos, r + eps, -r * 0.8, t),
        EYE_COLOR);

    res = upd(res,
        sdCutHollowSphere(pos, r + 2.0 * eps, -r * 0.95, t),
        BLACK);

    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.17, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    // return eye;
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
