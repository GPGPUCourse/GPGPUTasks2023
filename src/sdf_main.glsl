float smin_exp(float a, float b, float k) {
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
}

/// @name Scene
/// @{

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
float lazysin(float angle)
{
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return sin(angle + 1.57079632679);
    }
    
    return 1.0;
}



vec4 sdBody(vec3 p)
{
    float d = 1e10;

    vec3 body_center = vec3(0.0, 0.35, -0.7);
    d = sdSphere((p - body_center), 0.35);
    
    vec3 head_center = body_center + vec3(0.0, 0.23, 0.0);
    d = smin_exp(d, sdSphere(p - head_center, 0.29), 32.0);
    
    float arm_radius = 0.06;
    float arm_smin_exp_k = 200.0;
    
    vec3 left_arm_pivot = body_center - vec3(0.3, 0.0, -0.1);
    vec3 left_arm_end = left_arm_pivot - vec3(0.06 + 0.02 * lazycos(iTime * 10.0), 0.1 * lazysin(iTime * 10.0), 0.0);
    d = smin_exp(d, sdCapsule(p, left_arm_pivot, left_arm_end, arm_radius), arm_smin_exp_k);
    
    vec3 right_arm_pivot = body_center - vec3(-0.3, 0.0, -0.1);
    vec3 right_arm_end = right_arm_pivot + vec3(0.06, -0.1, 0.0);
    d = smin_exp(d, sdCapsule(p, right_arm_pivot, right_arm_end, arm_radius), arm_smin_exp_k);
    
    vec3 left_leg_pivot = body_center + vec3(0.1, -0.3, 0.1);
    vec3 left_leg_end = left_leg_pivot + vec3(0.0, -0.1, 0.0);
    d = smin_exp(d, sdCapsule(p, left_leg_pivot, left_leg_end, arm_radius), arm_smin_exp_k);
    
    vec3 right_leg_pivot = body_center + vec3(-0.1, -0.3, 0.1);
    vec3 right_leg_end = right_leg_pivot + vec3(0.0, -0.1, 0.0);
    d = smin_exp(d, sdCapsule(p, right_leg_pivot, right_leg_end, arm_radius), arm_smin_exp_k);
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);
    
    vec3 center = vec3(0.0, 0.58, -0.48);
    
    float white_dist = sdSphere(p - center, 0.18);
    if (res.x > white_dist) {
        res = vec4(white_dist, 1.0, 1.0, 1.0);
    }
    
    float cyan_dist = sdSphere(p - center - vec3(0, 0, 0.049), 0.14);
    if (res.x > cyan_dist) {
        res = vec4(cyan_dist, 0.0, 1.0, 1.0);
    }
    
    float black_dist = sdSphere(p - center - vec3(0, 0, 0.083), 0.11);
    if (res.x > black_dist) {
        res = vec4(black_dist, 0.0, 0.0, 0.0);
    }
    
    return res;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec4 sdLabel(vec3 p) {
    float d = sdBox(p - vec3(-1.1, 0, -0.5), vec3(0.03, 0.4, 0.03));
    d = min(d, sdBox(p - vec3(-0.9, 0, -0.5), vec3(0.03, 0.4, 0.03)));
    d = min(d, sdBox(p - vec3(-1, 0.17, -0.5), vec3(0.1, 0.03, 0.03)));
    d = min(d, sdBox(p - vec3(-0.65, 0, -0.5), vec3(0.03, 0.4, 0.03)));
    d = min(d, sdBox(p - vec3(-0.65, 0.37, -0.5), vec3(0.1, 0.03, 0.03)));
    d = min(d, sdBox(p - vec3(-0.65, -0.05, -0.5), vec3(0.1, 0.03, 0.03)));
    return vec4(d, 0, 0, 1);
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
    
    vec4 other = sdLabel(p);
    if (other.x < res.x)
        res = other;
    
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

/// @}


/// \pre length(ray_direction) = 1
/// \return (dist, color)
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

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
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
    vec2 uv = fragCoord / iResolution.y;
    
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.00));
    

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

