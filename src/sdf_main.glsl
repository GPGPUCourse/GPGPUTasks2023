// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

float sdDeathStar( vec3 p2, float ra, float rb, float d )
{
  // sampling independent computations (only depend on shape)
  float a = (ra*ra - rb*rb + d*d)/(2.0*d);
  float b = sqrt(max(ra*ra-a*a,0.0));
	
  // sampling dependant computations
  vec2 p = vec2( p2.x, length(p2.yz) );
  if( p.x*b-p.y*a > d*max(b-p.y,0.0) )
    return length(p-vec2(a,b));
  else
    return max( (length(p            )-ra),
               -(length(p-vec2(d,0.0))-rb));
}

float smin( float a, float b, float k )
{
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
}

float opU( float d1, float d2 )
{
    return smin( d1, d2, 32.0 );
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 5;
    
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
    float d = 1e10;

    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    d = opU(d, sdEllipsoid((p - vec3(0.0, 0.57, -0.7)), vec3(0.27, 0.25, 0.25)));
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdBackground(vec3 p) {
    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);
    return res;
}

vec4 sdEyeball(vec3 p) {
    float d = 1e10;
    d = sdDeathStar((vec3(p.z, p.x, p.y) - vec3(-0.45, 0.0, 0.6)), 0.144, 0.1008, 0.1152);
    return vec4(d, vec3(1.0, 1.0, 1.0));
}

vec4 sdIris(vec3 p) {
    float d = 1e10;
    d = sdDeathStar((vec3(p.z, p.x, p.y) - vec3(-0.35, 0.0, 0.6)), 0.1008, 0.07, 0.100);
    return vec4(d, vec3(0.18, 0.83, 0.78));
}

vec4 sdPupil(vec3 p) {
    float d = 1e10;
    d = sdSphere((p - vec3(0.0, 0.6, -0.3)), 0.07);
    return vec4(d, vec3(0.0, 0.0, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec4 eyeball = sdEyeball(p);
    
    vec4 iris = sdIris(p);
    
    vec4 pupil = sdPupil(p);
    
    vec4 eye = eyeball;
    
    if (eyeball.x < eye.x) {
        eye = eyeball;
    }
    if (iris.x < eye.x) {
        eye = iris;
    }
    if (pupil.x < eye.x) {
        eye = pupil;
    }
    
    return eye;
}

vec4 sdLegs(vec3 p) {
    float d = 1e10;
    float leftD = 1e10;
    float rightD = 1e10;
    
    leftD = sdVerticalCapsule(p - vec3(-0.06, 0.0, -0.55), 0.2, 0.05);
    rightD = sdVerticalCapsule(p - vec3(0.06, 0.0, -0.55), 0.2, 0.05);
    
    d = min(leftD, rightD);
    
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdArms(vec3 p) {
    float d = 1e10;
    float leftD = 1e10;
    float rightD = 1e10;
    
    leftD = sdCapsule(p - vec3(-0.35, 0.3, -0.59), vec3(0.1, 0.15, 0.0), 
        vec3(0.0, -lazycos(iTime * 4.0) / 8.0 + 0.125, 0.0), 0.05);
    rightD = sdCapsule(p - vec3(0.25, 0.45, -0.59), vec3(0.1, -0.15, 0.0), vec3(0.0, 0.0, 0.0), 0.05);
       
    d = min(leftD, rightD);
    
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 body = sdBody(p);
    
    vec4 eye = sdEye(p);
    
    vec4 background = sdBackground(p);
    
    vec4 legs = sdLegs(p);
    
    vec4 arms = sdArms(p);
    
    vec4 res = body;
      
    if (eye.x < res.x) {
        res = eye;
    }
    if (background.x < res.x) {
        res = background;
    }
    if (legs.x < res.x) {
        res = legs;
    }
    if (arms.x < res.x) {
        res = arms;
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