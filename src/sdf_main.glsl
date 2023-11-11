// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// sphere with center in (0, 0.25, 0)
float sdSphereHead(vec3 p, float r)
{
    return length(p - vec3(0, 0.25, 0)) - r;
}

// sphere with center in (0, 0.2, 0.1)
float sdSphereEye(vec3 p, vec3 c, float r)
{
    return length(p - c) - r;
}

float dot2(in vec3 v ) { return dot(v,v); }
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

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// root smooth min (k=0.01)
float smin( float a, float b, float k )
{
    float h = a-b;
    return 0.5*( (a+b) - sqrt(h*h+k) );
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

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    vec3 pt = p - vec3(0.0, 0.35, -0.7);
    d = min(min(smin(sdSphere(pt, 0.35), sdSphereHead(pt, 0.25), 0.01),
            sdRoundCone(pt, vec3(-0.33, 0.0, 0.0), vec3(-0.38, -0.1 * lazycos(iTime * 3.0), 0.1), 0.06, 0.04)),
            sdRoundCone(pt, vec3(0.33, 0.0, 0.0), vec3(0.38, -0.1, 0.1), 0.06, 0.04));
    d = min(d, min(sdCapsule(pt, vec3(-0.1, -0.33, 0.0), vec3(-0.1, -0.38, 0.0), 0.05),
                   sdCapsule(pt, vec3(0.1, -0.33, 0.0), vec3(0.1, -0.38, 0.0), 0.05)));
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 eyeCenter = vec3(0, 0.2, 0.2);
    vec3 pt = p - vec3(0.0, 0.35, -0.7);
    float d = 1e10;
    d = sdSphereEye(pt, eyeCenter, 0.2);
    
    vec4 res;
    if (lazycos(iTime * 10.0) < -0.8) {
        return vec4(d, vec3(0.1, 1.0, 0.0));
    }
    if (length(vec2(eyeCenter.x, eyeCenter.y) - vec2(pt.x, pt.y)) < 0.05) {
        res = vec4(d, vec3(0.0, 0.0, 0.0));
    } else if (length(vec2(eyeCenter.x, eyeCenter.y) - vec2(pt.x, pt.y)) < 0.1) {
        res = vec4(d, vec3(0.0, 0.8, 1.0));
    } else {
        res = vec4(d, vec3(1.0, 1.0, 1.0));
    }
    
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
