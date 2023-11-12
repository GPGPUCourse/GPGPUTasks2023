float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }

float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

// https://stackoverflow.com/questions/45597118/fastest-way-to-do-min-max-based-on-specific-component-of-vectors-in-glsl
vec4 minx(vec4 a, vec4 b)
{
    return mix(a, b, step(b.x, a.x));
}

float sdRoundCone(vec3 p, float r1, float r2, float h)
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


float sdRoundCone2( vec3 p, vec3 a, vec3 b, float r1, float r2 )
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

float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

// XZ plane
vec4 sdPlane(vec3 p)
{
    return vec4(p.y, vec3(1.0, 0.0, 0.0));
}

vec4 sdSky(vec3 p)
{
    return vec4(1e10, 0.0, 0.0, 1.0);
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

// возможно, для конструирования тела пригодятся какие-то примитивы из набора 
//   https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: 
//   https://iquilezles.org/articles/smin/


float sdBodyHigh(vec3 p)
{
    return sdEllipsoid((p - vec3(0.0, 0.6, -0.7)), vec3(0.35, 0.45, 0.45));
}

float sdBodyLow(vec3 p)
{
    return sdEllipsoid((p - vec3(0.0, 0.3, -0.7)), vec3(0.45, 0.3, 0.45));
}


float sdLeg1(vec3 p)
{
    return sdRoundCone(p - vec3(-0.2, 0.0, -0.5), 0.1, 0.1, 0.1);
}

float sdLeg2(vec3 p)
{
    return sdRoundCone(p - vec3(0.2, 0.0, -0.5), 0.1, 0.1, 0.1);
}

float sdArm1(vec3 p)
{
    return sdRoundCone2(
        p - vec3(-0.35, 0.5, -0.5),
        //vec3(-0.15, 0.0, 0.0), 
        vec3(-0.15 - abs(0.07*cos(1.57 + iTime*0.9)), 0.2*cos(iTime*0.9), 0.0),
        vec3(0.0, 0.0, 0.0), 
        0.08,
        0.06
    );
}

float sdArm2(vec3 p)
{
    return sdRoundCone2(
        p - vec3(0.35, 0.5, -0.5),
        vec3(0.15, -0.1, 0.0), 
        vec3(0.0, 0.0, 0.0), 
        0.08,
        0.06
    );
}


    

vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    //d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    d = opSmoothUnion(sdBodyHigh(p), sdBodyLow(p), 0.1);
    d = opSmoothUnion(d, sdLeg1(p), 0.1);
    d = opSmoothUnion(d, sdLeg2(p), 0.1);
    d = opSmoothUnion(d, sdArm1(p), 0.04);
    d = opSmoothUnion(d, sdArm2(p), 0.04);
    
    // return distance and color
    return vec4(d, vec3(0.0, 0.8, 0.0));
}

vec4 sdEye(vec3 p)
{
    const float eyeRadius = 0.2;
    const vec3 eyeCenter = vec3(0.0, 0.57, -0.4);
    const float irisRadius = 0.1;
    const float pupilRadius = 0.05;
    float eyelidHeight = eyeCenter.y + eyeRadius * cos(1.57 + iTime*0.6);
    vec3 color = vec3(1.0, 1.0, 1.0);
    float d = sdSphere((p - eyeCenter), eyeRadius);
    if (p.y > eyelidHeight) {
        color = vec3(0.0, 0.8, 0.0);
    }
    else if (distance(p.xy, eyeCenter.xy) < pupilRadius) {
        color = vec3(0.0, 0.0, 0.0);
    } else if (distance(p.xy, eyeCenter.xy) < irisRadius) {
        color = vec3(0.3, 0.0, 0.0);
    }
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
    vec4 monster = sdMonster(p);    
    vec4 plane = sdPlane(p);
    vec4 sky = sdSky(p);
    
    return minx(monster, minx(plane, sky));
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