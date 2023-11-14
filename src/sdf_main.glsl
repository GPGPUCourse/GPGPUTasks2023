// we need this
float dot2( in vec3 v ) { return dot(v,v); }

float opU(float a, float b)
{
    return min(a, b);
}

// polynomial smooth min 2 (k=0.1)
float opSoftU( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}

float sdVesicaSegment( in vec3 p, in vec3 a, in vec3 b, in float w )
{
    vec3  c = (a+b)*0.5;
    float l = length(b-a);
    vec3  v = (b-a)/l;
    float y = dot(p-c,v);
    vec2  q = vec2(length(p-c-y*v),abs(y));
    
    float r = 0.5*l;
    float d = 0.5*(r*r-w*w)/w;
    vec3  h = (r*q.x<d*(q.y-r)) ? vec3(0.0,r,0.0) : vec3(-d,0.0,d+w);
 
    return length(q-h.xy) - h.z;
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdBoxTx(vec3 p, vec3 b, mat3 t)
{
    return sdBox( inverse(t)*p, b);
}

float sdVesicaSegmentTx(vec3 p,vec3 a,vec3 b,float w ,mat3 t)
{
    return sdVesicaSegment( inverse(t)*p, a, b, w);
}

#define PI 3.1415926538
vec2 circle(float t, float r, float speed) {
    t = mod(t, (PI * 2.0)) - PI;
    t *= speed;
    float x = r * sin(t);
    float y = r * cos(t);
    return vec2(x, y);
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

float ass(vec3 p)
{
    float assR = 0.35, assAmp = 0.05, assSpeed = 10.0;
    vec2 circlePoint = circle(iTime, assAmp, assSpeed);
    // TODO
    vec3 startPoint = -vec3(0.0, 0.35, -0.2);
    startPoint += p;
    startPoint.x += circlePoint.x; startPoint.y += circlePoint.y;
    return sdSphere(startPoint, assR);
}

float head(vec3 p)
{
    float headR = 0.21, headAmp = 0.03, headSpeed = 4.0, headY = 0.7, headX = -0.1, headZ = -0.20;
    vec2 circlePoint = circle(iTime, headAmp, headSpeed);
    vec3 startPoint = -vec3(headX, headY - circlePoint.y, headZ);
    startPoint += p;
    return sdSphere(startPoint, headR);
}

float legs(vec3 p)
{
    float legH = 0.25, legBotR = 0.1, legTopR = 0.11, legZ = -0.19;
    float legDistance = 0.13;
    
    float lLeg = sdRoundCone(
    (p - vec3(-legDistance, 0.0, legZ)),
    legTopR,legBotR,legH);
    float rLeg = sdRoundCone(
    (p - vec3(legDistance, 0.0, legZ)),
    legTopR,legBotR,legH);
    return opU(lLeg, rLeg);
}

float backPack(vec3 p)
{    
    // ass circle
    float assAmp = 0.05, assSpeed = 10.0;
    vec2 assCirclePoint = circle(iTime, assAmp, assSpeed);
    
    float bpR = 0.08, bpZ = 0.05, bpY = 0.5, bpX = 0.0;
    vec3 startPoint = -vec3(bpX, bpY, bpZ);
    startPoint += p;
    
    startPoint.x += assCirclePoint.x;
    startPoint.y += assCirclePoint.y;
    
    float xRotationA = 0.2;
    mat3 transform;
    transform[0] = vec3(1.0, 0.0, 0.0);
    transform[1] = vec3(0.0, cos(xRotationA), -sin(xRotationA));
    transform[2] = vec3(0.0, cos(xRotationA), sin(xRotationA));
    
    return sdBoxTx(
    startPoint, 
    vec3(0.1, 0.15, 0.1), 
    transform);
    /*
    sdCappedCylinder(
        startPoint,
        vec3(-0.05, 0.08 - headCirclePoint.y, 0.0),
        vec3(0.05 - assCirclePoint.x, -0.08 - assCirclePoint.y, 0.06),
        bpR
    );
    */
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;
    
    // TODO
    float smoothFactor = 0.05;
    
    float assD = ass(p);
    float headD = head(p);
    float legsD = legs(p);
    float bpD = backPack(p);
    d = opSoftU(assD, headD, smoothFactor);
    d = opSoftU(d, legsD, smoothFactor);
    d = opU(d, bpD);
    // d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    
    // return distance and color
    return vec4(d, vec3(1.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    // head circle
    float headR = 0.21, headAmp = 0.03, headSpeed = 4.0;
    vec2 headCirclePoint = circle(iTime, headAmp, headSpeed);
    
    float eyeR = 0.1, eyeX = -0.20, eyeY = 0.75, eyeZ = 0.0;
    vec3 startPoint = -vec3(eyeX, eyeY - headCirclePoint.y, eyeZ);
    startPoint += p;
    mat3 transform;
    float yTransform = abs(headCirclePoint.y) / headAmp / 2.0;
    transform[0] = vec3(1.0, 0.0, 0.0);
    transform[1] = vec3(0.0, 0.5 + yTransform, 0.0);
    transform[2] = vec3(0.0, 0.0, 1.0);
    
    float eyeD = sdVesicaSegmentTx(startPoint,
    vec3(-0.02, 0.0, 0.0),
    vec3(0.02, 0.0, 0.0),
    eyeR,
    transform
    );

    vec4 res = vec4(eyeD, vec3(1.0, 1.0, 1.0));
    
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
