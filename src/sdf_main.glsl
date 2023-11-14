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

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

#define PI 3.1415926538
vec2 circle(float t, float r, float speed) {
    t = mod(t, (PI * 2.0)) - PI;
    t *= speed;
    float x = r * sin(t);
    float y = r * cos(t);
    return vec2(x, y);
}

// capsule
float sdCappedCylinder( vec3 p, vec3 a, vec3 b, float r )
{
  vec3  ba = b - a;
  vec3  pa = p - a;
  float baba = dot(ba,ba);
  float paba = dot(pa,ba);
  float x = length(pa*baba-ba*paba) - r*baba;
  float y = abs(paba-baba*0.5)-baba*0.5;
  float x2 = x*x;
  float y2 = y*y*baba;
  float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
  return sign(d)*sqrt(abs(d))/baba;
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
    float assR = 0.35, assAmp = 0.05, assSpeed = 6.0;
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
    // head circle
    float headR = 0.21, headAmp = 0.03, headSpeed = 4.0;
    vec2 headCirclePoint = circle(iTime, headAmp, headSpeed);
    
    // ass circle
    float assAmp = 0.05, assSpeed = 6.0;
    vec2 assCirclePoint = circle(iTime, assAmp, assSpeed);
    
    float bpR = 0.08, bpZ = 0.05, bpY = 0.61, bpX = 0.0;
    vec3 startPoint = -vec3(bpX, bpY, bpZ);
    startPoint += p;
    
    return sdCappedCylinder(
        startPoint,
        vec3(-0.05, 0.08 - headCirclePoint.y, 0.0),
        vec3(0.05 - assCirclePoint.x, -0.08 - assCirclePoint.y, 0.06),
        bpR
    );
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
    d = opSoftU(d, bpD, smoothFactor);
    // d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    
    // return distance and color
    return vec4(d, vec3(1.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    // head circle
    float headR = 0.21, headAmp = 0.03, headSpeed = 4.0;
    vec2 headCirclePoint = circle(iTime, headAmp, headSpeed);

    float eyeR = 0.05, eyeX = -0.25, eyeY = 0.75, eyeZ = -0.1;
    vec3 startPoint = -vec3(eyeX, eyeY - headCirclePoint.y, eyeZ);
    startPoint += p;
    
    
    float eyeD = sdSphere(startPoint, eyeR);

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
