// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

vec2 opRevolution( in vec3 p, float w )
{
    return vec2( length(p.xz) - w, p.y );
}

// https://iquilezles.org/articles/distfunctions2d
float sdVesica(vec2 p, float r, float d)
{
    p = abs(p);

    float b = sqrt(r*r-d*d); // can delay this sqrt
    return ((p.y-b)*d > p.x*b) 
            ? length(p-vec2(0.0,b))
            : length(p-vec2(-d,0.0))-r;
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

// (ко)синус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazysin(float angle)
{
    int nsleep = 4;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return sin(angle);
    }

    return 0.0;
}

// polynomial smooth min
float smin( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}

float sdHand(vec3 p, bool isLeft)
{
    float rotate = (isLeft ? 0.4 : -0.4) * lazysin(2. * iTime);
    vec3 start = vec3(isLeft ? 0.3 : -0.3, 0.7, -1.6);
    vec3 angleStart = vec3(0., 0., 0.);
    vec3 angleEnd = vec3(isLeft ? 0.2 : -0.2, rotate, 0.2);
    return sdCapsule(p - start, angleStart, angleEnd, 0.05);
}

float sdLeg(vec3 p, bool isLeft)
{
    float rotate = (isLeft ? 0.2 : -0.2) * lazysin(2. * iTime);
    vec3 start = vec3(isLeft ? 0.2 : -0.2, 0.3, -1.6);
    vec3 angleStart = vec3(0., 0., 0.);
    vec3 angleEnd = vec3(0., -0.3, rotate);
    return sdCapsule(p - start, angleStart, angleEnd, 0.06);
}

float sdCone( vec3 p, vec2 c, float h )
{
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);
    
  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;
    
    // Main body
    vec3 q = p - vec3(0.0, 0.6, -1.65);
    d = min(d, sdVesica(opRevolution(q, 0.15), 0.4, 0.2));
    
    float rightHand = sdHand(p, false);
    float leftHand = sdHand(p, true);
    d = smin(smin(d, rightHand, 0.1), leftHand, 0.1);
    
    float rightLeg = sdLeg(p, false);
    float leftLeg = sdLeg(p, true);
    d = smin(smin(d, rightLeg, 0.1), leftLeg, 0.1);
    
    //Party hat?!
    float hat = sdCone(p - vec3(0.0, 1.09, -1.6), vec2(1., 1.), 0.15); 
    if (hat < d)
    {
        d = smin(d, hat, 0.05);
        return vec4(d, vec3(1., 0., 1.));
    }

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float white_d = sdSphere((p - vec3(0.0, 0.6, -1.45)), 0.2);
    float black_d = sdSphere((p - vec3(0.0, 0.6, -1.3)), 0.1);
    float d = white_d;
    if (black_d < white_d)
    {
        d = smin(black_d, white_d, 0.2);
        return vec4(d, vec3(0., 0., 0.));
    }
    return vec4(d, vec3(1.0, 1.0, 1.0));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    
    if (eye.x < res.x) {
        eye.x = smin(eye.x, res.x, 0.2);
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
