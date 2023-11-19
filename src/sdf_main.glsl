// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// exponential smooth min (k=32)
float smin( float a, float b, float k )
{
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
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


vec3 hash( vec3 p )
{
    return fract(sin(vec3(dot(p,vec3(127.1,311.7, 61.4)),dot(p,vec3(269.5,183.3, 431.4)),dot(p,vec3(263.8,165.4, 343.7))))*43758.5453);
}

float noise( in vec3 x )
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);

    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);


    // gradients
    vec3 ga = hash( p+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( p+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( p+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( p+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( p+vec3(0.0,0.0,1.0) );
    vec3 gf = hash( p+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( p+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( p+vec3(1.0,1.0,1.0) );

    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );

    // interpolation
    return va +
           u.x*(vb-va) +
           u.y*(vc-va) +
           u.z*(ve-va) +
           u.x*u.y*(va-vb-vc+vd) +
           u.y*u.z*(va-vc-ve+vg) +
           u.z*u.x*(va-vb-ve+vf) +
           u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

float fbm( in vec3 x, in float H )
{
    const int numOctaves = 2;
    float G = exp2(-H);
    float f = 6.0;
    float a = 0.25;
    float t = 0.0;
    for( int i=0; i<numOctaves; i++ )
    {
        t += a*noise(f*x);
        f *= 2.0;
        a *= G;
    }
    return t;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    d = sdSphere((p - vec3(0.0, 0.35, 0.0)), 0.35);

    float d2 = sdSphere(p - vec3(0.0, 0.68, 0.0), 0.27);
    d = smin(d, d2, 20.0) + 0.05 * fbm(p, .125);

    const int limb_num = 6;
    vec3 limbsA[limb_num];
    vec3 limbsB[limb_num];
    limbsA[0] = vec3(0.40, 0.36, 0.11);
    limbsB[0] = vec3(0.31, 0.45, 0.06);
    limbsA[2] = vec3(0.11, 0.16, 0.1);
    limbsB[2] = vec3(0.12, -0.02, 0.1);
    limbsA[4] = vec3(0.155, 0.87, 0.11);
    limbsB[4] = vec3(0.16, 0.90, 0.11);

    limbsA[1] = vec3(-limbsA[0].x, limbsA[0].yz);
    limbsB[1] = vec3(-limbsB[0].x, limbsB[0].yz);
    limbsA[3] = vec3(-limbsA[2].x, limbsA[2].yz);
    limbsB[3] = vec3(-limbsB[2].x, limbsB[2].yz);
    limbsA[5] = vec3(-limbsA[4].x, limbsA[4].yz);
    limbsB[5] = vec3(-limbsB[4].x, limbsB[4].yz);
    const float thickness = 0.055;

    for (int i = 0; i < limb_num; ++i) {
        d2 = sdCapsule(p, limbsA[i], limbsB[i], i < 4 ? thickness : thickness - 0.01);
        d = smin(d, d2, 160.0);
    }

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(sdSphere(p - vec3(0.0, 0.55, 0.2), 0.2), vec3(1.0));

    vec4 res2 = vec4(sdSphere(p - vec3(0.0, 0.565, 0.3), 0.12), vec3(0.0, 0.9, 1.0));
    if (res2.x < res.x)
        res = res2;
    res2 = vec4(sdSphere(p - vec3(0.0, 0.57, 0.35), 0.08), vec3(0.0));
    if (res2.x < res.x)
        res = res2;

    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, -0.7);

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
