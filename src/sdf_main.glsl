
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// ellipsoid with center in (0, 0, 0)
float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
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

// exponential smooth min (k=32)
float smin( float a, float b, float k )
{
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
}

mat3 rotate_mat(float theta) {
   return mat3(vec3(cos(theta), -sin(theta), 0),
               vec3(sin(theta), cos(theta), 0),
               vec3(0, 0, 1));

  // return mat3(vec3(1, 0, 0),
  //            vec3(0, cos(theta), -sin(theta)),
  //             vec3(0, sin(theta), cos(theta)));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    d = sdEllipsoid((p - vec3(0.0, 0.55, -0.7)), vec3(0.33, 0.5, 0.4));

    float toBottom = sdSphere(p - vec3(0.0, 0.4, -0.7) ,0.36);

    d = smin(d, toBottom, 32.0);

    float toLegLeft   = sdRoundCone(p-vec3(-0.15,0.0,-0.7), 0.1, 0.1, 0.1);
    float toLegRight  = sdRoundCone(p-vec3(0.15, 0.0,-0.7), 0.1, 0.1, 0.1);


    mat3 rotLeft = rotate_mat( 1.0 + (1.0-lazycos(iTime*6.0)) * 0.5 );
    mat3 rotRight = rotate_mat(4.0);

    float toHandLeft  = sdRoundCone(rotLeft * (p-vec3(-0.38,0.5,-0.8)), 0.07, 0.1, 0.2);
    float toHandRight = sdRoundCone(rotRight * (p-vec3(0.38, 0.5,-0.8)), 0.07, 0.1, 0.2);

    d = smin(d, toLegLeft, 128.0);
    d = smin(d, toLegRight, 128.0);

    d = min(d, toHandLeft);
    d = min(d, toHandRight);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    const int sz = 3;

    float r[sz];
    r[0] = 0.2;
    r[1] = 0.13;
    r[2] = 0.07;

    vec3 c[sz];
    c[0] = vec3(0.0, 0.65, -0.4);
    c[1] = vec3(0.0, 0.65, -0.3);
    c[2] = vec3(0.0, 0.65, -0.2);

    vec3 col[sz];
    col[0] = vec3(1.0, 1.0, 1.0);
    col[1] = vec3(0.0, 0.95, 1.0);
    col[2] = vec3(0.0, 0.0, 0.0);

    vec4 res = vec4(1e10, 1.0, 1.0, 1.0);
    for (int i = 0; i < sz; ++i) {
         float now = sdSphere(p - c[i], r[i]);
         if (now < res.x) {
             res.x = now;
             res.y = col[i].x;
             res.z = col[i].y;
             res.w = col[i].z;
         }
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