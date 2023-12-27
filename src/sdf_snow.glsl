float Hash11(float p)
{
  vec3 p3 = fract(vec3(p) * 0.1);
  p3 += dot(p3, p3.yzx + 19.19);
  return fract((p3.x + p3.y) * p3.z); 
}

vec2 Hash22(vec2 p)
{
  vec3 p3 = fract(vec3(p.xyx) * 0.3);
  p3 += dot(p3, p3.yzx+19.19);
  return fract((p3.xx+p3.yz)*p3.zy);
}

vec2 Rand22(vec2 co)
{
  float x = fract(sin(dot(co.xy ,vec2(122.9898,783.233))) * 43758.5453);
  float y = fract(sin(dot(co.xy ,vec2(457.6537,537.2793))) * 37573.5913);
  return vec2(x,y);
}

vec3 SnowSingleLayer(vec2 uv,float layer){
  vec3 acc = vec3(0.0,0.0,0.0);
  uv = uv * (2.0 + layer);
  float xOffset = uv.y * (((Hash11(layer)*2.-1.)*0.5+1.) * 0.5);
  float yOffset = 0.75 * iTime;
  uv += vec2(xOffset,yOffset);
  vec2 rgrid = Hash22(floor(uv)+(31.1759*layer));
  uv = fract(uv) - (rgrid*2.-1.0) * 0.35 - 0.5;
  float r = length(uv);
  float circleSize = 0.04*(1.5+0.3*sin(iTime*0.1));
  float val = smoothstep(circleSize,-circleSize,r);
  vec3 col = vec3(val,val,val)* rgrid.x ;
  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;

  vec3 acc = vec3(0,0,0);
  for (float i = 0.; i < 20.0; i++) {
    acc += SnowSingleLayer(uv,i); 
  }

  fragColor = vec4(acc,1.0);
}
