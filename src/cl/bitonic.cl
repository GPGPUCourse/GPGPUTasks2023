__kernel void bitonic(
    __global float *as,
    const unsigned int lvl,
    const unsigned int outer_lvl) 
{
    const unsigned x = get_global_id(0);

    int shift = (1<<lvl);
    int gid = x/shift, lid = x - gid*shift; 
    int a = gid*(shift<<1) + lid, b = a + shift;    
    if (((x/(1<<outer_lvl))&1) > 0) {
        int dop = b;
        b = a;
        a = dop;
    }

    if(as[a]>as[b]) {
        float dop = as[b];
        as[b] = as[a];
        as[a] = dop;
    }
}
