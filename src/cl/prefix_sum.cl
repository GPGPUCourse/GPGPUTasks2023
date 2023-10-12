__kernel void accum(
    __global unsigned int *ss,
    const unsigned int block,
    const unsigned int n) 
{
    const unsigned int x = get_global_id(0);
    unsigned int i = x*block, j = i+block/2;
    if(i>=n) 
        return;

    ss[i] += ss[j];
}

__kernel void prefix_sum(
    __global unsigned int *as,
    __global const unsigned int *ss,
    const unsigned int bit) 
{
    const unsigned int x = get_global_id(0);

    if(((x+1)&bit)!=0) {
        as[x] += ss[((x+1)/bit-1)*bit];
    }
}
