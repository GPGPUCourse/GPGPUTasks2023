__kernel void fill(
                __global unsigned int *as,
                const unsigned int val,
                const unsigned int n)
{
    const unsigned i = get_global_id(0);
    if(i>=n) return;
    as[i] = val;
}

__kernel void count(
                __global const unsigned int *as,
                __global unsigned int *cs,
                const unsigned int shift,
                const unsigned int k)
{
    const unsigned i = get_global_id(0);
    const unsigned li = get_local_id(0);
    const unsigned gi = get_group_id(0);

    unsigned int WPT = SZ/WG;
    if(WPT<1) {    
        WPT = 1;
    }

    unsigned int a = (as[i] >> shift) & (SZ-1);

    __local unsigned int ct[SZ];

    if(li<SZ) ct[li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE); 

    atomic_add(ct+a, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int j = li*WPT; j < SZ && j < (li+1)*WPT; ++j) {
        cs[gi*SZ+j] = ct[j];
    }
}

__kernel void radix(
                __global const unsigned int *as,
                __global const unsigned int *cs,
                __global const unsigned int *cs_t,
                __global const unsigned int *ps,
                __global const unsigned int *ps_t,
                __global unsigned int *ss,
                const unsigned int shift,
                const unsigned int k)
{
    const unsigned i = get_global_id(0);
    const unsigned li = get_local_id(0);
    const unsigned gi = get_group_id(0);
    const unsigned ni = get_num_groups(0);

    unsigned int _a = as[i];
    unsigned int a = (_a >> shift) & (SZ-1);

    ss[(ps_t[a*ni+gi]-cs_t[a*ni+gi]) + (cs[gi*SZ+a] - (ps[gi*SZ+a]-li))] = _a;
}