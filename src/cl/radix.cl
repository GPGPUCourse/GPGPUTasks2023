__kernel void count(
                __global const unsigned int *as, 
                __global unsigned int *cs, 
                const unsigned int shift, 
                const unsigned int k) 
{
    const unsigned i = get_global_id(0);
    const unsigned li = get_local_id(0);
    const unsigned gi = get_group_id(0);
    
    unsigned int a = (as[i] >> shift) & (WG-1); // <=> & (2^k - 1)

    __local unsigned int ct[WG];

    ct[li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE); 

    atomic_add(ct+a, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    cs[gi*WG+li] = ct[li];
}

__kernel void radix(__global const unsigned int *as, 
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
    unsigned int a = (_a >> shift) & (WG-1); // <=> & (2^k - 1)

    ss[(ps_t[a*ni+gi]-cs_t[a*ni+gi]) + (cs[gi*WG+a] - (ps[gi*WG+a]-li))] = _a;
}