#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge_two_iters(
                    __global const float* as, 
                    __global float* ss, 
                    const unsigned int n,
                    const unsigned int len)
{
    const unsigned x = get_global_id(0);
    
    if (x*len >= n)
        return;
    
    unsigned int l1 = x*len, r1 = l1+len/2;
    unsigned int l2 = r1, r2 = l2 + len/2;
    unsigned int it = x*len;
    while(l1<r1 && l2<r2) {
        float a = as[l1], b = as[l2];
        ss[it++] = (a<=b?a:b);
        if(a<=b) ++l1;
        else ++l2;
    }
    while(l1<r1)
        ss[it++] = as[l1++];
    while(l2<r2)
        ss[it++] = as[l2++];
}

void binary_search(
        __global const float* as, 
        const unsigned int l1, 
        const unsigned int l2, 
        const unsigned int len, 
        const unsigned int x,
        unsigned int* ra,
        unsigned int* rb) 
{
    int l = 0, r = (x+1)*WORK_PER_WORKITEM - 1, lst_zero = 0; 

    if(r>=len/2) {
        r = len - r - 2;
    }

    if(r>=0) {
        int sa = (x+1)*WORK_PER_WORKITEM - 1, sb = 0;
        if(sa>=len/2) {
            sb += sa - len/2 + 1;
            sa = len/2 - 1;
        }

        while(r-l>=0) {
            unsigned int mid = (l+r)>>1;

            int a = sa - mid, b = sb + mid;
            bool comp = as[l2+b] <= as[l1+a];

            if(comp) {
                l = mid + 1;
                lst_zero = mid;
            } else {
                r = mid - 1;
            }
        }
        
        *ra = sa - lst_zero;
        *rb = sb + lst_zero + 1;
    } else {
        *ra = len/2;
        *rb = len/2;
    }
}

__kernel void merge_local(
                    __global const float* as, 
                    __global float* ss,
                    const unsigned int n,
                    const unsigned int len,
                    const unsigned int shift)
{
    const unsigned x  = get_global_id(0);

    const unsigned int l1 = shift, r1 = l1+len/2;
    const unsigned int l2 = r1, r2 = l2 + len/2;

    unsigned int ssl1, ssr1;
    unsigned int ssl2, ssr2;

    if(x)
        binary_search(as, l1, l2, len, x-1, &ssl1, &ssl2);
    else
        ssl1=ssl2=0;

    binary_search(as, l1, l2, len, x, &ssr1, &ssr2);

    __local float suba[WORK_PER_WORKITEM]; 
    for(unsigned int i=0;i<WORK_PER_WORKITEM;++i) {
        suba[i] = 0;
    }

    __local float subb[WORK_PER_WORKITEM]; 
    for(unsigned int i=0;i<WORK_PER_WORKITEM;++i) {
        subb[i] = 0;
    }

    for(unsigned int i=ssl1;i<ssr1;++i) {
        suba[i-ssl1] = as[l1+i];
    }

    for(unsigned int i=ssl2;i<ssr2;++i) {
        subb[i-ssl2] = as[l2+i];
    }

    ssr1 -= ssl1;
    ssr2 -= ssl2;

    ssl1 = 0;
    ssl2 = 0;
    
    unsigned int it = l1 + x*WORK_PER_WORKITEM;
    while(ssl1<ssr1 && ssl2<ssr2) {
        float a = suba[ssl1], b = subb[ssl2];
        ss[it++] = (a<=b?a:b);
        if(a<=b) ++ssl1;
        else ++ssl2;
    }
    
    while(ssl1<ssr1) {
        ss[it++] = suba[ssl1++];
    }
        
    while(ssl2<ssr2) {
        ss[it++] = subb[ssl2++];
    }
}