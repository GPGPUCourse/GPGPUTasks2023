__kernel void matrix_transpose(
                    __global const float* as, 
                    __global float* as_t,
                    const unsigned int width, const unsigned int height)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);

    const unsigned w = get_local_size(0);
    const unsigned h = get_local_size(1);
    const unsigned i = get_local_id(0);
    const unsigned j = get_local_id(1);

    __local float subrectangle[WG1][WG0];
    subrectangle[j][i] = as[y*width+x];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const int gi = get_group_id(0);
    const int gj = get_group_id(1);

    as_t[gi*height*w+gj*h + ((j*w+i)/h)*height + ((j*w+i)%h)] = subrectangle[(j*w+i)%h][(j*w+i)/h];
}