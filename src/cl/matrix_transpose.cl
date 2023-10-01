#define WORKGROUP_X_SIZE (32                                 )
#define WORKGROUP_Y_SIZE (32                                 )
#define WORKGROUP_SIZE   (WORKGROUP_X_SIZE * WORKGROUP_Y_SIZE)

__kernel void matrix_transpose(
    __global float const *matrixIn , 
    __global float       *matrixOut, 
    unsigned int          K        , 
    unsigned int          M         ) {
    
    __local float buffer[WORKGROUP_SIZE];

    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);

    const unsigned int groupX = get_group_id(0);
    const unsigned int groupY = get_group_id(1);

    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(1);

    unsigned int localId = localY * WORKGROUP_X_SIZE + localX;
    
    buffer[localId] = globalX < K && globalY < M ? matrixIn[K * globalY + globalX] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int const transposedId = M * (groupX * WORKGROUP_Y_SIZE + localY) + groupY * WORKGROUP_X_SIZE + localX;

    if (globalX < M && globalY < K) {
        matrixOut[transposedId] = buffer[localX * WORKGROUP_Y_SIZE + localY];
    }
}