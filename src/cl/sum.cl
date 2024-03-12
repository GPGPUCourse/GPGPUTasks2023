// TODO
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 256
#define WORKGROUP_SIZE 256

__kernel void compute_sum_baseline(global const unsigned int* inputArray, global unsigned int* outputSum, const unsigned int arraySize) {
    const unsigned int global_id = get_global_id(0);
    if (global_id >= arraySize) {
        return;
    }

    atomic_add(outputSum, inputArray[global_id]);
}

__kernel void compute_sum_coalesced(global const unsigned int* inputArray, global unsigned int* outputSum, const unsigned int arraySize) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int workgroup_id = get_group_id(0);
    const unsigned int localSize = get_local_size(0);

    int partialSum = 0;
    for (unsigned long long i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned long long global_id = workgroup_id * localSize * VALUES_PER_WORKITEM + i * localSize + local_id;
        if (global_id < arraySize) {
            partialSum += inputArray[global_id];
        }
    }

    atomic_add(outputSum, partialSum);
}

__kernel void compute_sum_uncoalesced(global const unsigned int* inputArray, global unsigned int* totalSum, const unsigned int arrayLength) {
    const unsigned int global_id = get_global_id(0);

    int partialSum = 0;
    for (unsigned long long i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned long long element_id = global_id * VALUES_PER_WORKITEM + i;
        if (element_id < arrayLength) {
            partialSum += inputArray[element_id];
        }
    }

    atomic_add(totalSum, partialSum);
}

__kernel void compute_sum_using_local_memory(global const unsigned int* inputArray, global unsigned int* totalSum, const unsigned int arraySize) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int localBuffer[WORKGROUP_SIZE];

    if (global_id < arraySize) {
        localBuffer[local_id] = inputArray[global_id];
    } else {
        localBuffer[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        unsigned int groupSum = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            groupSum += localBuffer[i];
        }
        atomic_add(totalSum, groupSum);
    }
}

__kernel void compute_tree_sum(global const unsigned int* inputArray, global unsigned int* outputSum, const unsigned int arraySize) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int localBuffer[WORKGROUP_SIZE];

    if (global_id < arraySize) {
        localBuffer[local_id] = inputArray[global_id];
    } else {
        localBuffer[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int activeElements = WORKGROUP_SIZE; activeElements > 1; activeElements /= 2) {
        if (2 * local_id < activeElements) {
            unsigned int firstValue = localBuffer[local_id];
            unsigned int secondValue = localBuffer[local_id + activeElements/2];
            localBuffer[local_id] = firstValue + secondValue;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(outputSum, localBuffer[0]);
    }
}

