#ifndef CL_DEFINES_H
#define CL_DEFINES_H

#define RADIX_BITS 2
#define WORKGROUP_SIZE 32 // size of a single chunk
#define WORK_PER_THREAD (WORKGROUP_SIZE / (1 << RADIX_BITS))
static_assert(WORK_PER_THREAD > 0);

#define WITH_LOCAL_SORT 0

#define TRANSPOSE_WORKGROUP_SIZE 16

#endif //CL_DEFINES_H
