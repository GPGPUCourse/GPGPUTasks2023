#define TILE_SIZE 16
#define WPT 4
#define RTS TILE_SIZE / WPT

__kernel void matrix_multiplication_naive(const __global float* as, const __global float* bs, __global float* cs, const uint m, const uint k, const uint n) {
    const uint gi = get_global_id(0);
    const uint gj = get_global_id(1);

    float res = 0;
    for (int i = 0; i < k; ++i) {
        res += as[gj * k + i] * bs[i * n + gi];
    }

    cs[gj * n + gi] = res;
}

__kernel void matrix_multiplication_local(__global const float* as, __global const float* bs, __global float* cs, const uint m, const uint k, const uint n) {
    const uint gi = get_global_id(0);
    const uint gj = get_global_id(1);

    const uint li = get_local_id(0);
    const uint lj = get_local_id(1);

    __local float as_tile[TILE_SIZE][TILE_SIZE];
    __local float bs_tile[TILE_SIZE][TILE_SIZE];

    float res = 0;
    for (uint step = 0; step * TILE_SIZE < k; ++step) {
        as_tile[lj][li] = as[gj * k + (li + step * TILE_SIZE)];
        bs_tile[lj][li] = bs[(lj + step * TILE_SIZE) * n + gi];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint idx = 0; idx < TILE_SIZE; ++idx) {
            res += as_tile[lj][idx] * bs_tile[idx][li];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    cs[gj * n + gi] = res;
}

__kernel void matrix_multiplication_local_bigger(__global const float* as, __global const float* bs, __global float* cs, const uint m, const uint k, const uint n) {
    const uint li = get_local_id(0);
    const uint lj = get_local_id(1);

    const uint gi = TILE_SIZE * get_group_id(0) + li;
    const uint gj = TILE_SIZE * get_group_id(1) + lj;

    __local float as_tile[TILE_SIZE][TILE_SIZE];
    __local float bs_tile[TILE_SIZE][TILE_SIZE];

    float res[WPT];
    for (uint i = 0; i < WPT; ++i) {
        res[i] = 0.0;
    }

    for (uint step = 0; step * TILE_SIZE < k; ++step) {
        for (uint i = 0; i < WPT; ++i) {
            const uint idx = lj + i * RTS;
            as_tile[idx][li] = as[(gj + i * RTS) * k + (step * TILE_SIZE + li)];
            bs_tile[idx][li] = bs[(step * TILE_SIZE + idx) * n + gi];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint s = 0; s < TILE_SIZE; ++s) {
            for (uint i = 0; i < WPT; ++i) {
                res[i] += as_tile[lj + i * RTS][s] * bs_tile[s][li];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < WPT; ++i) {
        cs[(i * RTS + gj) * n + gi] = res[i];
    }
}
