#define MOVE (1 << 4)
#define TILE_SIZE 16

__kernel void radix(__global const unsigned int *as,
                    __global unsigned int *res,
                    __global const unsigned int *p,
                    __global const unsigned int *cnt,
                    int n,
                    unsigned int spride) {
    int idx = get_global_id(0);
    int gid = get_group_id(0);

    int number_of_groups = n / get_local_size(0);
    unsigned int value = as[idx];
    unsigned int digit = (value & ((MOVE - 1) << spride)) >> spride;
    int new_idx = get_local_id(0);

    if (digit)
        new_idx -= cnt[gid * MOVE + digit - 1];
    if (gid || digit) {
        new_idx += p[digit * number_of_groups + gid - 1];
    }
    res[new_idx] = value;
}

__kernel void count(__global const unsigned int *as,
                    __global unsigned int *result,
                    unsigned int spride) {
    int idx = get_global_id(0);
    int gid = get_group_id(0);

    int start = gid * MOVE;
    unsigned int mask = (MOVE - 1) << spride;
    unsigned int value = (as[idx] & mask) >> spride;
    
    atomic_add(result + (start + value), 1);
}

__kernel void transpose(__global unsigned int *matrix,
                          __global unsigned int *matrixTransposed,
                          const unsigned int m,
                          const unsigned int k)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    
    int i = col - localCol;
    int j = row - localRow;
    
    __local unsigned int tile[TILE_SIZE * TILE_SIZE];
    
    if (col < k && row < m) {
        tile[localRow * TILE_SIZE + (localCol + localRow) % TILE_SIZE] = matrix[row * k + col];
        matrix[row * k + col] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int value = tile[localCol * TILE_SIZE + (localRow + localCol) % TILE_SIZE];

    if (j + localCol < m && i + localRow < k)
        matrixTransposed[(i + localRow) * m + j + localCol] = value;
}

__kernel void prefix(__global const unsigned int *as,
                     __global unsigned int *res,
                     unsigned int mask, unsigned int n) {
    unsigned int i = get_global_id(0);
    i = ((i & (~(mask - 1))) << 1) + mask + (i & (mask - 1));
    if (i > n) return;
    res[i - 1] += as[i - (i & (mask - 1)) - 1];
}

__kernel void reduce(__global unsigned int *as,
                       unsigned int n,
                       unsigned int stride) {
    int idx = get_global_id(0) * stride * 2;
    int step = idx + stride * 2;
    if (step > n) return;
    as[step - 1] += as[idx + stride - 1];
}

__kernel void merge(__global unsigned int *a,
                    __global unsigned int *b,
                    unsigned int spride) {
    int id = get_local_id(0);
    int sz = get_local_size(0);
    int gid = get_global_id(0);
    int start = gid - gid % sz;

    __global unsigned int *src = a + start;
    __global unsigned int *dst = b + start;

    int cnt = 0;
    int mask = (MOVE - 1) << spride;
    for (int block = 1; block <= sz / 2; block <<= 1, ++cnt) {
        unsigned int value = src[id];
        int k = id / (2 * block);
        int i = id % (2 * block);
        int j;

        if (i < block) {
            int l0 = k * (2 * block) + block - 1;
            int l = l0;
            int r = (k + 1) * (2 * block);
            if (l > sz)
                l = sz;
            if (r > sz)
                r = sz;
            while (r - l > 1) {
                int m = (l + r) / 2;
                if ((src[m] & mask) < (value & mask)) {
                    l = m;
                } else {
                    r = m;
                }
            }
            j = i + (l - l0);
        } else {
            int l0 = k * (2 * block) - 1;
            int l = l0;
            int r = k * (2 * block) + block;
            while (r - l > 1) {
                int m = (l + r) / 2;
                if ((src[m] & mask) <= (value & mask)) {
                    l = m;
                } else {
                    r = m;
                }
            }
            j = (i - block) + (l - l0);
        }

        dst[k * (2 * block) + j] = value;
        __global unsigned int *tmp = src;
        src = dst;
        dst = tmp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (cnt % 2 == 0)
        b[id] = a[id];
}

__kernel void zero(__global unsigned int *as) {
    int i = get_global_id(0);
    as[i] = 0;
}

__kernel void move(const __global unsigned int *src, __global unsigned int *dst) {
    int i = get_global_id(0);
    dst[i] = src[i];
}


__kernel void local_prefix(__global unsigned int *cnt) {
    int gid = get_group_id(0);
    int i = get_local_id(0);

    __global unsigned int *ptr = cnt + gid * MOVE;
    unsigned int result = 0;
    
    int start = 0;
    int border = 1 << (4 - 1);
    
    if (i >= border)
        start = border;
    
    for (int j = start; j <= i; j++)
        result += ptr[j];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    ptr[i] = result;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (i >= border)
        ptr[i] += ptr[border - 1];
}
