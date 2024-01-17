#line 2

// Сейчас не используется, но показывает чуть лучше результаты на CPU
__kernel void small_linear_sort(__global unsigned int *as, unsigned int n, unsigned int curr_digit)
{
    const unsigned int g_id = get_group_id(0);
    const unsigned int left = g_id * WG_SIZE;
    const unsigned int right = min((g_id + 1) * WG_SIZE, n);
    const unsigned int local_size = right - left;
    const unsigned int mask = ~((~0u) << BITS_IN_DIGIT);
    const unsigned int digit_set = (1 << BITS_IN_DIGIT);

    __local unsigned int buf[WG_SIZE];
    __local unsigned int res[WG_SIZE];
    __local unsigned int counters[1 << BITS_IN_DIGIT];
    __local unsigned int positions[1 << BITS_IN_DIGIT];

    for (unsigned int it = 0; it < local_size; it++)
    {
        buf[it] = as[left + it];
    }
    for (unsigned int it = 0; it < digit_set; it++)
    {
        counters[it] = 0;
        positions[it] = 0;
    }

    for (unsigned int it = 0; it < local_size; it++)
    {
        unsigned int digit = (buf[it] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;
        counters[digit]++;
    }
    for (int it = 1; it < digit_set; it++)
    {
        positions[it] = positions[it - 1] + counters[it - 1];
    }
    for (int it = 0; it < local_size; it++)
    {
        unsigned int digit = (buf[it] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;
        res[positions[digit]++] = buf[it];
    }

    for (unsigned int it = 0; it < local_size; it++)
    {
        as[left + it] = res[it];
    }
}

__kernel void small_sort(__global unsigned int *as, unsigned int n, unsigned int curr_digit)
{
    const unsigned int i = get_global_id(0);
    const unsigned int local_i = get_local_id(0);
    const unsigned int g_id = get_group_id(0);

    const unsigned int left = g_id * WG_SIZE;
    const unsigned int right = min((g_id + 1) * WG_SIZE, n);
    const unsigned int local_size = right - left;

    const unsigned int mask = ~((~0u) << BITS_IN_DIGIT);
    const unsigned int digit_set = (1 << BITS_IN_DIGIT);

    __local unsigned int buf[WG_SIZE];
    __local unsigned int res[WG_SIZE];
    if (i < n)
    {
        buf[local_i] = as[i];
    }

    __local unsigned int counters[1 << BITS_IN_DIGIT];
    __local unsigned int positions[1 << BITS_IN_DIGIT];
    if (local_i < digit_set)
    {
        counters[local_i] = 0;
        positions[local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n)
    {
        unsigned int digit = (buf[local_i] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;
        atomic_add(&counters[digit], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Я не смог придумать получше :(
    if (local_i == 0)
    {
        for (int it = 1; it < digit_set; it++)
        {
            positions[it] = positions[it - 1] + counters[it - 1];
        }
        for (int it = 0; it < local_size; it++)
        {
            unsigned int digit = (buf[it] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;
            res[positions[digit]++] = buf[it];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n)
    {
        as[i] = res[local_i];
    }
}

__kernel void counters(__global const unsigned int *as, unsigned int n, __global unsigned int *counters, unsigned int curr_digit)
{
    const unsigned int digit_set = 1 << BITS_IN_DIGIT;
    unsigned int i = get_global_id(0);
    unsigned int local_i = get_local_id(0);

    __local unsigned int local_counters[1 << BITS_IN_DIGIT];
    if (local_i < digit_set)
    {
        local_counters[local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int mask = ~((~0u) << BITS_IN_DIGIT);
    unsigned int digit = (as[i] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;
    atomic_add(&local_counters[digit], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int cnt_idx = get_group_id(0) * digit_set;
    if (local_i < digit_set)
    {
        counters[cnt_idx + local_i] = local_counters[local_i];
    }
}

__kernel void transpose(__global const unsigned int *as, __global unsigned int *ast, unsigned int M, unsigned int K)
{
    int i = get_global_id(0);  // 0 to K
    int j = get_global_id(1);  // 0 to M

    __local unsigned int tile[TILE_SIZE_M][TILE_SIZE_K + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < K && j < M && local_j < TILE_SIZE_M && local_i < TILE_SIZE_K)
    {
        tile[local_j][local_i] = as[j * K + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < K && j < M && local_j < TILE_SIZE_M && local_i < TILE_SIZE_K)
    {
        ast[i * M + j] = tile[local_j][local_i];
    }
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *result, unsigned int n, unsigned int take_id)
{
    unsigned int i = get_global_id(0);
    if (i < n && (i & take_id))
    {
        result[i] += as[i / take_id - 1];
    }
}

__kernel void reduce(__global const unsigned int *as, __global unsigned int *bs, unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i >= n)
    {
        return;
    }
    unsigned int x = (2 * i >= n) ? 0 : as[2 * i];
    unsigned int y = (2 * i + 1 >= n) ? 0 : as[2 * i + 1];
    bs[i] = x + y;
}

__kernel void local_prefix(__global const unsigned int *as, __global unsigned int *bs, unsigned int M, unsigned int K)
{
    const unsigned int i = get_global_id(0);
    if (i >= K)
    {
        return;
    }

    unsigned int accum = 0;
    for (unsigned int it = 0; it < M; it++)
    {
        bs[it * K + i] = accum;
        accum += as[it * K + i];
    }
}

__kernel void radix(__global const unsigned int *as, unsigned int n, __global const unsigned int *counters,
                    __global const unsigned int *prefixes, __global const unsigned int *row_prefixes,
                    unsigned int n_wg, unsigned int n_cnt, unsigned int curr_digit,
                    __global unsigned int *to)
{
    unsigned int i = get_global_id(0);
    unsigned int local_i = get_local_id(0);
    unsigned int group_no = get_group_id(0);

    unsigned int mask = ~((~0u) << BITS_IN_DIGIT);
    unsigned int digit = (as[i] >> (BITS_IN_DIGIT * (curr_digit - 1))) & mask;

    unsigned int in_our_wg = row_prefixes[digit * n_wg + group_no];
    unsigned int in_prev_wgs = prefixes[digit * n_wg + group_no];
    unsigned int pos = in_prev_wgs + local_i - in_our_wg;

    to[pos] = as[i];
}
