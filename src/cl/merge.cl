// vim: syntax=c

#define WORK_GROUP_SIZE 128


__kernel void merge_naive(
    __global const float * a,
    __global float * b,
    unsigned int k
) {
    const unsigned int id = get_global_id(0);
    const unsigned int base_idx = 2 * k * id;

    __global const float * const a1 = a + base_idx;
    __global const float * const a2 = a1 + k;

    __global float * ptr = b + base_idx;
    for (unsigned int i = 0, j = 0; i < k || j < k; ) {
        if (i >= k || j < k && a2[j] < a1[i]) {
            *ptr = a2[j];
            ++j;
        } else {
            *ptr = a1[i];
            ++i;
        }

        ++ptr;
    }
}

void binsearch_diag_indexes(
    __global const float * a1,
    __global const float * a2,
    unsigned int k,
    unsigned int diag_abs,
    __local unsigned int * ri,
    __local unsigned int * rj
) {
    unsigned int l = 0, r = diag_abs <= k ? diag_abs : 2 * k - diag_abs;

    unsigned int iter = 1000;

    while (l + 1 < r) {
        const unsigned int m = l + (r - l) / 2; // текущая засечка на диагонали

        // номер в массиве a1
        const unsigned int i = diag_abs <= k ? m : diag_abs - k + m;

        // номер в массиве a2
        const unsigned int j = diag_abs <= k ? diag_abs - m - 1 : k - m - 1;

        if (a1[i] > a2[j]) {
            r = m;
        } else {
            l = m;
        }

        --iter;
        if (iter == 0) {
            return;
        }
    }

    // здесь r -- это номер первой единицы на диагонали (оно же число нулей)
    // кроме случая, когда r = 1, его проверяем отдельно
    if (r == 1) {
        const unsigned int i = diag_abs <= k ? 0 : diag_abs - k;
        const unsigned int j = diag_abs <= k ? diag_abs - 1 : k - 1;

        if (a1[i] <= a2[j]) {
            r = 0;
        }
    }

    *ri = diag_abs <= k ? r : diag_abs - k + r;
    *rj = diag_abs <= k ? diag_abs - r : k - r;
}

__kernel void merge_diag(
    __global const float * a,
    __global float * b,
    unsigned int k      // размер одного блока
) {
    const unsigned int gid = get_global_id(0); // номер элемента массива
    const unsigned int lid = get_local_id(0); // номер элемента массива в группе
    const unsigned int grid = get_group_id(0); // номер группы

    const unsigned int groups_per_block = k / WORK_GROUP_SIZE; // число групп на блок
    const unsigned int block_pair = grid / groups_per_block; // номер пары блоков данной группы
    const unsigned int group_in_pair = grid % groups_per_block; // номер группы внутри пары блоков
    const unsigned int diag = group_in_pair * 2; // номер диагонали среди выделенных
    const unsigned int diag_abs = diag * WORK_GROUP_SIZE; // номер диагонали среди всех
    const unsigned int base_idx = 2 * k * block_pair; // индекс начала пары блоков в массиве

    __global const float * const a1 = a + base_idx; // указатель на начало первого блока массива
    __global const float * const a2 = a1 + k; // указатель на начало второго блока массива

    // if (lid == 0) {
    //     printf("gid = %d\n", gid);
    //     printf("lid = %d\n", lid);
    //     printf("grid = %d\n", grid);
    //     printf("k = %d\n", k);

    //     printf("groups_per_block = %d\n", groups_per_block);
    //     printf("block_pair = %d\n", block_pair);
    //     printf("group_in_pair = %d\n", group_in_pair);
    //     printf("diag = %d\n", diag);
    //     printf("diag_abs = %d\n", diag_abs);
    //     printf("base_idx = %d\n", base_idx);
    // }

    // i1 -- индекс начала диапазона в a1
    // j1 -- индекс начала диапазона в a2
    // i2 -- индекс конца диапазона в a1
    // j2 -- индекс конца диапазона в a2

    __local unsigned int i1, j1, i2, j2;

    if (lid == 0) {
        binsearch_diag_indexes(a1, a2, k, diag_abs, &i1, &j1);
        binsearch_diag_indexes(a1, a2, k, diag_abs + 2 * WORK_GROUP_SIZE, &i2, &j2);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int len1 = i2 - i1;
    const unsigned int len2 = j2 - j1;

    if (len1 + len2 != 2 * WORK_GROUP_SIZE) {
        printf("KERNEL PANIC! AMOUNT OF ITEMS = %d\n", len1 + len2);
        return;
    }

    __local float buf[2 * WORK_GROUP_SIZE];

    for (unsigned int i = 0; i < 2; ++i) {
        const unsigned int idx = i * WORK_GROUP_SIZE + lid;

        __global const float * my_item = (idx < len1 ? a1 + i1 + idx : a2 + j1 + idx - len1);
        buf[idx] = *my_item;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float res[2 * WORK_GROUP_SIZE];

    if (lid == 0) {
        for (unsigned int i = 0, j = 0; i < len1 || j < len2; ) {
            if (i >= len1 || j < len2 && buf[len1 + j] < buf[i]) {
                res[i + j] = buf[len1 + j];
                ++j;
            } else {
                res[i + j] = buf[i];
                ++i;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __global float * const b_res = b + base_idx + diag * WORK_GROUP_SIZE; // указатель на начало результата
    for (unsigned int i = 0; i < 2; ++i) {
        const unsigned int idx = i * WORK_GROUP_SIZE + lid;

        b_res[idx] = res[idx];
    }
}
