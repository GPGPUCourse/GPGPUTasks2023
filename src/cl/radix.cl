#define Nbits 2u
#define Ncols (1u << Nbits)
#define Mask ((1u << Nbits) - 1)
#define GetKey(value, shift) (((value) >> (shift)) & Mask)

kernel void count(global unsigned *result, global const unsigned *a, unsigned shift) {
    unsigned gid = get_group_id(0);
    unsigned id = get_global_id(0);
    atomic_add(result + gid * Ncols + GetKey(a[id], shift), 1);
}

kernel void reduce(global unsigned *a, unsigned k) {
    unsigned id = get_global_id(0);
    unsigned base = (1 << k) * id;
    a[base] += a[base + (1 << k - 1)];
}
// k = 1
// 2^k = 2
// 2^k * id -> 0 2 4 6 8 10
// 0 + 1, 2 + 3, 4 + 5
// k = 2, 2^k = 4, 0 4 8 12
// 0 + 2, 4 + 6, 8 + 10
// k = 3 0 8 16 24
// 0 + 4, 8 + 12, 16 + 20
// k = log n
// 2^k = n
// n * id -> 0
// 0 + n / 2

kernel void pick(global unsigned *result, global const unsigned *a, unsigned k) {
    unsigned id = get_global_id(0) + 1;
    // get the cell if we need it
    if (id >> k & 1)
        result[id - 1] += a[id >> k + 1 << k + 1];
}
// 0 1 2 3 4 5 6 7 8
// 1 2 3 4 5 6 7 8 9
// 3 = 2^1 + 2^0
// 0 -> 0 1 2 3 4
// 1 -> 0 2 4 6 8
// 2 -> 0 4 8 12 16
// 3 -> 0 8 16 24 32
// 4 ->                       0 (k = 2)
// 5 -> 4 (k = 0),            0 (k = 2)
// 6 ->            4 (k = 1), 0 (k = 2) 100
// 7 -> 6 (k = 0), 4 (k = 1), 0 (k = 2) 111
// 8 ->                       0 (k = 3)
// log n -> 0
// 00010 -> 1
// 00010
// 00010 -> 2

#define gsize 4
kernel void transpose(global unsigned* result, global const unsigned* a, unsigned n, unsigned m) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    local float buffer[gsize][gsize + 1]; // +1 for banking
    buffer[ly][lx] = a[y * m + x];

    barrier(CLK_LOCAL_MEM_FENCE);
    int writeY = x - lx + ly;
    int writeX = y - ly + lx;
    result[writeY * n + writeX] = buffer[lx][ly];
}

int bsearch(float key, global const unsigned *data, unsigned n, unsigned shift) {
    int a = 0, b = n;
    while (a < b) {
        int mid = (a + b) / 2;
        if (GetKey(data[mid], shift) < key) a = mid + 1;
        else b = mid; // key <= data[mid]
    }
    return a;
}
int bsearch2(float key, global const unsigned *data, unsigned n, unsigned shift) {
    int a = 0, b = n;
    while (a < b) {
        int mid = (a + b) / 2;
        if (GetKey(data[mid], shift) <= key) a = mid + 1;
        else b = mid; // key < data[mid]
    }
    return a;
}
kernel void merge(global unsigned *result, global const unsigned *a, unsigned size, unsigned shift) {
    unsigned id = get_global_id(0);
    unsigned group = id / size;

    int resultStart, j;
    if (group % 2) {
        resultStart = (group - 1) * size;
        j = bsearch2(GetKey(a[id], shift), a + (group - 1) * size, size, shift);
    } else {
        resultStart = group * size;
        j = bsearch(GetKey(a[id], shift), a + (group + 1) * size, size, shift);
    }

    result[resultStart + (id - group * size) + j] = a[id];
}

kernel void radix(global unsigned *result, global const unsigned *a, unsigned n,
                  global const unsigned *offsets, unsigned rows, unsigned shift) {
    unsigned id = get_global_id(0);
    unsigned lid = get_local_id(0);
    unsigned group = get_group_id(0);

    unsigned num = GetKey(a[id], shift);

    unsigned offsetLower = 0, offsetEqual = 0;
    if (group == 0) {
        if (num) offsetLower = offsets[(rows - 1) * Ncols + num - 1];
    } else {
        offsetLower = offsets[(group - 1) * Ncols + num];
    }
    for (int i = 0; i < num; ++i) {
        offsetEqual += offsets[group * Ncols + i];
        if (group == 0) {
            if (i) offsetEqual -= offsets[(rows - 1) * Ncols + i - 1];
        } else {
            offsetEqual -= offsets[(group - 1) * Ncols + i];
        }
    }
    result[lid - offsetEqual + offsetLower] = a[id];
}
