#define Nbits 2u
#define Ncols (1u << Nbits)
#define Mask ((1u << Nbits) - 1u)
#define GetKey(value, shift) (((value) >> (shift)) & Mask)

kernel void reset(global unsigned *a) {
    a[get_global_id(0)] = 0;
}

kernel void count(global unsigned *result, global const unsigned *a, unsigned shift) {
    unsigned gid = get_group_id(0);
    unsigned id = get_global_id(0);
    atomic_add(result + gid * Ncols + GetKey(a[id], shift), 1);
}

kernel void reduce(global unsigned *a, unsigned k) {
    unsigned id = get_global_id(0);
    unsigned base = (1u << k) * id;
    a[base] += a[base + (1u << k - 1)];
}
kernel void pick(global unsigned *result, global const unsigned *a, unsigned k) {
    unsigned id = get_global_id(0) + 1;
    // get the cell if we need it
    if (id >> k & 1u)
        result[id - 1] += a[id >> k + 1 << k + 1];
}

#define gsize 4u
kernel void transpose(global unsigned* result, global const unsigned* a, unsigned n, unsigned m) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    unsigned lx = get_local_id(0);
    unsigned ly = get_local_id(1);

    local unsigned buffer[gsize][gsize + 1]; // +1 for banking
    buffer[ly][lx] = a[y * m + x];

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned writeY = x - lx + ly;
    unsigned writeX = y - ly + lx;
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
