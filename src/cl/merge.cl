int bsearch(float key, global const float* data, unsigned n) {
    int a = 0, b = n;
    while (a < b) {
        int mid = (a + b) / 2;
        if (data[mid] < key) a = mid + 1;
        else b = mid; // key <= data[mid]
    }
    return a;
}
int bsearch2(float key, global const float* data, unsigned n) {
    int a = 0, b = n;
    while (a < b) {
        int mid = (a + b) / 2;
        if (data[mid] <= key) a = mid + 1;
        else b = mid; // key < data[mid]
    }
    return a;
}
kernel void merge(global float* result, global const float* a, unsigned n, unsigned k) {
    int i = get_global_id(0);
    int size = n / k;
    int group = i / size;

    int resultStart, j;
    if (group % 2) {
        resultStart = (group - 1) * size;
        j = bsearch(a[i], a + (group - 1) * size, size);
    } else {
        resultStart = group * size;
        j = bsearch2(a[i], a + (group + 1) * size, size);
    }

    result[resultStart + (i - group * size) + j] = a[i];
}
