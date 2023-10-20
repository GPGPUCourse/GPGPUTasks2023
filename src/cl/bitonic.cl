// vim: syntax=c

// as - array
// bs - block size
// it - iteration number
// sn - step number
__kernel void bitonic(__global float * as, unsigned int bs, unsigned int sn) {
    const unsigned int id = get_global_id(0);

    const unsigned int bs2 = bs / 2;
    const unsigned int bi = id / bs2; // block index
    const unsigned int bo = id % bs2; // block offset

    const unsigned int cs = bs >> sn; // chunk size
    const unsigned int cs2 = cs / 2;
    const unsigned int ci = bo / cs2; // chunk index
    const unsigned int co = bo % cs2; // chunk offset

    const unsigned int i = bi * bs + ci * cs + co;
    const unsigned int j = i + cs2;

    float a = as[i];
    float b = as[j];

    if (bi % 2 ? a < b : a > b) {
        const float c = b;
        b = a;
        a = c;
    }

    as[i] = a;
    as[j] = b;
}
