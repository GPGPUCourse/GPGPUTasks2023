__kernel void mergenew(
    __global const float *arr,
    __global float *permutation,
    unsigned int size,
    unsigned int blocksize
) {
    // we assume size % (2*blocksize) == 0

    const unsigned int index = get_global_id(0);
    if (index >= size) return;

    unsigned int big_block_start = index - (index % (2*blocksize)); // Big block = A in concatenation with B
    unsigned int this_block_start = index - (index % blocksize); // This block = A or B depending on index

    // If index is in A, we count items in B with value < arr[index].  
    // If index is in B, we count items in A with value <= arr[index]. This condition is equivalent
    // to value being less than next float after arr[index]
    float threshold = (big_block_start==this_block_start) ? arr[index] : nextafter(arr[index], arr[index]+1);

    unsigned int lo = (big_block_start*2+blocksize) - this_block_start;
    unsigned int hi = lo + blocksize;

    while (lo != hi) {
        // Invariant: index < lo => arr[index] < threshold; index >= hi => arr[index] >= threshold
        unsigned int mid = lo + (hi-lo)/2;

        if (arr[mid] < threshold) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    permutation[big_block_start+index+lo - big_block_start - (big_block_start+blocksize)] = arr[index];
}
