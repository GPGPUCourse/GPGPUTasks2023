__kernel void prefix(
  __global unsigned int* as, 
  __global unsigned int* res, 
  unsigned int offset
) {
    unsigned i = get_global_id(0);
    if (i >= offset) {
      res[i] = as[i] + as[i - offset];
    } else {
      res[i] = as[i];
    }
}
