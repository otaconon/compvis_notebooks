
template<typename T>
__global__ void vec_add(T* a, T* b, T* c, int n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}