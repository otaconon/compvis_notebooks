template<typename T>
struct MatrixView {
  T* __restrict__ ptr;
  int rows;
  int cols;
  int ld;
};

template<typename T>
__global__ void add_kernel(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < C.rows && col < C.cols) {
    int idx = row*C.cols + col;
    C.ptr[idx] = A.ptr[idx] + B.ptr[idx];
  }
}