import cupy as cp
import numpy as np

matrix_view_dtype = np.dtype([
  ("ptr", np.uintp),
  ("rows", np.int32),
  ("cols", np.int32),
  ("ld", np.int32),
], align=True)

def to_view(arr: cp.ndarray):
  if arr.ndim not in (1, 2): raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")
  rows, cols = arr.shape[0], arr.shape[1] if arr.ndim == 2 else 1
  ld = arr.strides[0] // arr.itemsize
  return np.array((arr.data.ptr, rows, cols, ld), dtype=matrix_view_dtype)[()]