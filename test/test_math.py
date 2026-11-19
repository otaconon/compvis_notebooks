import cupy as cp
import numpy as np
import pytest
from cuda_helpers.matrix import to_view

def run_add(kernel, a, b):
  c = cp.empty_like(a)
  rows = a.shape[0]
  cols = a.shape[1] if a.ndim == 2 else 1
  block = (16, 16)
  grid = ((rows + 15) // 16, (cols + 15) // 16)
  kernel(grid, block, (to_view(a), to_view(b), to_view(c)))
  return c

@pytest.mark.parametrize("shape", [(10,), (1,), (257,), (5, 7), (33, 1), (64, 64)])
def test_add_matches_cupy(add_kernel, shape):
  a = cp.random.rand(*shape).astype(cp.float32)
  b = cp.random.rand(*shape).astype(cp.float32)
  c = run_add(add_kernel, a, b)
  cp.testing.assert_allclose(c, a + b)

def test_to_view_layout():
  a = cp.zeros((4, 6), dtype=cp.float32)
  v = to_view(a)
  assert (v["rows"], v["cols"], v["ld"]) == (4, 6, 6)
  assert v["ptr"] == a.data.ptr