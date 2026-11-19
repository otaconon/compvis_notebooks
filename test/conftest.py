from pathlib import Path
import cupy as cp
import pytest

KERNELS = Path(__file__).resolve().parents[1] / "kernels"

@pytest.fixture(scope="session")
def module():
  return cp.RawModule(
    code=(KERNELS / "math.cu").read_text(),
    options=(f"-I{KERNELS}",),
    name_expressions=["add_kernel<float>"],
  )

@pytest.fixture(scope="session")
def add_kernel(module):
  return module.get_function("add_kernel<float>")