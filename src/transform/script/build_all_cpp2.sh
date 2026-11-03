#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$HERE"

# Native CUDA is required by default for field_flag=4.  Set REQUIRE_CUDA=0 only
# when intentionally building the CPU-only fallback on a machine without nvcc.
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export CXXFLAGS="${CXXFLAGS:-} -O3 -DNDEBUG -fopenmp"
export LDFLAGS="${LDFLAGS:-} -fopenmp"

rm -f collision_lin_cpp*.so
"$PYTHON_BIN" setup_collision_lin_cpp.py build_ext --inplace --force

"$PYTHON_BIN" - <<'PY'
import collision_lin_cpp as m
api = int(getattr(m, "COLLISION_RESIDUAL_API", 0))
assert api >= 34, "collision_lin_cpp API >= 34 required, got %d" % api
assert bool(getattr(m, "OPENMP_ENABLED", False)), "collision_lin_cpp was built without OpenMP"
assert int(getattr(m, "BP_FIT_API", 0)) >= 2, "BP_FIT_API >= 2 required"
assert int(getattr(m, "DYNAMICS_LINEARIZATION_API", 0)) >= 1, "DYNAMICS_LINEARIZATION_API >= 1 required"
assert hasattr(m, "build_dynamics_lin_seq"), "build_dynamics_lin_seq is required"
assert int(getattr(m, "PROPOSED_CUDA_API", 0)) >= 2, "native proposed CUDA profiling API missing"
import os
compiled = bool(getattr(m, "NATIVE_CUDA_COMPILED", False))
available = bool(m.cuda_available()) if hasattr(m, "cuda_available") else False
require_cuda = os.environ.get("REQUIRE_CUDA", "1").strip().lower() not in ("0", "false", "no", "off")
if require_cuda:
    assert compiled, "native CUDA backend was requested but not compiled"
print("[build] collision_lin_cpp API=%d OpenMP=on max_threads=%s CUDA_compiled=%s CUDA_available=%s" %
      (api, m.get_omp_max_threads(), compiled, available))
if compiled and not available:
    print("[build] CUDA runtime note:", m.cuda_last_error())
PY
