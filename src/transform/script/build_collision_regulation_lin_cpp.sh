#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SETUP_FILE="${SETUP_FILE:-$HERE/setup_collision_regulation_lin_cpp.py}"
CPP_FILE="${CPP_FILE:-$HERE/collision_regulation_lin_cpp.cpp}"

resolve_candidate() {
    local target="$1"
    local pattern="$2"
    local label="$3"

    shopt -s nullglob
    local candidates=("$HERE"/$pattern)
    shopt -u nullglob
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "[collision_regulation_lin_cpp] ERROR: missing $label matching $pattern" >&2
        exit 1
    fi

    # Always use the newest revision. This prevents an old canonical file from
    # shadowing a newly downloaded '(N)' or timestamped source/setup file.
    local newest
    newest="$(ls -1t "${candidates[@]}" | head -n 1)"
    if [[ "$newest" != "$target" ]]; then
        echo "[collision_regulation_lin_cpp] using newest $label:"
        echo "[collision_regulation_lin_cpp]   $newest -> $target"
        cp -f -- "$newest" "$target"
    else
        echo "[collision_regulation_lin_cpp] using canonical $label: $target"
    fi
}

resolve_candidate "$SETUP_FILE" 'setup_collision_regulation_lin_cpp*.py' 'setup file'
resolve_candidate "$CPP_FILE" 'collision_regulation_lin_cpp*.cpp' 'C++ source'

cd "$HERE"
echo "[collision_regulation_lin_cpp] Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[collision_regulation_lin_cpp] Building in: $HERE"

# Remove stale binaries so a nominally successful command can never import an
# older ABI from this directory.
rm -rf "$HERE/build"
find "$HERE" -maxdepth 1 -type f \( -name 'collision_regulation_lin_cpp*.so' -o -name 'collision_regulation_lin_cpp*.pyd' \) -delete

"$PYTHON_BIN" "$SETUP_FILE" build_ext --inplace --force

PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from __future__ import print_function

import numpy as np
import collision_regulation_lin_cpp as m

api = int(getattr(m, "COLLISION_REGULATION_RESIDUAL_API", 0))
if api < 7:
    raise RuntimeError(
        "collision_regulation_lin_cpp API {} is older than required API 7".format(api)
    )
if not hasattr(m, "build_collision_regulation_lin_seq"):
    raise RuntimeError("build_collision_regulation_lin_seq is missing")

H = 2
X = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.05, 0.0, 1.0, 0.0],
], dtype=np.float64)
V = np.array([[1.5, 0.0]], dtype=np.float64)
N = np.array([[-1.0, 0.0]], dtype=np.float64)
types = np.array([1], dtype=np.int32)
obs = np.array([0], dtype=np.int32)
h_steps = np.arange(1, H + 1, dtype=np.float64)
dt = np.full(H, 0.05, dtype=np.float64)
rays = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
axis = np.array([0.30, 0.20], dtype=np.float64)

def run(Xarg):
    return np.asarray(m.build_collision_regulation_lin_seq(
        Xarg, V, N, types, obs, h_steps, dt, rays, axis,
        True, True, 80,
        0.05, 0.05, 0.05, 1.0e15,
        0.30, -0.2, 0.001,
        0.60, 0.40, 0.0,
        True, "outward", 0.0, 0.0,
        0.001, 0.001, 1.0e-9,
        0.2, 0.2, 0.8, True, 0.0,
    ), dtype=np.float64)

out = run(X)
if out.shape != (H, 7) or not np.all(np.isfinite(out)):
    raise RuntimeError("collision_regulation_lin_cpp substituted Eq. (34) smoke test returned invalid output")

eps = 1.0e-6
Xp = X.copy(); Xp[0, 0] += eps
Xm = X.copy(); Xm[0, 0] -= eps
fd = (run(Xp)[0, 0] - run(Xm)[0, 0]) / (2.0 * eps)
if not np.isclose(fd, out[0, 1], rtol=2.0e-3, atol=2.0e-5):
    raise RuntimeError("API-7 Eq. (34) Jacobian mismatch: analytic={}, fd={}".format(out[0, 1], fd))

print("[collision_regulation_lin_cpp] module:", m.__file__)
print("[collision_regulation_lin_cpp] API:", api)
print("[collision_regulation_lin_cpp] API-7 substituted Eq. (34) smoke test: OK")
print("[collision_regulation_lin_cpp] build successful")
PY
