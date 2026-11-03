#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/vertex_selector.cpp"
OUT_MOTION="${SCRIPT_DIR}/libvertex_selector.so"
OUT_COMPAT="${SCRIPT_DIR}/libvertex_selector2.so"
CXX="${CXX:-g++}"
"${CXX}" -O3 -DNDEBUG -std=c++17 -fPIC -shared -march=native -flto -fopenmp "${SRC}" -o "${OUT_MOTION}"
ln -sfn "$(basename "${OUT_MOTION}")" "${OUT_COMPAT}"
echo "built ${OUT_MOTION}"
echo "compat symlink ${OUT_COMPAT} -> $(basename "${OUT_MOTION}")"
nm -D "${OUT_MOTION}" | grep -E 'vertex_selector_(process|motion_update|motion_reset|finalize_selected)' || true
