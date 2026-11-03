#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"
OUT="${SCRIPT_DIR}/libexact_geometry_cpp.so"
SRC="${SCRIPT_DIR}/exact_geometry_cpp.cpp"

"${CXX}" -O3 -DNDEBUG -std=c++14 -fPIC -shared -fopenmp \
  "${SRC}" -o "${OUT}"

echo "built ${OUT}"
