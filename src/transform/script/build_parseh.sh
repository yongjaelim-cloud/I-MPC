#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
if command -v nvcc >/dev/null 2>&1 && [[ "${PARSEH_DISABLE_CUDA:-0}" != "1" ]]; then
  nvcc -O3 --use_fast_math -std=c++14 -Xcompiler -fPIC -c parseh_cuda.cu -o parseh_cuda.o
  nvcc -O3 --use_fast_math -std=c++14 -Xcompiler -fPIC -c parseh_cuda_best.cu -o parseh_cuda_best.o
  "$CXX" -std=c++14 -O3 -fPIC -fopenmp -DPARSEH_WITH_CUDA -c parseh_backend.cpp -o parseh_backend.o
  nvcc -shared -o libparseh_accel.so parseh_backend.o parseh_cuda.o parseh_cuda_best.o -Xcompiler -fopenmp
else
  "$CXX" -std=c++14 -O3 -fPIC -fopenmp -shared parseh_backend.cpp -o libparseh_accel.so
fi
printf 'Built %s/libparseh_accel.so\n' "$PWD"
