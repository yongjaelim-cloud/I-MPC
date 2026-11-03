#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 setup_collision_lin_cpp.py build_ext --inplace
python3 - <<'PY2'
import collision_lin_cpp
api = int(getattr(collision_lin_cpp, "COLLISION_RESIDUAL_API", 0))
surface_api = int(getattr(collision_lin_cpp, "SURFACE_DISTANCE_API", 0))
if api < 32:
    raise RuntimeError("collision_lin_cpp API {} loaded, expected >=32".format(api))
if surface_api < 4:
    raise RuntimeError("surface-distance API {} loaded, expected >=4".format(surface_api))
print("collision_lin_cpp API", api, "surface API", surface_api, "OK")
PY2
