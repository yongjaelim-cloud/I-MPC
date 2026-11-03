#include "collision_lin_cuda.hpp"

// API-32 backend output is Hx21; the stub keeps the same pointer-only ABI.

bool collision_proposed_cuda_compiled() { return false; }
bool collision_proposed_cuda_available() { return false; }
const char* collision_proposed_cuda_last_error() { return "native CUDA backend was not compiled"; }
void collision_proposed_cuda_reset_cache() {}
void collision_proposed_cuda_set_profiling(bool) {}
bool collision_proposed_cuda_get_last_timing(CollisionProposedCudaTiming* out) {
    if (out) *out = CollisionProposedCudaTiming{};
    return false;
}


void collision_bp_cuda_reset_cache() {}
void collision_bp_cuda_set_profiling(bool) {}
bool collision_bp_cuda_get_last_timing(CollisionBPCudaTiming* out) {
    if (out) *out = CollisionBPCudaTiming{};
    return false;
}

bool collision_bp_cuda_run(
    const double*, const double*, const double*, const double*, const double*, const double*,
    const int32_t*, int, int, int, int, double, double, double, double, double, double, double,
    double, double, double, double, bool, int, double*) {
    return false;
}

bool collision_proposed_cuda_run(
    const double*, const double*, int, const double*, int, const double*,
    const int32_t*, const int32_t*, const double*, const double*, int, int, int,
    double, double, double, double, double, double, double, bool, double, double, double,
    double, double, bool, bool, double, double, double, double, double, double, double,
    double, int, double*) {
    return false;
}
