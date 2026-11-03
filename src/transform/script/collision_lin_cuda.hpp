#pragma once

#include <cstddef>
#include <cstdint>

// Native CUDA backend for the proposed (field_flag=4) dual linearization plus J_global impact-aligned value block.
// The caller supplies dense row-major host buffers.  Geometry can be shared
// across the horizon (stage_count=1) or stage-wise (stage_count=H).

struct CollisionProposedCudaTiming {
    double h2d_ms = 0.0;
    double kernel_group_ms = 0.0;
    double kernel_stage_ms = 0.0;
    double d2h_ms = 0.0;
    double total_device_ms = 0.0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    bool feature_points_copied = false;
    bool feature_normals_copied = false;
};

bool collision_proposed_cuda_compiled();
bool collision_proposed_cuda_available();
const char* collision_proposed_cuda_last_error();
void collision_proposed_cuda_reset_cache();
void collision_proposed_cuda_set_profiling(bool enabled);
bool collision_proposed_cuda_get_last_timing(CollisionProposedCudaTiming* out);


// Native CUDA backend for the world-frame Bernstein-polynomial SDF (field_flag=2).
// The BP fit remains on CPU/cache; this backend accelerates the repeated metric
// SDF + gradient queries and the Hx21 collision/regulation linearization used
// during planning. Dense model arrays follow the BP_FIT_API=9 layout:
//   W[Hm,O,Q,Q], T_ref[Hm,O,2,2], p_ref/x_range/y_range[Hm,O,2], empty[Hm,O],
// with Hm either 1 (shared/static) or H (stage-wise translated models).
struct CollisionBPCudaTiming {
    double h2d_ms = 0.0;
    double kernel_eval_ms = 0.0;
    double kernel_assemble_ms = 0.0;
    double d2h_ms = 0.0;
    double total_device_ms = 0.0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    bool model_copied = false;
};

void collision_bp_cuda_reset_cache();
void collision_bp_cuda_set_profiling(bool enabled);
bool collision_bp_cuda_get_last_timing(CollisionBPCudaTiming* out);

bool collision_bp_cuda_run(
    const double* x_h4,
    const double* W,
    const double* T_ref,
    const double* p_ref,
    const double* x_range,
    const double* y_range,
    const int32_t* empty,
    int H,
    int model_stage_count,
    int O,
    int Q,
    double safety_margin,
    double h_upper_bound,
    double d_safe,
    double Ns,
    double regulation_lambda,
    double normal_eps,
    double fd_position_eps_m,
    double fd_yaw_eps_rad,
    double lcbx,
    double lcby,
    bool impact_normal_enabled,
    int threads,
    double* out_h21);

bool collision_proposed_cuda_run(
    const double* x_h4,
    const double* points,
    int point_stage_count,
    const double* normals,
    int normal_stage_count,
    const double* dynamic_mask,
    const int32_t* group_offsets,
    const int32_t* group_indices,
    const double* h_steps,
    const double* dt_vals,
    int H,
    int K,
    int O,
    double sx,
    double sy,
    double geom_buffer,
    double lambda,
    double regulation_lambda,
    // Paper Appendix A: sqrt(sigma_vx^2 + sigma_vy^2), supplied by the host.
    double sigma_iso,
    double confidence,
    bool use_probability_margin,
    double pred_rate,
    double dt_init,
    double dt_min,
    double dt_max,
    double h_upper_bound,
    bool flip_normal,
    bool use_scaled,
    double d_safe,
    double Ns,
    double normal_eps,
    double lcbx,
    double lcby,
    double lc_local_x,
    double lc_local_y,
    double disabled,
    int threads,
    double* out_h21);
