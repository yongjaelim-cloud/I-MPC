#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATLAB-aligned MPC utilities with selectable nonlinear/linearized RK4 shooting dynamics
and sequentially linearized collision constraints.

The /setup field ``mpc_flag`` selects the MPC formulation while ``field_flag``
selects the collision-field evaluator.  For mpc_flag=4 the OCP is deliberately
compact:

    x_k = [p_x, p_y, v, yaw] for every formulation.
    u_k = [delta_yawrate, delta_acc, s^a, s^r] for mpc_flag=4 when regulation
          is enabled, [delta_yawrate, delta_acc, s^a] when it is disabled.

There is no dt decision state, boundary-slack state b_k, or unused s^3/s^4
decision variable. N and dt_init are fixed configuration values.  With
``nonlinear_dyn_flag=true`` acados uses the nonlinear unicycle/point-mass RK4
map.  With ``nonlinear_dyn_flag=false`` each outer SQP iteration freezes the
first-order RK4 model around the current nominal trajectory, so the acados
subproblem is a convex QP (LINEAR_LS cost, affine dynamics/constraints) solved
by the configured acados QP solver.

For mpc_flag=4 every outer collision-refinement iteration uses the same
phase-0 scalarized objective J_local plus the configured weighted slack cost.
The impact-aligned term rho_impact (-grad Phi_k(0))^T n(phi_c) is evaluated only in
J_global for post-solve candidate ranking. There is no final lexicographic phase
and no horizon slack-budget state.
Returned candidate cost/feasibility are finalized here.

Disabling ``enable_impact_normal_constraint`` removes the collision-regulation
constraint from the proposed OCP at build time.  In that mode mpc_flag=4 uses
only u=[delta_yawrate, delta_acc, s^a] and the softened avoidance constraint;
there is no s^r decision variable or regulation constraint row.

The local q_pos term tracks stage-wise initial-guidance positions when valid
guidance is supplied; otherwise it tracks the global goal at every stage.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import os
import time
import hashlib
from collections import OrderedDict
import numpy as np





_COLLISION_LIN_CPP_REQUIRED_API = 39
_SURFACE_DISTANCE_CPP_REQUIRED_API = 4

_collision_lin_cpp_import_error = None
_collision_lin_cpp_runtime_error = None
_collision_lin_cpp_regulation_error = None  # legacy integrated helper only
_collision_lin_cpp_regulation_disabled = False

try:
    import collision_lin_cpp as _collision_lin_cpp_candidate
except Exception as exc:  # pragma: no cover - optional C++ dependency
    _collision_lin_cpp_candidate = None
    _collision_lin_cpp_import_error = repr(exc)

_collision_lin_cpp_api = int(
    getattr(_collision_lin_cpp_candidate, "COLLISION_RESIDUAL_API", 0)
) if _collision_lin_cpp_candidate is not None else 0
if (
    _collision_lin_cpp_candidate is not None
    and _collision_lin_cpp_api >= _COLLISION_LIN_CPP_REQUIRED_API
    and hasattr(_collision_lin_cpp_candidate, "build_collision_multifield_dual_lin_hk")
    and hasattr(_collision_lin_cpp_candidate, "fit_bp_models_seq")
    and int(getattr(_collision_lin_cpp_candidate, "BP_FIT_API", 0)) >= 9
    and hasattr(_collision_lin_cpp_candidate, "evaluate_bp_model_phi_seq")
    and int(getattr(_collision_lin_cpp_candidate, "SURFACE_DISTANCE_API", 0)) >= _SURFACE_DISTANCE_CPP_REQUIRED_API
    and hasattr(_collision_lin_cpp_candidate, "build_surface_distance_lin_hk")
    and hasattr(_collision_lin_cpp_candidate, "evaluate_surface_distance_seq")
):
    collision_lin_cpp = _collision_lin_cpp_candidate
else:
    collision_lin_cpp = None


def cpp_acceleration_status() -> Dict[str, Any]:
    """Return status for the unified dual-constraint C++ accelerator."""
    rejection = None
    if _collision_lin_cpp_candidate is not None and collision_lin_cpp is None:
        if _collision_lin_cpp_api < _COLLISION_LIN_CPP_REQUIRED_API:
            rejection = "API {} is older than required API {}".format(
                _collision_lin_cpp_api, _COLLISION_LIN_CPP_REQUIRED_API
            )
        elif not hasattr(_collision_lin_cpp_candidate, "build_collision_multifield_dual_lin_hk"):
            rejection = "build_collision_multifield_dual_lin_hk is missing"
        elif not hasattr(_collision_lin_cpp_candidate, "fit_bp_models_seq"):
            rejection = "fit_bp_models_seq is missing"
        elif int(getattr(_collision_lin_cpp_candidate, "BP_FIT_API", 0)) < 9:
            rejection = "BP_FIT_API {} is older than required 9".format(int(getattr(_collision_lin_cpp_candidate, "BP_FIT_API", 0)))
        elif not hasattr(_collision_lin_cpp_candidate, "evaluate_bp_model_phi_seq"):
            rejection = "evaluate_bp_model_phi_seq is missing"
        elif int(getattr(_collision_lin_cpp_candidate, "SURFACE_DISTANCE_API", 0)) < _SURFACE_DISTANCE_CPP_REQUIRED_API:
            rejection = "SURFACE_DISTANCE_API {} is older than required {}".format(
                int(getattr(_collision_lin_cpp_candidate, "SURFACE_DISTANCE_API", 0)),
                _SURFACE_DISTANCE_CPP_REQUIRED_API,
            )
        elif not hasattr(_collision_lin_cpp_candidate, "build_surface_distance_lin_hk"):
            rejection = "build_surface_distance_lin_hk is missing"
        elif not hasattr(_collision_lin_cpp_candidate, "evaluate_surface_distance_seq"):
            rejection = "evaluate_surface_distance_seq is missing"
    return {
        "collision_lin_cpp": {
            "enabled": collision_lin_cpp is not None,
            "api": _collision_lin_cpp_api,
            "required_api": _COLLISION_LIN_CPP_REQUIRED_API,
            "path": getattr(_collision_lin_cpp_candidate, "__file__", None),
            "import_error": _collision_lin_cpp_import_error,
            "rejection_reason": rejection,
            "runtime_error": _collision_lin_cpp_runtime_error,
            "native_cuda_compiled": bool(
                getattr(_collision_lin_cpp_candidate, "NATIVE_CUDA_COMPILED", False)
            ) if _collision_lin_cpp_candidate is not None else False,
            "native_cuda_available": bool(
                _collision_lin_cpp_candidate.cuda_available()
            ) if (_collision_lin_cpp_candidate is not None and hasattr(_collision_lin_cpp_candidate, "cuda_available")) else False,
            "bp_cuda_api": int(getattr(_collision_lin_cpp_candidate, "BP_CUDA_API", 0))
                if _collision_lin_cpp_candidate is not None else 0,
            "bp_cuda_enabled": bool(
                collision_lin_cpp is not None
                and int(getattr(_collision_lin_cpp_candidate, "BP_CUDA_API", 0)) >= 1
                and hasattr(_collision_lin_cpp_candidate, "set_bp_cuda_enabled")
            ),
            "native_cuda_last_error": (
                _collision_lin_cpp_candidate.cuda_last_error()
            ) if (_collision_lin_cpp_candidate is not None and hasattr(_collision_lin_cpp_candidate, "cuda_last_error")) else None,
            "dual_constraint_enabled": bool(
                collision_lin_cpp is not None
                and hasattr(collision_lin_cpp, "build_collision_multifield_dual_lin_hk")
            ),
            "horizon_rollout_api": int(getattr(_collision_lin_cpp_candidate, "HORIZON_ROLLOUT_API", 0))
                if _collision_lin_cpp_candidate is not None else 0,
            "horizon_rollout_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "rollout_ego_seq")
            ),
            "dynamics_linearization_api": int(getattr(_collision_lin_cpp_candidate, "DYNAMICS_LINEARIZATION_API", 0))
                if _collision_lin_cpp_candidate is not None else 0,
            "dynamics_linearization_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "build_dynamics_lin_seq")
            ),
            "collision_hk_batch_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "build_collision_multifield_dual_lin_hk")
            ),
            "legacy_proposed_dual_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "build_collision_dual_lin_hk")
            ),
            "multifield_collision_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "build_collision_multifield_dual_lin_hk")
            ),
            "multifield_phi_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "build_collision_multifield_phi_seq")
            ),
            "bp_cpp_fit_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "fit_bp_models_seq")
            ),
            "bp_cpp_eval_enabled": bool(
                collision_lin_cpp is not None and hasattr(collision_lin_cpp, "evaluate_bp_model_phi_seq")
            ),
            "bp_fit_api": int(getattr(_collision_lin_cpp_candidate, "BP_FIT_API", 0))
                if _collision_lin_cpp_candidate is not None else 0,
            "bp_tensor_gram_enabled": bool(
                getattr(_collision_lin_cpp_candidate, "BP_TENSOR_GRAM_ENABLED", False)
            ) if _collision_lin_cpp_candidate is not None else False,
            "bp_tensor_qr_enabled": bool(
                getattr(_collision_lin_cpp_candidate, "BP_TENSOR_QR_ENABLED", False)
            ) if _collision_lin_cpp_candidate is not None else False,
            "bp_parallel_sampling_enabled": bool(
                getattr(_collision_lin_cpp_candidate, "BP_PARALLEL_SAMPLING_ENABLED", False)
            ) if _collision_lin_cpp_candidate is not None else False,
            "surface_distance_api": int(getattr(_collision_lin_cpp_candidate, "SURFACE_DISTANCE_API", 0))
                if _collision_lin_cpp_candidate is not None else 0,
            "surface_distance_cpp_enabled": bool(
                collision_lin_cpp is not None
                and hasattr(collision_lin_cpp, "build_surface_distance_lin_hk")
                and hasattr(collision_lin_cpp, "evaluate_surface_distance_seq")
            ),
        }
    }


def require_cpp_acceleration(require_regulation: bool = False) -> None:
    """Raise if the unified C++ dual-constraint accelerator is unavailable.

    ``require_regulation`` is retained as an ignored compatibility argument;
    there is no separate regulation module anymore.
    """
    status = cpp_acceleration_status()["collision_lin_cpp"]
    if not status["enabled"]:
        raise RuntimeError("collision_lin_cpp is unavailable: {!r}".format(status))


def _configure_native_cuda(cfg: "MPCConfig") -> None:
    """Push proposed-field CUDA policy into the C++ extension.

    The actual GPU implementation, persistent device buffers and kernels are
    entirely native C++/CUDA; this function only updates four scalar policies.
    """
    if collision_lin_cpp is None:
        return
    if hasattr(collision_lin_cpp, "set_cuda_enabled"):
        collision_lin_cpp.set_cuda_enabled(
            _bool_compat(getattr(cfg, "use_native_cuda_proposed", True), True)
        )
    if hasattr(collision_lin_cpp, "set_cuda_min_hf"):
        collision_lin_cpp.set_cuda_min_hf(
            max(int(getattr(cfg, "native_cuda_min_hf", 1024)), 0)
        )
    if hasattr(collision_lin_cpp, "set_cuda_threads"):
        collision_lin_cpp.set_cuda_threads(
            int(getattr(cfg, "native_cuda_threads", 128))
        )
    if hasattr(collision_lin_cpp, "set_cuda_fallback_to_cpu"):
        collision_lin_cpp.set_cuda_fallback_to_cpu(
            _bool_compat(getattr(cfg, "native_cuda_fallback_to_cpu", True), True)
        )
    if hasattr(collision_lin_cpp, "set_cuda_profile"):
        collision_lin_cpp.set_cuda_profile(
            _bool_compat(getattr(cfg, "native_cuda_profile", False), False),
            _bool_compat(getattr(cfg, "native_cuda_compare_cpu", False), False),
            max(int(getattr(cfg, "native_cuda_profile_every", 1)), 1),
        )
    if hasattr(collision_lin_cpp, "set_bp_cuda_enabled"):
        collision_lin_cpp.set_bp_cuda_enabled(
            _bool_compat(getattr(cfg, "use_native_cuda_bp", True), True)
        )
    if hasattr(collision_lin_cpp, "set_bp_cuda_min_ho"):
        collision_lin_cpp.set_bp_cuda_min_ho(
            max(int(getattr(cfg, "native_cuda_bp_min_ho", 16)), 0)
        )
    if hasattr(collision_lin_cpp, "set_bp_cuda_threads"):
        collision_lin_cpp.set_bp_cuda_threads(
            int(getattr(cfg, "native_cuda_bp_threads", 32))
        )
    if hasattr(collision_lin_cpp, "set_bp_cuda_fallback_to_cpu"):
        collision_lin_cpp.set_bp_cuda_fallback_to_cpu(
            _bool_compat(getattr(cfg, "native_cuda_bp_fallback_to_cpu", True), True)
        )
    if hasattr(collision_lin_cpp, "set_bp_cuda_profile"):
        collision_lin_cpp.set_bp_cuda_profile(
            _bool_compat(getattr(cfg, "native_cuda_bp_profile", False), False),
            _bool_compat(getattr(cfg, "native_cuda_bp_compare_cpu", False), False),
            max(int(getattr(cfg, "native_cuda_bp_profile_every", 1)), 1),
        )


# Runtime policy: this module is C++-accelerated only.  Do not silently
# continue with the NumPy/Python collision kernels when the extension is
# missing, stale, or ABI-incompatible.  Fail at import so the build problem is
# visible before the planner starts.
require_cpp_acceleration()

if os.environ.get("UTILS_LINEARIZED_CPP_VERBOSE", "0").strip().lower() in (
    "1", "true", "yes", "on"
):
    print("[utils_mpc] C++ status: {}".format(cpp_acceleration_status()))


# -----------------------------------------------------------------------------
# Config / result containers
# -----------------------------------------------------------------------------


class MPCConfig(object):
    """MPC/acados configuration.

    This intentionally avoids @dataclass for Python 3.6 compatibility.
    """

    __dataclass_fields__ = {
        "planner_id": None,
        "hz": None,
        "N": None,
        "dt_init": None,
        "ego_L": None,
        "ego_W": None,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "ego_x": None,
        "ego_y": None,
        "ego_z": None,
        "ego_v0": None,
        "ego_yaw0": None,
        "goal_x": None,
        "goal_y": None,
        "goal_z": None,
        "yawrate_min": None,
        "yawrate_max": None,
        "yawrate_time_constant": None,
        "ego_yawrate0": None,
        "a_min": None,
        "a_max": None,
        "v_min": None,
        "v_max": None,
        "no_acc_flag": None,
        "nonlinear_dyn_flag": None,
        "q_pos": None,
        "q_pos_terminal": None,
        "q_yaw": None,
        "q_v": None,
        "v_des": None,
        "r_yawrate": None,
        "r_a": None,
        "r_du": None,
        "obs_sensing_R": None,
        "obs_fov_deg": None,
        "msg_max_vertices": None,
        "fallback_polygon_vertices": None,
        "perception_rate": None,
        "disabled_c_value": None,
        "h_upper_bound": None,
        "acados_model_name": None,
        "acados_json_file": None,
        "acados_code_export_dir": None,
        "acados_qp_solver": None,
        "acados_qp_solver_iter_max": None,
        "acados_hessian_approx": None,
        "acados_integrator_type": None,
        "acados_nlp_solver_type": None,
        "acados_sim_method_num_stages": None,
        "acados_sim_method_num_steps": None,
        "acados_nlp_solver_max_iter": None,
        "acados_tol": None,
        "acados_rebuild_solver": None,
        "use_acados_cython_wrapper": None,
        "acados_fixed_hess": None,
        "acados_qp_solver_warm_start": None,
        "ORDER": None,
        "SQP_iter": None,
        "realtime_max_sqp_iter": None,
        "sqp_min_iter": None,
        "sqp_early_stop_tol": None,
        "use_acados_flat_io": None,
        "frame_id": None,
        "t_scale": None,
        "marker_width": None,
        "ego_trail_max": None,
        "moving_obstacle_R": None,
        "ego_R": None,
        "safety_margin": None,
        "enable_boundary_constraint": None,
        "enable_normal_mag_avoidance_constraint": None,
        "enable_impact_normal_constraint": None,
        "collsion_avoidance_off_over_20": None,
        "collision_avoidance_off_over_20": None,
        "sector_id": None,
        "sector_dir": None,
        "designated_surface_id": None,
        "collision_slack_max": None,
        "normal_alignment_slack_max": None,
        "enable_stage1_bspline_tracking": None,
        "stage1_position_gain": None,
        "stage1_heading_gain": None,
        "stage1_velocity_gain": None,
        "d_safe": None,
        "rho_Phi_global": None,
        "rho_s": None,
        "rho_c": None,
        "rho_impact": None,
        "rho_terminal_goal_global": None,
        "q_global_goal": None,
        "Ns": None,  # derived at runtime from theta and sector half-width alpha
        "theta": None,
        "reg_margin": None,
        "delta_o": None,
        "delta_f": None,
        "M_max": None,
        "F_max": None,
        "epsilon_o": None,
        "epsilon_f": None,
        "lambda_r": None,
        "debug_mode": None,
        "regulation_theta": None,
        "regulation_lambda": None,
        "beta": None,  # deprecated pre-Eq.-(34) compatibility field
        "mitigation_alpha": None,  # deprecated pre-paper gate parameter
        "alpha": None,  # deprecated compatibility alias
        # MPC formulation selector.
        #   0: Uniform-slack surface-wise MPC
        #   1: Surface-wise MPC (vulnerable surface hard)
        #   2: Weighted vulnerability-aware surface-wise MPC
        #   3: Lexicographic surface-wise MPC
        #   4: Proposed minimum-distance + impact-location-conditioned relaxation,
        #      solved once per outer SQP iteration with a scalarized local objective
        "mpc_flag": None,
        "surface_slack_max": None,
        "q_surface_slack": None,
        "q_vulnerable_slack": None,
        "q_nonvulnerable_slack": None,
        "lexicographic_fix_tolerance": None,
        # Collision-field configuration (flattened into the single MPC config).
        # field_flag mapping used by the proposed OCP collision path:
        #   0 circle-overestimation SDF
        #   1 DCSAT
        #   2 Geometry-Aware Bernstein-polynomial SDF
        #   3 Wu et al. (2025) smooth polygon field
        #   4 proposed proximity field with configured N_n
        "field_flag": None,
        "N_n": None,
        "epsilon_tot": None,
        "normal_orientation": None,
        "lambda_f": None,
        "lambda_o": None,
        "normal_eps": None,
        "fd_position_eps_m": None,
        "fd_yaw_eps_rad": None,
        "use_probability_propagation_margin": None,
        "sigma_vx": None,
        "sigma_vy": None,
        "probability_collision": None,
        "circle_union_lambda": None,
        "dcsat_alpha": None,
        "dcsat_union_lambda": None,
        "wu_kappa": None,
        "wu_union_lambda": None,
        "wu_buffer_offset": None,
        "bp_order": None,
        "bp_training_n_x": None,
        "bp_training_n_y": None,
        "bp_safety_margin_l": None,
        "bp_domain_padding": None,
        "use_ego_scaled_local_frame": None,
        "use_external_collision_linearization": None,
        "cpp_omp_threads": None,
        "ego_axis_scale": None,
        "use_native_cuda_proposed": None,
        "native_cuda_min_hf": None,
        "native_cuda_threads": None,
        "native_cuda_fallback_to_cpu": None,
        "native_cuda_profile": None,
        "native_cuda_compare_cpu": None,
        "native_cuda_profile_every": None,
        "use_native_cuda_bp": None,
        "native_cuda_bp_min_ho": None,
        "native_cuda_bp_threads": None,
        "native_cuda_bp_fallback_to_cpu": None,
        "native_cuda_bp_profile": None,
        "native_cuda_bp_compare_cpu": None,
        "native_cuda_bp_profile_every": None,
    }

    @property
    def vertex_capacity(self):
        """Derived buffer capacity, never a separate vertex-count setting.

        Proposed field: each complete convex polygon vertex generates one
        edge and N_n pseudo-normal features.  The selector enforces the
        per-original-obstacle F_max limit before publication.
        """
        if int(getattr(self, "field_flag", 4)) == 4:
            return max(0, int(self.M_max)) * max(0, int(self.F_max) // (max(0, int(self.N_n)) + 1))
        return max(0, int(self.M_max)) * max(1, int(self.msg_max_vertices))

    def __init__(self, **kwargs):
        self.planner_id = 1
        self.hz = 20.0

        # Horizon.
        self.N = 10
        # Fixed MPC sampling period. It is a configuration parameter, not an acados state.
        self.dt_init = 0.1

        # Ego / map.
        self.ego_L = 0.6
        self.ego_W = 0.35
        self.x_min = -6.0
        self.x_max = 6.0
        self.y_min = -6.0
        self.y_max = 6.0
        self.ego_x = -5.99
        self.ego_y = -5.99
        self.ego_z = 8.5
        self.ego_v0 = 5.0
        self.ego_yaw0 = math.pi / 4.0
        self.goal_x = 6.0
        self.goal_y = 6.0
        self.goal_z = 8.5

        # Bounds.
        self.yawrate_min = -0.6
        self.yawrate_max = 0.6
        # First-order yaw-rate actuator lag used by every MPC prediction model:
        #   yaw' = yawrate,  yawrate' = (yawrate_cmd - yawrate) / yawrate_time_constant.
        # The MPC state is therefore x = [px, py, v, yaw, yawrate] and the
        # control u[0] is the commanded yaw rate.
        self.yawrate_time_constant = 0.7
        # Actual (lagged) yaw rate at x0. The ROS planners refresh this every
        # planning cycle; rollouts given a four-state x0 use it as x0[4].
        self.ego_yawrate0 = 0.0
        self.a_min = -0.0
        self.a_max = 0.0
        self.v_min = -10.0
        self.v_max = 10.0
        self.no_acc_flag = False
        # true: nonlinear RK4 map in acados; false: stage-wise affine RK4
        # linearization, yielding a convex QP for each outer refinement step.
        self.nonlinear_dyn_flag = True

        # Costs.
        self.q_pos = 1.0
        # Eq. (35) uses the same Q at k=1,...,H. A separate terminal
        # tracking weight is retained only as a legacy config field.
        self.q_pos_terminal = 1.0
        self.q_yaw = 0.0
        self.q_v = 0.0
        self.v_des = 1.2
        self.r_yawrate = 1.0e-2
        self.r_a = 1.0e-2
        self.r_du = 0.0
        # Legacy initialization helper settings. They do not add a B-spline
        # tracking term to the OCP; q_pos tracks the global goal directly.
        self.enable_stage1_bspline_tracking = True
        self.stage1_position_gain = 1.5
        self.stage1_heading_gain = 2.0
        self.stage1_velocity_gain = 2.0

        # Local vertex selection / message parsing.
        self.obs_sensing_R = 5.0
        self.obs_fov_deg = 80.0
        self.msg_max_vertices = 10
        self.fallback_polygon_vertices = 8
        self.perception_rate = 1.0

        # Half-space construction.
        self.disabled_c_value = -1.0e9
        self.h_upper_bound = 1.0e15

        # acados options.
        self.acados_model_name = "switchable_mpc_compact_delta_dynamics"
        self.acados_json_file = "acados_switchable_mpc_compact_delta_dynamics.json"
        self.acados_code_export_dir = "c_generated_code_switchable_mpc_compact_delta_dynamics"
        self.acados_qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.acados_qp_solver_iter_max = 30
        self.acados_hessian_approx = "GAUSS_NEWTON"
        self.acados_integrator_type = "DISCRETE"
        # nonlinear_dyn_flag=true uses the nonlinear CasADi RK4 map.  The
        # false branch still uses SQP_RTI as a thin one-QP driver around HPIPM;
        # because the generated dynamics/constraints are affine, one solve is
        # the exact convex subproblem solve.
        self.acados_nlp_solver_type = "SQP_RTI"
        self.acados_sim_method_num_stages = 4
        self.acados_sim_method_num_steps = 1
        # For SQP_RTI this is the runtime cap on repeated RTI steps performed
        # under ONE externally constructed collision linearization. The generated
        # acados SQP_RTI solver itself is kept at one NLP iteration per solve()
        # call; AcadosMPCPlanner explicitly repeats solve() up to this many times.
        self.acados_nlp_solver_max_iter = 1
        self.acados_tol = 1.0e-6
        self.acados_rebuild_solver = False
        # The Cython wrapper implements the same solver API with much lower
        # per-stage set/get overhead. If it cannot be built or imported, the
        # planner automatically keeps the standard ctypes wrapper.
        self.use_acados_cython_wrapper = True
        # With LINEAR_LS + GAUSS_NEWTON the cost Hessian is constant; nonlinear
        # dynamics/constraint Jacobians are still re-evaluated by acados.
        self.acados_fixed_hess = True
        # Reuse the preceding HPIPM iterate between outer collision-refinement SQP steps.
        self.acados_qp_solver_warm_start = 1

        # Collision-refinement loop.
        # ORDER selects the nominal input bias:
        #   1: yawrate = 0
        #   2: yawrate = yawrate_max
        #   3: yawrate = yawrate_min
        self.ORDER = 1
        self.SQP_iter = 2
        # Total collision-refinement solver budget per control cycle.
        # Each outer iteration solves either the nonlinear RK4 OCP or, when
        # nonlinear_dyn_flag=false, its affine-dynamics convex QP approximation
        # using collision-field linearization built at the current nominal.
        self.realtime_max_sqp_iter = 3
        self.sqp_min_iter = 1
        self.sqp_early_stop_tol = 1.0e-3
        # Recent acados_template versions expose set_flat/get_flat.  Using them
        # removes O(N) Python/C boundary calls for p, x, and u.
        self.use_acados_flat_io = True

        # Visualization / ROS helpers.
        self.frame_id = "map"
        self.t_scale = 1.0
        self.marker_width = 0.05
        self.ego_trail_max = 300
        self.moving_obstacle_R = 0.2
        self.ego_R = 0.1
        self.safety_margin = 0.0
        # Toggle softened x/y map-boundary constraints.  When False, the
        # planner no longer constrains px/py by x_min/x_max/y_min/y_max.
        self.enable_boundary_constraint = True

        # Paper constraints: avoidance plus impact-location regulation.
        #   Phi_safe^lin >= -s^a, where d_safe [m] is applied in world-frame feature inflation
        #   lambda_r*LSE(Phi_safe/lambda_r,
        #                (Ns-N^T n(phi_c))/lambda_r) >= lambda_r*log(2).
        # The proposed field itself uses lambda_o=lambda_f, so its nested
        # log/exp terms are collapsed before Phi and N are linearized.
        # lambda_r is independent and only controls the regulation LSE.
        # The local SQP may soften only that row with s^r.
        self.enable_normal_mag_avoidance_constraint = True
        # Toggle the impact-normal alignment/regulation constraint independently
        # from the distance/avoidance constraint above.
        self.enable_impact_normal_constraint = True
        # If enabled, skip collision-avoidance constraints whenever the actual
        # MPC horizon cfg.N (Np) is 20 or larger.  The misspelled key is kept
        # for JSON compatibility with existing launch/config files.
        self.collsion_avoidance_off_over_20 = False
        self.collision_avoidance_off_over_20 = False
        self.d_safe = 0.3
        self.rho_Phi_global = 1.0
        # Proposed weights (mpc_flag=4):
        #   + rho_s * sum (s_k^a)^2
        #   + rho_c * sum (s_k^r)^2
        #   + rho_impact * sum [(-grad Phi_k(0))^T n(phi_c)] in J_global.
        # rho_s and rho_c are local OCP LINEAR_LS slack weights; rho_impact is
        # used only by the post-solve impact-alignment ranking term.
        self.rho_s = 1.0e8
        self.rho_c = 1.0e8
        self.rho_impact = 100.0
        self.rho_terminal_goal_global = 1.0e3  # legacy alias
        self.q_global_goal = self.rho_terminal_goal_global
        # Paper Eqs. (32)--(33): alpha is the vulnerable-sector half-width and
        # theta is the impact-normal cone half-angle.  Ns is derived, not tuned:
        #     Ns = cos(theta + alpha) + reg_margin.
        # theta and reg_margin are configured from /setup (ultimately JSON).
        self.Ns = None
        self.theta = math.pi / 4.0
        self.reg_margin = 0.0
        self.delta_o = 0.008
        self.delta_f = 0.005
        self.M_max = 200
        self.F_max = 36

        self.epsilon_o = 0.999
        self.epsilon_f = None
        self.lambda_r = 0.01
        # Global diagnostic switch. Keep disabled by default for real-time runs.
        # When enabled from /setup.mpc.debug_mode, detailed regulation
        # linearization diagnostics such as JPhi, Jalign, and Jreg are printed.
        self.debug_mode = False
        self.regulation_analysis_log_max_stages = 64
        self.regulation_theta = None  # optional explicit alias; overrides theta when set
        self.regulation_lambda = 0.01  # compatibility alias of lambda_r
        self.beta = 0.3  # deprecated; retained only as a migration fallback
        self.mitigation_alpha = 0.0  # deprecated pre-paper gate parameter
        self.alpha = None
        self.sector_id = 1
        # "outward": keep sector rays as originally defined.
        # "inward": flip la/lb (and the derived center lc) to point inward.
        self.sector_dir = "outward"
        self.designated_surface_id = 1

        # MPC formulation.  Surface-wise baselines use four surface constraints
        # ordered [left, rear, right, front].  sector_id selects which one is
        # impact-vulnerable; the other three are non-vulnerable.
        self.mpc_flag = 4
        self.surface_slack_max = 1.0e3
        self.q_surface_slack = 1.0e8
        self.q_nonvulnerable_slack = 1.0e8
        self.q_vulnerable_slack = 1.0e10
        self.lexicographic_fix_tolerance = 1.0e-7

        self.collision_slack_max = 1.0e3
        self.normal_alignment_slack_max = 1.0e3

        # Smooth collision-field parameters live in this same config object.
        # Keep field_flag independent of the acados OCP: only the externally
        # generated collision residual/Jacobian changes between methods.
        self.field_flag = 4
        self.N_n = 5
        # Dominant-feature/obstacle weight lower bound used by the angular-error
        # guarantee. For epsilon_tot > 1/2, N_n is automatically raised to the
        # smallest integer satisfying theta_err + Delta_theta_n <= theta.
        self.epsilon_tot = 0.7321  # overwritten by /setup auto-derivation
        self.normal_orientation = "inward"
        self.lambda_f = 0.001  # proposed-field feature smoothing
        self.lambda_o = 0.001  # proposed-field obstacle smoothing
        self.normal_eps = 1.0e-9
        self.fd_position_eps_m = 1.0e-4
        self.fd_yaw_eps_rad = 1.0e-4
        self.use_probability_propagation_margin = True
        self.sigma_vx = 0.2
        self.sigma_vy = 0.2
        self.probability_collision = 0.8
        self.circle_union_lambda = 0.0
        self.dcsat_alpha = 35.0
        self.dcsat_union_lambda = 0.02
        self.wu_kappa = 35.0
        self.wu_union_lambda = 0.02
        self.wu_buffer_offset = 0.0
        self.bp_order = 12
        self.bp_training_n_x = 30
        self.bp_training_n_y = 30
        self.bp_safety_margin_l = 0.13
        self.bp_domain_padding = 1.0  # obstacle-local padding for fitted BP boundary
        self.use_ego_scaled_local_frame = True
        self.use_external_collision_linearization = True
        # Per-planner OpenMP cap for C++ horizon-batched collision kernels.
        # Multiple MPC_planner processes are commonly run in parallel; capping
        # each process avoids severe CPU oversubscription while retaining native
        # parallelism inside one H-stage linearization call.
        self.cpp_omp_threads = 2
        self.ego_axis_scale = [1.0, 1.0]
        # field_flag=4 native CUDA backend inside collision_lin_cpp.
        # No CuPy/Python GPU path remains.
        self.use_native_cuda_proposed = True
        self.native_cuda_min_hf = 1024
        self.native_cuda_threads = 128
        self.native_cuda_fallback_to_cpu = True
        # Benchmark mode: when enabled, emit [COLLISION-BENCH] timing lines.
        # compare_cpu evaluates both backends on the exact same input and returns
        # whichever backend the normal HF threshold policy selected.
        self.native_cuda_profile = False
        self.native_cuda_compare_cpu = False
        self.native_cuda_profile_every = 1
        # field_flag=2 BP-SDF CUDA backend. The polynomial fit stays CPU/cached;
        # repeated metric-SDF/gradient queries and Hx21 collision linearization
        # are offloaded when H * obstacle_count reaches native_cuda_bp_min_ho.
        self.use_native_cuda_bp = True
        self.native_cuda_bp_min_ho = 16
        self.native_cuda_bp_threads = 32
        self.native_cuda_bp_fallback_to_cpu = True
        self.native_cuda_bp_profile = False
        self.native_cuda_bp_compare_cpu = False
        self.native_cuda_bp_profile_every = 1

        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)


class PlanResult(object):
    def __init__(
        self,
        U,
        X,
        dt,
        exitflag,
        cost,
        vertex_set=None,
        halfspaces=None,
        slack=None,
        normal_slack=None,
        surface_slack=None,
        boundary_slack=0.0,
        vertex_types=None,
        dynamic_halfspace_mask=None,
    ):
        self.U = U
        self.X = X
        self.dt = float(dt)
        self.exitflag = int(exitflag)
        self.cost = float(cost) if cost is not None else float("nan")
        # Separate post-solve candidate-ranking objective.  ``cost`` remains the
        # local/scalarized MPC objective and is never overwritten by J_global.
        self.J_global = float("inf")
        self.global_cost = float("inf")
        self.vertex_set = np.asarray(vertex_set if vertex_set is not None else np.zeros((0, 2)), dtype=float)
        self.halfspaces = np.asarray(halfspaces if halfspaces is not None else np.zeros((0, 3)), dtype=float)
        self.slack = np.asarray(slack if slack is not None else np.zeros(0), dtype=float)
        self.normal_slack = np.asarray(normal_slack if normal_slack is not None else np.zeros(0), dtype=float)
        if surface_slack is None:
            Hs = max(self.slack.size, self.normal_slack.size)
            ss = np.zeros((4, Hs), dtype=float)
            if self.slack.size:
                ss[0, :self.slack.size] = self.slack
            if self.normal_slack.size:
                ss[1, :self.normal_slack.size] = self.normal_slack
            self.surface_slack = ss
        else:
            ss = np.asarray(surface_slack, dtype=float)
            if ss.ndim == 1:
                ss = ss.reshape(1, -1)
            self.surface_slack = ss.reshape(ss.shape[0], -1)
        self.boundary_slack = float(boundary_slack)
        self.vertex_types = np.asarray(vertex_types if vertex_types is not None else np.zeros(0), dtype=np.int32)
        self.dynamic_halfspace_mask = np.asarray(
            dynamic_halfspace_mask if dynamic_halfspace_mask is not None else np.zeros(0),
            dtype=bool,
        )
        # Metadata consumed by ego_mover's Algorithm-1 candidate selection.
        self.avoidance_residual_max = float("inf")
        self.normal_alignment_residual_max = float("inf")
        self.phi_min = float("-inf")
        self.min_clearance = float("-inf")
        self.ocp_feasible = False
        self.paper_feasible = False  # legacy runtime-checkable subset of Algorithm-1 feasibility
        # Candidate check used by MPC_planner1~6 before publishing to ego_mover.
        # "Zero slack" refers only to the auxiliary regulation slack s^r.
        # The paper avoidance slack s^a (= s_k) remains active in the avoidance
        # constraint and in J_global.
        self.zero_slack_avoidance_residual_max = float("inf")
        self.zero_slack_normal_alignment_residual_max = float("inf")
        self.zero_slack_feasible = False

        # Legacy feasibility fields consumed by the planner-selection pipeline.
        # For mpc_flag=4 the runtime annotation uses the proposed-field
        # quantities Phi_k(0) and N_k(0); it does not recompute true geometric
        # clearance or a true impact normal.
        self.original_ocp_avoidance_residual_max = float("inf")
        self.original_ocp_slack_nonneg_residual_max = float("inf")
        self.original_ocp_mitigation_residual_max = float("inf")
        self.original_ocp_feasible = False
        self.nonsmooth_ocp_feasible = False  # compatibility alias; no geometric re-check is performed
        self.selection_eligible = False
        self.sqp_iteration = -1
        # Successful outer-SQP iterates returned by AcadosMPCPlanner.solve().
        # Candidate feasibility/cost are finalized inside utils_linearized so the
        # ROS wrappers only select and publish the returned candidates.
        self.intermediate_results = []
        self.feasible_candidates = []
        self.successful_iteration_count = 0
        self.returned_candidate_count = 0
        self.returned_candidate_costs = []
        self.returned_candidate_J_globals = []
        # Runtime RTI diagnostics for the outer SQP iterate that produced this
        # candidate.  rti_steps is the total number of SQP_RTI solve() calls
        # performed without rebuilding the external collision linearization.
        self.rti_steps = 0
        self.rti_converged = False
        self.rti_step_inf = float("inf")
        self.rti_phase_steps = {}
        # Preserve the raw acados exit while allowing ACADOS_MAXITER to reuse
        # its finite current iterate as the next outer-SQP nominal trajectory.
        self.raw_solver_status = 0
        self.maxiter_recovered = False
        # Wall-clock phase timings populated by AcadosMPCPlanner.solve().
        # timing_ms stores cycle-aggregated phase time; iteration_timing_ms
        # stores the same breakdown for each outer SQP iteration.
        self.timing_ms = {}
        self.iteration_timing_ms = []
        self.zero_slack_feasible_mask = np.zeros(0, dtype=bool)
        self.nonsmooth_feasible_mask = np.zeros(0, dtype=bool)  # compatibility alias
        self.feasible_candidate_count = 0
        self.feasible_candidate_costs = []
        self.feasible_candidate_J_globals = []


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _bool_compat(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def collision_safe_distance(cfg: Any) -> float:
    """Return d_safe for the two collision constraints."""
    return max(abs(float(getattr(cfg, "d_safe", 0.2))), 1.0e-12)


def regulation_theta_value(cfg: Any) -> float:
    """Return theta in Eqs. (23), (28), and (33), in radians."""
    raw = getattr(cfg, "regulation_theta", None)
    if raw is None:
        raw = getattr(cfg, "theta", math.pi / 4.0)
    try:
        theta = float(raw)
    except Exception:
        theta = math.pi / 4.0
    if not math.isfinite(theta):
        theta = math.pi / 4.0
    return min(max(theta, 1.0e-9), math.pi / 2.0)


def pseudo_normal_angular_bound(cfg: Any, N_n: int) -> Tuple[float, float, float]:
    """Return (theta_err, Delta_theta_n, total_bound) for a pseudo-normal count.

    The bound follows Proposition 2 / Eq. (18) in the paper:
        Delta_theta_n = pi / (2 (N_n + 1)),
        theta_err = atan2(1-epsilon_tot, 2*epsilon_tot - 1).
    A finite useful guarantee requires epsilon_tot > 1/2.
    """
    eps = float(getattr(cfg, "epsilon_tot", 0.7321))
    if not math.isfinite(eps) or not (0.5 < eps <= 1.0):
        raise ValueError(
            "epsilon_tot must satisfy 0.5 < epsilon_tot <= 1 for the angular bound; got {!r}".format(eps)
        )
    n = max(0, int(N_n))
    delta_theta_n = math.pi / (2.0 * float(n + 1))
    theta_err = math.atan2(1.0 - eps, 2.0 * eps - 1.0)
    return float(theta_err), float(delta_theta_n), float(theta_err + delta_theta_n)


def required_pseudo_normal_count(cfg: Any) -> int:
    """Smallest N_n satisfying theta_err + Delta_theta_n <= configured theta."""
    theta = regulation_theta_value(cfg)
    # In the paper, theta_err depends only on epsilon_tot; N_n only changes
    # Delta_theta_n.  Check the irreducible theta_err before searching N_n.
    eps = float(getattr(cfg, "epsilon_tot", 0.7321))
    if not math.isfinite(eps) or not (0.5 < eps <= 1.0):
        raise ValueError(
            "epsilon_tot must satisfy 0.5 < epsilon_tot <= 1 for automatic N_n; got {!r}".format(eps)
        )
    theta_err = math.atan2(1.0 - eps, 2.0 * eps - 1.0)
    if theta <= theta_err:
        raise ValueError(
            "configured theta={:.9g} rad is too small for epsilon_tot={:.9g}; "
            "it must exceed theta_err={:.9g} rad".format(theta, eps, theta_err)
        )
    for n in range(0, 100000):
        _, _, total = pseudo_normal_angular_bound(cfg, n)
        if total <= theta + 1.0e-12:
            return int(n)
    raise ValueError("failed to find a finite N_n satisfying the configured angular bound")


def effective_pseudo_normal_count(cfg: Any) -> int:
    """Validate/raise cfg.N_n against the theorem-required lower bound and return it.

    With the new /setup path, epsilon_tot and N_n are normally already derived
    from theta, delta_o, and delta_f by setup_config.py.
    """
    configured = max(0, int(getattr(cfg, "N_n", 0)))
    if collision_field_flag(cfg) != 4:
        return configured
    required = required_pseudo_normal_count(cfg)
    effective = max(configured, required)
    try:
        cfg.N_n = int(effective)
    except Exception:
        pass
    return int(effective)


def regulation_margin_value(cfg: Any) -> float:
    """Return the configured additive regulation threshold margin."""
    raw = getattr(cfg, "reg_margin", 0.0)
    try:
        margin = float(raw)
    except Exception:
        margin = 0.0
    return margin if math.isfinite(margin) else 0.0


def regulation_sector_half_width(
    cfg: Any,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """Return the vulnerable-sector half-width in the unit-disk frame.

    The proposed field and its impact normal are defined in

        q = S^{-1} R(yaw)^T (P-p),

    so the sector boundary rays must be expressed in the same q frame before
    measuring their angular opening.  Since the rays are body-fixed, only the
    anisotropic scale S^{-1} is required here; the common rotation cancels.
    """
    if sector_rays_body is None:
        la, lb, _ = sector_rays_body_from_cfg(cfg)
    else:
        rays = np.asarray(sector_rays_body, dtype=float).reshape(3, 2)
        la, lb = rays[0], rays[1]

    la = np.asarray(la, dtype=float).reshape(2)
    lb = np.asarray(lb, dtype=float).reshape(2)

    if use_ego_scaled_local_frame(cfg) and collision_field_flag(cfg) != 2:
        scale = _ego_axis_scale_array(cfg)
        sx = max(abs(float(scale[0])), 1.0e-12)
        sy = max(abs(float(scale[1])), 1.0e-12)
        la = np.array([la[0] / sx, la[1] / sy], dtype=float)
        lb = np.array([lb[0] / sx, lb[1] / sy], dtype=float)

    la = la / max(float(np.linalg.norm(la)), 1.0e-12)
    lb = lb / max(float(np.linalg.norm(lb)), 1.0e-12)
    opening = math.acos(min(max(float(la @ lb), -1.0), 1.0))
    return 0.5 * opening


def regulation_alignment_Ns(
    cfg: Any,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """Return Ns = cos(theta + alpha) + reg_margin.

    ``reg_margin`` is an additive threshold margin configured through JSON -> /setup.
    The legacy
    cfg.Ns field is intentionally ignored so JSON files cannot silently restore
    the previous independently tuned threshold.
    """
    theta = regulation_theta_value(cfg)
    alpha = regulation_sector_half_width(cfg, sector_rays_body)
    reg_margin = regulation_margin_value(cfg)
    value = float(math.cos(theta + alpha) + reg_margin)
    try:
        cfg.Ns = value
    except Exception:
        pass
    return value


def collision_smoothing_lambda(cfg: Any) -> float:
    """Return the common proposed-field smoothing lambda.

    ``lambda_f`` is the source of truth for field_flag=4 and is copied only to
    ``lambda_o``.  This enforces lambda_o=lambda_f for the field-level log/exp
    cancellation while leaving the regulation parameter ``lambda_r``
    independent. DCSAT, Wu, and circle-union smoothing use their own
    field-specific parameters.
    """
    value = getattr(cfg, "lambda_f", 1.0e-3)
    try:
        value = float(value)
    except Exception:
        value = 1.0e-3
    if not math.isfinite(value) or value <= 0.0:
        value = 1.0e-3
    value = max(value, 1.0e-9)
    for name in ("lambda_f", "lambda_o"):
        try:
            setattr(cfg, name, value)
        except Exception:
            pass
    return value


def regulation_smoothing_lambda(cfg: Any) -> float:
    """Return the independent regulation LSE smoothing parameter lambda_r."""
    value = getattr(cfg, "lambda_r", getattr(cfg, "regulation_lambda", 1.0e-2))
    try:
        value = float(value)
    except Exception:
        value = 1.0e-2
    if not math.isfinite(value) or value <= 0.0:
        value = 1.0e-2
    value = max(value, 1.0e-9)
    for name in ("lambda_r", "regulation_lambda"):
        try:
            setattr(cfg, name, value)
        except Exception:
            pass
    return value


def collision_field_flag(cfg: Any) -> int:
    """Return the selected collision field in the unified 0..4 mapping."""
    try:
        flag = int(round(float(getattr(cfg, "field_flag", 4))))
    except Exception:
        flag = 4
    if flag not in (0, 1, 2, 3, 4):
        raise ValueError("field_flag must be one of 0, 1, 2, 3, or 4; got {!r}".format(flag))
    return flag


def collision_field_name(cfg: Any) -> str:
    return {
        0: "Circle-overestimation SDF",
        1: "DCSAT field",
        2: "Geometry-Aware Bernstein-polynomial SDF",
        3: "Wu et al. (2025) smooth polygon field",
        4: "Proposed field (effective N_n={})".format(effective_pseudo_normal_count(cfg)),
    }[collision_field_flag(cfg)]


def mpc_formulation_flag(cfg: Any) -> int:
    """Return the selected MPC formulation in the 0..4 mapping."""
    try:
        flag = int(round(float(getattr(cfg, "mpc_flag", 4))))
    except Exception:
        flag = 4
    if flag not in (0, 1, 2, 3, 4):
        raise ValueError("mpc_flag must be one of 0, 1, 2, 3, or 4; got {!r}".format(flag))
    return flag


def mpc_formulation_name(cfg: Any) -> str:
    return {
        0: "Uniform-slack MPC",
        1: "Surface-wise MPC",
        2: "Weighted Vulnerability-Aware MPC",
        3: "Lexicographic MPC",
        4: "Proposed minimum-distance + impact-location-conditioned MPC",
    }[mpc_formulation_flag(cfg)]


def vulnerable_surface_index(cfg: Any) -> int:
    """Return zero-based vulnerable surface index [left,rear,right,front]."""
    return normalized_sector_id(getattr(cfg, "sector_id", 1)) - 1


def proposed_active_slack_count(cfg: Any) -> int:
    """Return 1 for avoidance-only proposed OCP, otherwise 2 for [s^a, s^r]."""
    return 2 if impact_normal_constraint_enabled(cfg) else 1


def surface_slack_upper_bounds(cfg: Any) -> np.ndarray:
    """Return per-surface/constraint slack upper bounds for the selected MPC.

    For mpc_flag=4 this is [ub(s^a)] when impact-normal regulation is disabled,
    otherwise [ub(s^a), ub(s^r)].  Surface-wise baselines 0..3 return four
    bounds [left,rear,right,front].
    """
    flag = mpc_formulation_flag(cfg)
    surface_ub = max(_finite_or_default(getattr(cfg, "surface_slack_max", getattr(cfg, "collision_slack_max", 1.0e3)), 1.0e3), 0.0)
    if flag == 4:
        avoid_ub = max(_finite_or_default(getattr(cfg, "collision_slack_max", surface_ub), surface_ub), 0.0)
        if proposed_active_slack_count(cfg) == 1:
            return np.array([avoid_ub], dtype=float)
        reg_ub = max(_finite_or_default(getattr(cfg, "normal_alignment_slack_max", avoid_ub), avoid_ub), 0.0)
        return np.array([avoid_ub, reg_ub], dtype=float)

    ub = np.full(4, surface_ub, dtype=float)
    if flag == 1:
        ub[vulnerable_surface_index(cfg)] = 0.0
    return ub


def global_impact_alignment_weight(cfg: Any) -> float:
    """Return rho_impact for J_global's impact-aligned field-gradient term."""
    return max(_finite_or_default(getattr(cfg, "rho_impact", 0.0), 0.0), 0.0)


def proposed_slack_cost_weights(cfg: Any) -> np.ndarray:
    """Return the proposed OCP LINEAR_LS weights [rho_s, rho_c].

    rho_s penalizes the avoidance slack s^a.  When impact-normal regulation is
    enabled, rho_c penalizes the regulation slack s^r.
    """
    rho_s = max(_finite_or_default(getattr(cfg, "rho_s", 0.0), 0.0), 0.0)
    if proposed_active_slack_count(cfg) == 1:
        return np.array([rho_s], dtype=float)
    rho_c = max(_finite_or_default(getattr(cfg, "rho_c", 0.0), 0.0), 0.0)
    return np.array([rho_s, rho_c], dtype=float)

def surface_slack_cost_weights(cfg: Any, lexicographic_phase: int = 0) -> np.ndarray:
    """Return slack LS weights for the selected formulation.

    mpc_flag=4 always uses the phase-0 scalarized J_local + weighted-slack
    objective, regardless of ``lexicographic_phase``.  The phase argument is
    retained only for the mpc_flag=3 lexicographic surface-wise baseline.
    Surface-wise baselines 0..3 retain four surface slacks.
    """
    flag = mpc_formulation_flag(cfg)
    if flag == 4:
        active = proposed_active_slack_count(cfg)
        return proposed_slack_cost_weights(cfg)[:active].copy()

    q_base = float(getattr(cfg, "q_surface_slack", 1.0e8))
    q_nv = float(getattr(cfg, "q_nonvulnerable_slack", q_base))
    q_v = float(getattr(cfg, "q_vulnerable_slack", max(q_nv, q_base)))
    w = np.full(4, q_nv, dtype=float)
    v_idx = vulnerable_surface_index(cfg)
    if flag == 0:
        w[:] = q_base
    elif flag == 1:
        w[v_idx] = 0.0
    elif flag == 2:
        w[v_idx] = q_v
    elif flag == 3:
        if int(lexicographic_phase) == 1:
            w[:] = 0.0
            w[v_idx] = max(q_v, 1.0)
        elif int(lexicographic_phase) == 2:
            w[v_idx] = 0.0
    return np.maximum(np.where(np.isfinite(w), w, 0.0), 0.0)

def regulation_relax_beta(cfg: Any) -> float:
    """Deprecated compatibility alias; use :func:`collision_smoothing_lambda`."""
    return collision_smoothing_lambda(cfg)


def mitigation_alignment_alpha(cfg: Any) -> float:
    """Deprecated compatibility helper for older JSON/config files."""
    alpha_alias = getattr(cfg, "alpha", None)
    value = alpha_alias if alpha_alias is not None else getattr(cfg, "mitigation_alpha", 0.0)
    try:
        alpha = float(value)
    except Exception:
        alpha = 0.0
    return alpha if math.isfinite(alpha) else 0.0


def ego_overestimated_ellipse_axes(ego_L: float, ego_W: float) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return rectangle and overestimated-ellipse axes for ego geometry.

    ego_L/ego_W are full rectangle lengths.  The axis-aligned ellipse used to
    overestimate the rectangle is chosen to pass through the four rectangle
    corners.  Its full axes are sqrt(2)*ego_L and sqrt(2)*ego_W.

    Returns:
        rect_L, rect_W, ellipse_L, ellipse_W,
        ellipse_minor_axis, ellipse_major_axis,
        ellipse_minor_radius, ellipse_major_radius.
    """
    rect_L = max(abs(float(ego_L)), 1.0e-12)
    rect_W = max(abs(float(ego_W)), 1.0e-12)
    ellipse_L = math.sqrt(2.0) * rect_L
    ellipse_W = math.sqrt(2.0) * rect_W
    ellipse_minor = min(ellipse_L, ellipse_W)
    ellipse_major = max(ellipse_L, ellipse_W)
    ellipse_minor_radius = 0.5 * ellipse_minor
    ellipse_major_radius = 0.5 * ellipse_major
    return rect_L, rect_W, ellipse_L, ellipse_W, ellipse_minor, ellipse_major, ellipse_minor_radius, ellipse_major_radius


def normalize_ellipse_local_frame_config(cfg: MPCConfig) -> None:
    """Configure the exact MATLAB ego-centered unit-disk frame.

    cfg.ego_L/cfg.ego_W are the rectangle's full dimensions.  The
    corner-passing circumscribed ellipse has full axes sqrt(2)*L and
    sqrt(2)*W, hence
        z = diag(sqrt(2)/L, sqrt(2)/W) R(yaw)^T (p-q).
    The ego geometry is represented by the explicit -1 feature-score offset,
    so no legacy boundary/initial/geometric buffer is applied here.
    """
    L = max(abs(float(getattr(cfg, "ego_L", 0.6))), 1.0e-12)
    W = max(abs(float(getattr(cfg, "ego_W", 0.35))), 1.0e-12)
    sqrt2 = math.sqrt(2.0)

    cfg.ego_rect_L_m = L
    cfg.ego_rect_W_m = W
    cfg.ego_ellipse_L_axis_m = sqrt2 * L
    cfg.ego_ellipse_W_axis_m = sqrt2 * W
    cfg.ego_minor_axis_m = sqrt2 * min(L, W)
    cfg.ego_major_axis_m = sqrt2 * max(L, W)
    cfg.ego_minor_radius_m = min(L, W) / sqrt2
    cfg.ego_major_radius_m = max(L, W) / sqrt2
    cfg.local_frame_base_buffer_m = 0.0

    # S contains the circumscribed ellipse radii.  Thus S^{-1} maps the
    # ellipse exactly to the unit disk with gains sqrt(2)/L and sqrt(2)/W.
    cfg.ego_axis_scale = [
        L / sqrt2,
        W / sqrt2,
    ]
    cfg.use_ego_scaled_local_frame = True
    cfg.use_external_collision_linearization = True


def nonlinear_dynamics_enabled(cfg: MPCConfig) -> bool:
    """Return true when acados should keep the nonlinear RK4 shooting map."""
    return _bool_compat(getattr(cfg, "nonlinear_dyn_flag", True), True)


def force_delta_sqp_solver_names(cfg: MPCConfig) -> None:
    """Force generated acados artifacts to names consistent with OCP dimensions."""
    if mpc_formulation_flag(cfg) == 4:
        base = "switchable_mpc_proposed_phase0_ls_delta_dynamics"
        if not impact_normal_constraint_enabled(cfg):
            # The avoidance-only proposed OCP has nu=3 instead of nu=4.
            base += "_avoidance_only"
    else:
        base = "switchable_mpc_compact_delta_dynamics"
    if not nonlinear_dynamics_enabled(cfg):
        base += "_linearized_dyn_qp"
    cfg.acados_model_name = base
    cfg.acados_json_file = "acados_{}.json".format(base)
    cfg.acados_code_export_dir = "c_generated_code_{}".format(base)


def normalize_horizon_timing_config(cfg: MPCConfig) -> None:
    """Validate the fixed MPC horizon and sampling period from /setup."""
    try:
        horizon = int(getattr(cfg, "N", 10))
    except Exception:
        horizon = 10
    if horizon < 2:
        raise ValueError("N must be >= 2; got {!r}".format(horizon))
    cfg.N = horizon

    try:
        dt_init = float(getattr(cfg, "dt_init", 0.1))
    except Exception:
        dt_init = 0.1
    if not math.isfinite(dt_init) or dt_init <= 0.0:
        raise ValueError("dt_init must be finite and > 0; got {!r}".format(dt_init))
    cfg.dt_init = dt_init



def load_planner_config(
    raw_config: Dict[str, Any],
    planner_id: Optional[int] = None,
) -> Tuple[MPCConfig, MPCConfig, Dict[str, Any]]:
    """Build one flattened planner config from setter.py's /setup message.

    This utility performs no configuration-file I/O. The ROS wrapper must
    receive ``drone_msgs/Setup`` from the latched ``/setup`` topic, convert it
    with ``setup_msg_to_raw()``, and pass that dictionary as ``raw_config``.
    """
    if not isinstance(raw_config, dict) or not raw_config:
        raise RuntimeError(
            "load_planner_config requires non-empty configuration reconstructed "
            "from setter.py's /setup message"
        )
    raw: Dict[str, Any] = dict(raw_config)

    flat: Dict[str, Any] = {}
    for section_name in ("config", "ego", "guidance", "normal_cfg", "mpc"):
        section = raw.get(section_name, {})
        if isinstance(section, dict):
            flat.update(section)

    # Also accept an already-flattened in-memory setup dictionary. Nested objects are
    # filtered out when MPCConfig is constructed below.
    for key, value in raw.items():
        if key in MPCConfig.__dataclass_fields__:
            flat.setdefault(key, value)

    scenario = raw.get("config", {}) if isinstance(raw.get("config"), dict) else {}

    # The ego node uses shorter field names than MPCConfig.
    ego = raw.get("ego", {}) if isinstance(raw.get("ego"), dict) else {}
    if "ego_v" in ego and "ego_v0" not in flat:
        flat["ego_v0"] = ego["ego_v"]
    if "ego_yaw" in ego and "ego_yaw0" not in flat:
        flat["ego_yaw0"] = ego["ego_yaw"]

    if "goal_x" in scenario:
        flat["goal_x"] = scenario["goal_x"]
    elif "x_end" in scenario:
        flat["goal_x"] = scenario["x_end"]
    if "goal_y" in scenario:
        flat["goal_y"] = scenario["goal_y"]
    elif "y_end" in scenario:
        flat["goal_y"] = scenario["y_end"]
    if "goal_z" in scenario:
        flat["goal_z"] = scenario["goal_z"]
    elif "z_end" in scenario:
        flat["goal_z"] = scenario["z_end"]
    if "FOV" in scenario:
        flat["obs_sensing_R"] = scenario["FOV"]
    if "ego_x0" in scenario and "ego_x" not in flat:
        flat["ego_x"] = scenario["ego_x0"]
    if "ego_y0" in scenario and "ego_y" not in flat:
        flat["ego_y"] = scenario["ego_y0"]

    if "sector_id" in flat and "designated_surface_id" not in flat:
        flat["designated_surface_id"] = flat["sector_id"]
    if "designated_surface_id" in flat and "sector_id" not in flat:
        flat["sector_id"] = flat["designated_surface_id"]
    if "dt" in flat and "dt_init" not in flat:
        flat["dt_init"] = float(flat["dt"])
    if "obs_fov_angle_deg" in flat and "obs_fov_deg" not in flat:
        flat["obs_fov_deg"] = float(flat["obs_fov_angle_deg"])

    vertex_selector = raw.get("vertex_selector", {}) if isinstance(raw.get("vertex_selector"), dict) else {}
    if "N_n" not in flat and "pseudo_normal_num" in vertex_selector:
        flat["N_n"] = int(vertex_selector["pseudo_normal_num"])

    if planner_id is None:
        planner_id = int(flat.get("planner_id", 1))
    else:
        planner_id = int(planner_id)
    planner_name = "MPC_planner{}".format(planner_id)
    planners = raw.get("planners", {})
    if isinstance(planners, dict):
        planner_section = planners.get(planner_name)
        if isinstance(planner_section, dict):
            flat.update(planner_section)
    flat["planner_id"] = planner_id

    cfg = MPCConfig(**{k: v for k, v in flat.items() if k in MPCConfig.__dataclass_fields__})
    normalize_horizon_timing_config(cfg)
    if cfg.no_acc_flag:
        cfg.a_min = -0.0
        cfg.a_max = 0.0

    normalize_ellipse_local_frame_config(cfg)
    normalize_sector_config(cfg)
    effective_pseudo_normal_count(cfg)
    return cfg, cfg, raw





def ros_param_override(cfg: MPCConfig, *args: Any) -> None:
    """Deprecated compatibility no-op.

    Configuration is already loaded from setter.py's latched /setup message.
    Legacy nodes may still call this function; intentionally do nothing so no
    ROS private parameter can override the /setup-derived configuration.
    """
    return None


def signed_polygon_area(V: np.ndarray) -> float:
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    if V.shape[0] < 3:
        return 0.0
    x = V[:, 0]
    y = V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    if V.shape[0] >= 3 and signed_polygon_area(V) < 0.0:
        return np.flipud(V)
    return V


def circle_fallback_polygon(cx: float, cy: float, radius: float, n: int = 8) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    return np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))


def polygon_radius(V: np.ndarray, center: Optional[np.ndarray] = None) -> float:
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    if V.size == 0:
        return 0.0
    if center is None:
        center = np.mean(V, axis=0)
    return float(np.max(np.linalg.norm(V - np.asarray(center, dtype=float).reshape(1, 2), axis=1)))


def build_obstacle_from_polygon(
    V: np.ndarray,
    vel: Sequence[float] = (0.0, 0.0),
    obs_type: str = "dyn",
    center: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    V = ensure_ccw(np.asarray(V, dtype=float).reshape(-1, 2))
    c = np.asarray(center, dtype=float).reshape(2) if center is not None else np.mean(V, axis=0)
    return {
        "V": V,
        "center": c,
        "radius": polygon_radius(V, c),
        "vel": np.asarray(vel, dtype=float).reshape(2),
        "type": str(obs_type),
        "vertex_count": int(V.shape[0]),
    }


# -----------------------------------------------------------------------------
# Local vertex set and half-space parameterization
# -----------------------------------------------------------------------------


def obstacle_type_to_int(obs_type: Any, vel: Optional[Sequence[float]] = None) -> int:
    """Return 0 for static obstacles and 1 for dynamic obstacles."""
    text = str(obs_type).strip().lower()
    if text in ("1", "dyn", "dynamic", "moving", "true", "mobile"):
        return 1
    if text in ("0", "static", "fixed", "false"):
        return 0
    if vel is not None:
        try:
            vv = np.asarray(vel, dtype=float).reshape(-1)
            if vv.size >= 2 and float(np.linalg.norm(vv[:2])) > 1.0e-12:
                return 1
        except Exception:
            pass
    return 0


def select_vertex_set_by_fov_radius_with_types(
    obstacles: Sequence[Dict[str, Any]],
    x_ego: np.ndarray,
    sensing_radius: float,
    fov_deg: float,
    vertex_capacity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select obstacle vertices and preserve each selected row's obstacle type.

    Returns:
        vertex_set: K x 2 selected vertices.
        vertex_types: K integers, where 0=static and 1=dynamic.

    The configured fov_deg is interpreted as a half-angle, matching the older
    planner code that checked abs(relative_angle) <= radians(obs_fov_deg).
    """
    if not obstacles or vertex_capacity <= 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=np.int32)

    x_ego = np.asarray(x_ego, dtype=float).reshape(-1)
    ego_pos = x_ego[:2]
    ego_yaw = float(x_ego[3]) if x_ego.size >= 4 else 0.0
    radius = max(float(sensing_radius), 0.0)
    fov_half_rad = math.radians(max(float(fov_deg), 0.0))

    candidates: List[Tuple[float, float, int, int, np.ndarray, int]] = []
    for oi, obs in enumerate(obstacles):
        V = np.asarray(obs.get("V", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        if V.shape[0] == 0:
            continue
        c = np.asarray(obs.get("center", np.mean(V, axis=0)), dtype=float).reshape(2)
        r = float(obs.get("radius", polygon_radius(V, c)))
        d_center = float(np.linalg.norm(c - ego_pos))
        if max(0.0, d_center - r) > radius:
            continue

        obs_type_i = obstacle_type_to_int(obs.get("type", "static"), obs.get("vel", None))
        for vi, vtx in enumerate(V):
            rel = vtx - ego_pos
            d = float(np.linalg.norm(rel))
            if d > radius + r:
                continue
            rel_angle = wrap_to_pi(math.atan2(rel[1], rel[0]) - ego_yaw)
            if abs(rel_angle) > fov_half_rad:
                continue
            candidates.append((d, abs(rel_angle), oi, vi, vtx.copy(), obs_type_i))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    selected = candidates[: int(vertex_capacity)]
    if not selected:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=np.int32)
    vertices = np.asarray([item[4] for item in selected], dtype=float).reshape(-1, 2)
    types = np.asarray([item[5] for item in selected], dtype=np.int32).reshape(-1)
    return vertices, types


def select_vertex_set_by_fov_radius(
    obstacles: Sequence[Dict[str, Any]],
    x_ego: np.ndarray,
    sensing_radius: float,
    fov_deg: float,
    vertex_capacity: int,
) -> np.ndarray:
    """Backward-compatible vertex-only wrapper."""
    vertices, _ = select_vertex_set_by_fov_radius_with_types(
        obstacles,
        x_ego,
        sensing_radius,
        fov_deg,
        vertex_capacity,
    )
    return vertices


def use_ego_scaled_local_frame(cfg: MPCConfig) -> bool:
    return _bool_compat(getattr(cfg, "use_ego_scaled_local_frame", False), False)


def use_external_collision_linearization(cfg: MPCConfig) -> bool:
    return use_ego_scaled_local_frame(cfg) or _bool_compat(
        getattr(cfg, "use_external_collision_linearization", False), False
    )


def selected_feature_normals_to_outward(normals: Optional[np.ndarray], cfg: MPCConfig) -> np.ndarray:
    """Convert legacy selected-feature normals to outward support normals.

    The current selector publishes raw polygon vertices and does not construct
    collision-feature normals.  This helper is retained for legacy/non-external
    paths that still provide explicit normals.
    """
    if normals is None:
        return np.zeros((0, 2), dtype=float)
    N = np.asarray(normals, dtype=float).reshape(-1, 2).copy()
    if str(getattr(cfg, "normal_orientation", "inward")).strip().lower() == "inward":
        N *= -1.0
    nrm = np.linalg.norm(N, axis=1)
    good = nrm > 1.0e-12
    if np.any(good):
        N[good, :] = N[good, :] / nrm[good].reshape(-1, 1)
    if np.any(~good):
        N[~good, :] = np.array([1.0, 0.0], dtype=float)
    return N


def ego_scaled_local_transform_matrix(
    x_state: np.ndarray,
    cfg: MPCConfig,
) -> np.ndarray:
    x = np.asarray(x_state, dtype=float).reshape(-1)

    yaw = float(x[3]) if x.size >= 4 else 0.0
    c = math.cos(yaw)
    s = math.sin(yaw)

    R = np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)

    axis_scale = getattr(cfg, "ego_axis_scale", None)

    if axis_scale is None:
        sqrt2 = math.sqrt(2.0)
        axis_scale_arr = np.array([
            max(abs(float(cfg.ego_L)), 1.0e-12) / sqrt2,
            max(abs(float(cfg.ego_W)), 1.0e-12) / sqrt2,
        ], dtype=float)
    else:
        axis_scale_arr = np.asarray(axis_scale, dtype=float).reshape(-1)

        if axis_scale_arr.size < 2:
            axis_scale_arr = np.array([
                max(abs(float(cfg.ego_L)), 1.0e-12) / math.sqrt(2.0),
                max(abs(float(cfg.ego_W)), 1.0e-12) / math.sqrt(2.0),
            ], dtype=float)
        else:
            axis_scale_arr = axis_scale_arr[:2]

    axis_scale_arr = np.maximum(axis_scale_arr, 1.0e-12)

    S_inv = np.diag(1.0 / axis_scale_arr)

    return S_inv @ R.T


def transform_points_to_ego_scaled_local(P_world: np.ndarray, T_local: np.ndarray, ego_pos: np.ndarray) -> np.ndarray:
    P = np.asarray(P_world, dtype=float).reshape(-1, 2)
    ego = np.asarray(ego_pos, dtype=float).reshape(1, 2)
    return (P - ego) @ np.asarray(T_local, dtype=float).reshape(2, 2).T


def transform_points_from_ego_scaled_local(P_local: np.ndarray, T_local: np.ndarray, ego_pos: np.ndarray) -> np.ndarray:
    """Inverse of transform_points_to_ego_scaled_local, vectorized."""
    Z = np.asarray(P_local, dtype=float).reshape(-1, 2)
    T = np.asarray(T_local, dtype=float).reshape(2, 2)
    ego = np.asarray(ego_pos, dtype=float).reshape(1, 2)
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        T_inv = np.eye(2, dtype=float)
    return Z @ T_inv.T + ego


def transform_normals_to_ego_scaled_local(N_world_out: np.ndarray, T_local: np.ndarray) -> np.ndarray:
    """Transform support-normal covectors by T^{-T} for z=T(p-q)."""
    N = np.asarray(N_world_out, dtype=float).reshape(-1, 2)
    T = np.asarray(T_local, dtype=float).reshape(2, 2)
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        T_inv = np.eye(2, dtype=float)
    # Column form: n_local_raw=T^{-T}n_world.  Row-vector equivalent: N@T^{-1}.
    N_local = N @ T_inv
    nrm = np.linalg.norm(N_local, axis=1)
    good = nrm > 1.0e-12
    if np.any(good):
        N_local[good, :] /= nrm[good].reshape(-1, 1)
    if np.any(~good):
        N_local[~good, :] = np.array([1.0, 0.0], dtype=float)
    return N_local


def selected_vertices_normals_to_halfspaces(
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    x_state: np.ndarray,
    vertex_capacity: int,
    cfg: MPCConfig,
    disabled_c_value: Optional[float] = None,
) -> np.ndarray:
    """Build support-feature halfspace rows from selected vertices and normals.

    If use_ego_scaled_local_frame is true, points use
    z = S^{-1}R^T(p-q_ego), support normals use T^{-T}, and world-frame
    buffers are shifted before transformation.  The resulting local affine
    feature is then mapped back to an equivalent world row for diagnostics.
    """
    M = int(vertex_capacity)
    disabled = float(cfg.disabled_c_value if disabled_c_value is None else disabled_c_value)
    hs = np.zeros((M, 3), dtype=float)
    hs[:, 2] = disabled
    V = np.asarray(vertex_set if vertex_set is not None else np.zeros((0, 2)), dtype=float).reshape(-1, 2)
    count = min(M, V.shape[0])
    if count <= 0:
        return hs
    if vertex_normals is None or np.asarray(vertex_normals).size == 0:
        return vertex_set_to_halfspaces(
            V[:count], np.asarray(x_state, dtype=float).reshape(-1)[:2], M,
            buffer_m=0.0,
            disabled_c_value=disabled,
        )

    N_out_world = selected_feature_normals_to_outward(vertex_normals, cfg)[:count]
    V = V[:count]
    x_arr = np.asarray(x_state, dtype=float).reshape(-1)
    ego_pos = x_arr[:2]

    if use_ego_scaled_local_frame(cfg):
        T = ego_scaled_local_transform_matrix(x_arr, cfg)
        N_local = transform_normals_to_ego_scaled_local(N_out_world, T)
        P_safe_local = transform_points_to_ego_scaled_local(V, T, ego_pos)
        A_world = N_local @ T
        C_world = np.einsum("ij,j->i", A_world, ego_pos) + np.einsum("ij,ij->i", N_local, P_safe_local)
        hs[:count, 0:2] = A_world
        hs[:count, 2] = C_world
    else:
        hs[:count, 0:2] = N_out_world
        hs[:count, 2] = np.einsum("ij,ij->i", N_out_world, V)
    return hs






def _sector_lc_world_from_state(x_state: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    _, _, lc_body = sector_rays_body_from_cfg(cfg)
    yaw = float(np.asarray(x_state, dtype=float).reshape(-1)[3])
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    lc = R @ lc_body
    return lc / max(float(np.linalg.norm(lc)), 1.0e-12)


def _sector_lc_local_from_state(x_state: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    T = ego_scaled_local_transform_matrix(x_state, cfg)
    lc_world = _sector_lc_world_from_state(x_state, cfg)
    lc = T @ lc_world
    return lc / max(float(np.linalg.norm(lc)), 1.0e-12)


def _stage_state_matrix_from_augmented(X_seq: np.ndarray) -> np.ndarray:
    """Return H x 4 [px, py, v, yaw] from core or augmented state matrices."""
    X = np.asarray(X_seq, dtype=float)
    if X.ndim != 2:
        X = X.reshape(-1, 4)
    # H x 4 is the common internal representation used by the batch residual
    # and finite-difference calls.  Prefer it whenever the second dimension is
    # exactly 4, even if H happens to be 4 or 6.
    if X.shape[1] == 4:
        out = X[:, :4].copy()
    elif X.shape[1] in (5, 6, 7) and X.shape[0] not in (4, 5, 6, 7):
        out = X[:, :4].copy()
    elif X.shape[0] in (4, 5, 6, 7):
        out = X[:4, :].T.copy()
    elif X.shape[1] >= 4:
        out = X[:, :4].copy()
    elif X.shape[0] >= 4:
        out = X[:4, :].T.copy()
    else:
        out = X.reshape(-1, 4).copy()
    out[:, 3] = np.arctan2(np.sin(out[:, 3]), np.cos(out[:, 3]))
    return out


def _stage_scalar_sequence(values: Any, H: int, default_value: float) -> np.ndarray:
    arr = np.asarray(values if values is not None else default_value, dtype=float).reshape(-1)
    if arr.size == 0:
        arr = np.asarray([default_value], dtype=float)
    if arr.size == 1 and H > 1:
        arr = np.repeat(arr, H)
    elif arr.size < H:
        arr = np.concatenate((arr, np.repeat(arr[-1], H - arr.size)))
    return arr[:H].astype(float)


def _ego_axis_scale_array(cfg: MPCConfig) -> np.ndarray:
    """Return S diagonal entries: circumscribed-ellipse semi-axes [m]."""
    axis_scale = getattr(cfg, "ego_axis_scale", None)
    if axis_scale is None:
        sqrt2 = math.sqrt(2.0)
        arr = np.array([
            max(abs(float(cfg.ego_L)), 1.0e-12) / sqrt2,
            max(abs(float(cfg.ego_W)), 1.0e-12) / sqrt2,
        ], dtype=float)
    else:
        arr = np.asarray(axis_scale, dtype=float).reshape(-1)
        if arr.size < 2:
            sqrt2 = math.sqrt(2.0)
            arr = np.array([
                max(abs(float(cfg.ego_L)), 1.0e-12) / sqrt2,
                max(abs(float(cfg.ego_W)), 1.0e-12) / sqrt2,
            ], dtype=float)
        else:
            arr = arr[:2]
    return np.maximum(arr.astype(float), 1.0e-12)




def _prepare_collision_feature_tensor(values: Optional[np.ndarray], H: int) -> np.ndarray:
    """Return contiguous Kx2 or HxKx2 collision features for one C++ batch call.

    A 2-D Kx2 array is shared across the horizon.  A 3-D array is stage-wise;
    it is clipped/padded along the horizon only, never flattened across stages.
    The underlying memory remains contiguous, so the C++ accessor traverses the
    complete flattened H*K feature block without per-stage Python calls.
    """
    if values is None:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            arr = arr.reshape(-1, 2)
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError("collision feature array must be Kx2 or HxKx2")
    H = max(0, int(H))
    if H == 0:
        return np.ascontiguousarray(arr[:0, :, :], dtype=np.float64)
    if arr.shape[0] < H:
        if arr.shape[0] == 0:
            return np.zeros((H, 0, 2), dtype=np.float64)
        arr = np.concatenate((arr, np.repeat(arr[-1:, :, :], H - arr.shape[0], axis=0)), axis=0)
    elif arr.shape[0] > H:
        arr = arr[:H, :, :]
    return np.ascontiguousarray(arr, dtype=np.float64)


def batch_transform_points_normals_to_ego_scaled_local(
    points_world: np.ndarray,
    normals_world_out: np.ndarray,
    X_core_seq: np.ndarray,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply z=S^-1 R^-1(yaw)(p-q) to all horizon/feature rows at once.

    Returns:
        P_local: H x K x 2
        N_local: H x K x 2, normalized after the covector transform T^{-T}

    Points/directions transform with T=S^{-1}R^T, whereas support normals
    transform with T^{-T}.  Runtime execution is intentionally C++-only; the
    Python/NumPy transform fallback is disabled.
    """
    X = _stage_state_matrix_from_augmented(X_core_seq)
    H = int(X.shape[0])
    P = _prepare_collision_feature_tensor(points_world, H)
    N = _prepare_collision_feature_tensor(normals_world_out, H)
    Kp = int(P.shape[-2] if P.ndim == 3 else P.shape[0])
    Kn = int(N.shape[-2] if N.ndim == 3 else N.shape[0])
    K = min(Kp, Kn)
    if H <= 0 or K <= 0:
        return np.zeros((H, 0, 2), dtype=float), np.zeros((H, 0, 2), dtype=float)
    P = P[:, :K, :] if P.ndim == 3 else P[:K, :]
    N = N[:, :K, :] if N.ndim == 3 else N[:K, :]

    # API-18 runtime policy: shared or stage-wise horizon transforms stay in one C++ call.
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "transform_points_normals_seq"):
        raise RuntimeError("collision_lin_cpp.transform_points_normals_seq is required; Python fallback is disabled")
    try:
        scale_cpp = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
        transformed = collision_lin_cpp.transform_points_normals_seq(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(N, dtype=np.float64),
            scale_cpp,
        )
        P_local_cpp = np.asarray(transformed[0], dtype=float).reshape(H, K, 2)
        N_local_cpp = np.asarray(transformed[1], dtype=float).reshape(H, K, 2)
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "transform_points_normals_seq: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.transform_points_normals_seq failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc
    if not np.all(np.isfinite(P_local_cpp)) or not np.all(np.isfinite(N_local_cpp)):
        msg = "transform_points_normals_seq returned non-finite values"
        globals()["_collision_lin_cpp_runtime_error"] = msg
        raise RuntimeError(msg + "; Python fallback is disabled")
    return P_local_cpp, N_local_cpp

def _sector_lc_world_sequence(
    X_core_seq: np.ndarray,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    X = _stage_state_matrix_from_augmented(X_core_seq)
    H = int(X.shape[0])
    if sector_rays_body is None:
        _, _, lc_body = sector_rays_body_from_cfg(cfg)
    else:
        lc_body = np.asarray(sector_rays_body, dtype=float).reshape(3, 2)[2]
    lc_body = np.asarray(lc_body, dtype=float).reshape(2)
    yaw = X[:, 3]
    c = np.cos(yaw)
    s = np.sin(yaw)
    lc = np.empty((H, 2), dtype=float)
    lc[:, 0] = c * lc_body[0] - s * lc_body[1]
    lc[:, 1] = s * lc_body[0] + c * lc_body[1]
    nrm = np.linalg.norm(lc, axis=1)
    good = nrm > 1.0e-12
    if np.any(good):
        lc[good, :] = lc[good, :] / nrm[good].reshape(-1, 1)
    if np.any(~good):
        lc[~good, :] = np.array([1.0, 0.0], dtype=float)
    return lc


def _sector_lc_local_sequence(
    X_core_seq: np.ndarray,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    X = _stage_state_matrix_from_augmented(X_core_seq)
    H = int(X.shape[0])
    lc_world = _sector_lc_world_sequence(X, cfg, sector_rays_body=sector_rays_body)
    scale = _ego_axis_scale_array(cfg)
    inv_sx = 1.0 / float(scale[0])
    inv_sy = 1.0 / float(scale[1])
    yaw = X[:, 3]
    c = np.cos(yaw)
    s = np.sin(yaw)
    lc = np.empty((H, 2), dtype=float)
    lc[:, 0] = inv_sx * (c * lc_world[:, 0] + s * lc_world[:, 1])
    lc[:, 1] = inv_sy * (-s * lc_world[:, 0] + c * lc_world[:, 1])
    nrm = np.linalg.norm(lc, axis=1)
    good = nrm > 1.0e-12
    if np.any(good):
        lc[good, :] = lc[good, :] / nrm[good].reshape(-1, 1)
    if np.any(~good):
        lc[~good, :] = np.array([1.0, 0.0], dtype=float)
    return lc




# -----------------------------------------------------------------------------
# Multi-field collision evaluators for field_flag = 0..3.
#
# Every field is defined from the same perceived/augmented obstacle geometry in
# the ego body-fixed normalized frame, where the ego circumscribed ellipse is
# mapped to the unit disk.  field_flag only changes the signed-distance / smooth
# proximity construction applied to that common body-fixed geometry.
# field_flag=4 continues to use the optimized C++ proposed-field kernel.
# -----------------------------------------------------------------------------


_PROPOSED_FEATURE_CACHE = OrderedDict()
_PROPOSED_FEATURE_CACHE_MAX = 8


def clear_proposed_feature_cache() -> None:
    """Drop cached mpc_flag=4 edge/pseudo-vertex feature tensors."""
    _PROPOSED_FEATURE_CACHE.clear()


def _prepare_proposed_features_cpp(
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    X_core_seq: np.ndarray,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool, int]:
    """Build proposed-field features with pseudo normals uniform in q-frame.

    For every horizon stage, raw polygon vertices are interpreted in the ego
    body-fixed normalized frame

        q = S^{-1} R(yaw)^T (P - p).

    Edge normals are constructed from the transformed polygon geometry and the
    pseudo-vertex normals are then sampled *uniformly in angle in that q-frame*.
    The C++ preprocessor converts those local directions back to equivalent
    world-frame covectors so the existing CPU/CUDA collision kernels recover the
    same q-frame directions with their normal transform.

    ``vertex_set`` may be Kx2, HxKx2, or 1xKx2.  A single stage is preserved as
    a broadcastable geometry slice instead of being expanded unnecessarily.
    """
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "build_polygon_features_seq"):
        raise RuntimeError(
            "collision_lin_cpp.build_polygon_features_seq is required by API-33; rebuild the C++ extension"
        )

    X_feature = _stage_state_matrix_from_augmented(X_core_seq)
    H = int(X_feature.shape[0])
    if H <= 0:
        return (
            np.zeros((0, 0, 2), dtype=np.float64),
            np.zeros((0, 0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            vertex_types is not None,
            True,
            0,
        )

    raw_arr = np.asarray(vertex_set, dtype=np.float64)
    if raw_arr.ndim == 3 and raw_arr.shape[-1] == 2 and raw_arr.shape[0] == 1:
        # C++ FeatureAccessor intentionally broadcasts H==1 geometry.
        raw_vertices = np.ascontiguousarray(raw_arr, dtype=np.float64)
    else:
        raw_vertices = _prepare_collision_feature_tensor(vertex_set, H)

    has_vertex_types = vertex_types is not None
    raw_types = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = feature_obs_indices is not None
    raw_groups = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )
    axis_scale = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)

    # Feature directions depend on yaw and the anisotropic axis scale, but not on
    # ego translation.  Hash only the state information that can change the
    # body-fixed angular subdivision.
    yaw_key = np.ascontiguousarray(
        np.arctan2(np.sin(X_feature[:, 3]), np.cos(X_feature[:, 3])),
        dtype=np.float64,
    )

    hkey = hashlib.blake2b(digest_size=16)
    for arr in (raw_vertices, raw_types, raw_groups, yaw_key, axis_scale):
        carr = np.ascontiguousarray(arr)
        hkey.update(str(carr.shape).encode("ascii"))
        hkey.update(str(carr.dtype).encode("ascii"))
        hkey.update(carr.tobytes(order="C"))
    hkey.update(repr((
        int(H), int(getattr(cfg, "vertex_capacity", 0)),
        effective_pseudo_normal_count(cfg),
        str(getattr(cfg, "normal_orientation", "inward")),
        bool(has_vertex_types), bool(has_feature_obs_indices),
    )).encode("ascii"))
    cache_key = hkey.digest()
    cached = _PROPOSED_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        _PROPOSED_FEATURE_CACHE.move_to_end(cache_key)
        return cached

    points, normals, out_types, out_groups = collision_lin_cpp.build_polygon_features_seq(
        np.ascontiguousarray(X_feature, dtype=np.float64),
        raw_vertices, raw_types, raw_groups, axis_scale,
        bool(has_vertex_types), bool(has_feature_obs_indices),
        int(getattr(cfg, "vertex_capacity", raw_vertices.shape[-2] if raw_vertices.ndim >= 2 else 0)),
        effective_pseudo_normal_count(cfg),
        str(getattr(cfg, "normal_orientation", "inward")),
        int(H),
    )
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    out_types = np.ascontiguousarray(out_types, dtype=np.int32).reshape(-1)
    out_groups = np.ascontiguousarray(out_groups, dtype=np.int32).reshape(-1)
    if points.ndim != 3 or normals.shape != points.shape or points.shape[0] != int(H) or points.shape[2] != 2:
        raise RuntimeError(
            "invalid proposed feature tensor from C++: points={}, normals={}, H={}".format(
                points.shape, normals.shape, H
            )
        )
    F = int(points.shape[1])
    if out_types.size != F or out_groups.size != F:
        raise RuntimeError(
            "invalid proposed feature metadata from C++: F={}, types={}, groups={}".format(
                F, out_types.size, out_groups.size
            )
        )

    result = (points, normals, out_types, out_groups, bool(has_vertex_types), True, F)
    _PROPOSED_FEATURE_CACHE[cache_key] = result
    _PROPOSED_FEATURE_CACHE.move_to_end(cache_key)
    while len(_PROPOSED_FEATURE_CACHE) > _PROPOSED_FEATURE_CACHE_MAX:
        _PROPOSED_FEATURE_CACHE.popitem(last=False)
    return result


def _field_stage_values(values: Optional[np.ndarray], h: int, H: int) -> np.ndarray:
    arr = _prepare_collision_feature_tensor(values, H)
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(arr[min(max(int(h), 0), arr.shape[0] - 1)], dtype=float).reshape(-1, 2)
    return np.asarray(arr, dtype=float).reshape(-1, 2)


def _finite_unique_points(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float).reshape(-1, 2)
    if V.size == 0:
        return np.zeros((0, 2), dtype=float)
    V = V[np.all(np.isfinite(V), axis=1)]
    if V.shape[0] <= 1:
        return V.copy()
    key = np.round(V, 12)
    _, idx = np.unique(key, axis=0, return_index=True)
    return V[np.sort(idx)]


def _convex_hull_points(V: np.ndarray) -> np.ndarray:
    """Monotone-chain hull, returned CCW without repeating the first point."""
    P = _finite_unique_points(V)
    if P.shape[0] <= 2:
        return P
    order = np.lexsort((P[:, 1], P[:, 0]))
    pts = P[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-12:
            lower.pop()
        lower.append(p.copy())
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-12:
            upper.pop()
        upper.append(p.copy())
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float).reshape(-1, 2)
    return ensure_ccw(hull)


def _group_world_polygons(
    points: np.ndarray,
    feature_obs_indices: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    vertex_capacity: int,
    convexify: bool = True,
) -> List[Dict[str, Any]]:
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    K = min(max(int(vertex_capacity), 0), P.shape[0])
    P = P[:K]
    if K <= 0:
        return []

    if feature_obs_indices is not None and np.asarray(feature_obs_indices).size >= K:
        obs_ids = np.asarray(feature_obs_indices, dtype=np.int32).reshape(-1)[:K]
    else:
        # Polygon methods need obstacle grouping.  If an older caller does not
        # provide IDs, preserve compatibility by treating the available set as
        # one obstacle rather than changing the OCP or failing the planner.
        obs_ids = np.zeros(K, dtype=np.int32)

    if vertex_types is not None and np.asarray(vertex_types).size >= K:
        types = np.asarray(vertex_types, dtype=np.int32).reshape(-1)[:K]
    else:
        types = np.zeros(K, dtype=np.int32)

    groups: List[Dict[str, Any]] = []
    for oid in np.unique(obs_ids):
        if int(oid) < 0:
            continue
        idx = np.where(obs_ids == oid)[0]
        # Circle over-estimation (field 0) must use the ordered boundary of the
        # complete augmented obstacle.  Other polygon fields operate on convex
        # groups, so they retain the previous convex-hull normalization.
        V = _convex_hull_points(P[idx]) if bool(convexify) else _finite_unique_points(P[idx])
        if V.shape[0] == 0:
            continue
        groups.append({
            "id": int(oid),
            "V": V,
            "dynamic": bool(np.any(types[idx] == 1)),
        })
    return groups


def _surface_arc_theta_interval(cfg: MPCConfig, surface_id: int) -> Tuple[float, float]:
    """Return the exact ellipse-parameter interval for one ego surface.

    Surface ids follow sector_id: 1=left, 2=rear, 3=right, 4=front.  The
    boundaries are the intersections of the two MATLAB-compatible corner rays
    with the circumscribed ellipse.  For an ellipse point

        e(theta) = [a cos(theta), b sin(theta)],

    a body-frame ray r intersects at an ellipse parameter satisfying
    theta = atan2(r_y / b, r_x / a).  Using atan2(r_y, r_x) directly is only
    correct for a circle and was the source of the previous sampled-boundary
    approximation error.
    """
    sid = normalized_sector_id(surface_id)
    la, lb, _ = sector_rays_body_from_id(float(cfg.ego_L), float(cfg.ego_W), sid)
    scale = _ego_axis_scale_array(cfg)
    a = float(scale[0]); b = float(scale[1])
    theta_a = math.atan2(float(la[1]) / b, float(la[0]) / a)
    theta_b = math.atan2(float(lb[1]) / b, float(lb[0]) / a)
    while theta_b <= theta_a:
        theta_b += 2.0 * math.pi
    return float(theta_a), float(theta_b)


def _ellipse_arc_point(theta: float, a: float, b: float) -> np.ndarray:
    return np.array([a * math.cos(float(theta)), b * math.sin(float(theta))], dtype=float)


def _angle_lifts_in_interval(theta: float, theta_lo: float, theta_hi: float, tol: float = 1.0e-10) -> List[float]:
    """Return all 2*pi-equivalent copies of theta in an unwrapped interval."""
    two_pi = 2.0 * math.pi
    base = float(theta)
    k0 = int(math.floor((float(theta_lo) - base) / two_pi))
    out: List[float] = []
    for k in range(k0 - 1, k0 + 4):
        val = base + two_pi * float(k)
        if val >= float(theta_lo) - tol and val <= float(theta_hi) + tol:
            out.append(min(max(val, float(theta_lo)), float(theta_hi)))
    return out


def _point_segment_distance(point: np.ndarray, a_pt: np.ndarray, b_pt: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(2)
    a0 = np.asarray(a_pt, dtype=float).reshape(2)
    b0 = np.asarray(b_pt, dtype=float).reshape(2)
    d = b0 - a0
    denom = float(np.dot(d, d))
    if denom <= 1.0e-24:
        return float(np.linalg.norm(p - a0))
    t = float(np.dot(p - a0, d) / denom)
    t = min(max(t, 0.0), 1.0)
    return float(np.linalg.norm(p - (a0 + t * d)))


def _point_inside_convex_polygon(point: np.ndarray, polygon: np.ndarray, tol: float = 1.0e-10) -> bool:
    p = np.asarray(point, dtype=float).reshape(2)
    V = ensure_ccw(np.asarray(polygon, dtype=float).reshape(-1, 2))
    if V.shape[0] < 3:
        return False
    E = np.roll(V, -1, axis=0) - V
    P = p.reshape(1, 2) - V
    cross = E[:, 0] * P[:, 1] - E[:, 1] * P[:, 0]
    return bool(np.all(cross >= -float(tol)))


def _ellipse_arc_segment_intersects(
    a_pt: np.ndarray,
    b_pt: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
    tol: float = 1.0e-10,
) -> bool:
    """Analytically test whether a line segment intersects an ellipse arc."""
    A = np.asarray(a_pt, dtype=float).reshape(2)
    B = np.asarray(b_pt, dtype=float).reshape(2)
    D = B - A
    aa = max(float(ellipse_a), 1.0e-12)
    bb = max(float(ellipse_b), 1.0e-12)
    q2 = (D[0] * D[0]) / (aa * aa) + (D[1] * D[1]) / (bb * bb)
    q1 = 2.0 * (A[0] * D[0] / (aa * aa) + A[1] * D[1] / (bb * bb))
    q0 = (A[0] * A[0]) / (aa * aa) + (A[1] * A[1]) / (bb * bb) - 1.0

    roots: List[float] = []
    if abs(q2) <= 1.0e-24:
        if abs(q1) <= 1.0e-24:
            if abs(q0) <= tol:
                roots = [0.0]
        else:
            roots = [-q0 / q1]
    else:
        disc = q1 * q1 - 4.0 * q2 * q0
        if disc >= -tol:
            disc = max(disc, 0.0)
            sq = math.sqrt(disc)
            roots = [(-q1 - sq) / (2.0 * q2), (-q1 + sq) / (2.0 * q2)]

    for t in roots:
        if t < -tol or t > 1.0 + tol:
            continue
        tc = min(max(float(t), 0.0), 1.0)
        P = A + tc * D
        theta = math.atan2(float(P[1]) / bb, float(P[0]) / aa)
        if _angle_lifts_in_interval(theta, theta_lo, theta_hi, tol=tol):
            return True
    return False


def _ellipse_arc_closest_to_point_exact(
    point: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
) -> Tuple[float, np.ndarray]:
    """Exact continuous closest point on an ellipse arc to a point."""
    P = np.asarray(point, dtype=float).reshape(2)
    a = max(float(ellipse_a), 1.0e-12)
    b = max(float(ellipse_b), 1.0e-12)
    px = float(P[0]); py = float(P[1])
    D = b * b - a * a

    candidates: List[float] = [float(theta_lo), float(theta_hi)]
    # Explicit cardinal candidates make the degenerate quartic cases robust,
    # including tan(theta/2)=infinity at theta=pi.
    for base in (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi):
        candidates.extend(_angle_lifts_in_interval(base, theta_lo, theta_hi))

    # d/dtheta ||e(theta)-p||^2 = 0 under t=tan(theta/2).
    coeff = np.array([
        b * py,
        2.0 * (a * px - D),
        0.0,
        2.0 * (D + a * px),
        -b * py,
    ], dtype=float)
    scale_c = max(float(np.max(np.abs(coeff))), 1.0)
    nz = np.where(np.abs(coeff) > 1.0e-14 * scale_c)[0]
    if nz.size:
        poly = coeff[int(nz[0]):]
        if poly.size >= 2:
            for root in np.roots(poly):
                if abs(float(np.imag(root))) <= 1.0e-9 * (1.0 + abs(float(np.real(root)))):
                    theta = 2.0 * math.atan(float(np.real(root)))
                    candidates.extend(_angle_lifts_in_interval(theta, theta_lo, theta_hi))

    best = float("inf")
    best_e = _ellipse_arc_point(theta_lo, a, b)
    for theta in candidates:
        E = _ellipse_arc_point(theta, a, b)
        d = float(np.linalg.norm(E - P))
        if d < best:
            best = d
            best_e = E
    return float(best), np.asarray(best_e, dtype=float).reshape(2)


def _ellipse_arc_distance_to_point_exact(
    point: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
) -> float:
    """Exact continuous minimum distance from an ellipse arc to a point."""
    return _ellipse_arc_closest_to_point_exact(
        point, ellipse_a, ellipse_b, theta_lo, theta_hi,
    )[0]

def _ellipse_arc_distance_to_segment_exact(
    a_pt: np.ndarray,
    b_pt: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
) -> float:
    """Exact continuous minimum distance from an ellipse arc to a segment.

    KKT candidates consist of arc endpoints, segment endpoints, and interior
    line-projection points where the ellipse tangent is parallel to the segment.
    Segment/ellipse intersections are solved analytically and return zero.
    """
    A = np.asarray(a_pt, dtype=float).reshape(2)
    B = np.asarray(b_pt, dtype=float).reshape(2)
    a = max(float(ellipse_a), 1.0e-12)
    b = max(float(ellipse_b), 1.0e-12)

    if _ellipse_arc_segment_intersects(A, B, a, b, theta_lo, theta_hi):
        return 0.0

    best = min(
        _ellipse_arc_distance_to_point_exact(A, a, b, theta_lo, theta_hi),
        _ellipse_arc_distance_to_point_exact(B, a, b, theta_lo, theta_hi),
        _point_segment_distance(_ellipse_arc_point(theta_lo, a, b), A, B),
        _point_segment_distance(_ellipse_arc_point(theta_hi, a, b), A, B),
    )

    D = B - A
    len2 = float(np.dot(D, D))
    if len2 <= 1.0e-24:
        return float(best)
    length = math.sqrt(len2)
    # Either of the two unit normals gives the same stationary theta pair.
    n = np.array([-D[1] / length, D[0] / length], dtype=float)
    theta0 = math.atan2(b * float(n[1]), a * float(n[0]))
    for base in (theta0, theta0 + math.pi):
        for theta in _angle_lifts_in_interval(base, theta_lo, theta_hi):
            E = _ellipse_arc_point(theta, a, b)
            t = float(np.dot(E - A, D) / len2)
            if t >= -1.0e-10 and t <= 1.0 + 1.0e-10:
                tc = min(max(t, 0.0), 1.0)
                best = min(best, float(np.linalg.norm(E - (A + tc * D))))
    return float(best)


def _ellipse_arc_to_convex_polygon_closest_exact(
    polygon_body: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Exact closest pair between one continuous ellipse arc and a convex polygon.

    The candidate set is complete for the constrained distance problem: ellipse
    arc endpoints, polygon vertices (quartic stationary equation), interior
    polygon-edge projections whose line normal is an ellipse normal, and exact
    segment/ellipse intersections.  No angular or spatial sampling is used.
    """
    V = ensure_ccw(_convex_hull_points(np.asarray(polygon_body, dtype=float).reshape(-1, 2)))
    n_v = int(V.shape[0])
    if n_v <= 0:
        z = np.zeros(2, dtype=float)
        return float("inf"), z.copy(), z.copy()
    if n_v == 1:
        d, E = _ellipse_arc_closest_to_point_exact(V[0], ellipse_a, ellipse_b, theta_lo, theta_hi)
        return float(d), E, V[0].copy()

    a = max(float(ellipse_a), 1.0e-12)
    b = max(float(ellipse_b), 1.0e-12)
    E0 = _ellipse_arc_point(theta_lo, a, b)
    E1 = _ellipse_arc_point(theta_hi, a, b)
    if n_v >= 3 and _point_inside_convex_polygon(E0, V):
        return 0.0, E0.copy(), E0.copy()
    if n_v >= 3 and _point_inside_convex_polygon(E1, V):
        return 0.0, E1.copy(), E1.copy()

    edge_count = n_v if n_v >= 3 else 1

    # Exact edge/ellipse intersections imply zero clearance.
    for i in range(edge_count):
        j = (i + 1) % n_v if n_v >= 3 else 1
        A = V[i]; B = V[j]; Dseg = B - A
        aa = a; bb = b
        q2 = (Dseg[0] * Dseg[0]) / (aa * aa) + (Dseg[1] * Dseg[1]) / (bb * bb)
        q1 = 2.0 * (A[0] * Dseg[0] / (aa * aa) + A[1] * Dseg[1] / (bb * bb))
        q0 = (A[0] * A[0]) / (aa * aa) + (A[1] * A[1]) / (bb * bb) - 1.0
        roots: List[float] = []
        if abs(q2) <= 1.0e-24:
            if abs(q1) > 1.0e-24:
                roots = [-q0 / q1]
            elif abs(q0) <= 1.0e-10:
                roots = [0.0]
        else:
            disc = q1 * q1 - 4.0 * q2 * q0
            if disc >= -1.0e-10:
                sq = math.sqrt(max(disc, 0.0))
                roots = [(-q1 - sq) / (2.0 * q2), (-q1 + sq) / (2.0 * q2)]
        for t in roots:
            if t < -1.0e-10 or t > 1.0 + 1.0e-10:
                continue
            tc = min(max(float(t), 0.0), 1.0)
            P = A + tc * Dseg
            theta = math.atan2(float(P[1]) / bb, float(P[0]) / aa)
            if _angle_lifts_in_interval(theta, theta_lo, theta_hi):
                return 0.0, P.copy(), P.copy()

    best = float("inf")
    best_e = E0.copy()
    best_q = V[0].copy()

    # Arc endpoints against every polygon segment.
    for E in (E0, E1):
        for i in range(edge_count):
            j = (i + 1) % n_v if n_v >= 3 else 1
            A = V[i]; B = V[j]; Dseg = B - A
            den = float(np.dot(Dseg, Dseg))
            if den <= 1.0e-24:
                Q = A.copy()
            else:
                t = min(max(float(np.dot(E - A, Dseg) / den), 0.0), 1.0)
                Q = A + t * Dseg
            d = float(np.linalg.norm(E - Q))
            if d < best:
                best = d; best_e = E.copy(); best_q = Q.copy()

    # Polygon vertices against the continuous ellipse arc.
    for i in range(n_v):
        d, E = _ellipse_arc_closest_to_point_exact(V[i], a, b, theta_lo, theta_hi)
        if d < best:
            best = float(d); best_e = E.copy(); best_q = V[i].copy()

    # Interior edge-projection KKT candidates.  At an interior closest pair,
    # the ellipse normal is perpendicular to the polygon edge.
    for i in range(edge_count):
        j = (i + 1) % n_v if n_v >= 3 else 1
        A = V[i]; B = V[j]; Dseg = B - A
        len2 = float(np.dot(Dseg, Dseg))
        if len2 <= 1.0e-24:
            continue
        length = math.sqrt(len2)
        n = np.array([-Dseg[1] / length, Dseg[0] / length], dtype=float)
        theta0 = math.atan2(b * float(n[1]), a * float(n[0]))
        for base in (theta0, theta0 + math.pi):
            for theta in _angle_lifts_in_interval(base, theta_lo, theta_hi):
                E = _ellipse_arc_point(theta, a, b)
                t = float(np.dot(E - A, Dseg) / len2)
                if t < -1.0e-10 or t > 1.0 + 1.0e-10:
                    continue
                tc = min(max(t, 0.0), 1.0)
                Q = A + tc * Dseg
                d = float(np.linalg.norm(E - Q))
                if d < best:
                    best = d; best_e = E.copy(); best_q = Q.copy()

    return float(best), best_e, best_q


def _ellipse_arc_to_convex_polygon_distance_exact(
    polygon_body: np.ndarray,
    ellipse_a: float,
    ellipse_b: float,
    theta_lo: float,
    theta_hi: float,
) -> float:
    return _ellipse_arc_to_convex_polygon_closest_exact(
        polygon_body, ellipse_a, ellipse_b, theta_lo, theta_hi,
    )[0]

def _surface_distances_and_gradients_single(
    x_state: np.ndarray,
    points: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_step: float,
    dt_val: float,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact four-surface distances and their local state/time derivatives.

    Returns
        d_surf: [4]
        dd_dx:  [4,4] for [px,py,v,yaw]
        dd_ddt: [4]

    The distance itself is exact for the continuous ellipse arcs.  At a smooth
    closest pair the derivatives follow from the envelope theorem, so the SQP
    linearization does not need repeated finite-difference geometry evaluations.
    At zero clearance (where closed-set distance is nonsmooth) a zero geometric
    derivative is used.
    """
    pts = np.asarray(points if points is not None else np.zeros((0, 2)), dtype=float).reshape(-1, 2)
    groups = _group_world_polygons(
        pts, feature_obs_indices, vertex_types,
        vertex_capacity=min(int(getattr(cfg, "vertex_capacity", pts.shape[0])), pts.shape[0]),
        convexify=True,
    )
    far = max(float(getattr(cfg, "h_upper_bound", 1.0e15)), 1.0e6)
    if not groups:
        return np.full(4, far, dtype=float), np.zeros((4, 4), dtype=float), np.zeros(4, dtype=float)

    x = np.asarray(x_state, dtype=float).reshape(-1)
    c = math.cos(float(x[3])); s = math.sin(float(x[3]))
    R = np.array([[c, -s], [s, c]], dtype=float)  # column-vector body -> world
    pos = x[:2].reshape(1, 2)
    scale = _ego_axis_scale_array(cfg)
    ellipse_a = float(scale[0]); ellipse_b = float(scale[1])
    intervals = [_surface_arc_theta_interval(cfg, sid) for sid in (1, 2, 3, 4)]

    use_margin = _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False)
    uncertainty = _horizon_uncertainty_margin_world(h_step, dt_val, cfg)
    margin_rate = 0.0
    if use_margin and float(dt_val) >= 0.0:
        sigma_iso_rate = math.hypot(
            max(float(getattr(cfg, "sigma_vx", 0.0)), 0.0),
            max(float(getattr(cfg, "sigma_vy", 0.0)), 0.0),
        ) * max(float(h_step), 0.0)
        p_col = min(max(float(getattr(cfg, "probability_collision", 0.8)), 0.0), 1.0 - 1.0e-12)
        margin_rate = sigma_iso_rate * math.sqrt(max(-2.0 * math.log(1.0 - p_col), 0.0))

    out = np.full(4, float("inf"), dtype=float)
    dd_dx = np.zeros((4, 4), dtype=float)
    dd_ddt = np.zeros(4, dtype=float)
    J2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)

    for g in groups:
        V_world = np.asarray(g["V"], dtype=float).reshape(-1, 2)
        # Row-vector inverse rigid transform: p_body = (p_world-pos) R.
        V_body = (V_world - pos) @ R
        dynamic = bool(g.get("dynamic", False))
        margin = uncertainty if dynamic else 0.0
        margin_dt = margin_rate if dynamic else 0.0

        for j, interval in enumerate(intervals):
            d_geom, E_body, Q_body = _ellipse_arc_to_convex_polygon_closest_exact(
                V_body, ellipse_a, ellipse_b, interval[0], interval[1],
            )
            d_value = float(d_geom) - margin
            if d_value >= out[j]:
                continue

            out[j] = d_value
            dd_dx[j, :] = 0.0
            dd_ddt[j] = -margin_dt
            if float(d_geom) > 1.0e-10:
                u_body = (np.asarray(E_body, dtype=float) - np.asarray(Q_body, dtype=float)) / float(d_geom)
                # u_world = R u_body for column vectors.
                u_world = R @ u_body.reshape(2)
                dd_dx[j, 0] = float(u_world[0])
                dd_dx[j, 1] = float(u_world[1])
                dd_dx[j, 2] = 0.0
                # d/dyaw R*E = R*J*E; rotation invariance gives u_body^T J E.
                dd_dx[j, 3] = float(np.dot(u_body, J2 @ np.asarray(E_body, dtype=float).reshape(2)))

    out = np.where(np.isfinite(out), out, far)
    dd_dx = np.where(np.isfinite(dd_dx), dd_dx, 0.0)
    dd_ddt = np.where(np.isfinite(dd_ddt), dd_ddt, 0.0)
    return out.astype(float), dd_dx.astype(float), dd_ddt.astype(float)


def _surface_distances_single(
    x_state: np.ndarray,
    points: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_step: float,
    dt_val: float,
    cfg: MPCConfig,
) -> np.ndarray:
    """Return exact continuous [d_left,d_rear,d_right,d_front] in meters."""
    return _surface_distances_and_gradients_single(
        x_state, points, vertex_types, feature_obs_indices, h_step, dt_val, cfg,
    )[0]


def build_surface_distance_linearization_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
) -> np.ndarray:
    """Return H x 28 exact-surface affine residual blocks via C++ API-25.

    g_j(x) = d_safe - d_surf,j(x) <= s_j, j=1..4.

    The exact continuous ellipse-arc/polygon closest-pair calculation and its
    envelope-theorem Jacobian execute as one horizon-batched C++ call.
    """
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "build_surface_distance_lin_hk"):
        raise RuntimeError("collision_lin_cpp.build_surface_distance_lin_hk is required; Python fallback is disabled")

    X_arr = np.ascontiguousarray(X_bar, dtype=np.float64)
    H = int(_stage_state_matrix_from_augmented(X_arr).shape[0])
    if H <= 0:
        return np.zeros((0, 28), dtype=float)
    vertex_set_arr = np.ascontiguousarray(_prepare_collision_feature_tensor(vertex_set, H), dtype=np.float64)
    K_used = min(int(getattr(cfg, "vertex_capacity", vertex_set_arr.shape[-2])), int(vertex_set_arr.shape[-2]))
    raw_types = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_vertex_types = bool(raw_types.size >= K_used)
    vertex_types_arr = raw_types if has_vertex_types else np.zeros(0, dtype=np.int32)
    raw_obs = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = bool(raw_obs.size >= K_used)
    feature_obs_indices_arr = raw_obs if has_feature_obs_indices else np.zeros(0, dtype=np.int32)
    h_arr = np.ascontiguousarray(_stage_scalar_sequence(h_steps, H, 1.0), dtype=np.float64)
    dt_arr = np.ascontiguousarray(_stage_scalar_sequence(dt_vals, H, float(cfg.dt_init)), dtype=np.float64)
    axis_scale_arr = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
    try:
        out = collision_lin_cpp.build_surface_distance_lin_hk(
            X_arr, vertex_set_arr, vertex_types_arr, feature_obs_indices_arr,
            h_arr, dt_arr, axis_scale_arr, bool(has_vertex_types),
            bool(has_feature_obs_indices), int(cfg.vertex_capacity),
            float(getattr(cfg, "h_upper_bound", 1.0e15)), collision_safe_distance(cfg),
            float(getattr(cfg, "ego_L", 0.6)), float(getattr(cfg, "ego_W", 0.4)),
            float(getattr(cfg, "sigma_vx", 0.0)), float(getattr(cfg, "sigma_vy", 0.0)),
            float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "build_surface_distance_lin_hk: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.build_surface_distance_lin_hk failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc
    out = np.asarray(out, dtype=float).reshape(-1, 28)
    if out.shape[0] != H or not np.all(np.isfinite(out)):
        raise RuntimeError("invalid C++ surface-distance linearization output: {}".format(out.shape))
    return out


def evaluate_surface_distance_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
) -> np.ndarray:
    """Evaluate exact nonlinear [left,rear,right,front] distances via C++."""
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "evaluate_surface_distance_seq"):
        raise RuntimeError("collision_lin_cpp.evaluate_surface_distance_seq is required; Python fallback is disabled")

    X_arr = np.ascontiguousarray(X_bar, dtype=np.float64)
    H = int(_stage_state_matrix_from_augmented(X_arr).shape[0])
    if H <= 0:
        return np.zeros((0, 4), dtype=float)
    vertex_set_arr = np.ascontiguousarray(_prepare_collision_feature_tensor(vertex_set, H), dtype=np.float64)
    K_used = min(int(getattr(cfg, "vertex_capacity", vertex_set_arr.shape[-2])), int(vertex_set_arr.shape[-2]))
    raw_types = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_vertex_types = bool(raw_types.size >= K_used)
    vertex_types_arr = raw_types if has_vertex_types else np.zeros(0, dtype=np.int32)
    raw_obs = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = bool(raw_obs.size >= K_used)
    feature_obs_indices_arr = raw_obs if has_feature_obs_indices else np.zeros(0, dtype=np.int32)
    h_arr = np.ascontiguousarray(_stage_scalar_sequence(h_steps, H, 1.0), dtype=np.float64)
    dt_arr = np.ascontiguousarray(_stage_scalar_sequence(dt_vals, H, float(cfg.dt_init)), dtype=np.float64)
    axis_scale_arr = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
    try:
        out = collision_lin_cpp.evaluate_surface_distance_seq(
            X_arr, vertex_set_arr, vertex_types_arr, feature_obs_indices_arr,
            h_arr, dt_arr, axis_scale_arr, bool(has_vertex_types),
            bool(has_feature_obs_indices), int(cfg.vertex_capacity),
            float(getattr(cfg, "h_upper_bound", 1.0e15)),
            float(getattr(cfg, "ego_L", 0.6)), float(getattr(cfg, "ego_W", 0.4)),
            float(getattr(cfg, "sigma_vx", 0.0)), float(getattr(cfg, "sigma_vy", 0.0)),
            float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "evaluate_surface_distance_seq: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.evaluate_surface_distance_seq failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc
    out = np.asarray(out, dtype=float).reshape(-1, 4)
    if out.shape[0] != H or not np.all(np.isfinite(out)):
        raise RuntimeError("invalid C++ surface-distance evaluation output: {}".format(out.shape))
    return out


def _groups_to_local(
    groups_world: Sequence[Dict[str, Any]],
    x_state: np.ndarray,
    cfg: MPCConfig,
    convexify: bool = True,
) -> List[Dict[str, Any]]:
    """Transform obstacle groups to the ego body-fixed normalized frame.

    The transform is ``S^{-1} R(psi)^T (P-p)``.  For field 0 we preserve the
    ordered augmented-obstacle boundary so its vertex and edge points define one
    enclosing circle.  Fields 1--4 use convex groups.
    """
    x = np.asarray(x_state, dtype=float).reshape(-1)
    T = ego_scaled_local_transform_matrix(x, cfg)
    ego = x[:2]
    out: List[Dict[str, Any]] = []
    for g in groups_world:
        V_local = transform_points_to_ego_scaled_local(g["V"], T, ego)
        V_local = _convex_hull_points(V_local) if bool(convexify) else _finite_unique_points(V_local)
        if V_local.shape[0] == 0:
            continue
        out.append({"id": g["id"], "V": V_local, "dynamic": bool(g["dynamic"])})
    return out


def _polygon_vertex_edge_points_local(V: np.ndarray) -> np.ndarray:
    """Return ordered polygon vertices followed by one midpoint per edge.

    This is the geometry used by field_flag=0 to define the enclosing circle of
    each occluded-region-augmented obstacle in the body-fixed normalized frame.
    """
    V = _finite_unique_points(np.asarray(V, dtype=float).reshape(-1, 2))
    if V.shape[0] == 0:
        return V
    if V.shape[0] == 1:
        return V.copy()
    edge_points = 0.5 * (V + np.roll(V, -1, axis=0))
    return np.ascontiguousarray(np.vstack((V, edge_points)), dtype=float)


def _horizon_uncertainty_margin_world(h_step: float, dt_val: float, cfg: MPCConfig) -> float:
    if not _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False):
        return 0.0
    t_pred = max(float(h_step), 0.0) * max(float(dt_val), 0.0)
    sigma_x = max(float(getattr(cfg, "sigma_vx", 0.0)), 0.0) * t_pred
    sigma_y = max(float(getattr(cfg, "sigma_vy", 0.0)), 0.0) * t_pred
    # Paper Appendix A: sigma_h = sqrt(sigma_x,h^2 + sigma_y,h^2).
    sigma_iso = math.hypot(sigma_x, sigma_y)
    p_col = min(max(float(getattr(cfg, "probability_collision", 0.8)), 0.0), 1.0 - 1.0e-12)
    return sigma_iso * math.sqrt(max(-2.0 * math.log(1.0 - p_col), 0.0))


def _horizon_uncertainty_margin_local(h_step: float, dt_val: float, cfg: MPCConfig) -> float:
    r_world = _horizon_uncertainty_margin_world(h_step, dt_val, cfg)
    return r_world * max(2.0 / max(abs(float(cfg.ego_L)), 1.0e-12), 2.0 / max(abs(float(cfg.ego_W)), 1.0e-12))


def _smooth_min_value_grad(values: np.ndarray, grads: np.ndarray, tau: float) -> Tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    grads = np.asarray(grads, dtype=float).reshape(-1, 2)
    if values.size == 0:
        return float("inf"), np.zeros(2, dtype=float)
    if float(tau) <= 0.0:
        idx = int(np.argmin(values))
        return float(values[idx]), grads[idx].copy()
    tau = max(float(tau), 1.0e-12)
    scaled = -values / tau
    m = float(np.max(scaled))
    a = np.exp(scaled - m)
    Z = max(float(np.sum(a)), 1.0e-300)
    w = a / Z
    return -tau * (m + math.log(Z)), np.asarray(w @ grads, dtype=float).reshape(2)


def _normalized_smooth_max_value_grad(values: np.ndarray, grads: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    grads = np.asarray(grads, dtype=float).reshape(-1, 2)
    if values.size == 0:
        return float("-inf"), np.zeros(2, dtype=float)
    alpha = max(float(alpha), 1.0e-12)
    scaled = alpha * values
    m = float(np.max(scaled))
    a = np.exp(scaled - m)
    Z = max(float(np.sum(a)), 1.0e-300)
    w = a / Z
    value = (m + math.log(Z) - math.log(max(values.size, 1))) / alpha
    return float(value), np.asarray(w @ grads, dtype=float).reshape(2)


def _circle_union_field_local(groups_local: Sequence[Dict[str, Any]], cfg: MPCConfig) -> Tuple[float, np.ndarray]:
    """C-SDF in the ego body-fixed normalized frame with ego unit disk.

    Each obstacle is fit by one enclosing circle from its transformed vertices,
    and the field at the ego origin is ``||c|| - r - 1``.  The ``-1`` is the
    circumscribed ego ellipse after ``q=S^{-1}R(yaw)^T(P-p)`` maps it to a unit disk.
    """
    vals = []
    grads = []
    q = np.zeros(2, dtype=float)
    for g in groups_local:
        V = np.asarray(g["V"], dtype=float).reshape(-1, 2)
        if V.shape[0] == 0:
            continue
        c = np.mean(V, axis=0)
        radius = float(np.max(np.linalg.norm(V - c.reshape(1, 2), axis=1)))
        delta = q - c
        dist = float(np.linalg.norm(delta))
        vals.append(dist - radius - 1.0)
        grads.append(delta / dist if dist > 1.0e-12 else np.zeros(2, dtype=float))
    return _smooth_min_value_grad(np.asarray(vals), np.asarray(grads).reshape(-1, 2), 0.0)

def _dcsat_union_field_local(groups_local: Sequence[Dict[str, Any]], margin_local: float, cfg: MPCConfig) -> Tuple[float, np.ndarray]:
    vals = []
    grads = []
    alpha = float(getattr(cfg, "dcsat_alpha", 35.0))
    q = np.zeros(2, dtype=float)
    for g in groups_local:
        V = ensure_ccw(_convex_hull_points(g["V"]))
        if V.shape[0] < 3:
            continue
        Vn = np.roll(V, -1, axis=0)
        edge = Vn - V
        axes = np.column_stack((edge[:, 1], -edge[:, 0]))
        nrm = np.maximum(np.linalg.norm(axes, axis=1), 1.0e-12)
        axes = axes / nrm.reshape(-1, 1)
        point_proj = q @ axes.T
        vertex_proj = V @ axes.T
        d1 = point_proj - np.max(vertex_proj, axis=0)
        d2 = np.min(vertex_proj, axis=0) - point_proj
        terms = np.concatenate((d1, d2))
        deriv = np.vstack((axes, -axes))
        sm, gsm = _normalized_smooth_max_value_grad(terms, deriv, alpha)
        margin = float(margin_local) if g["dynamic"] else 0.0
        vals.append(sm - 1.0 - margin)
        grads.append(gsm)
    return _smooth_min_value_grad(np.asarray(vals), np.asarray(grads).reshape(-1, 2), float(getattr(cfg, "dcsat_union_lambda", 0.02)))


def _wu_union_field_local(groups_local: Sequence[Dict[str, Any]], margin_local: float, cfg: MPCConfig) -> Tuple[float, np.ndarray]:
    vals = []
    grads = []
    kappa = max(float(getattr(cfg, "wu_kappa", 35.0)), 1.0e-12)
    buffer_offset = float(getattr(cfg, "wu_buffer_offset", 0.0))
    q = np.zeros(2, dtype=float)
    for g in groups_local:
        V = ensure_ccw(_convex_hull_points(g["V"]))
        if V.shape[0] < 3:
            continue
        Vn = np.roll(V, -1, axis=0)
        edge = Vn - V
        A = np.column_stack((edge[:, 1], -edge[:, 0]))
        nrm = np.maximum(np.linalg.norm(A, axis=1), 1.0e-12)
        A = A / nrm.reshape(-1, 1)
        b = np.sum(A * V, axis=1)
        atomic = q @ A.T - b
        scaled = kappa * atomic
        m = float(np.max(scaled))
        a = np.exp(scaled - m)
        Z = max(float(np.sum(a)), 1.0e-300)
        w = a / Z
        buffer = math.log(max(A.shape[0], 1)) + buffer_offset
        h_i = (m + math.log(Z) - buffer) / kappa
        margin = float(margin_local) if g["dynamic"] else 0.0
        vals.append(h_i - 1.0 - margin)
        grads.append(w @ A)
    return _smooth_min_value_grad(np.asarray(vals), np.asarray(grads).reshape(-1, 2), float(getattr(cfg, "wu_union_lambda", 0.02)))


def _signed_distance_polygon_points(points: np.ndarray, V: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    V = ensure_ccw(_convex_hull_points(V))
    if V.shape[0] < 3:
        return np.full(P.shape[0], np.inf, dtype=float)
    min_sq = np.full(P.shape[0], np.inf, dtype=float)
    inside = np.zeros(P.shape[0], dtype=bool)
    xq = P[:, 0]
    yq = P[:, 1]
    for i in range(V.shape[0]):
        a = V[i]
        b = V[(i + 1) % V.shape[0]]
        e = b - a
        ee = float(e @ e)
        if ee <= 1.0e-14:
            tau = np.zeros(P.shape[0], dtype=float)
        else:
            tau = ((P - a.reshape(1, 2)) @ e) / ee
            tau = np.clip(tau, 0.0, 1.0)
        cp = a.reshape(1, 2) + tau.reshape(-1, 1) * e.reshape(1, 2)
        min_sq = np.minimum(min_sq, np.sum((P - cp) ** 2, axis=1))

        yi, yj = float(a[1]), float(b[1])
        xi, xj = float(a[0]), float(b[0])
        crossing = ((yi > yq) != (yj > yq)) & (
            xq < (xj - xi) * (yq - yi) / (yj - yi + 1.0e-300) + xi
        )
        inside ^= crossing
    d = np.sqrt(np.maximum(min_sq, 0.0))
    d[inside] *= -1.0
    return d


def _exact_union_clearance_points(points: np.ndarray, groups_local: Sequence[Dict[str, Any]], margin_local: float) -> np.ndarray:
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if not groups_local:
        return np.full(P.shape[0], np.inf, dtype=float)
    per = []
    for g in groups_local:
        sd = _signed_distance_polygon_points(P, g["V"])
        margin = float(margin_local) if g["dynamic"] else 0.0
        per.append(sd - 1.0 - margin)
    return np.min(np.column_stack(per), axis=1)



def _exact_union_sdf_points_world(points: np.ndarray, groups_world: Sequence[Dict[str, Any]]) -> np.ndarray:
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    if not groups_world:
        return np.full(P.shape[0], np.inf, dtype=float)
    per = [_signed_distance_polygon_points(P, g["V"]) for g in groups_world]
    return np.min(np.column_stack(per), axis=1)

def _binomial_coeff(n: int, k: int) -> int:
    """Return the exact binomial coefficient without requiring math.comb.

    ROS installations frequently run Python < 3.8, where ``math.comb`` is not
    available.  This multiplicative integer implementation is exact and keeps
    the Bernstein basis numerically identical to the Python-3.8+ expression.
    """
    n = int(n)
    k = int(k)
    if n < 0 or k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for i in range(1, k + 1):
        value = (value * (n - k + i)) // i
    return int(value)


def _bernstein_basis_matrix(xi: np.ndarray, Q: int) -> np.ndarray:
    xi = np.clip(np.asarray(xi, dtype=float).reshape(-1), 0.0, 1.0)
    degree = max(int(Q) - 1, 0)
    B = np.zeros((xi.size, degree + 1), dtype=float)
    for q in range(degree + 1):
        B[:, q] = _binomial_coeff(degree, q) * (xi ** q) * ((1.0 - xi) ** (degree - q))
    return B


def _bernstein_basis_and_derivative(xi: float, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    x = min(max(float(xi), 0.0), 1.0)
    degree = max(int(Q) - 1, 0)
    B = np.array([_binomial_coeff(degree, q) * x ** q * (1.0 - x) ** (degree - q) for q in range(degree + 1)], dtype=float)
    dB = np.zeros_like(B)
    if degree <= 0:
        return B, dB
    Blow = np.array([_binomial_coeff(degree - 1, q) * x ** q * (1.0 - x) ** (degree - 1 - q) for q in range(degree)], dtype=float)
    for q in range(degree + 1):
        left = Blow[q - 1] if q >= 1 else 0.0
        right = Blow[q] if q <= degree - 1 else 0.0
        dB[q] = degree * (left - right)
    return B, dB


def _fit_bp_field_model(
    x_ref: np.ndarray,
    groups_world: Sequence[Dict[str, Any]],
    h_step: float,
    dt_val: float,
    cfg: MPCConfig,
) -> Dict[str, Any]:
    """Reference world-frame BP-SDF with one Bernstein fit per obstacle.

    Runtime planning uses the native C++ implementation.  The obstacle union is
    evaluated as the pointwise minimum of the independently fitted models.
    """
    del x_ref, h_step, dt_val
    Q = max(int(getattr(cfg, "bp_order", 12)), 2)
    xr = np.array([float(cfg.x_min), float(cfg.x_max)], dtype=float)
    yr = np.array([float(cfg.y_min), float(cfg.y_max)], dtype=float)
    common = {
        "T_ref": np.eye(2, dtype=float),
        "p_ref": np.zeros(2, dtype=float),
        "safety_margin_l": float(getattr(cfg, "bp_safety_margin_l", 0.13)),
        "order": Q,
        "x_range": xr,
        "y_range": yr,
    }
    if not groups_world:
        model = dict(common)
        model.update({"empty": True, "models": []})
        return model

    nx = max(int(getattr(cfg, "bp_training_n_x", 30)), Q + 2)
    ny = max(int(getattr(cfg, "bp_training_n_y", 30)), Q + 2)
    xx = np.linspace(xr[0], xr[1], nx)
    yy = np.linspace(yr[0], yr[1], ny)
    Xg, Yg = np.meshgrid(xx, yy)
    sample_points = np.column_stack((Xg.ravel(), Yg.ravel()))
    Bx = _bernstein_basis_matrix((xx - xr[0]) / (xr[1] - xr[0]), Q)
    By = _bernstein_basis_matrix((yy - yr[0]) / (yr[1] - yr[0]), Q)

    models = []
    for group in groups_world:
        # One obstacle only: this target is S_{O_m}(p), not S_{union O}(p).
        target_grid = _exact_union_sdf_points_world(sample_points, [group]).reshape(ny, nx)
        Z = np.linalg.lstsq(Bx, target_grid.T, rcond=None)[0].T
        W_yx = np.linalg.lstsq(By, Z, rcond=None)[0]
        child = dict(common)
        child.update({"empty": False, "W": W_yx.T.copy()})
        models.append(child)

    model = dict(common)
    model.update({"empty": False, "models": models})
    return model

def _evaluate_bp_field_model(x_state: np.ndarray, model: Dict[str, Any], cfg: MPCConfig) -> Tuple[float, np.ndarray]:
    children = model.get("models", None)
    if children is not None:
        if not children:
            return float(getattr(cfg, "h_upper_bound", 1.0e15)), np.zeros(2, dtype=float)
        values = [_evaluate_bp_field_model(x_state, child, cfg) for child in children]
        return min(values, key=lambda item: float(item[0]))
    if bool(model.get("empty", False)):
        return float(getattr(cfg, "h_upper_bound", 1.0e15)), np.zeros(2, dtype=float)
    x = np.asarray(x_state, dtype=float).reshape(-1)[:4]
    T_ref = np.asarray(model["T_ref"], dtype=float).reshape(2, 2)
    q_ref = T_ref @ (x[:2] - np.asarray(model["p_ref"], dtype=float).reshape(2))
    xr = np.asarray(model["x_range"], dtype=float).reshape(2)
    yr = np.asarray(model["y_range"], dtype=float).reshape(2)
    Q = int(model["order"])
    xi_raw = (q_ref[0] - xr[0]) / (xr[1] - xr[0])
    eta_raw = (q_ref[1] - yr[0]) / (yr[1] - yr[0])
    Bx, dBx = _bernstein_basis_and_derivative(xi_raw, Q)
    By, dBy = _bernstein_basis_and_derivative(eta_raw, Q)
    W = np.asarray(model["W"], dtype=float).reshape(Q, Q)
    # Fit/design layout is W[qx, qy] with basis Bx[qx] * By[qy].
    # Evaluating as By @ W @ Bx transposes the x/y coefficient axes.
    raw = float(Bx @ W @ By)
    gx = float(dBx @ W @ By) / (xr[1] - xr[0])
    gy = float(Bx @ W @ dBy) / (yr[1] - yr[0])
    if xi_raw < 0.0 or xi_raw > 1.0:
        gx = 0.0
    if eta_raw < 0.0 or eta_raw > 1.0:
        gy = 0.0
    grad_ref = np.array([gx, gy], dtype=float)
    grad_world = T_ref.T @ grad_ref
    return raw - float(model.get("safety_margin_l", 0.13)), grad_world


def _evaluate_nonproposed_field_single(
    x_state: np.ndarray,
    points_world: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_step: float,
    dt_val: float,
    cfg: MPCConfig,
    bp_model: Optional[Dict[str, Any]] = None,
) -> Tuple[float, np.ndarray]:
    flag = collision_field_flag(cfg)
    groups_world = _group_world_polygons(
        points_world, feature_obs_indices, vertex_types, int(cfg.vertex_capacity),
        convexify=(flag != 0),
    )
    if flag == 2:
        model = bp_model if bp_model is not None else _fit_bp_field_model(x_state, groups_world, h_step, dt_val, cfg)
        return _evaluate_bp_field_model(x_state, model, cfg)
    groups_local = _groups_to_local(groups_world, x_state, cfg, convexify=(flag != 0))
    if flag == 0:
        return _circle_union_field_local(groups_local, cfg)
    if not groups_local:
        return float(getattr(cfg, "h_upper_bound", 1.0e15)), np.zeros(2, dtype=float)
    margin_local = _horizon_uncertainty_margin_local(h_step, dt_val, cfg)
    if flag == 1:
        return _dcsat_union_field_local(groups_local, margin_local, cfg)
    if flag == 3:
        return _wu_union_field_local(groups_local, margin_local, cfg)
    raise ValueError("non-proposed evaluator received field_flag={}".format(flag))


def _stable_two_logsumexp(a: float, b: float) -> float:
    m = max(float(a), float(b))
    return m + math.log(math.exp(float(a) - m) + math.exp(float(b) - m))


def _nonproposed_residual_pair(
    x_state: np.ndarray,
    points_world: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_step: float,
    dt_val: float,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]],
    bp_model: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float]:
    phi, grad_local = _evaluate_nonproposed_field_single(
        x_state, points_world, vertex_types, feature_obs_indices, h_step, dt_val, cfg, bp_model=bp_model
    )
    disabled = -float(getattr(cfg, "h_upper_bound", 1.0e15))
    inactive_regulation = -1.0
    if not math.isfinite(float(phi)):
        return float(getattr(cfg, "h_upper_bound", 1.0e15)), inactive_regulation, float(getattr(cfg, "h_upper_bound", 1.0e15))
    ds = collision_safe_distance(cfg)
    g_a = ds - float(phi)
    if not impact_normal_constraint_enabled(cfg):
        return g_a, inactive_regulation, float(phi)

    grad_native = np.asarray(grad_local, dtype=float).reshape(2)
    eps_n = max(float(getattr(cfg, "normal_eps", 1.0e-9)), 1.0e-12)
    gnorm = float(np.linalg.norm(grad_native))
    if (not math.isfinite(gnorm)) or gnorm <= eps_n:
        return g_a, inactive_regulation, float(phi)
    N_field = -grad_native / gnorm
    if collision_field_flag(cfg) == 2:
        _, _, lc_body = sector_rays_body_from_cfg(cfg) if sector_rays_body is None else np.asarray(sector_rays_body, dtype=float).reshape(3, 2)
        lc_body = np.asarray(lc_body, dtype=float).reshape(2)
        lc_body /= max(float(np.linalg.norm(lc_body)), 1.0e-12)
        yaw = float(np.asarray(x_state, dtype=float).reshape(-1)[3])
        c, ss = math.cos(yaw), math.sin(yaw)
        lc_native = np.array([c * lc_body[0] - ss * lc_body[1], ss * lc_body[0] + c * lc_body[1]], dtype=float)
    else:
        lc_native = _sector_lc_local_sequence(
            np.asarray(x_state, dtype=float).reshape(1, 4), cfg,
            sector_rays_body=sector_rays_body,
        )[0]
    align = float(N_field @ lc_native)
    Ns = regulation_alignment_Ns(cfg, sector_rays_body)
    lam_r = regulation_smoothing_lambda(cfg)
    # Regulation in solver residual form:
    #   lambda_r*LSE((Phi-d_safe)/lambda_r,
    #                (Ns-align)/lambda_r) >= lambda_r*log(2)
    # <=>
    #   gbar_r = lambda_r*(log(2)-LSE(a,b)).
    # The final OCP residual is g_r = gbar_r - s^r <= 0; this evaluator
    # returns gbar_r because no decision slack is available here.
    za = (float(phi) - ds) / lam_r
    zb = (Ns - align) / lam_r
    zmax = max(za, zb)
    lse = zmax + math.log(math.exp(za - zmax) + math.exp(zb - zmax))
    gbar_r = lam_r * (math.log(2.0) - lse)
    if not math.isfinite(gbar_r):
        return g_a, inactive_regulation, float(phi)
    return g_a, float(gbar_r), float(phi)


def _prepare_bp_models_for_sequence(
    X: np.ndarray,
    points_tensor: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: np.ndarray,
    dt_vals: np.ndarray,
    cfg: MPCConfig,
) -> List[Optional[Dict[str, Any]]]:
    """Fit BP-SDF once at k=0, then translate each obstacle model over the horizon.

    For obstacle m, the k-th field is evaluated as
        phi_m,k(p) = phi_m,0(p - Delta p_m,k),
    implemented by shifting only the fitted model's p_ref.  Bernstein
    coefficients and fitting domains are never recomputed for k > 0.
    """
    H = int(X.shape[0])
    if collision_field_flag(cfg) != 2:
        return [None] * H
    if H <= 0:
        return []

    pts0 = _field_stage_values(points_tensor, 0, H)
    groups0 = _group_world_polygons(
        pts0, feature_obs_indices, vertex_types, int(cfg.vertex_capacity)
    )
    base_model = _fit_bp_field_model(
        X[0], groups0, float(h_steps[0]), float(dt_vals[0]), cfg
    )

    # Shared Kx2 geometry has no stage-wise obstacle translation.
    if points_tensor.ndim == 2:
        return [base_model] * H

    base_children = list(base_model.get("models", []))
    if len(base_children) != len(groups0):
        raise RuntimeError("BP-SDF k=0 obstacle/model count mismatch")
    base_centroids = {
        int(g["id"]): np.mean(np.asarray(g["V"], dtype=float), axis=0)
        for g in groups0
    }

    models: List[Optional[Dict[str, Any]]] = [None] * H
    for h in range(H):
        if h == 0:
            models[h] = base_model
            continue
        pts = _field_stage_values(points_tensor, h, H)
        groups_h = _group_world_polygons(
            pts, feature_obs_indices, vertex_types, int(cfg.vertex_capacity)
        )
        current = {int(g["id"]): g for g in groups_h}

        stage_children = []
        for g0, child0 in zip(groups0, base_children):
            oid = int(g0["id"])
            if oid not in current:
                raise RuntimeError(
                    "BP-SDF translation reuse requires the same obstacle IDs at every horizon stage"
                )
            ck = np.mean(np.asarray(current[oid]["V"], dtype=float), axis=0)
            delta = ck - base_centroids[oid]
            child = dict(child0)
            child["p_ref"] = np.asarray(child0["p_ref"], dtype=float).reshape(2) + delta
            stage_children.append(child)

        stage_model = dict(base_model)
        stage_model["models"] = stage_children
        models[h] = stage_model
    return models




class BodyFixedBernsteinSDFModel(object):
    """Fitted Geometry-Aware Bernstein SDF in one or more WORLD-frame charts.

    Compatibility class name retained for callers.  The k=0 BP-SDF is fitted
    with ``T_ref=I`` and ``p_ref=0`` on the fixed world-map domain.  For later
    stages, the same coefficients/domain are reused and only ``p_ref`` is shifted
    by each obstacle's world-frame translation.
    """

    __slots__ = (
        "W", "T_ref", "p_ref", "x_range", "y_range", "empty", "order",
        "safety_margin_l", "cache_key",
    )

    def __init__(self, W, T_ref, p_ref, x_range, y_range, empty, order, safety_margin_l, cache_key=None):
        self.W = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
        self.T_ref = np.ascontiguousarray(np.asarray(T_ref, dtype=np.float64))
        self.p_ref = np.ascontiguousarray(np.asarray(p_ref, dtype=np.float64))
        self.x_range = np.ascontiguousarray(np.asarray(x_range, dtype=np.float64))
        self.y_range = np.ascontiguousarray(np.asarray(y_range, dtype=np.float64))
        self.empty = np.ascontiguousarray(np.asarray(empty, dtype=np.int32))
        self.order = int(order)
        self.safety_margin_l = float(safety_margin_l)
        self.cache_key = cache_key

    @property
    def model_count(self):
        """Number of stage model sets (compatibility property)."""
        return int(self.W.shape[0]) if self.W.ndim == 4 else 0

    @property
    def obstacle_count(self):
        return int(self.W.shape[1]) if self.W.ndim == 4 else 0

    def packed(self):
        return (
            self.W, self.T_ref, self.p_ref, self.x_range, self.y_range,
            self.empty, int(self.order),
        )


_BP_MODEL_CACHE = OrderedDict()
_BP_MODEL_CACHE_MAX = 8


def clear_body_fixed_bp_sdf_cache() -> None:
    """Drop cached field_flag=2 world-frame Bernstein models."""
    _BP_MODEL_CACHE.clear()


def _bp_hash_array(hasher, value, dtype) -> None:
    if value is None:
        hasher.update(b"<none>")
        return
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    hasher.update(str(arr.shape).encode("ascii"))
    hasher.update(str(arr.dtype).encode("ascii"))
    hasher.update(arr.tobytes(order="C"))


def _bp_model_cache_key(
    points_tensor: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: np.ndarray,
    dt_vals: np.ndarray,
    cfg: MPCConfig,
    reference_yaws: Optional[np.ndarray] = None,
):
    """Key the world-frame fitted-boundary BP-SDF model.

    Ego state/yaw, body axis scale, horizon timing, and prediction uncertainty do
    not define this fit. Geometry, basis order/grid, l, and local-domain padding do.
    """
    del h_steps, dt_vals, reference_yaws
    h = hashlib.blake2b(digest_size=20)
    _bp_hash_array(h, points_tensor, np.float64)
    _bp_hash_array(h, vertex_types, np.int32)
    _bp_hash_array(h, feature_obs_indices, np.int32)
    params = (
        int(getattr(cfg, "vertex_capacity", 0)),
        int(getattr(cfg, "bp_order", 12)),
        int(getattr(cfg, "bp_training_n_x", 30)),
        int(getattr(cfg, "bp_training_n_y", 30)),
        float(getattr(cfg, "bp_safety_margin_l", 0.13)),
        float(getattr(cfg, "bp_domain_padding", 1.0)),
        float(getattr(cfg, "x_min", -6.0)), float(getattr(cfg, "x_max", 6.0)),
        float(getattr(cfg, "y_min", -6.0)), float(getattr(cfg, "y_max", 6.0)),
    )
    h.update(repr(params).encode("ascii"))
    return h.digest()


def _bp_model_contains_world_states(model: BodyFixedBernsteinSDFModel, X: np.ndarray, tol: float = 1.0e-9) -> bool:
    """Return whether a cached fitted-boundary model can evaluate the states.

    The final BP-SDF is metric distance to the extracted fitted boundary, so a
    query no longer needs to lie inside the obstacle-local polynomial fit box.
    """
    del tol
    states = _stage_state_matrix_from_augmented(X)
    if states.shape[0] <= 0:
        return True
    if not np.all(np.isfinite(states[:, :2])):
        return False
    return int(model.model_count) > 0 and int(model.obstacle_count) > 0

def _cache_bp_model(key, model: BodyFixedBernsteinSDFModel) -> None:
    _BP_MODEL_CACHE[key] = model
    _BP_MODEL_CACHE.move_to_end(key)
    while len(_BP_MODEL_CACHE) > _BP_MODEL_CACHE_MAX:
        _BP_MODEL_CACHE.popitem(last=False)


def build_body_fixed_bp_sdf_model(
    reference_states: np.ndarray,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
    use_cache: bool = True,
) -> BodyFixedBernsteinSDFModel:
    """Construct world-frame per-obstacle fitted-boundary BP-SDF models.

    Each Bernstein polynomial fits exact SDF samples on an obstacle-local chart.
    Its l-level set is treated as the fitted obstacle boundary; C++ then returns
    signed Euclidean distance to that boundary. Shared static fits are cached.
    """
    X = _stage_state_matrix_from_augmented(np.asarray(reference_states, dtype=np.float64))
    H = int(X.shape[0])
    points = _prepare_collision_feature_tensor(vertex_set, H)
    h_arr = _stage_scalar_sequence(h_steps, H, 1.0)
    dt_arr = _stage_scalar_sequence(dt_vals, H, float(cfg.dt_init))
    packed = _pack_bp_models_for_cpp(
        X, points, vertex_types, feature_obs_indices, h_arr, dt_arr, cfg,
        use_cache=bool(use_cache),
    )
    W, T_ref, p_ref, x_range, y_range, empty, Q = packed
    return BodyFixedBernsteinSDFModel(
        W, T_ref, p_ref, x_range, y_range, empty, Q,
        float(getattr(cfg, "bp_safety_margin_l", 0.13)),
    )


def evaluate_body_fixed_bp_sdf_world(
    X_world: np.ndarray,
    model: BodyFixedBernsteinSDFModel,
    cfg: MPCConfig,
) -> np.ndarray:
    """Evaluate signed Euclidean distance to fitted Bernstein boundaries."""
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "evaluate_bp_model_phi_seq"):
        raise RuntimeError("collision_lin_cpp.evaluate_bp_model_phi_seq is required")
    X = np.ascontiguousarray(_stage_state_matrix_from_augmented(X_world), dtype=np.float64)
    if X.shape[0] <= 0:
        return np.zeros(0, dtype=np.float64)
    axis_scale = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
    out = collision_lin_cpp.evaluate_bp_model_phi_seq(
        X, axis_scale, float(getattr(cfg, "h_upper_bound", 1.0e15)),
        model.W, model.T_ref, model.p_ref, model.x_range, model.y_range,
        model.empty, int(model.order), float(model.safety_margin_l),
    )
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if out.size != X.shape[0] or not np.all(np.isfinite(out)):
        raise RuntimeError("invalid world-frame Bernstein SDF evaluation")
    return out


def _pack_bp_models_for_cpp(
    X: np.ndarray,
    points_tensor: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: np.ndarray,
    dt_vals: np.ndarray,
    cfg: MPCConfig,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit/cache world-frame Bernstein boundary models entirely in C++.

    Each obstacle is fitted independently on bbox + bp_domain_padding. The
    polynomial l-level set defines the fitted boundary, and evaluation recomputes
    signed Euclidean distance to that boundary. Later stages reuse the same
    coefficients and translate p_ref by obstacle motion.
    """
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    H = int(X_arr.shape[0])
    Q = max(int(getattr(cfg, "bp_order", 12)), 2)
    # Empty placeholders remain part of the established multifield C++ ABI for
    # field flags other than 2.
    W0 = np.zeros((1, 1, Q, Q), dtype=np.float64)
    T0 = np.eye(2, dtype=np.float64).reshape(1, 1, 2, 2)
    p0 = np.zeros((1, 1, 2), dtype=np.float64)
    xr0 = np.array([[[-1.0, 1.0]]], dtype=np.float64)
    yr0 = xr0.copy()
    e0 = np.ones((1, 1), dtype=np.int32)
    if H <= 0 or collision_field_flag(cfg) != 2:
        return W0, T0, p0, xr0, yr0, e0, Q

    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "fit_bp_models_seq"):
        raise RuntimeError(
            "collision_lin_cpp.fit_bp_models_seq is required for field_flag=2; "
            "Python Bernstein fitting is disabled"
        )

    points_arr = np.ascontiguousarray(points_tensor, dtype=np.float64)
    # A predictor may publish static geometry as H identical slices. Collapse
    # that exact repetition back to Kx2 so C++ performs only one obstacle-wise
    # BP fit and broadcasts the resulting world-frame model across the horizon.
    if (points_arr.ndim == 3 and points_arr.shape[0] > 0
            and np.array_equal(points_arr, np.broadcast_to(points_arr[0:1], points_arr.shape))):
        points_arr = np.ascontiguousarray(points_arr[0], dtype=np.float64)
    cache_key = _bp_model_cache_key(
        points_arr, vertex_types, feature_obs_indices, h_steps, dt_vals, cfg
    )
    if use_cache:
        cached = _BP_MODEL_CACHE.get(cache_key)
        if cached is not None and _bp_model_contains_world_states(cached, X_arr):
            _BP_MODEL_CACHE.move_to_end(cache_key)
            return cached.packed()

    vertex_types_arr = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    obs_ids_arr = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )
    h_arr = np.ascontiguousarray(np.asarray(h_steps, dtype=np.float64).reshape(-1))
    dt_arr = np.ascontiguousarray(np.asarray(dt_vals, dtype=np.float64).reshape(-1))
    axis_scale = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)

    try:
        packed = collision_lin_cpp.fit_bp_models_seq(
            X_arr,
            points_arr,
            vertex_types_arr,
            obs_ids_arr,
            h_arr,
            dt_arr,
            axis_scale,
            bool(vertex_types is not None),
            bool(feature_obs_indices is not None),
            int(cfg.vertex_capacity),
            float(cfg.dt_init),
            float(getattr(cfg, "ego_L", 0.6)),
            float(getattr(cfg, "ego_W", 0.4)),
            float(getattr(cfg, "sigma_vx", 0.0)),
            float(getattr(cfg, "sigma_vy", 0.0)),
            float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False),
            int(Q),
            int(getattr(cfg, "bp_training_n_x", 30)),
            int(getattr(cfg, "bp_training_n_y", 30)),
            float(getattr(cfg, "bp_safety_margin_l", 0.13)),
            float(getattr(cfg, "bp_domain_padding", 1.0)),
            int(getattr(cfg, "N", max(H, 1))),
            float(getattr(cfg, "v_max", 0.0)),
            float(getattr(cfg, "a_min", 0.0)),
            float(getattr(cfg, "a_max", 0.0)),
            float(getattr(cfg, "x_min", -6.0)),
            float(getattr(cfg, "x_max", 6.0)),
            float(getattr(cfg, "y_min", -6.0)),
            float(getattr(cfg, "y_max", 6.0)),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "fit_bp_models_seq: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.fit_bp_models_seq failed; Python Bernstein fitting "
            "fallback is disabled: {!r}".format(exc)
        ) from exc

    if not isinstance(packed, tuple) or len(packed) != 6:
        raise RuntimeError("fit_bp_models_seq returned an invalid model tuple")
    W, T_ref, p_ref, x_range, y_range, empty = packed
    W = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
    T_ref = np.ascontiguousarray(np.asarray(T_ref, dtype=np.float64))
    p_ref = np.ascontiguousarray(np.asarray(p_ref, dtype=np.float64))
    x_range = np.ascontiguousarray(np.asarray(x_range, dtype=np.float64))
    y_range = np.ascontiguousarray(np.asarray(y_range, dtype=np.float64))
    empty = np.ascontiguousarray(np.asarray(empty, dtype=np.int32))
    expected_stages = H if points_arr.ndim == 3 else 1
    stage_count = int(W.shape[0]) if W.ndim == 4 else 0
    obstacle_count = int(W.shape[1]) if W.ndim == 4 else 0
    valid = (
        stage_count == expected_stages
        and obstacle_count >= 1
        and W.shape[2:] == (Q, Q)
        and T_ref.shape == (expected_stages, obstacle_count, 2, 2)
        and p_ref.shape == (expected_stages, obstacle_count, 2)
        and x_range.shape == (expected_stages, obstacle_count, 2)
        and y_range.shape == (expected_stages, obstacle_count, 2)
        and empty.shape == (expected_stages, obstacle_count)
        and np.all(np.isfinite(W))
        and np.all(np.isfinite(T_ref))
        and np.all(np.isfinite(p_ref))
        and np.all(np.isfinite(x_range))
        and np.all(np.isfinite(y_range))
    )
    if not valid:
        raise RuntimeError(
            "fit_bp_models_seq returned invalid per-obstacle shapes/values: W={}, T={}, p={}, xr={}, yr={}, empty={}, expected_stages={}".format(
                W.shape, T_ref.shape, p_ref.shape, x_range.shape, y_range.shape, empty.shape, expected_stages
            )
        )
    model = BodyFixedBernsteinSDFModel(
        W, T_ref, p_ref, x_range, y_range, empty, Q,
        float(getattr(cfg, "bp_safety_margin_l", 0.13)), cache_key=cache_key,
    )
    if use_cache:
        _cache_bp_model(cache_key, model)
    return model.packed()


def _build_nonproposed_dual_linearization_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    X = _stage_state_matrix_from_augmented(X_bar)
    H = int(X.shape[0])
    out = np.zeros((H, 14), dtype=float)
    if H <= 0:
        return out
    points_tensor = _prepare_collision_feature_tensor(vertex_set, H)
    h_arr = np.asarray(h_steps, dtype=float).reshape(-1)
    dt_arr = np.asarray(dt_vals, dtype=float).reshape(-1)
    if h_arr.size < H:
        h_arr = np.pad(h_arr, (0, H - h_arr.size), mode="edge") if h_arr.size else np.arange(1, H + 1, dtype=float)
    if dt_arr.size < H:
        dt_arr = np.pad(dt_arr, (0, H - dt_arr.size), mode="edge") if dt_arr.size else np.full(H, float(cfg.dt_init))
    h_arr = h_arr[:H]
    dt_arr = dt_arr[:H]
    bp_models = _prepare_bp_models_for_sequence(X, points_tensor, vertex_types, feature_obs_indices, h_arr, dt_arr, cfg)

    eps_pos = max(float(getattr(cfg, "fd_position_eps_m", 1.0e-4)), 1.0e-7)
    eps_yaw = max(float(getattr(cfg, "fd_yaw_eps_rad", 1.0e-4)), 1.0e-7)
    eps_dt = max(1.0e-6, min(1.0e-4, 0.1 * max(float(getattr(cfg, "dt_init", 0.1)), 1.0e-6)))
    disabled = -float(getattr(cfg, "h_upper_bound", 1.0e15))

    for h in range(H):
        pts = _field_stage_values(points_tensor, h, H)
        x0 = np.asarray(X[h], dtype=float).reshape(4).copy()
        dt0 = float(dt_arr[h])
        ga0, gr0, _ = _nonproposed_residual_pair(
            x0, pts, vertex_types, feature_obs_indices, h_arr[h], dt0, cfg,
            sector_rays_body, bp_model=bp_models[h],
        )
        out[h, 0] = ga0
        out[h, 7] = gr0

        # Derivative variables in the unchanged OCP are
        # Legacy C++ Jacobians include dt/boundary columns. The compact acados model uses only [px,py,v,yaw].
        # or boundary slack, so those two columns remain exactly zero.
        for d, eps in ((0, eps_pos), (1, eps_pos), (3, eps_yaw)):
            xp = x0.copy(); xm = x0.copy()
            xp[d] += eps; xm[d] -= eps
            gap, grp, _ = _nonproposed_residual_pair(
                xp, pts, vertex_types, feature_obs_indices, h_arr[h], dt0, cfg,
                sector_rays_body, bp_model=bp_models[h],
            )
            gam, grm, _ = _nonproposed_residual_pair(
                xm, pts, vertex_types, feature_obs_indices, h_arr[h], dt0, cfg,
                sector_rays_body, bp_model=bp_models[h],
            )
            out[h, 1 + d] = (gap - gam) / (2.0 * eps)
            if gr0 > disabled * 0.5 and grp > disabled * 0.5 and grm > disabled * 0.5:
                out[h, 8 + d] = (grp - grm) / (2.0 * eps)

        # Only the horizon uncertainty margin depends on dt.  BP models are held
        # fixed during one SQP linearization, exactly like the MATLAB fit.
        if collision_field_flag(cfg) != 2 and _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False):
            dtp = max(dt0 + eps_dt, 1.0e-9)
            dtm = max(dt0 - eps_dt, 1.0e-9)
            gap, grp, _ = _nonproposed_residual_pair(
                x0, pts, vertex_types, feature_obs_indices, h_arr[h], dtp, cfg,
                sector_rays_body, bp_model=bp_models[h],
            )
            gam, grm, _ = _nonproposed_residual_pair(
                x0, pts, vertex_types, feature_obs_indices, h_arr[h], dtm, cfg,
                sector_rays_body, bp_model=bp_models[h],
            )
            denom = max(dtp - dtm, 1.0e-12)
            out[h, 5] = (gap - gam) / denom
            if gr0 > disabled * 0.5 and grp > disabled * 0.5 and grm > disabled * 0.5:
                out[h, 12] = (grp - grm) / denom

    if not np.all(np.isfinite(out)):
        raise RuntimeError("{} linearization returned non-finite values".format(collision_field_name(cfg)))
    return out


def _evaluate_nonproposed_phi_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
) -> np.ndarray:
    X = _stage_state_matrix_from_augmented(X_bar)
    H = int(X.shape[0])
    if H <= 0:
        return np.zeros(0, dtype=float)
    points_tensor = _prepare_collision_feature_tensor(vertex_set, H)
    h_arr = np.asarray(h_steps, dtype=float).reshape(-1)
    dt_arr = np.asarray(dt_vals, dtype=float).reshape(-1)
    if h_arr.size < H:
        h_arr = np.pad(h_arr, (0, H - h_arr.size), mode="edge") if h_arr.size else np.arange(1, H + 1, dtype=float)
    if dt_arr.size < H:
        dt_arr = np.pad(dt_arr, (0, H - dt_arr.size), mode="edge") if dt_arr.size else np.full(H, float(cfg.dt_init))
    h_arr = h_arr[:H]; dt_arr = dt_arr[:H]
    bp_models = _prepare_bp_models_for_sequence(X, points_tensor, vertex_types, feature_obs_indices, h_arr, dt_arr, cfg)
    phi = np.empty(H, dtype=float)
    for h in range(H):
        pts = _field_stage_values(points_tensor, h, H)
        phi[h], _ = _evaluate_nonproposed_field_single(
            X[h], pts, vertex_types, feature_obs_indices, h_arr[h], dt_arr[h], cfg, bp_model=bp_models[h]
        )
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("{} Phi evaluation returned non-finite values".format(collision_field_name(cfg)))
    return phi


def build_external_avoidance_linearization_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Return H x 7 avoidance linearization from the unified C++ H x K kernel."""
    dual = build_external_dual_collision_linearization_sequence(
        X_bar, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
        h_steps, dt_vals, cfg, sector_rays_body=sector_rays_body,
    )
    return np.ascontiguousarray(np.asarray(dual, dtype=float)[:, :7])

def _log_simplified_regulation_linearization(dual: np.ndarray, cfg: MPCConfig) -> None:
    """Print diagnostics for the LSE regulation linearization.

    The native linearizer supplies the state-only core

        gbar_r = lambda_r*log(2)
                 - lambda_r*LSE(Phi_safe/lambda_r,
                                (Ns-N^T n_c)/lambda_r).

    The actual solver residual is ``g_r = gbar_r - s^r <= 0``.  For the
    proposed field, Phi and N are evaluated after the
    lambda_o=lambda_f field-level log/exp cancellation.  The Jacobian includes
    the derivative of N's complete normalization denominator, including the
    active max(||r||, Z*normal_eps) branch.
    """
    if not _bool_compat(getattr(cfg, "debug_mode", False), False):
        return
    arr = np.asarray(dual, dtype=float).reshape(-1, 14)
    if arr.size == 0:
        return
    lam_f = max(float(collision_smoothing_lambda(cfg)), 1.0e-12)
    lam_r = max(float(regulation_smoothing_lambda(cfg)), 1.0e-12)
    Ns = float(regulation_alignment_Ns(cfg))
    log2 = math.log(2.0)
    max_rows = max(int(getattr(cfg, "regulation_analysis_log_max_stages", 64)), 0)
    print("[REG-LIN] lambda_f=lambda_o={:.6e} | lambda_r={:.6e} | Ns={:+.6f}".format(
        lam_f, lam_r, Ns))
    for k, row in enumerate(arr[:max_rows]):
        phi = -float(row[0])
        gbar_r = float(row[7])
        jphi = -np.asarray(row[1:7], dtype=float)
        jreg = np.asarray(row[8:14], dtype=float)
        if not (math.isfinite(phi) and math.isfinite(gbar_r)):
            print("[REG-LIN] {:02d} | non-finite row".format(k))
            continue

        za = phi / lam_r
        lse = log2 - gbar_r / lam_r
        wa = math.exp(min(0.0, za - lse)) if za <= lse + 1.0e-12 else float("nan")
        wa = min(max(wa, 0.0), 1.0) if math.isfinite(wa) else float("nan")
        wb = 1.0 - wa if math.isfinite(wa) else float("nan")

        align = float("nan")
        jalign_norm = float("nan")
        if math.isfinite(wa) and wb > 1.0e-12:
            zb = lse + math.log(max(wb, 1.0e-300))
            align = Ns - lam_r * zb
            jalign = (jreg + wa * jphi) / wb
            jalign_norm = float(np.linalg.norm(jalign))

        print("[REG-LIN] {:02d} | Phi_safe={:+.6e} | a=Phi_safe={:+.6e} | align={:+.6f} | b=Ns-align={:+.6e} | g={:+.6e} | wa={:.3e} | wb={:.3e} | ||JPhi||={:.3e} | ||Jalign||={:.3e} | ||Jreg||={:.3e}".format(
            k, phi, phi, align, Ns-align if math.isfinite(align) else float("nan"),
            gbar_r, wa, wb, float(np.linalg.norm(jphi)), jalign_norm, float(np.linalg.norm(jreg))))

    if arr.shape[0] > 0:
        finite = np.all(np.isfinite(arr), axis=1)
        print("[REG-QP] stages={} finite={} | gbar_reg[min,max]=[{:+.3e},{:+.3e}] | |Jreg|max={:.3e}".format(
            int(arr.shape[0]), int(np.count_nonzero(finite)),
            float(np.min(arr[:, 7])), float(np.max(arr[:, 7])),
            float(np.max(np.abs(arr[:, 8:14]))) if arr[:, 8:14].size else 0.0))


def build_external_dual_collision_linearization_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
    return_alignment: bool = False,
):
    """Return avoidance/regulation linearizations via API-32 C++.

    Native API-32 returns three 7-value blocks per stage:
      [avoidance | regulation | impact-aligned value].
    The public default remains the historical H x 14 dual array; setting
    ``return_alignment=True`` additionally returns the H x 7 third block.
    Only column 0 is used by J_global; derivative columns are intentionally zero.

    All field_flag=0..4 values use one C++ entry point. C-SDF/DCSAT/Wu/proposed
    use body-fixed normalized geometry, while field_flag=2 keeps a world-frame
    SDF. For field_flag=2, world-frame SDF sampling, Bernstein coefficient fitting, field
    evaluation and linearization execute in the native backend; when enabled and
    large enough, field_flag=2 uses the CUDA BP-SDF path with CPU fallback.
    """
    require_cpp_acceleration()
    _configure_native_cuda(cfg)
    api = int(getattr(collision_lin_cpp, "COLLISION_RESIDUAL_API", 0))
    if api < _COLLISION_LIN_CPP_REQUIRED_API or not hasattr(collision_lin_cpp, "build_collision_multifield_dual_lin_hk"):
        raise RuntimeError(
            "collision_lin_cpp API {} with build_collision_multifield_dual_lin_hk is required; Python fallback is disabled".format(api)
        )

    X_arr = np.ascontiguousarray(X_bar, dtype=np.float64)
    X_stage = _stage_state_matrix_from_augmented(X_arr)
    H = int(X_stage.shape[0])
    vertex_set_arr = _prepare_collision_feature_tensor(vertex_set, H)
    vertex_normals_arr = _prepare_collision_feature_tensor(vertex_normals, H)
    has_vertex_types = vertex_types is not None
    vertex_types_arr = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = feature_obs_indices is not None
    feature_obs_indices_arr = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )

    cpp_vertex_set_arr = vertex_set_arr
    cpp_vertex_normals_arr = vertex_normals_arr
    cpp_vertex_types_arr = vertex_types_arr
    cpp_feature_obs_indices_arr = feature_obs_indices_arr
    cpp_has_vertex_types = bool(has_vertex_types)
    cpp_has_feature_obs_indices = bool(has_feature_obs_indices)
    cpp_vertex_capacity = int(cfg.vertex_capacity)
    if collision_field_flag(cfg) == 4:
        (
            cpp_vertex_set_arr, cpp_vertex_normals_arr, cpp_vertex_types_arr,
            cpp_feature_obs_indices_arr, cpp_has_vertex_types,
            cpp_has_feature_obs_indices, cpp_vertex_capacity,
        ) = _prepare_proposed_features_cpp(
            vertex_set_arr, vertex_types, feature_obs_indices, X_stage, cfg
        )
    sector = np.ascontiguousarray(
        sector_rays_body_from_cfg(cfg) if sector_rays_body is None else sector_rays_body,
        dtype=np.float64,
    ).reshape(3, 2)
    axis_scale_arr = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
    h_arr = _stage_scalar_sequence(h_steps, H, 1.0)
    dt_arr = _stage_scalar_sequence(dt_vals, H, float(cfg.dt_init))

    bp_W, bp_T_ref, bp_p_ref, bp_x_range, bp_y_range, bp_empty, bp_Q = _pack_bp_models_for_cpp(
        X_stage, vertex_set_arr, vertex_types, feature_obs_indices, h_arr, dt_arr, cfg
    )

    try:
        out = collision_lin_cpp.build_collision_multifield_dual_lin_hk(
            X_arr, cpp_vertex_set_arr, cpp_vertex_normals_arr, cpp_vertex_types_arr, cpp_feature_obs_indices_arr,
            np.ascontiguousarray(h_arr, dtype=np.float64), np.ascontiguousarray(dt_arr, dtype=np.float64),
            sector, axis_scale_arr, bool(cpp_has_vertex_types), bool(cpp_has_feature_obs_indices),
            int(cpp_vertex_capacity), float(cfg.dt_init), float(cfg.dt_init), float(cfg.dt_init),
            float(getattr(cfg, "h_upper_bound", 1.0e15)), collision_safe_distance(cfg),
            regulation_alignment_Ns(cfg, sector), regulation_smoothing_lambda(cfg),
            float(getattr(cfg, "ego_L", 0.6)), float(getattr(cfg, "ego_W", 0.4)),
            0.0, bool(use_ego_scaled_local_frame(cfg)), str(getattr(cfg, "normal_orientation", "inward")),
            0.0, 0.0, collision_smoothing_lambda(cfg), collision_smoothing_lambda(cfg),
            float(getattr(cfg, "normal_eps", 1.0e-9)), float(getattr(cfg, "sigma_vx", 0.0)),
            float(getattr(cfg, "sigma_vy", 0.0)), float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False), 0.0,
            collision_field_flag(cfg), float(getattr(cfg, "circle_union_lambda", 0.0)),
            float(getattr(cfg, "dcsat_alpha", 35.0)), float(getattr(cfg, "dcsat_union_lambda", 0.02)),
            float(getattr(cfg, "wu_kappa", 35.0)), float(getattr(cfg, "wu_union_lambda", 0.02)),
            float(getattr(cfg, "wu_buffer_offset", 0.0)), bp_W, bp_T_ref, bp_p_ref, bp_x_range,
            bp_y_range, bp_empty, int(bp_Q), float(getattr(cfg, "bp_safety_margin_l", 0.13)),
            float(getattr(cfg, "fd_position_eps_m", 1.0e-4)), float(getattr(cfg, "fd_yaw_eps_rad", 1.0e-4)),
            bool(impact_normal_constraint_enabled(cfg)),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "build_collision_multifield_dual_lin_hk: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.build_collision_multifield_dual_lin_hk failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc

    out = np.asarray(out, dtype=float).reshape(-1, 21)
    if out.shape[0] != H or not np.all(np.isfinite(out)):
        msg = "invalid multifield collision output: shape={}, expected=({}, 21), finite={}".format(
            out.shape, H, bool(np.all(np.isfinite(out)))
        )
        globals()["_collision_lin_cpp_runtime_error"] = msg
        raise RuntimeError(msg + "; Python fallback is disabled")
    dual = np.ascontiguousarray(out[:, 0:14], dtype=float)
    alignment = np.ascontiguousarray(out[:, 14:21], dtype=float)
    if collision_field_flag(cfg) == 4 and impact_normal_constraint_enabled(cfg):
        _log_simplified_regulation_linearization(dual, cfg)
    if return_alignment:
        return dual, alignment
    return dual


def build_selected_mpc_collision_linearization_sequence(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
    return_alignment: bool = False,
):
    """Return H x 28 collision blocks for the selected mpc_flag.

    For mpc_flag=4, ``return_alignment=True`` also returns the H x 7 impact
    block; only column 0 contains (-grad Phi_k(0))^T n(phi_c).

    mpc_flag 0..3:
        four surface-wise residuals d_safe-d_surf,j.
    mpc_flag 4:
        block 0 = proposed minimum-distance residual,
        block 1 = proposed impact-location-conditioned relaxation residual,
        blocks 2..3 disabled.
    """
    flag = mpc_formulation_flag(cfg)
    if flag != 4:
        surface = build_surface_distance_linearization_sequence(
            X_bar, vertex_set, vertex_types, feature_obs_indices,
            h_steps, dt_vals, cfg,
        )
        if return_alignment:
            return surface, np.zeros((np.asarray(surface).reshape(-1, 28).shape[0], 7), dtype=float)
        return surface

    if return_alignment:
        dual, alignment = build_external_dual_collision_linearization_sequence(
            X_bar, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
            h_steps, dt_vals, cfg, sector_rays_body=sector_rays_body, return_alignment=True,
        )
        alignment = np.asarray(alignment, dtype=float).reshape(-1, 7)
    else:
        dual = build_external_dual_collision_linearization_sequence(
            X_bar, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
            h_steps, dt_vals, cfg, sector_rays_body=sector_rays_body, return_alignment=False,
        )
        alignment = None
    dual = np.asarray(dual, dtype=float).reshape(-1, 14)
    H = dual.shape[0]
    disabled = -float(getattr(cfg, "h_upper_bound", 1.0e15))
    out = np.zeros((H, 28), dtype=float)
    out[:, 0:14] = dual
    out[:, 14] = disabled
    out[:, 21] = disabled
    if return_alignment:
        return out, alignment
    return out


def evaluate_external_phi_sequence_fast(
    X_bar: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    h_steps: Sequence[int],
    dt_vals: Sequence[float],
    cfg: MPCConfig,
) -> np.ndarray:
    """Return Phi over the horizon through the unified C++ path.

    C-SDF/DCSAT/Wu/proposed use the ego body-fixed normalized geometry path.
    BP-SDF (field_flag=2) is intentionally evaluated in the fixed world frame
    from a reusable per-obstacle Bernstein fit.
    """
    X_arr = np.ascontiguousarray(X_bar, dtype=np.float64)
    X_stage = _stage_state_matrix_from_augmented(X_arr)
    H = int(X_stage.shape[0])
    if H <= 0:
        return np.zeros(0, dtype=float)

    vertex_set_arr = _prepare_collision_feature_tensor(vertex_set, H)
    vertex_normals_arr = _prepare_collision_feature_tensor(vertex_normals, H)
    has_vertex_types = vertex_types is not None
    vertex_types_arr = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = feature_obs_indices is not None
    feature_obs_indices_arr = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )

    cpp_vertex_set_arr = vertex_set_arr
    cpp_vertex_normals_arr = vertex_normals_arr
    cpp_vertex_types_arr = vertex_types_arr
    cpp_feature_obs_indices_arr = feature_obs_indices_arr
    cpp_has_vertex_types = bool(has_vertex_types)
    cpp_has_feature_obs_indices = bool(has_feature_obs_indices)
    cpp_vertex_capacity = int(cfg.vertex_capacity)
    if collision_field_flag(cfg) == 4:
        (
            cpp_vertex_set_arr, cpp_vertex_normals_arr, cpp_vertex_types_arr,
            cpp_feature_obs_indices_arr, cpp_has_vertex_types,
            cpp_has_feature_obs_indices, cpp_vertex_capacity,
        ) = _prepare_proposed_features_cpp(
            vertex_set_arr, vertex_types, feature_obs_indices, X_stage, cfg
        )
    h_arr = _stage_scalar_sequence(h_steps, H, 1.0)
    dt_arr = _stage_scalar_sequence(dt_vals, H, float(cfg.dt_init))
    axis_scale_arr = np.ascontiguousarray(_ego_axis_scale_array(cfg), dtype=np.float64).reshape(2)
    bp_W, bp_T_ref, bp_p_ref, bp_x_range, bp_y_range, bp_empty, bp_Q = _pack_bp_models_for_cpp(
        X_stage, vertex_set_arr, vertex_types, feature_obs_indices, h_arr, dt_arr, cfg
    )

    # field_flag=2 has an explicit model-evaluation path: after the world-frame
    # obstacle SDF has been sampled/fitted (or retrieved from cache), Phi evaluation
    # must not re-enter the generic obstacle-field dispatcher.
    if collision_field_flag(cfg) == 2:
        bp_model = BodyFixedBernsteinSDFModel(
            bp_W, bp_T_ref, bp_p_ref, bp_x_range, bp_y_range, bp_empty, bp_Q,
            float(getattr(cfg, "bp_safety_margin_l", 0.13)),
        )
        return evaluate_body_fixed_bp_sdf_world(X_stage, bp_model, cfg)

    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "build_collision_multifield_phi_seq"):
        raise RuntimeError("collision_lin_cpp.build_collision_multifield_phi_seq is required; Python fallback is disabled")
    try:
        phi = collision_lin_cpp.build_collision_multifield_phi_seq(
            X_arr, cpp_vertex_set_arr, cpp_vertex_normals_arr, cpp_vertex_types_arr, cpp_feature_obs_indices_arr,
            np.ascontiguousarray(h_arr, dtype=np.float64), np.ascontiguousarray(dt_arr, dtype=np.float64),
            axis_scale_arr, bool(cpp_has_vertex_types), bool(cpp_has_feature_obs_indices), int(cpp_vertex_capacity),
            float(cfg.dt_init), float(cfg.dt_init), float(cfg.dt_init), float(getattr(cfg, "h_upper_bound", 1.0e15)),
            collision_safe_distance(cfg), float(getattr(cfg, "ego_L", 0.6)), float(getattr(cfg, "ego_W", 0.4)), 0.0,
            bool(use_ego_scaled_local_frame(cfg)), str(getattr(cfg, "normal_orientation", "inward")),
            0.0, 0.0, collision_smoothing_lambda(cfg), collision_smoothing_lambda(cfg),
            float(getattr(cfg, "sigma_vx", 0.0)), float(getattr(cfg, "sigma_vy", 0.0)),
            float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(getattr(cfg, "use_probability_propagation_margin", False), False), 0.0,
            collision_field_flag(cfg), float(getattr(cfg, "circle_union_lambda", 0.0)),
            float(getattr(cfg, "dcsat_alpha", 35.0)), float(getattr(cfg, "dcsat_union_lambda", 0.02)),
            float(getattr(cfg, "wu_kappa", 35.0)), float(getattr(cfg, "wu_union_lambda", 0.02)),
            float(getattr(cfg, "wu_buffer_offset", 0.0)), bp_W, bp_T_ref, bp_p_ref, bp_x_range,
            bp_y_range, bp_empty, int(bp_Q), float(getattr(cfg, "bp_safety_margin_l", 0.13)),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "build_collision_multifield_phi_seq: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.build_collision_multifield_phi_seq failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc

    phi = np.asarray(phi, dtype=float).reshape(-1)
    if phi.size != H or not np.all(np.isfinite(phi)):
        msg = "invalid multifield Phi output: size={}, expected={}, finite={}".format(
            phi.size, H, bool(np.all(np.isfinite(phi)))
        )
        globals()["_collision_lin_cpp_runtime_error"] = msg
        raise RuntimeError(msg + "; Python fallback is disabled")
    return phi


def evaluate_external_phi_grid_cpp(
    X_grid: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
) -> np.ndarray:
    """Evaluate a current-time SDF/RViz query grid in one C++ call.

    This path is intentionally specialized for a dense visualization grid at
    h=0.  Unlike :func:`evaluate_external_phi_sequence_fast`, the obstacle
    geometry is *shared* by every query state.  In particular for field_flag=4
    the raw polygons are converted to edge/pseudo-vertex features only for H=1;
    collision_lin_cpp's FeatureAccessor broadcasts that single feature slice to
    all grid queries.  This avoids materializing an H x F x 2 feature tensor for
    a 101x101 (or larger) grid.

    field_flag=2 is excluded because the plotter first fits/reuses per-obstacle
    world-frame Bernstein models and then calls evaluate_body_fixed_bp_sdf_world.
    """
    X_arr = np.ascontiguousarray(X_grid, dtype=np.float64)
    X_stage = _stage_state_matrix_from_augmented(X_arr)
    H = int(X_stage.shape[0])
    if H <= 0:
        return np.zeros(0, dtype=np.float64)

    flag = collision_field_flag(cfg)
    if flag == 2:
        raise ValueError(
            "evaluate_external_phi_grid_cpp is not the Bernstein fitting path; "
            "use build_body_fixed_bp_sdf_model + evaluate_body_fixed_bp_sdf_world"
        )

    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "build_collision_multifield_phi_seq"):
        raise RuntimeError(
            "collision_lin_cpp.build_collision_multifield_phi_seq is required; "
            "rebuild the C++ extension"
        )

    # Grid visualization uses the same current obstacle geometry at every query
    # point.  Keep shared Kx2 geometry whenever possible instead of expanding it
    # along H.
    raw_vertex_set = _prepare_collision_feature_tensor(vertex_set, 1)
    raw_vertex_normals = _prepare_collision_feature_tensor(vertex_normals, 1)
    has_vertex_types = vertex_types is not None
    vertex_types_arr = (
        np.zeros(0, dtype=np.int32) if vertex_types is None
        else np.ascontiguousarray(vertex_types, dtype=np.int32).reshape(-1)
    )
    has_feature_obs_indices = feature_obs_indices is not None
    feature_obs_indices_arr = (
        np.zeros(0, dtype=np.int32) if feature_obs_indices is None
        else np.ascontiguousarray(feature_obs_indices, dtype=np.int32).reshape(-1)
    )

    cpp_vertex_set_arr = raw_vertex_set
    cpp_vertex_normals_arr = raw_vertex_normals
    cpp_vertex_types_arr = vertex_types_arr
    cpp_feature_obs_indices_arr = feature_obs_indices_arr
    cpp_has_vertex_types = bool(has_vertex_types)
    cpp_has_feature_obs_indices = bool(has_feature_obs_indices)
    cpp_vertex_capacity = int(cfg.vertex_capacity)

    if flag == 4:
        # Pseudo-normal directions depend on yaw but not on query position.
        # The visualization grid normally has one common yaw, so retain the
        # H==1 broadcast optimization in that common case.
        yaw_grid = np.arctan2(np.sin(X_stage[:, 3]), np.cos(X_stage[:, 3]))
        yaw_delta = np.arctan2(
            np.sin(yaw_grid - yaw_grid[0]), np.cos(yaw_grid - yaw_grid[0])
        )
        if yaw_grid.size <= 1 or np.max(np.abs(yaw_delta)) <= 1.0e-12:
            feature_X = X_stage[:1, :]
        else:
            feature_X = X_stage
        (
            cpp_vertex_set_arr, cpp_vertex_normals_arr, cpp_vertex_types_arr,
            cpp_feature_obs_indices_arr, cpp_has_vertex_types,
            cpp_has_feature_obs_indices, cpp_vertex_capacity,
        ) = _prepare_proposed_features_cpp(
            raw_vertex_set, vertex_types, feature_obs_indices, feature_X, cfg
        )

    axis_scale_arr = np.ascontiguousarray(
        _ego_axis_scale_array(cfg), dtype=np.float64
    ).reshape(2)

    # h=0 is common to the complete visualization grid.  A single scalar entry
    # is enough because the C++ accessor clamps every query to the last entry.
    h_arr = np.zeros(1, dtype=np.float64)
    dt_arr = np.array([max(float(cfg.dt_init), 1.0e-9)], dtype=np.float64)

    # Non-Bernstein fields only need ABI placeholders for the BP parameters.
    Q = max(int(getattr(cfg, "bp_order", 12)), 2)
    bp_W = np.zeros((1, Q, Q), dtype=np.float64)
    bp_T_ref = np.eye(2, dtype=np.float64).reshape(1, 2, 2)
    bp_p_ref = np.zeros((1, 2), dtype=np.float64)
    bp_x_range = np.array([[-1.0, 1.0]], dtype=np.float64)
    bp_y_range = bp_x_range.copy()
    bp_empty = np.ones(1, dtype=np.int32)

    try:
        phi = collision_lin_cpp.build_collision_multifield_phi_seq(
            np.ascontiguousarray(X_stage, dtype=np.float64),
            np.ascontiguousarray(cpp_vertex_set_arr, dtype=np.float64),
            np.ascontiguousarray(cpp_vertex_normals_arr, dtype=np.float64),
            np.ascontiguousarray(cpp_vertex_types_arr, dtype=np.int32),
            np.ascontiguousarray(cpp_feature_obs_indices_arr, dtype=np.int32),
            h_arr, dt_arr, axis_scale_arr,
            bool(cpp_has_vertex_types), bool(cpp_has_feature_obs_indices),
            int(cpp_vertex_capacity),
            float(cfg.dt_init), float(cfg.dt_init), float(cfg.dt_init),
            float(getattr(cfg, "h_upper_bound", 1.0e15)), collision_safe_distance(cfg),
            float(getattr(cfg, "ego_L", 0.6)),
            float(getattr(cfg, "ego_W", 0.4)),
            0.0,
            bool(use_ego_scaled_local_frame(cfg)),
            str(getattr(cfg, "normal_orientation", "inward")),
            0.0, 0.0,
            collision_smoothing_lambda(cfg), collision_smoothing_lambda(cfg),
            float(getattr(cfg, "sigma_vx", 0.0)),
            float(getattr(cfg, "sigma_vy", 0.0)),
            float(getattr(cfg, "probability_collision", 0.8)),
            _bool_compat(
                getattr(cfg, "use_probability_propagation_margin", False), False
            ),
            0.0,
            flag,
            float(getattr(cfg, "circle_union_lambda", 0.0)),
            float(getattr(cfg, "dcsat_alpha", 35.0)),
            float(getattr(cfg, "dcsat_union_lambda", 0.02)),
            float(getattr(cfg, "wu_kappa", 35.0)),
            float(getattr(cfg, "wu_union_lambda", 0.02)),
            float(getattr(cfg, "wu_buffer_offset", 0.0)),
            bp_W, bp_T_ref, bp_p_ref, bp_x_range, bp_y_range, bp_empty,
            int(Q), float(getattr(cfg, "bp_safety_margin_l", 0.13)),
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = (
            "build_collision_multifield_phi_seq(grid): " + repr(exc)
        )
        raise RuntimeError(
            "C++ SDF-grid evaluation failed: {!r}".format(exc)
        ) from exc

    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    if phi.size != H or not np.all(np.isfinite(phi)):
        raise RuntimeError(
            "invalid C++ SDF-grid output: size={}, expected={}, finite={}".format(
                int(phi.size), H, bool(np.all(np.isfinite(phi)))
            )
        )
    return phi


def _surface_distance_result_cache(
    result: PlanResult,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
) -> Optional[np.ndarray]:
    """Return one cached Hx4 exact-surface evaluation for a PlanResult.

    mpc_flag 0..3 use the same nonlinear surface distances for objective
    ranking, original-OCP feasibility, zero-slack diagnostics, and final
    metadata.  Computing them separately repeated the expensive exact
    ellipse-arc/polygon geometry several times per SQP candidate.
    """
    X_full = np.asarray(getattr(result, "X", np.zeros((4, 0))), dtype=float)
    if X_full.ndim != 2 or X_full.size == 0 or not np.all(np.isfinite(X_full)):
        return None
    X_core_all = _stage_state_matrix_from_augmented(X_full)
    if X_core_all.shape[0] <= 1:
        return None
    H_eval = min(int(getattr(cfg, "N", X_core_all.shape[0] - 1)), X_core_all.shape[0] - 1)
    if H_eval <= 0:
        return None

    dt_val = float(getattr(result, "dt", getattr(cfg, "dt_init", 0.1)))
    if not np.isfinite(dt_val) or dt_val <= 0.0:
        dt_val = float(getattr(cfg, "dt_init", 0.1))

    cached = getattr(result, "_surface_distance_cache", None)
    cached_h = int(getattr(result, "_surface_distance_cache_h", -1))
    cached_dt = float(getattr(result, "_surface_distance_cache_dt", float("nan")))
    if cached is not None and cached_h == H_eval and np.isfinite(cached_dt) and abs(cached_dt - dt_val) <= 1.0e-12:
        arr = np.asarray(cached, dtype=float)
        if arr.shape == (H_eval, 4) and np.all(np.isfinite(arr)):
            return arr

    X_stage = np.ascontiguousarray(X_core_all[1:H_eval + 1, :], dtype=float)
    h_steps = np.arange(1, H_eval + 1, dtype=float)
    dt_vals = np.full(H_eval, dt_val, dtype=float)
    dist = evaluate_surface_distance_sequence(
        X_stage, vertex_set, vertex_types, feature_obs_indices,
        h_steps, dt_vals, cfg,
    )
    dist = np.asarray(dist, dtype=float).reshape(H_eval, 4)
    if not np.all(np.isfinite(dist)):
        return None

    result._surface_distance_cache = dist
    result._surface_distance_cache_h = int(H_eval)
    result._surface_distance_cache_dt = float(dt_val)
    result.surface_distance = dist.copy()
    result.surface_distance_min = np.min(dist, axis=1)
    result.phi_min = float(np.min(dist))
    result.min_clearance = float(result.phi_min)
    return dist


def _proposed_phi_from_dual(dual_eval: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """Recover the proposed Phi sequence from the dual kernel intercept.

    For enabled stages the dual kernel stores ``g_a = -Phi_safe`` in column
    zero, because d_safe [m] is already applied as world-frame feature inflation.
    Empty/disabled stages use ``-h_upper_bound`` instead; preserve the
    legacy evaluator's ``+h_upper_bound`` convention for those rows.
    """
    dual = np.asarray(dual_eval, dtype=float).reshape(-1, 14)
    hub = float(getattr(cfg, "h_upper_bound", 1.0e15))
    phi = -dual[:, 0]
    disabled = dual[:, 0] <= (-0.5 * hub)
    if np.any(disabled):
        phi = phi.copy()
        phi[disabled] = hub
    return phi


def _proposed_alignment_from_dual(dual_eval: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """Recover ``N_k(0)^T n(phi_c)`` from the proposed Hx14 dual kernel.

    The native dual kernel stores the state-only regulation core

        gbar_r = lambda_r*log(2)
                 - lambda_r*LSE(Phi_safe/lambda_r,
                                (Ns-align)/lambda_r).

    The actual OCP residual is ``g_r = gbar_r - s^r <= 0``.  Since Phi_safe is stored through ``g_a=-Phi_safe``, the second LSE
    argument can be recovered by a stable log-difference.
    """
    dual = np.asarray(dual_eval, dtype=float).reshape(-1, 14)
    if dual.shape[0] == 0:
        return np.zeros(0, dtype=float)

    lam_r = max(float(regulation_smoothing_lambda(cfg)), 1.0e-12)
    Ns = float(regulation_alignment_Ns(cfg))
    log2 = math.log(2.0)

    phi = -dual[:, 0]
    gbar_r = dual[:, 7]
    za = phi / lam_r
    lse = log2 - gbar_r / lam_r
    gap = za - lse

    alignment = np.full(phi.shape, np.nan, dtype=float)
    valid = np.isfinite(phi) & np.isfinite(gbar_r) & np.isfinite(gap) & (gap < 0.0)
    if np.any(valid):
        exp_gap = np.exp(np.clip(gap[valid], -745.0, -1.0e-15))
        wb = 1.0 - exp_gap
        good = wb > 0.0
        zb = np.full(wb.shape, np.nan, dtype=float)
        zb[good] = lse[valid][good] + np.log(wb[good])
        recovered = Ns - lam_r * zb
        tmp = alignment[valid]
        tmp[good] = recovered[good]
        alignment[valid] = tmp

    valid_alignment = np.isfinite(alignment)
    return np.where(valid_alignment, np.clip(alignment, -1.0, 1.0), 0.0)


def _proposed_grad_q_phi_sector_dot_from_dual(
    dual_eval: np.ndarray,
    X_stage: np.ndarray,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Recover ``grad_q Phi_k(0)^T n(phi_c)`` from the proposed dual kernel.

    The dual kernel stores ``-dPhi/dx`` in columns 1:3.  For the ego-scaled
    local frame used by the proposed field,

        dPhi/dp = R(yaw) S^{-1} grad_q Phi,

    so ``grad_q Phi`` is recovered exactly as

        grad_q Phi = S R(yaw)^T dPhi/dp.

    The sector-center direction is expressed in the same q frame before taking
    the dot product.  Disabled/invalid stages contribute zero.
    """
    dual = np.asarray(dual_eval, dtype=float).reshape(-1, 14)
    X_arr = _stage_state_matrix_from_augmented(X_stage)
    H = min(int(dual.shape[0]), int(X_arr.shape[0]))
    if H <= 0:
        return np.zeros(0, dtype=float)

    dual = dual[:H, :]
    X_arr = X_arr[:H, :]
    dphi_dp = -np.asarray(dual[:, 1:3], dtype=float)

    if sector_rays_body is None:
        _, _, lc_body = sector_rays_body_from_cfg(cfg)
    else:
        rays = np.asarray(sector_rays_body, dtype=float).reshape(3, 2)
        lc_body = rays[2].copy()
    lc_body = np.asarray(lc_body, dtype=float).reshape(2)
    lc_body = lc_body / max(float(np.linalg.norm(lc_body)), 1.0e-12)

    yaw = np.asarray(X_arr[:, 3], dtype=float)
    c = np.cos(yaw)
    s = np.sin(yaw)

    if use_ego_scaled_local_frame(cfg):
        scale = _ego_axis_scale_array(cfg)
        # R^T dPhi/dp, followed by S, gives grad_q Phi in the scaled q frame.
        grad_body_x = c * dphi_dp[:, 0] + s * dphi_dp[:, 1]
        grad_body_y = -s * dphi_dp[:, 0] + c * dphi_dp[:, 1]
        grad_q = np.column_stack((
            float(scale[0]) * grad_body_x,
            float(scale[1]) * grad_body_y,
        ))

        # Match the sector-center representation used by the proposed-field
        # regulation constraint in the scaled q frame.
        lc_q = np.array([
            lc_body[0] / float(scale[0]),
            lc_body[1] / float(scale[1]),
        ], dtype=float)
        lc_q = lc_q / max(float(np.linalg.norm(lc_q)), 1.0e-12)
        score = grad_q @ lc_q
    else:
        lc_world = np.empty((H, 2), dtype=float)
        lc_world[:, 0] = c * lc_body[0] - s * lc_body[1]
        lc_world[:, 1] = s * lc_body[0] + c * lc_body[1]
        score = np.einsum("hi,hi->h", dphi_dp, lc_world)

    hub = float(getattr(cfg, "h_upper_bound", 1.0e15))
    valid = (
        np.all(np.isfinite(dphi_dp), axis=1)
        & np.isfinite(score)
        & (dual[:, 0] > (-0.5 * hub))
    )
    return np.where(valid, score, 0.0)



def _proposed_normal_alignment_from_dual_gradient(
    dual_eval: np.ndarray,
    X_stage: np.ndarray,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Return ``N_k(0)^T n(phi_c)`` directly from the proposed-field gradient.

    The dual kernel stores the avoidance residual Jacobian in columns 1:3,
    hence ``dPhi/dp = -dual[:,1:3]``.  For the ego-scaled local coordinates
    ``q = S^{-1} R^T (P-p)``, recover ``grad_q Phi`` by

        grad_q Phi = S R^T dPhi/dp,

    then use ``N_k(0) = -grad_q Phi / ||grad_q Phi||``.  This matches the
    obstacle-induced inward-normal convention used by the solver.  Recovering
    the alignment directly also avoids numerical loss of the non-dominant LSE
    branch when one argument is much larger than the other.
    """
    dual = np.asarray(dual_eval, dtype=float).reshape(-1, 14)
    X_arr = _stage_state_matrix_from_augmented(X_stage)
    H = min(int(dual.shape[0]), int(X_arr.shape[0]))
    if H <= 0:
        return np.zeros(0, dtype=float)

    dual = dual[:H, :]
    X_arr = X_arr[:H, :]
    dphi_dp = -np.asarray(dual[:, 1:3], dtype=float)

    if sector_rays_body is None:
        _, _, lc_body = sector_rays_body_from_cfg(cfg)
    else:
        rays = np.asarray(sector_rays_body, dtype=float).reshape(3, 2)
        lc_body = rays[2].copy()
    lc_body = np.asarray(lc_body, dtype=float).reshape(2)
    lc_body /= max(float(np.linalg.norm(lc_body)), 1.0e-12)

    yaw = np.asarray(X_arr[:, 3], dtype=float)
    c = np.cos(yaw)
    ss = np.sin(yaw)

    if use_ego_scaled_local_frame(cfg):
        scale = _ego_axis_scale_array(cfg)
        grad_body_x = c * dphi_dp[:, 0] + ss * dphi_dp[:, 1]
        grad_body_y = -ss * dphi_dp[:, 0] + c * dphi_dp[:, 1]
        grad_q = np.column_stack((
            float(scale[0]) * grad_body_x,
            float(scale[1]) * grad_body_y,
        ))

        lc_q = np.array([
            lc_body[0] / float(scale[0]),
            lc_body[1] / float(scale[1]),
        ], dtype=float)
        lc_q /= max(float(np.linalg.norm(lc_q)), 1.0e-12)
        center = np.repeat(lc_q.reshape(1, 2), H, axis=0)
    else:
        grad_q = dphi_dp
        center = np.empty((H, 2), dtype=float)
        center[:, 0] = c * lc_body[0] - ss * lc_body[1]
        center[:, 1] = ss * lc_body[0] + c * lc_body[1]

    gnorm = np.linalg.norm(grad_q, axis=1)
    eps_n = max(float(getattr(cfg, "normal_eps", 1.0e-9)), 1.0e-12)
    hub = float(getattr(cfg, "h_upper_bound", 1.0e15))
    valid = (
        np.all(np.isfinite(grad_q), axis=1)
        & np.all(np.isfinite(center), axis=1)
        & np.isfinite(gnorm)
        & (gnorm > eps_n)
        & (dual[:, 0] > (-0.5 * hub))
    )

    alignment = np.full(H, np.nan, dtype=float)
    if np.any(valid):
        N_q = -grad_q[valid, :] / gnorm[valid].reshape(-1, 1)
        alignment[valid] = np.einsum("hi,hi->h", N_q, center[valid, :])
    return alignment

def _proposed_dual_result_cache(
    result: PlanResult,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[np.ndarray]:
    """Return one cached Hx14 proposed-field evaluation for a PlanResult.

    The same nonlinear C++ pass also returns the direct impact-aligned quantity
    ``(-grad Phi_k(0))^T n(phi_c)`` in its third 7-value block. Cache that value
    alongside the dual block so J_global can use the actual candidate field
    evaluation without an additional kernel call.
    """
    if mpc_formulation_flag(cfg) != 4:
        return None

    X_full = np.asarray(getattr(result, "X", np.zeros((4, 0))), dtype=float)
    if X_full.ndim != 2 or X_full.size == 0 or not np.all(np.isfinite(X_full)):
        return None
    X_core_all = _stage_state_matrix_from_augmented(X_full)
    if X_core_all.shape[0] <= 1:
        return None
    H_eval = min(int(getattr(cfg, "N", X_core_all.shape[0] - 1)), X_core_all.shape[0] - 1)
    if H_eval <= 0:
        return None

    dt_val = float(getattr(result, "dt", getattr(cfg, "dt_init", 0.1)))
    if not np.isfinite(dt_val) or dt_val <= 0.0:
        dt_val = float(getattr(cfg, "dt_init", 0.1))

    cached = getattr(result, "_proposed_dual_cache", None)
    cached_h = int(getattr(result, "_proposed_dual_cache_h", -1))
    cached_dt = float(getattr(result, "_proposed_dual_cache_dt", float("nan")))
    cached_x = getattr(result, "_proposed_dual_cache_x", None)
    if (
        cached is not None
        and cached_h == H_eval
        and np.isfinite(cached_dt)
        and abs(cached_dt - dt_val) <= 1.0e-12
        and cached_x is not None
    ):
        arr = np.asarray(cached, dtype=float)
        impact_cached = getattr(result, "_proposed_impact_alignment_cache", None)
        impact_arr = None if impact_cached is None else np.asarray(impact_cached, dtype=float).reshape(-1)
        x_arr = np.asarray(cached_x, dtype=float)
        current_x = np.asarray(X_core_all[1:H_eval + 1, :], dtype=float)
        if (
            arr.shape == (H_eval, 14)
            and impact_arr is not None and impact_arr.shape == (H_eval,)
            and x_arr.shape == current_x.shape
            and np.array_equal(x_arr, current_x)
            and np.all(np.isfinite(arr))
            and np.all(np.isfinite(impact_arr))
        ):
            return arr

    X_stage = np.ascontiguousarray(X_core_all[1:H_eval + 1, :], dtype=float)
    h_steps = np.arange(1, H_eval + 1, dtype=float)
    dt_vals = np.full(H_eval, dt_val, dtype=float)
    dual, impact_block = build_external_dual_collision_linearization_sequence(
        X_stage, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
        h_steps, dt_vals, cfg, sector_rays_body=sector_rays_body, return_alignment=True,
    )
    dual = np.asarray(dual, dtype=float).reshape(H_eval, 14)
    impact_block = np.asarray(impact_block, dtype=float).reshape(H_eval, 7)
    impact_alignment = impact_block[:, 0].copy()
    if not (np.all(np.isfinite(dual)) and np.all(np.isfinite(impact_alignment))):
        return None

    result._proposed_dual_cache = dual
    result._proposed_impact_alignment_cache = impact_alignment
    result._proposed_dual_cache_h = int(H_eval)
    result._proposed_dual_cache_dt = float(dt_val)
    result._proposed_dual_cache_x = X_stage.copy()

    phi = _proposed_phi_from_dual(dual, cfg)
    result._proposed_phi_cache = phi
    result.phi_min = float(np.min(phi))
    result.min_clearance = float(result.phi_min)
    return dual


def _annotate_surface_mpc_feasibility(
    result: PlanResult,
    vertex_set: np.ndarray,
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
    zero_slack: bool,
) -> PlanResult:
    """Evaluate the nonlinear four-surface constraints used by mpc_flag 0..3."""
    if zero_slack:
        result.zero_slack_avoidance_residual_max = float("inf")
        result.zero_slack_normal_alignment_residual_max = float("-inf")
        result.zero_slack_feasible = False
    else:
        result.original_ocp_avoidance_residual_max = float("inf")
        result.original_ocp_slack_nonneg_residual_max = float("inf")
        result.original_ocp_mitigation_residual_max = float("-inf")
        result.original_ocp_feasible = False

    X_full = np.asarray(getattr(result, "X", np.zeros((4, 0))), dtype=float)
    if X_full.ndim != 2 or X_full.size == 0 or not np.all(np.isfinite(X_full)):
        return result
    X_core_all = _stage_state_matrix_from_augmented(X_full)
    if X_core_all.shape[0] <= 1:
        return result
    H_eval = min(int(getattr(cfg, "N", X_core_all.shape[0] - 1)), X_core_all.shape[0] - 1)
    if H_eval <= 0:
        return result

    dist = _surface_distance_result_cache(
        result, vertex_set, vertex_types, feature_obs_indices, cfg
    )
    if dist is None:
        return result

    solver_ok = bool(int(getattr(result, "exitflag", 0)) > 0)
    if not collision_avoidance_constraints_enabled(cfg):
        if zero_slack:
            result.zero_slack_avoidance_residual_max = float("-inf")
            result.zero_slack_feasible = True
        else:
            result.original_ocp_avoidance_residual_max = float("-inf")
            result.original_ocp_slack_nonneg_residual_max = float("-inf")
            result.original_ocp_feasible = solver_ok
            result.ocp_feasible = solver_ok
        return result

    ds = collision_safe_distance(cfg)
    tol = max(float(getattr(cfg, "acados_tol", 1.0e-6)), 1.0e-6)
    if zero_slack:
        residual = ds - dist
        rmax = float(np.max(residual))
        result.zero_slack_avoidance_residual_max = rmax
        result.zero_slack_feasible = bool(np.isfinite(rmax) and rmax <= tol)
        return result

    ss = np.zeros((4, H_eval), dtype=float)
    raw = np.asarray(getattr(result, "surface_slack", np.zeros((4, 0))), dtype=float)
    if raw.size:
        try:
            raw = raw.reshape(4, -1)
            count = min(H_eval, raw.shape[1])
            ss[:, :count] = raw[:, :count]
        except Exception:
            return result
    if not np.all(np.isfinite(ss)):
        return result

    residual = ds - dist - ss.T
    slack_nonneg = -ss
    result.original_ocp_avoidance_residual_max = float(np.max(residual))
    result.original_ocp_slack_nonneg_residual_max = float(np.max(slack_nonneg))
    feasible = bool(
        solver_ok
        and result.original_ocp_avoidance_residual_max <= tol
        and result.original_ocp_slack_nonneg_residual_max <= tol
    )
    # Surface-wise MPC has no impact-normal conditional constraint.  Its policy
    # is encoded entirely by slack bounds/costs (hard vulnerable surface for
    # flag 1, weighted penalties for flag 2, and lexicographic solves for flag 3).
    result.original_ocp_mitigation_residual_max = float("-inf")
    result.original_ocp_feasible = feasible
    result.ocp_feasible = feasible
    return result

def annotate_zero_slack_nonlinear_feasibility(
    result: PlanResult,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> PlanResult:
    """Evaluate candidate feasibility with only regulation slack ``s^r`` set to zero.

    The paper avoidance slack ``s^a_k = s_k`` remains active:

        g_a,k = -Phi_safe,k - s^a_k <= 0.

    The auxiliary regulation slack is removed for the feasibility test:

        g_r,k = lambda_r*log(2)
                - lambda_r*LSE(Phi_safe,k/lambda_r,
                               (Ns-N_k^T n(phi_c))/lambda_r) <= 0,

    i.e. ``s^r_k = 0``.  This is the feasibility flag used by ego_mover's
    paper selection rule.  ``phi_min`` is refreshed from the same nonlinear
    trajectory and is used by the no-feasible fallback:

        argmax_i min_k Phi_{i,k}.
    """
    if mpc_formulation_flag(cfg) != 4:
        return _annotate_surface_mpc_feasibility(
            result, vertex_set, vertex_types, feature_obs_indices, cfg, zero_slack=False
        )

    result.zero_slack_avoidance_residual_max = float("inf")
    result.zero_slack_normal_alignment_residual_max = float("inf")
    result.zero_slack_feasible = False

    X_full = np.asarray(getattr(result, "X", np.zeros((4, 0))), dtype=float)
    if X_full.ndim != 2 or X_full.size == 0 or not np.all(np.isfinite(X_full)):
        return result

    X_core_all = _stage_state_matrix_from_augmented(X_full)
    if X_core_all.shape[0] <= 1:
        return result
    H_eval = min(int(getattr(cfg, "N", X_core_all.shape[0] - 1)), X_core_all.shape[0] - 1)
    if H_eval <= 0:
        return result

    dual_eval = _proposed_dual_result_cache(
        result, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
        cfg, sector_rays_body=sector_rays_body,
    )
    if dual_eval is None:
        return result
    dual_eval = np.asarray(dual_eval, dtype=float).reshape(H_eval, 14)
    phi_eval = _proposed_phi_from_dual(dual_eval, cfg)
    if phi_eval.size > 0 and np.all(np.isfinite(phi_eval)):
        result.phi_min = float(np.min(phi_eval))
        result.min_clearance = float(result.phi_min)

    solver_ok = bool(int(getattr(result, "exitflag", 0)) > 0)
    if not collision_avoidance_constraints_enabled(cfg):
        result.zero_slack_avoidance_residual_max = float("-inf")
        result.zero_slack_normal_alignment_residual_max = float("-inf")
        result.zero_slack_feasible = solver_ok
        return result

    # Keep the optimized paper avoidance slack s^a_k (= s_k).
    s_a = np.zeros(H_eval, dtype=float)
    slack_arr = np.asarray(getattr(result, "slack", np.zeros(0)), dtype=float).reshape(-1)
    count = min(H_eval, slack_arr.size)
    if count > 0:
        s_a[:count] = slack_arr[:count]
    if not np.all(np.isfinite(s_a)):
        return result

    g_a = dual_eval[:, 0] - s_a
    g_a_max = float(np.max(g_a))
    result.zero_slack_avoidance_residual_max = g_a_max

    tol = max(float(getattr(cfg, "acados_tol", 1.0e-6)), 1.0e-6)
    feasible = bool(
        solver_ok
        and np.isfinite(g_a_max)
        and g_a_max <= tol
        and float(np.min(s_a)) >= -tol
    )

    if impact_normal_constraint_enabled(cfg):
        # Do NOT subtract result.normal_slack: zero slack means s^r_k = 0.
        g_r_max = float(np.max(dual_eval[:, 7]))
        result.zero_slack_normal_alignment_residual_max = g_r_max
        feasible = bool(feasible and np.isfinite(g_r_max) and g_r_max <= tol)
    else:
        result.zero_slack_normal_alignment_residual_max = float("-inf")

    result.zero_slack_feasible = bool(feasible)
    return result


def annotate_original_ocp_feasibility(
    result: PlanResult,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> PlanResult:
    """Evaluate candidate feasibility using the proposed field ``Phi_k(0), N_k(0)``.

    For mpc_flag=4 this check intentionally avoids the separate exact/true
    ellipse-arc distance and impact-normal geometry.  It reuses the nonlinear
    proposed-field evaluation already available through the Hx14 dual kernel:

        Phi_safe,k(0) >= -s_k,  s_k >= 0,
        N_k(0) not in S_E  when Phi_safe,k(0) <= 0.

    Here d_safe [m] is already included by world-frame inflation of each
    proposed-field support feature before body-fixed normalization.

    The vulnerable-sector test is performed directly with
    ``N_k(0)^T n(phi_c) <= cos(alpha + theta)``.  Thus no exact closest-surface search,
    exact-distance gradient, or true-impact-normal reconstruction is performed
    in this feasibility annotation.
    """
    if mpc_formulation_flag(cfg) != 4:
        return _annotate_surface_mpc_feasibility(
            result, vertex_set, vertex_types, feature_obs_indices, cfg, zero_slack=False
        )

    result.original_ocp_avoidance_residual_max = float("inf")
    result.original_ocp_slack_nonneg_residual_max = float("inf")
    result.original_ocp_mitigation_residual_max = float("inf")
    result.original_ocp_feasible = False

    X_full = np.asarray(getattr(result, "X", np.zeros((4, 0))), dtype=float)
    if X_full.ndim != 2 or X_full.size == 0 or not np.all(np.isfinite(X_full)):
        return result

    X_core_all = _stage_state_matrix_from_augmented(X_full)
    if X_core_all.shape[0] <= 1:
        return result
    H_eval = min(int(getattr(cfg, "N", X_core_all.shape[0] - 1)), X_core_all.shape[0] - 1)
    if H_eval <= 0:
        return result

    # Reuse the proposed-field nonlinear evaluation.  No exact surface-distance
    # or exact impact-normal geometry is evaluated here.
    dual_eval = _proposed_dual_result_cache(
        result, vertex_set, vertex_normals, vertex_types, feature_obs_indices,
        cfg, sector_rays_body=sector_rays_body,
    )
    if dual_eval is None:
        return result
    dual_eval = np.asarray(dual_eval, dtype=float).reshape(H_eval, 14)

    phi_eval = _proposed_phi_from_dual(dual_eval, cfg)
    if phi_eval.size != H_eval or not np.all(np.isfinite(phi_eval)):
        return result

    # Compatibility metadata: these legacy fields now contain the proposed-field
    # minimum rather than a separately computed true geometric clearance.
    result._proposed_phi_cache = phi_eval.copy()
    result.phi_min = float(np.min(phi_eval))
    result.min_clearance = float(result.phi_min)

    solver_ok = bool(int(getattr(result, "exitflag", 0)) > 0)
    tol = max(float(getattr(cfg, "acados_tol", 1.0e-6)), 1.0e-6)

    if not collision_avoidance_constraints_enabled(cfg):
        result.original_ocp_avoidance_residual_max = float("-inf")
        result.original_ocp_slack_nonneg_residual_max = float("-inf")
        result.original_ocp_mitigation_residual_max = float("-inf")
        result.original_ocp_feasible = solver_ok
        result.ocp_feasible = bool(result.original_ocp_feasible)
        return result

    s_a = np.zeros(H_eval, dtype=float)
    slack_arr = np.asarray(getattr(result, "slack", np.zeros(0)), dtype=float).reshape(-1)
    count = min(H_eval, slack_arr.size)
    if count > 0:
        s_a[:count] = slack_arr[:count]
    if not np.all(np.isfinite(s_a)):
        return result

    avoidance_residual = -phi_eval - s_a
    slack_nonneg_residual = -s_a
    result.original_ocp_avoidance_residual_max = float(np.max(avoidance_residual))
    result.original_ocp_slack_nonneg_residual_max = float(np.max(slack_nonneg_residual))

    feasible = bool(
        solver_ok
        and result.original_ocp_avoidance_residual_max <= tol
        and result.original_ocp_slack_nonneg_residual_max <= tol
    )

    if impact_normal_constraint_enabled(cfg):
        alpha = regulation_sector_half_width(cfg, sector_rays_body)
        theta = regulation_theta_value(cfg)
        vulnerable_threshold = math.cos(alpha + theta)
        close_indices = np.where(phi_eval <= tol)[0]

        if close_indices.size > 0:
            X_stage = np.ascontiguousarray(X_core_all[1:H_eval + 1, :], dtype=float)
            alignment_eval = _proposed_normal_alignment_from_dual_gradient(
                dual_eval, X_stage, cfg, sector_rays_body=sector_rays_body,
            )
            alignment_close = np.asarray(alignment_eval, dtype=float).reshape(-1)[close_indices]
            if alignment_close.size != close_indices.size or not np.all(np.isfinite(alignment_close)):
                # N_k is undefined when the proposed-field gradient magnitude is
                # numerically zero. Treat such an active stage conservatively.
                result.original_ocp_mitigation_residual_max = float("inf")
                feasible = False
            else:
                mitigation_residual = alignment_close - vulnerable_threshold
                result.original_ocp_mitigation_residual_max = float(np.max(mitigation_residual))
                feasible = bool(
                    feasible
                    and result.original_ocp_mitigation_residual_max <= tol
                )
        else:
            result.original_ocp_mitigation_residual_max = float("-inf")
    else:
        result.original_ocp_mitigation_residual_max = float("-inf")

    result.original_ocp_feasible = bool(feasible)
    result.ocp_feasible = bool(feasible)
    return result

def vertex_types_to_dynamic_mask(vertex_types: Sequence[int], vertex_capacity: int) -> np.ndarray:
    """Return a padded boolean mask for halfspace rows generated from vertices.

    True means the corresponding halfspace row belongs to a dynamic obstacle.
    The mask is retained only for result metadata / compatibility; collision
    for mpc_flag=4, d_safe is encoded as world-frame feature inflation before Phi_safe is evaluated.
    """
    M = int(vertex_capacity)
    mask = np.zeros(M, dtype=bool)
    if M <= 0:
        return mask
    arr = np.asarray(vertex_types if vertex_types is not None else [], dtype=np.int32).reshape(-1)
    count = min(M, arr.size)
    if count > 0:
        mask[:count] = arr[:count] == 1
    return mask

def vertex_set_to_halfspaces(
    vertex_set: np.ndarray,
    ego_pos: np.ndarray,
    vertex_capacity: int,
    buffer_m: float = 0.0,
    disabled_c_value: float = -1.0e9,
) -> np.ndarray:
    """Convert selected vertices to local separating half-space rows [A, B, C].

    For each selected vertex v_i, a safe-side normal is taken from the vertex
    toward the current ego position:
        n_i = normalize(ego_pos - v_i)
        n_i^T p - (n_i^T v_i + buffer) >= 0

    This is a local linearized collision-avoidance approximation. Rows beyond
    the selected vertex count are disabled with C=-1e9.
    """
    M = int(vertex_capacity)
    halfspaces = np.zeros((M, 3), dtype=float)
    halfspaces[:, 2] = float(disabled_c_value)

    V = np.asarray(vertex_set, dtype=float).reshape(-1, 2)
    ego = np.asarray(ego_pos, dtype=float).reshape(2)
    count = min(M, V.shape[0])
    for i in range(count):
        direction = ego - V[i]
        norm_direction = float(np.linalg.norm(direction))
        if norm_direction < 1.0e-9:
            continue
        n = direction / norm_direction
        C = float(n @ V[i] + max(float(buffer_m), 0.0))
        halfspaces[i, :] = [float(n[0]), float(n[1]), C]
    return halfspaces


def normalized_sector_id(sector_id: Any) -> int:
    try:
        sid = int(sector_id)
    except Exception:
        sid = 1
    if sid not in (1, 2, 3, 4):
        sid = 1
    return sid


def normalize_sector_dir(sector_dir: Any) -> str:
    text = str(sector_dir if sector_dir is not None else "outward").strip().lower()
    if text not in ("outward", "inward"):
        text = "outward"
    return text


def normalize_sector_config(cfg: MPCConfig) -> int:
    """Synchronize cfg.sector_id/designated_surface_id and sector_dir.

    Surface id mapping follows the MATLAB file:
        1: left, 2: rear, 3: right, 4: front.

    sector_dir controls the ray orientation:
        outward: keep la/lb as the original corner rays from the ego center.
        inward:  use -la/-lb, and therefore the center ray -lc.
    """
    sid = getattr(cfg, "sector_id", None)
    if sid is None:
        sid = getattr(cfg, "designated_surface_id", 1)
    sid = normalized_sector_id(sid)
    cfg.sector_id = sid
    cfg.designated_surface_id = sid
    cfg.sector_dir = normalize_sector_dir(getattr(cfg, "sector_dir", "outward"))
    return sid


def sector_rays_body_from_id(ego_L: float, ego_W: float, sector_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATLAB-consistent body-frame (l_a, l_b, l_c).

    The rays point from the ego center to the two corners of the selected
    surface.  l_c = normalize(l_a + l_b).
    Surface id mapping:
        1: left, 2: rear, 3: right, 4: front.
    """
    L = float(ego_L)
    W = float(ego_W)
    fl = np.array([ L / 2.0,  W / 2.0], dtype=float)
    fr = np.array([ L / 2.0, -W / 2.0], dtype=float)
    rr = np.array([-L / 2.0, -W / 2.0], dtype=float)
    rl = np.array([-L / 2.0,  W / 2.0], dtype=float)
    corners = {1: (fl, rl), 2: (rl, rr), 3: (rr, fr), 4: (fr, fl)}
    a_body, b_body = corners.get(normalized_sector_id(sector_id), corners[1])
    a_body = a_body / max(float(np.linalg.norm(a_body)), 1.0e-12)
    b_body = b_body / max(float(np.linalg.norm(b_body)), 1.0e-12)
    c_body = a_body + b_body
    c_body = c_body / max(float(np.linalg.norm(c_body)), 1.0e-12)
    return a_body, b_body, c_body


def sector_rays_body_from_cfg(cfg: MPCConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = normalize_sector_config(cfg)
    la, lb, lc = sector_rays_body_from_id(float(cfg.ego_L), float(cfg.ego_W), sid)
    if normalize_sector_dir(getattr(cfg, "sector_dir", "outward")) == "inward":
        la = -la
        lb = -lb
        lc = -lc
    return la, lb, lc


def sector_rays_to_parameter_vector(
    sector_rays_body: Optional[Sequence[Sequence[float]]],
    cfg: MPCConfig,
) -> np.ndarray:
    if sector_rays_body is None:
        la, lb, lc = sector_rays_body_from_cfg(cfg)
    else:
        arr = np.asarray(sector_rays_body, dtype=float)
        if arr.shape == (6,):
            arr = arr.reshape(3, 2)
        arr = arr.reshape(3, 2)
        la = arr[0].copy()
        lb = arr[1].copy()
        lc = arr[2].copy()
        la = la / max(float(np.linalg.norm(la)), 1.0e-12)
        lb = lb / max(float(np.linalg.norm(lb)), 1.0e-12)
        lc = lc / max(float(np.linalg.norm(lc)), 1.0e-12)
    return np.concatenate((la, lb, lc)).astype(float)


def build_parameter_vector(
    goal: np.ndarray,
    halfspaces: np.ndarray,
    cfg: MPCConfig,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    goal = np.asarray(goal, dtype=float).reshape(2)
    hs = np.asarray(halfspaces, dtype=float).reshape(-1, 3)
    if hs.shape[0] < cfg.vertex_capacity:
        pad = np.zeros((cfg.vertex_capacity - hs.shape[0], 3), dtype=float)
        pad[:, 2] = cfg.disabled_c_value
        hs = np.vstack((hs, pad))
    hs = hs[: cfg.vertex_capacity, :]
    sector_vec = sector_rays_to_parameter_vector(sector_rays_body, cfg)
    return np.concatenate((goal, hs.reshape(-1), sector_vec)).astype(float)




# -----------------------------------------------------------------------------
# Dynamics helper used by the ROS wrapper for one-step execution
# -----------------------------------------------------------------------------



def _as_bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def collision_avoidance_constraints_enabled(cfg: MPCConfig) -> bool:
    """Return whether the pair of smooth collision constraints is active.

    cfg.N is the actual acados horizon length used by this file, i.e., Np.
    When collsion_avoidance_off_over_20 is true, Np >= 20 disables the
    collision h-constraint and skips its external linearization.
    """
    avoidance_enabled = _as_bool_value(
        getattr(cfg, "enable_normal_mag_avoidance_constraint", True),
        True,
    )
    base_enabled = avoidance_enabled
    off_over_20 = _as_bool_value(
        getattr(cfg, "collsion_avoidance_off_over_20", None),
        _as_bool_value(getattr(cfg, "collision_avoidance_off_over_20", False), False),
    )
    try:
        np_horizon = int(getattr(cfg, "N", 0))
    except Exception:
        np_horizon = 0
    if off_over_20 and np_horizon >= 20:
        return False
    return bool(base_enabled)


def impact_normal_constraint_enabled(cfg: MPCConfig) -> bool:
    """Return whether the impact-normal alignment/regulation constraint is active."""
    return _as_bool_value(
        getattr(cfg, "enable_impact_normal_constraint", True),
        True,
    )


EGO_STATE_DIM = 5  # [px, py, v, yaw, yawrate]


def yawrate_time_constant(cfg: Any) -> float:
    """Return the first-order yaw-rate lag time constant tau_w [s] (> 0)."""
    tau = _finite_or_default(getattr(cfg, "yawrate_time_constant", 0.7), 0.7)
    return max(float(tau), 1.0e-3)


def augment_ego_state(x: np.ndarray, cfg: Any, yawrate0: Optional[float] = None) -> np.ndarray:
    """Return the five-state MPC vector [px, py, v, yaw, yawrate].

    A five-element input is returned as-is.  For a four-element pose/speed
    state the actual yaw rate is taken from ``yawrate0`` when given, otherwise
    from ``cfg.ego_yawrate0`` (refreshed by the ROS planners every cycle).
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size >= EGO_STATE_DIM:
        return arr[:EGO_STATE_DIM].copy()
    if arr.size != 4:
        raise ValueError("ego state must have 4 or 5 entries, got {}".format(arr.size))
    if yawrate0 is None:
        yawrate0 = _finite_or_default(getattr(cfg, "ego_yawrate0", 0.0), 0.0)
    return np.array([arr[0], arr[1], arr[2], arr[3], float(yawrate0)], dtype=float)


def build_linearized_dynamics_sequence(
    X_bar: np.ndarray,
    U_bar: np.ndarray,
    dt_vals: Sequence[float],
    cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """C++ RK4 first-order linearization of the yaw-rate-lag kinematics.

    State x=[px,py,v,yaw,w], input u=[w_cmd,a] with w'=(w_cmd-w)/tau_w.
    Returns ``(x_next_bar, A, B, F)`` with shapes Hx5, Hx5x5, Hx5x2, Hx5.
    The same coefficients serve two paths.  With nonlinear dynamics they are
    used only to compose the externally linearized collision residual at
    x_{k+1} with (x_k, delta_u_k).  With ``nonlinear_dyn_flag=false`` they are
    also packed into the acados parameter vector and define the affine shooting
    model x_{k+1}=Fbar_k+A_k(x_k-xbar_k)+B_k delta_u_k.
    """
    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "build_dynamics_lag_lin_seq"):
        raise RuntimeError(
            "collision_lin_cpp.build_dynamics_lag_lin_seq is required; rebuild with build_collision_lin_cpp.sh"
        )
    nx = EGO_STATE_DIM
    X = np.ascontiguousarray(np.asarray(X_bar, dtype=np.float64).reshape(-1, nx))
    U = np.ascontiguousarray(np.asarray(U_bar, dtype=np.float64).reshape(-1, 2))
    if X.shape[0] != U.shape[0]:
        raise ValueError("X_bar and U_bar must have the same horizon length")
    dt = np.ascontiguousarray(np.asarray(dt_vals, dtype=np.float64).reshape(-1))
    try:
        x_next, A, B, F = collision_lin_cpp.build_dynamics_lag_lin_seq(
            X, U, dt, float(cfg.v_min), float(cfg.v_max), yawrate_time_constant(cfg)
        )
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "build_dynamics_lag_lin_seq: " + repr(exc)
        raise RuntimeError("C++ dynamics linearization failed: {!r}".format(exc)) from exc
    H = X.shape[0]
    x_next = np.asarray(x_next, dtype=float).reshape(H, nx)
    A = np.asarray(A, dtype=float).reshape(H, nx, nx)
    B = np.asarray(B, dtype=float).reshape(H, nx, 2)
    F = np.asarray(F, dtype=float).reshape(H, nx)
    if not (np.all(np.isfinite(x_next)) and np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(F))):
        raise RuntimeError("C++ dynamics linearization returned non-finite coefficients")
    return x_next, A, B, F


def compose_collision_linearization_through_dynamics(
    collision_lin_seq: np.ndarray,
    collision_state_nominal: np.ndarray,
    dynamics_next_nominal: np.ndarray,
    dynamics_A: np.ndarray,
    dynamics_B: np.ndarray,
) -> np.ndarray:
    """Compose g(x_{k+1}) linearization with the RK4 dynamics Jacobian.

    ``collision_lin_seq`` contains four 7-value blocks per stage.  Before this
    composition each block is interpreted as

        [g_bar, dg/dx_next (4: px,py,v,yaw), auxiliary_0, auxiliary_1].

    The returned block has 8 values,

        [g_stage_bar, dg/dx_k (5: px,py,v,yaw,yawrate), dg/d(delta_u_k) (2)],

    where ``delta_u_k`` is [delta_yawrate_cmd, delta_acc].  The collision
    geometry does not depend on the yaw-rate state directly, but x_{k+1} does
    through the lag dynamics, so dg/dyawrate_k is generally nonzero.

    The constant correction ``dg/dx_next @ (F_bar-x_next_bar)`` preserves the
    original first-order model even when a supplied warm-start trajectory is
    not exactly dynamically consistent.
    """
    nx = EGO_STATE_DIM
    raw = np.asarray(collision_lin_seq, dtype=float).reshape(-1, 4, 7)
    H = int(raw.shape[0])
    x_col = np.asarray(collision_state_nominal, dtype=float).reshape(H, 4)
    x_dyn = np.asarray(dynamics_next_nominal, dtype=float).reshape(H, nx)
    A = np.asarray(dynamics_A, dtype=float).reshape(H, nx, nx)
    B = np.asarray(dynamics_B, dtype=float).reshape(H, nx, 2)

    out = np.zeros((H, 4, 1 + nx + 2), dtype=float)
    grad_next = raw[:, :, 1:5]
    dyn_mismatch = x_dyn[:, :4] - x_col

    out[:, :, 0] = raw[:, :, 0] + np.einsum(
        "hbi,hi->hb", grad_next, dyn_mismatch
    )
    out[:, :, 1:1 + nx] = np.einsum("hbi,hij->hbj", grad_next, A[:, :4, :])
    out[:, :, 1 + nx:3 + nx] = np.einsum("hbi,hij->hbj", grad_next, B[:, :4, :])

    if not np.all(np.isfinite(out)):
        raise RuntimeError("composed collision/dynamics linearization is non-finite")
    return out.reshape(H, 4 * (1 + nx + 2))


def step_ego_dynamics(x: np.ndarray, u: np.ndarray, dt: float, cfg: MPCConfig) -> np.ndarray:
    """One RK4 step.

    A five-state input [px,py,v,yaw,yawrate] is propagated with the
    first-order yaw-rate lag (tau_w = cfg.yawrate_time_constant) and a
    five-state result is returned.  A legacy four-state input keeps the
    original behaviour (u[0] is applied directly as the yaw rate) so the
    ego_mover simulators, which model the actuator lag themselves, are
    unaffected.
    """
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    lag = x.size >= EGO_STATE_DIM
    x = x[:EGO_STATE_DIM] if lag else x.reshape(4)
    u = np.asarray(u, dtype=float).reshape(2).copy()
    u[0] = min(max(float(u[0]), cfg.yawrate_min), cfg.yawrate_max)
    u[1] = min(max(float(u[1]), cfg.a_min), cfg.a_max)
    tau = yawrate_time_constant(cfg)

    def f(xl: np.ndarray) -> np.ndarray:
        v = min(max(float(xl[2]), cfg.v_min), cfg.v_max)
        yaw = float(xl[3])
        if lag:
            w = float(xl[4])
            return np.array([v * math.cos(yaw), v * math.sin(yaw), u[1], w, (u[0] - w) / tau], dtype=float)
        return np.array([v * math.cos(yaw), v * math.sin(yaw), u[1], u[0]], dtype=float)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[2] = min(max(float(x_next[2]), cfg.v_min), cfg.v_max)
    x_next[3] = wrap_to_pi(float(x_next[3]))
    return x_next


def rollout_ego(
    x0: np.ndarray,
    U: np.ndarray,
    dt: float,
    cfg: MPCConfig,
    yawrate0: Optional[float] = None,
    return_full_state: bool = False,
) -> np.ndarray:
    """Roll out the yaw-rate-lag RK4 horizon in the mandatory C++ kernel.

    ``x0`` may be [px,py,v,yaw] (actual yaw rate from ``yawrate0`` or
    ``cfg.ego_yawrate0``) or [px,py,v,yaw,yawrate].  ``U[0]`` is the commanded
    yaw rate, tracked with w'=(w_cmd-w)/cfg.yawrate_time_constant.

    Returns the 4 x (N+1) [px,py,v,yaw] trajectory used by all planners and
    messages, or the 5 x (N+1) solver state when ``return_full_state=True``.

    The horizon is dynamically sequential, so stages cannot be evaluated independently.
    Python rollout fallback is disabled to make accelerator/build failures explicit.
    """
    nx = EGO_STATE_DIM
    U_arr = np.asarray(U, dtype=float).reshape(2, -1)
    x0_arr = augment_ego_state(x0, cfg, yawrate0)
    N = int(U_arr.shape[1])

    require_cpp_acceleration()
    if not hasattr(collision_lin_cpp, "rollout_ego_lag_seq"):
        raise RuntimeError(
            "collision_lin_cpp.rollout_ego_lag_seq is required; rebuild with build_collision_lin_cpp.sh"
        )
    try:
        U_cpp = np.ascontiguousarray(U_arr.T, dtype=np.float64)
        dt_cpp = np.asarray([float(dt)], dtype=np.float64)
        X_cpp = collision_lin_cpp.rollout_ego_lag_seq(
            np.ascontiguousarray(x0_arr, dtype=np.float64),
            U_cpp, dt_cpp,
            float(cfg.yawrate_min), float(cfg.yawrate_max),
            float(cfg.a_min), float(cfg.a_max),
            float(cfg.v_min), float(cfg.v_max),
            yawrate_time_constant(cfg),
        )
        X_cpp = np.asarray(X_cpp, dtype=float).reshape(N + 1, nx)
    except Exception as exc:
        globals()["_collision_lin_cpp_runtime_error"] = "rollout_ego_lag_seq: " + repr(exc)
        raise RuntimeError(
            "collision_lin_cpp.rollout_ego_lag_seq failed; Python fallback is disabled: {!r}".format(exc)
        ) from exc
    if not np.all(np.isfinite(X_cpp)):
        msg = "rollout_ego_lag_seq returned non-finite values"
        globals()["_collision_lin_cpp_runtime_error"] = msg
        raise RuntimeError(msg + "; Python fallback is disabled")
    if return_full_state:
        return X_cpp.T
    return X_cpp[:, :4].T.copy()


def build_bspline_tracking_nominal(
    x0: np.ndarray,
    bspline_states: np.ndarray,
    dt: float,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy compatibility helper; solve() no longer calls guidance tracking.

    If called externally, this helper still builds the old dynamically feasible
    tracking nominal. It is intentionally outside the active MPC solve path.

    ``bspline_states`` may be 2/3/4/6 x M or its transpose.  Position is
    required; speed and yaw are used when supplied and otherwise reconstructed
    from the B-spline tangent.  The returned arrays have shapes ``2 x N`` and
    ``4 x (N+1)``.  Controls are bounded before the nonlinear RK4 rollout, so
    the returned state sequence is consistent with the exact control sequence
    used as the Stage-2 linearization point.
    """
    N = int(cfg.N)
    x0_arr = augment_ego_state(x0, cfg)
    x0_arr[3] = wrap_to_pi(float(x0_arr[3]))
    dt_nom = float(cfg.dt_init)

    ref = np.asarray(bspline_states, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.ndim != 2:
        raise ValueError("bspline_states must be a two-dimensional state sequence")
    if ref.shape[0] not in (2, 3, 4, 6, 7) and ref.shape[1] in (2, 3, 4, 6, 7):
        ref = ref.T
    if ref.shape[0] < 2 or ref.shape[1] < 1:
        raise ValueError("bspline_states must contain at least x/y samples")

    # Resample the guidance over the MPC horizon instead of padding its last
    # sample.  This keeps the Stage-1 tracker well-defined when the B-spline
    # sampler and the OCP use different numbers of nodes.
    src_s = np.linspace(0.0, 1.0, int(ref.shape[1]))
    dst_s = np.linspace(0.0, 1.0, N + 1)
    pos_ref = np.vstack((
        np.interp(dst_s, src_s, ref[0, :]),
        np.interp(dst_s, src_s, ref[1, :]),
    ))
    pos_ref[:, 0] = x0_arr[:2]

    seg = np.diff(pos_ref, axis=1)
    seg_len = np.linalg.norm(seg, axis=0)
    tangent_yaw = np.empty(N + 1, dtype=float)
    fallback_yaw = float(x0_arr[3])
    for k in range(N):
        if seg_len[k] > 1.0e-9:
            fallback_yaw = math.atan2(float(seg[1, k]), float(seg[0, k]))
        tangent_yaw[k] = fallback_yaw
    tangent_yaw[N] = tangent_yaw[N - 1] if N > 0 else fallback_yaw

    if ref.shape[0] >= 4:
        yaw_src = np.unwrap(np.asarray(ref[3, :], dtype=float))
        yaw_interp = np.interp(dst_s, src_s, yaw_src)
        valid_yaw = np.isfinite(yaw_interp)
        tangent_yaw[valid_yaw] = yaw_interp[valid_yaw]
    tangent_yaw = np.unwrap(tangent_yaw)

    speed_ref = np.zeros(N + 1, dtype=float)
    if ref.shape[0] >= 3:
        speed_src = np.asarray(ref[2, :], dtype=float)
        speed_interp = np.interp(dst_s, src_s, speed_src)
        if np.all(np.isfinite(speed_interp)):
            speed_ref[:] = speed_interp
        else:
            speed_ref[:N] = seg_len / dt_nom
    else:
        speed_ref[:N] = seg_len / dt_nom
    if N > 0 and (ref.shape[0] < 3 or not math.isfinite(float(speed_ref[N]))):
        speed_ref[N] = speed_ref[N - 1]
    speed_ref = np.clip(speed_ref, float(cfg.v_min), float(cfg.v_max))

    kp_pos = max(_finite_or_default(getattr(cfg, "stage1_position_gain", 1.5), 1.5), 0.0)
    kp_yaw = max(_finite_or_default(getattr(cfg, "stage1_heading_gain", 2.0), 2.0), 0.0)
    kp_vel = max(_finite_or_default(getattr(cfg, "stage1_velocity_gain", 2.0), 2.0), 0.0)

    U_track = np.zeros((2, N), dtype=float)
    X_track = np.zeros((EGO_STATE_DIM, N + 1), dtype=float)
    X_track[:, 0] = x0_arr
    for k in range(N):
        xk = X_track[:, k]
        pos_error = pos_ref[:, k + 1] - xk[:2]
        if float(np.linalg.norm(pos_error)) > 1.0e-9:
            heading_target = math.atan2(float(pos_error[1]), float(pos_error[0]))
        else:
            heading_target = float(tangent_yaw[k])

        yaw_ff = wrap_to_pi(float(tangent_yaw[k + 1] - tangent_yaw[k])) / dt_nom
        yaw_error = wrap_to_pi(heading_target - float(xk[3]))
        yawrate = yaw_ff + kp_yaw * yaw_error

        tangent = np.array([math.cos(float(tangent_yaw[k])), math.sin(float(tangent_yaw[k]))])
        along_error = float(np.dot(pos_error, tangent))
        speed_cmd = float(speed_ref[k + 1]) + kp_pos * along_error
        acc_ff = (float(speed_ref[k + 1]) - float(speed_ref[k])) / dt_nom
        acceleration = acc_ff + kp_vel * (speed_cmd - float(xk[2]))

        U_track[0, k] = np.clip(yawrate, float(cfg.yawrate_min), float(cfg.yawrate_max))
        if _bool_compat(getattr(cfg, "no_acc_flag", False), False):
            U_track[1, k] = 0.0
        else:
            U_track[1, k] = np.clip(acceleration, float(cfg.a_min), float(cfg.a_max))
        X_track[:, k + 1] = step_ego_dynamics(xk, U_track[:, k], dt_nom, cfg)

    # Use the mandatory horizon kernel for the authoritative nominal.  The
    # sequential Python loop above only computes feedback controls.
    X_track = rollout_ego(x0_arr, U_track, dt_nom, cfg)
    return U_track, X_track

def _finite_or_default(value: float, default: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value




def construct_acados_solver(cfg: MPCConfig, *_legacy_cfg: Any):
    """Build the compact fixed-N, fixed-dt acados OCP.

    ``nonlinear_dyn_flag=true`` keeps the nonlinear RK4 discrete map.
    ``nonlinear_dyn_flag=false`` uses a parameterized affine RK4 linearization;
    together with LINEAR_LS and affine path rows this is a convex QP per outer
    refinement iteration.

    mpc_flag=4:
        x = [px, py, v, yaw, yawrate]   (yawrate' = (yawrate_cmd - yawrate)/tau_w)
        u = [delta_yawrate, delta_acc, s^a, s^r] when impact-normal regulation is enabled
        u = [delta_yawrate, delta_acc, s^a] when it is disabled

    mpc_flag=0..3 keep four surface slacks, hence six inputs total. There is no
    dt state, slack-budget state, boundary slack, or s^3/s^4 in flag 4.
    """
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except Exception as exc:
        raise ImportError(
            "acados_template and casadi are required to construct the acados solver. "
            "Source your acados environment before running MPC_planner1.py."
        ) from exc

    if cfg is None:
        cfg = MPCConfig()

    enable_boundary_constraint = _as_bool_value(
        getattr(cfg, "enable_boundary_constraint", False), False
    )
    use_direct_collision_linearization = use_external_collision_linearization(cfg)
    formulation_flag = mpc_formulation_flag(cfg)
    proposed = formulation_flag == 4

    nx = EGO_STATE_DIM
    nx_param = EGO_STATE_DIM
    collision_state_nx = 4
    tau_w = yawrate_time_constant(cfg)
    # Each externally built collision row is affine in the current shooting
    # state x_k (5 variables) and the two physical control increments.  Slack
    # variables are appended exactly as -s_j when the complete residual
    # g_j = gbar_j - s_j is assembled below, so they are not part of this
    # native state/control gradient block.
    collision_decision_dim = nx + 2
    active_slack_count = proposed_active_slack_count(cfg) if proposed else 4
    nu = 2 + active_slack_count
    collision_block_count = active_slack_count

    nh = int(cfg.vertex_capacity)
    sector_np = 6
    param_nh = 0 if use_direct_collision_linearization else nh
    halfspace_np = 3 * param_nh
    sector_offset = halfspace_np
    base_np = halfspace_np + sector_np

    x_param_offset = base_np
    u_param_offset = x_param_offset + nx_param
    collision_state_offset = u_param_offset + nu
    collision_lin_np = collision_block_count * (1 + collision_decision_dim)
    collision_lin_offset = collision_state_offset + collision_state_nx
    linearized_dynamics = not nonlinear_dynamics_enabled(cfg)
    dyn_A_np = nx * nx if linearized_dynamics else 0
    dyn_B_np = nx * 2 if linearized_dynamics else 0
    dyn_F_np = nx if linearized_dynamics else 0
    dyn_A_offset = collision_lin_offset + collision_lin_np
    dyn_B_offset = dyn_A_offset + dyn_A_np
    dyn_F_offset = dyn_B_offset + dyn_B_np
    # J_local is pure LINEAR_LS.  No local rho_impact/alignment-cost or
    # stage-yaw-reference parameter block is needed.
    np_param = dyn_F_offset + dyn_F_np

    px = ca.SX.sym("px")
    py = ca.SX.sym("py")
    v = ca.SX.sym("v")
    yaw = ca.SX.sym("yaw")
    yawrate = ca.SX.sym("yawrate")
    x = ca.vertcat(px, py, v, yaw, yawrate)

    delta_yawrate = ca.SX.sym("delta_yawrate")
    delta_acc = ca.SX.sym("delta_acc")
    slack_syms = [ca.SX.sym("slack_{}".format(j + 1)) for j in range(active_slack_count)]
    collision_slacks = ca.vertcat(*slack_syms)
    u = ca.vertcat(delta_yawrate, delta_acc, collision_slacks)

    p = ca.SX.sym("p", np_param)
    x_bar = p[x_param_offset: x_param_offset + nx_param]
    u_bar = p[u_param_offset: u_param_offset + nu]

    yawrate_abs = u_bar[0] + delta_yawrate
    acc_abs = u_bar[1] + delta_acc
    dt_fixed = float(cfg.dt_init)
    x_core = ca.vertcat(px, py, v, yaw, yawrate)

    if linearized_dynamics:
        # Exact first-order affine model around the current nominal stage:
        #   x+ = Fbar + A (x-xbar) + B delta_u.
        # A, B, and Fbar are updated through p at every outer refinement.
        A_dyn = ca.reshape(p[dyn_A_offset:dyn_A_offset + dyn_A_np], nx, nx)
        B_dyn = ca.reshape(p[dyn_B_offset:dyn_B_offset + dyn_B_np], nx, 2)
        F_dyn = p[dyn_F_offset:dyn_F_offset + dyn_F_np]
        delta_u_core = ca.vertcat(delta_yawrate, delta_acc)
        disc_dyn_expr = F_dyn + ca.mtimes(A_dyn, x_core - x_bar) + ca.mtimes(B_dyn, delta_u_core)
    else:
        def nonlinear_core_rhs(xi):
            # yawrate_abs is the commanded yaw rate; the actual yaw rate xi[4]
            # follows it through a first-order lag with time constant tau_w.
            return ca.vertcat(
                xi[2] * ca.cos(xi[3]),
                xi[2] * ca.sin(xi[3]),
                acc_abs,
                xi[4],
                (yawrate_abs - xi[4]) / tau_w,
            )

        k1 = nonlinear_core_rhs(x_core)
        k2 = nonlinear_core_rhs(x_core + 0.5 * dt_fixed * k1)
        k3 = nonlinear_core_rhs(x_core + 0.5 * dt_fixed * k2)
        k4 = nonlinear_core_rhs(x_core + dt_fixed * k3)
        core_next = x_core + (dt_fixed / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        yaw_next = ca.atan2(ca.sin(core_next[3]), ca.cos(core_next[3]))
        disc_dyn_expr = ca.vertcat(core_next[0], core_next[1], core_next[2], yaw_next, core_next[4])

    collision_enabled = collision_avoidance_constraints_enabled(cfg) and nh > 0
    if collision_enabled and not use_direct_collision_linearization:
        raise ValueError(
            "mpc_flag={} requires use_external_collision_linearization=true".format(formulation_flag)
        )

    if collision_enabled:
        collision_terms = []
        # Collision residuals are evaluated at the next physical state, but
        # their first-order models have already been composed with the RK4
        # dynamics Jacobian outside acados.  Consequently each row below is
        # affine in the stage variables.  When nonlinear_dyn_flag=false the
        # shooting dynamics are affine as well, so the full subproblem is QP.
        collision_stage_delta = ca.vertcat(
            x - x_bar,
            delta_yawrate,
            delta_acc,
        )
        for j in range(collision_block_count):
            off_j = collision_lin_offset + j * (1 + collision_decision_dim)
            g_bar_j = p[off_j]
            grad_g_j = p[off_j + 1: off_j + 1 + collision_decision_dim]
            # Native rows contain only the state/control core.  Define the
            # complete OCP residual here so the slack is part of g_j itself:
            #   g_a = gbar_a - s^a <= 0,
            #   g_r = gbar_r - s^r <= 0.
            gbar_lin_j = g_bar_j + ca.dot(grad_g_j, collision_stage_delta)
            g_full_lin_j = gbar_lin_j - collision_slacks[j]
            collision_terms.append(g_full_lin_j)
        collision_h_expr = ca.vertcat(*collision_terms)
    else:
        collision_h_expr = ca.SX.zeros(0, 1)

    # Optional map bounds are hard.  No b_k decision variable exists.
    if enable_boundary_constraint:
        boundary_h_expr = ca.vertcat(
            float(cfg.x_min) - px,
            px - float(cfg.x_max),
            float(cfg.y_min) - py,
            py - float(cfg.y_max),
        )
    else:
        boundary_h_expr = ca.SX.zeros(0, 1)

    h_expr = ca.vertcat(collision_h_expr, boundary_h_expr)
    terminal_h_expr = boundary_h_expr

    model = AcadosModel()
    model.name = str(cfg.acados_model_name)
    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = disc_dyn_expr
    model.con_h_expr = h_expr
    model.con_h_expr_e = terminal_h_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = int(cfg.N)

    # Local objective: all formulations use the same pure LINEAR_LS structure.
    # For mpc_flag=4 this is tracking/control + active slack penalties only;
    # no extra sequence-tracking or rho_impact impact terms are present in the OCP.
    ny = 4 + nu
    ny_e = 4
    Vx = np.zeros((ny, nx), dtype=float)
    Vu = np.zeros((ny, nu), dtype=float)
    Vx[0:4, 0:4] = np.eye(4, dtype=float)
    Vu[4:4 + nu, 0:nu] = np.eye(nu, dtype=float)
    Vx_e = np.zeros((ny_e, nx), dtype=float)
    Vx_e[0:4, 0:4] = np.eye(4, dtype=float)

    slack_weights = surface_slack_cost_weights(cfg)
    active_slack_weights = slack_weights[:active_slack_count]

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e
    ocp.cost.W = np.diag([
        float(cfg.q_pos), float(cfg.q_pos), float(cfg.q_v), float(cfg.q_yaw),
        float(cfg.r_yawrate), float(cfg.r_a),
        *[float(w) for w in active_slack_weights],
    ])
    ocp.cost.W_e = np.diag([
        float(cfg.q_pos), float(cfg.q_pos), float(cfg.q_v), float(cfg.q_yaw)
    ])
    ocp.cost.yref = np.zeros(ny, dtype=float)
    ocp.cost.yref_e = np.zeros(ny_e, dtype=float)

    # State and input bounds.  Every formulation has the same five states [px,py,v,yaw,yawrate].
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int64)
    ocp.constraints.lbx_0 = np.array([
        cfg.ego_x, cfg.ego_y, cfg.ego_v0, cfg.ego_yaw0,
        _finite_or_default(getattr(cfg, "ego_yawrate0", 0.0), 0.0),
    ], dtype=float)
    ocp.constraints.ubx_0 = ocp.constraints.lbx_0.copy()
    ocp.constraints.idxbx = np.array([2], dtype=np.int64)
    ocp.constraints.lbx = np.array([cfg.v_min], dtype=float)
    ocp.constraints.ubx = np.array([cfg.v_max], dtype=float)
    ocp.constraints.idxbx_e = np.array([2], dtype=np.int64)
    ocp.constraints.lbx_e = np.array([cfg.v_min], dtype=float)
    ocp.constraints.ubx_e = np.array([cfg.v_max], dtype=float)

    slack_ub_vec = surface_slack_upper_bounds(cfg)
    ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)
    lbu = np.zeros(nu, dtype=float)
    ubu = np.zeros(nu, dtype=float)
    lbu[0] = float(cfg.yawrate_min)
    lbu[1] = float(cfg.a_min)
    ubu[0] = float(cfg.yawrate_max)
    ubu[1] = float(cfg.a_max)
    ubu[2:] = slack_ub_vec[:active_slack_count]
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu

    nh_collision = collision_block_count if collision_enabled else 0
    nh_boundary = 4 if enable_boundary_constraint else 0
    nh_path = nh_collision + nh_boundary
    nh_terminal = nh_boundary
    ocp.constraints.lh = -float(cfg.h_upper_bound) * np.ones(nh_path, dtype=float)
    ocp.constraints.uh = np.zeros(nh_path, dtype=float)
    ocp.constraints.lh_e = -float(cfg.h_upper_bound) * np.ones(nh_terminal, dtype=float)
    ocp.constraints.uh_e = np.zeros(nh_terminal, dtype=float)

    default_xbar = np.array([cfg.ego_x, cfg.ego_y, cfg.ego_v0, cfg.ego_yaw0, 0.0], dtype=float)
    default_ubar = np.zeros(nu, dtype=float)
    default_p = np.zeros(np_param, dtype=float)
    if param_nh > 0:
        default_hs = np.zeros((param_nh, 3), dtype=float)
        default_hs[:, 2] = cfg.disabled_c_value
        default_p[0:3 * param_nh] = default_hs.reshape(-1)
    default_p[sector_offset:sector_offset + sector_np] = sector_rays_to_parameter_vector(None, cfg)
    default_p[x_param_offset:x_param_offset + nx_param] = default_xbar
    default_p[u_param_offset:u_param_offset + nu] = default_ubar
    default_p[collision_state_offset:collision_state_offset + collision_state_nx] = default_xbar[:collision_state_nx]
    for j in range(collision_block_count):
        default_p[collision_lin_offset + j * (1 + collision_decision_dim)] = -float(
            getattr(cfg, "h_upper_bound", 1.0e15)
        )
    if linearized_dynamics:
        # Safe construction-time affine map. Runtime coefficients overwrite
        # these at every stage before solve(). CasADi reshape is column-major.
        default_p[dyn_A_offset:dyn_A_offset + dyn_A_np] = np.eye(nx, dtype=float).reshape(-1, order="F")
        default_p[dyn_B_offset:dyn_B_offset + dyn_B_np] = np.zeros(nx * 2, dtype=float)
        default_p[dyn_F_offset:dyn_F_offset + dyn_F_np] = default_xbar
    ocp.parameter_values = default_p

    ocp.solver_options.qp_solver = str(cfg.acados_qp_solver)
    ocp.solver_options.qp_solver_iter_max = int(cfg.acados_qp_solver_iter_max)
    ocp.solver_options.hessian_approx = str(cfg.acados_hessian_approx)
    ocp.solver_options.integrator_type = "DISCRETE"

    # acados exposes HPIPM through the OCP-QP backend.  For the affine mode,
    # SQP_RTI is intentionally used as a one-shot QP driver: there is no
    # nonlinear remainder inside the frozen subproblem.
    solver_nlp_type = str(cfg.acados_nlp_solver_type) if not linearized_dynamics else "SQP_RTI"
    ocp.solver_options.nlp_solver_type = solver_nlp_type
    ocp.solver_options.sim_method_num_stages = int(cfg.acados_sim_method_num_stages)
    ocp.solver_options.sim_method_num_steps = int(cfg.acados_sim_method_num_steps)
    # SQP_RTI performs one real-time iteration per solve() call.  Keep the
    # generated solver at one NLP iteration and let AcadosMPCPlanner repeat
    # solve() explicitly while the external collision linearization is frozen.
    # This makes cfg.acados_nlp_solver_max_iter mean exactly the requested inner
    # RTI-step budget per outer re-linearization.
    if solver_nlp_type.strip().upper() == "SQP_RTI":
        ocp.solver_options.nlp_solver_max_iter = 1
    else:
        ocp.solver_options.nlp_solver_max_iter = int(cfg.acados_nlp_solver_max_iter)
    ocp.solver_options.tol = float(cfg.acados_tol)
    ocp.solver_options.tf = float(cfg.N)
    try:
        ocp.solver_options.fixed_hess = int(
            _bool_compat(getattr(cfg, "acados_fixed_hess", False), False)
        )
    except Exception:
        pass
    try:
        ocp.solver_options.qp_solver_warm_start = int(
            getattr(cfg, "acados_qp_solver_warm_start", 1)
        )
    except Exception:
        pass

    root = Path(__file__).resolve().parent
    json_file = root / str(cfg.acados_json_file)
    ocp.code_export_directory = str(root / str(cfg.acados_code_export_dir))

    solver = AcadosOcpSolver(ocp, json_file=str(json_file))
    if _bool_compat(getattr(cfg, "use_acados_cython_wrapper", True), True):
        try:
            rebuild_cython = _bool_compat(getattr(cfg, "acados_rebuild_solver", False), False)
            if rebuild_cython:
                AcadosOcpSolver.build(
                    str(ocp.code_export_directory), with_cython=True, verbose=False,
                )
                cython_solver = AcadosOcpSolver.create_cython_solver(str(json_file))
            else:
                try:
                    cython_solver = AcadosOcpSolver.create_cython_solver(str(json_file))
                except Exception:
                    AcadosOcpSolver.build(
                        str(ocp.code_export_directory), with_cython=True, verbose=False,
                    )
                    cython_solver = AcadosOcpSolver.create_cython_solver(str(json_file))
            solver = cython_solver
        except Exception:
            pass
    return solver, ocp


class AcadosMPCPlanner:
    """Runtime wrapper for selectable MPC formulations and collision fields.

    mpc_flag=4 uses the same scalarized phase-0 objective on every outer-SQP
    collision-refinement iteration.  The mpc_flag=3 surface-wise lexicographic
    baseline is retained unchanged.
    """

    def __init__(self, cfg: Optional[MPCConfig] = None, *_legacy_cfg: Any):
        self.cfg = cfg if cfg is not None else MPCConfig()
        
        
        
        normalize_sector_config(self.cfg)
        effective_pseudo_normal_count(self.cfg)
        force_delta_sqp_solver_names(self.cfg)
        try:
            cpp_threads = max(1, int(getattr(self.cfg, "cpp_omp_threads", 2)))
        except Exception:
            cpp_threads = 2
        if hasattr(collision_lin_cpp, "set_omp_threads"):
            collision_lin_cpp.set_omp_threads(cpp_threads)
        self.cpp_omp_threads = cpp_threads
        self.dt = float(self.cfg.dt_init)  # fixed sampling period metadata; not a decision variable
        self.solver, self.ocp = construct_acados_solver(self.cfg)
        # Remember unsupported flat-I/O fields after the first failure so an
        # older acados version does not pay repeated exception overhead.
        self._flat_io_unsupported = set()
        self.U_warm = self._make_bias_control(int(getattr(self.cfg, "ORDER", 1)))
        self.X_warm = self._make_augmented_warm_start(
            np.array([self.cfg.ego_x, self.cfg.ego_y, self.cfg.ego_v0, self.cfg.ego_yaw0], dtype=float),
            self.U_warm,
        )

        # Latest single dual-constraint solve result.
        self.last_result = None
        # Successful outer-SQP iterates generated in the latest solve.
        # Feasibility masking and J_global evaluation are intentionally owned by
        # MPC_planner0~6, not by utils_linearized.
        self.last_iteration_results = []
        # Legacy compatibility field. It is no longer populated by solve().
        self.last_feasible_candidates = []

        # Exact collision affine model frozen for the currently solved outer-SQP
        # iteration.  These are copied onto each PlanResult immediately after the
        # solve so RViz can visualize the *same* separating hyperplane that acados
        # actually used, rather than re-evaluating/re-linearizing Phi later.
        self._active_collision_nominal = None
        self._active_collision_lin_seq = None

    def _set_flat_if_supported(self, field: str, values: np.ndarray) -> bool:
        """Set all stages in one acados call when the installed API supports it."""
        key = ("set", str(field))
        if not _bool_compat(getattr(self.cfg, "use_acados_flat_io", True), True):
            return False
        if key in self._flat_io_unsupported:
            return False
        setter = getattr(self.solver, "set_flat", None)
        if setter is None:
            self._flat_io_unsupported.add(key)
            return False
        try:
            setter(str(field), np.ascontiguousarray(values, dtype=float).reshape(-1))
            return True
        except Exception:
            self._flat_io_unsupported.add(key)
            return False

    def _get_flat_if_supported(self, field: str, expected_size: int) -> Optional[np.ndarray]:
        """Read all stages in one acados call, with a stage-wise fallback."""
        key = ("get", str(field))
        if not _bool_compat(getattr(self.cfg, "use_acados_flat_io", True), True):
            return None
        if key in self._flat_io_unsupported:
            return None
        getter = getattr(self.solver, "get_flat", None)
        if getter is None:
            self._flat_io_unsupported.add(key)
            return None
        try:
            out = np.asarray(getter(str(field)), dtype=float).reshape(-1)
            if out.size != int(expected_size):
                raise ValueError("unexpected flat acados field size")
            return out
        except Exception:
            self._flat_io_unsupported.add(key)
            return None

    def _solver_state_dim(self) -> int:
        return EGO_STATE_DIM

    def _solver_input_dim(self) -> int:
        if mpc_formulation_flag(self.cfg) == 4:
            return 2 + proposed_active_slack_count(self.cfg)
        return 6

    def _set_runtime_cost_profile(self, lexicographic_phase: int = 0) -> None:
        """Update LINEAR_LS weights without rebuilding the acados solver.

        mpc_flag=4 always uses the phase-0 scalarized objective.  The phase
        argument is honored only by the mpc_flag=3 lexicographic surface-wise
        baseline.
        """
        cfg = self.cfg
        flag = mpc_formulation_flag(cfg)
        # mpc_flag=4 now also uses LINEAR_LS; only mpc_flag=3 changes weights
        # across lexicographic phases.
        phase = int(lexicographic_phase) if flag == 3 else 0
        slack_w = surface_slack_cost_weights(cfg, lexicographic_phase=phase)
        nu = self._solver_input_dim()
        ny = 4 + nu

        if flag == 3 and phase == 1:
            diag_path = np.zeros(ny, dtype=float)
            diag_path[6:10] = slack_w
            diag_terminal = np.zeros(4, dtype=float)
        else:
            active_slack_count = proposed_active_slack_count(cfg) if flag == 4 else 4
            diag_path = np.array([
                float(cfg.q_pos), float(cfg.q_pos), float(cfg.q_v), float(cfg.q_yaw),
                float(cfg.r_yawrate), float(cfg.r_a),
                *[float(x) for x in slack_w[:active_slack_count]],
            ], dtype=float)
            diag_terminal = np.array([
                float(cfg.q_pos), float(cfg.q_pos), float(cfg.q_v), float(cfg.q_yaw)
            ], dtype=float)

        setter = getattr(self.solver, "cost_set", None)
        if setter is None:
            raise RuntimeError("Runtime MPC cost update requires acados solver.cost_set()")
        W = np.diag(diag_path)
        W_e = np.diag(diag_terminal)
        for k in range(int(cfg.N)):
            setter(k, "W", W)
        setter(int(cfg.N), "W", W_e)



    def _finalize_returned_candidate(
        self,
        result: PlanResult,
    ) -> PlanResult:
        """Attach the cheap feasibility flag and ranking cost to one SQP iterate.

        For the proposed OCP, a successful candidate is selection-feasible iff
        every auxiliary regulation slack s^r_k is zero up to the solver tolerance.
        No nonsmooth/exact-distance feasibility re-evaluation is performed here.

        The returned ``result.cost`` is the local MPC objective, including the
        rho_s/rho_c-weighted quadratic slack cost but no rho_impact impact-alignment term.
        J_global is evaluated separately in ``solve()`` and includes
        +rho_impact (-grad Phi_k(0))^T n(phi_c). It is stored in
        ``result.J_global`` / ``result.global_cost``.
        """
        cfg = self.cfg
        solver_ok = bool(
            int(getattr(result, "exitflag", 0)) > 0
            or bool(getattr(result, "maxiter_recovered", False))
        )
        flag = mpc_formulation_flag(cfg)

        if flag == 4:
            tol = max(float(getattr(cfg, "acados_tol", 1.0e-6)), 1.0e-6)
            if impact_normal_constraint_enabled(cfg):
                s_r = np.asarray(getattr(result, "normal_slack", np.zeros(0)), dtype=float).reshape(-1)
                sr_max = float(np.max(np.abs(s_r))) if s_r.size else 0.0
                feasible = bool(solver_ok and np.isfinite(sr_max) and sr_max <= tol)
                result.zero_slack_normal_alignment_residual_max = sr_max
            else:
                feasible = solver_ok
                result.zero_slack_normal_alignment_residual_max = 0.0

            # Compatibility diagnostics: feasibility is intentionally based only
            # on the auxiliary regulation slack, so no exact-distance residual is
            # evaluated. Keep legacy residual fields finite for downstream ROS
            # message definitions that still expose them.
            result.zero_slack_avoidance_residual_max = 0.0
            result.avoidance_residual_max = 0.0
            result.normal_alignment_residual_max = float(result.zero_slack_normal_alignment_residual_max)
            result.original_ocp_avoidance_residual_max = 0.0
            result.original_ocp_slack_nonneg_residual_max = 0.0
            result.original_ocp_mitigation_residual_max = float(result.zero_slack_normal_alignment_residual_max)
            result.zero_slack_feasible = feasible
            result.selection_eligible = feasible
            result.ocp_feasible = feasible
            result.paper_feasible = feasible
            result.original_ocp_feasible = feasible
            result.nonsmooth_ocp_feasible = feasible

            # acados LINEAR_LS uses the conventional 1/2 scaling, so doubling
            # recovers the quadratic J_local exactly for the proposed planner.
            raw_solver_cost = float(getattr(result, "cost", float("nan")))
            if np.isfinite(raw_solver_cost):
                # Path and terminal costs are both LINEAR_LS.
                result.cost = float(2.0 * raw_solver_cost)
            else:
                result.cost = float("inf")
        else:
            # Surface-wise baselines do not use s^r as an auxiliary regulation
            # slack. Keep their solver feasibility/cost unchanged.
            feasible = solver_ok
            result.zero_slack_feasible = feasible
            result.selection_eligible = feasible
            result.ocp_feasible = feasible
            result.paper_feasible = feasible
            result.original_ocp_feasible = feasible
            result.nonsmooth_ocp_feasible = feasible

        return result


    def _fix_lexicographic_vulnerable_slack(self, U_bar: np.ndarray, s_v_star: np.ndarray) -> None:
        """Fix stage-wise vulnerable slack to the phase-1 optimum for phase 2."""
        cfg = self.cfg
        N = int(cfg.N)
        U_nom = np.asarray(U_bar, dtype=float).reshape(6, N)
        yaw_nom = np.clip(U_nom[0, :], float(cfg.yawrate_min), float(cfg.yawrate_max))
        acc_nom = np.clip(U_nom[1, :], float(cfg.a_min), float(cfg.a_max))
        ub_slack = surface_slack_upper_bounds(cfg)
        v_idx = vulnerable_surface_index(cfg)
        s_star = np.asarray(s_v_star, dtype=float).reshape(-1)
        if s_star.size < N:
            s_star = np.pad(s_star, (0, N - s_star.size), mode="edge") if s_star.size else np.zeros(N)
        s_star = np.clip(s_star[:N], 0.0, float(ub_slack[v_idx]))
        tol = max(float(getattr(cfg, "lexicographic_fix_tolerance", 1.0e-7)), 0.0)
        for k in range(N):
            lbu = np.zeros(6, dtype=float)
            ubu = np.zeros(6, dtype=float)
            lbu[0] = float(cfg.yawrate_min) - yaw_nom[k]
            lbu[1] = float(cfg.a_min) - acc_nom[k]
            ubu[0] = float(cfg.yawrate_max) - yaw_nom[k]
            ubu[1] = float(cfg.a_max) - acc_nom[k]
            ubu[2:6] = ub_slack
            idx = 2 + v_idx
            # A tiny one-sided numerical tolerance avoids making HPIPM reject a
            # phase-1 optimum because of floating-point roundoff, while still
            # preventing any meaningful worsening of the vulnerable violation.
            lbu[idx] = max(0.0, s_star[k] - tol)
            ubu[idx] = min(float(ub_slack[v_idx]), s_star[k] + tol)
            self.solver.set(k, "lbu", lbu)
            self.solver.set(k, "ubu", ubu)



    def _make_bias_control(self, order: Optional[int] = None) -> np.ndarray:
        """Return U_bar for the requested ORDER bias.

        ORDER mapping:
            1 -> acc = 0, yawrate = 0
            2 -> acc = 0, yawrate = yawrate_max
            3 -> acc = 0, yawrate = yawrate_min
        """
        cfg = self.cfg
        N = int(cfg.N)
        try:
            order_i = int(order if order is not None else getattr(cfg, "ORDER", 1))
        except Exception:
            order_i = 1

        if order_i == 2:
            yawrate_bias = float(cfg.yawrate_max)
        elif order_i == 3:
            yawrate_bias = float(cfg.yawrate_min)
        else:
            yawrate_bias = 0.0

        yawrate_bias = min(max(yawrate_bias, float(cfg.yawrate_min)), float(cfg.yawrate_max))
        acc_bias = 0.0
        acc_bias = min(max(acc_bias, float(cfg.a_min)), float(cfg.a_max))

        U = np.zeros((6, N), dtype=float)
        U[0, :] = yawrate_bias
        U[1, :] = acc_bias
        U[2, :] = 0.0
        U[3, :] = 0.0
        return U

    def _make_augmented_warm_start(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return the fixed-dt five-state (yaw-rate lag) rollout used as SQP warm start."""
        cfg = self.cfg
        N = int(cfg.N)
        U_arr = np.asarray(U, dtype=float).reshape(6, N)
        return rollout_ego(
            augment_ego_state(x0, cfg), U_arr[:2, :], float(cfg.dt_init), cfg,
            return_full_state=True,
        )


    def reset_cycle_nominal(self, x0: np.ndarray, order: Optional[int] = None) -> None:
        """Reset X_bar/U_bar for a new fixed-dt planning cycle.

        The next cycle must not reuse the previous solution as a shifted
        warm-start.  ORDER alone defines the initial nominal input sequence:
            1 -> yawrate=0, 2 -> yawrate_max, 3 -> yawrate_min, acc=0.
        """
        cfg = self.cfg
        try:
            order_i = int(order if order is not None else getattr(cfg, "ORDER", 1))
        except Exception:
            order_i = 1
        self.U_warm = self._make_bias_control(order_i)
        self.X_warm = self._make_augmented_warm_start(augment_ego_state(x0, cfg), self.U_warm)

    def _coerce_initial_nominal(
        self,
        x0: np.ndarray,
        initial_U: Optional[np.ndarray],
        initial_X: Optional[np.ndarray],
        initial_dt: Optional[float],
        order: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert an external guidance nominal to compact solver dimensions.

        ``initial_dt`` is ignored because acados uses fixed ``cfg.dt_init``.
        ``initial_X`` is intentionally ignored for SQP linearization/warm start:
        X_bar is always rolled out from x0 and the coerced nominal control U_aug.
        Guidance states can still be supplied independently through reference_states.
        """
        if initial_U is None and initial_X is None:
            return None

        cfg = self.cfg
        N = int(cfg.N)
        x0_arr = augment_ego_state(x0, cfg)
        U_aug = self._make_bias_control(order)
        if initial_U is not None:
            try:
                U_arr = np.asarray(initial_U, dtype=float)
                if U_arr.ndim == 1:
                    U_arr = U_arr.reshape(1, -1)
                if U_arr.shape[0] not in (2, 3, 4, 5, 6) and U_arr.shape[1] in (2, 3, 4, 5, 6):
                    U_arr = U_arr.T
                rows = min(int(U_arr.shape[0]), 6)
                cols = min(int(U_arr.shape[1]), N)
                if rows >= 2 and cols > 0:
                    U_aug[:rows, :cols] = U_arr[:rows, :cols]
                    if cols < N:
                        U_aug[:rows, cols:] = U_aug[:rows, cols - 1:cols]
            except Exception:
                pass
        U_aug[0, :] = np.clip(U_aug[0, :], float(cfg.yawrate_min), float(cfg.yawrate_max))
        U_aug[1, :] = np.clip(U_aug[1, :], float(cfg.a_min), float(cfg.a_max))
        if bool(getattr(cfg, "no_acc_flag", False)):
            U_aug[1, :] = 0.0
        U_aug[2:6, :] = np.maximum(U_aug[2:6, :], 0.0)

        # Linearization/warm-start states are always generated from the nominal
        # control sequence.  A supplied initial_X may still be passed separately
        # as reference_states by the ROS wrapper, but it must never define X_bar.
        # This keeps the nominal state dynamically consistent with (x0, U_nom).
        X_aug = self._make_augmented_warm_start(x0_arr, U_aug)

        return U_aug, X_aug


    def _extract_solution(self, status: int, vertex_set_arr: np.ndarray, halfspaces_arr: np.ndarray, U_bar: np.ndarray, dynamic_halfspace_mask: Optional[np.ndarray] = None, vertex_types: Optional[np.ndarray] = None) -> PlanResult:
        cfg = self.cfg
        N = int(cfg.N)
        state_dim = self._solver_state_dim()
        solver_nu = self._solver_input_dim()
        X_aug = np.zeros((state_dim, N + 1), dtype=float)
        U_delta = np.zeros((solver_nu, N), dtype=float)
        U_nom = np.asarray(U_bar, dtype=float).reshape(6, N).copy()

        x_flat = self._get_flat_if_supported("x", (N + 1) * state_dim)
        if x_flat is not None:
            X_aug[:, :] = x_flat.reshape(N + 1, state_dim).T
        else:
            for k in range(N + 1):
                X_aug[:, k] = np.asarray(self.solver.get(k, "x"), dtype=float).reshape(state_dim)

        u_flat = self._get_flat_if_supported("u", N * solver_nu)
        if u_flat is not None:
            U_delta[:, :] = u_flat.reshape(N, solver_nu).T
        else:
            for k in range(N):
                U_delta[:, k] = np.asarray(self.solver.get(k, "u"), dtype=float).reshape(solver_nu)

        X_aug[3, :] = np.arctan2(np.sin(X_aug[3, :]), np.cos(X_aug[3, :]))
        X = X_aug[:4, :].copy()
        U = U_nom[:2, :].copy() + U_delta[:2, :]
        U[0, :] = np.clip(U[0, :], float(cfg.yawrate_min), float(cfg.yawrate_max))
        U[1, :] = np.clip(U[1, :], float(cfg.a_min), float(cfg.a_max))
        if bool(getattr(cfg, "no_acc_flag", False)):
            U[1, :] = 0.0

        surface_slack = np.zeros((4, N), dtype=float)
        if mpc_formulation_flag(cfg) == 4:
            active_slack_count = proposed_active_slack_count(cfg)
            surface_slack[0:active_slack_count, :] = U_delta[2:2 + active_slack_count, :]
        else:
            surface_slack[:, :] = U_delta[2:6, :]
        slack = surface_slack[0, :].copy()
        normal_slack = surface_slack[1, :].copy()
        opt_dt = float(cfg.dt_init)

        try:
            cost = float(self.solver.get_cost())
        except Exception:
            cost = float("nan")

        result = PlanResult(
            U=U,
            X=X,
            dt=opt_dt,
            exitflag=(1 if int(status) == 0 else -int(status)),
            cost=cost,
            vertex_set=vertex_set_arr,
            halfspaces=halfspaces_arr,
            slack=slack,
            normal_slack=normal_slack,
            surface_slack=surface_slack,
            boundary_slack=0.0,
            vertex_types=vertex_types,
            dynamic_halfspace_mask=dynamic_halfspace_mask,
        )
        if mpc_formulation_flag(cfg) == 4:

            # Preserve the exact outer-SQP affine model used for this candidate.
            # Copying is essential because the planner will overwrite its active
            # linearization on the next SQP iteration.
            active_nominal = getattr(self, "_active_collision_nominal", None)
            active_lin = getattr(self, "_active_collision_lin_seq", None)
            if active_nominal is not None:
                result.collision_nominal = np.asarray(
                    active_nominal, dtype=float
                ).reshape(N, 4).copy()
            if active_lin is not None:
                result.collision_linearization = np.asarray(
                    active_lin, dtype=float
                ).reshape(N, 4, 7).copy()
                # Proposed block 0: g_a = d_safe - Phi.
                result.avoidance_linearization = result.collision_linearization[:, 0, :].copy()
        return result


    def _make_fallback_result(
        self,
        x0: np.ndarray,
        order: int,
        vertex_set_arr: np.ndarray,
        halfspaces_arr: np.ndarray,
        dynamic_halfspace_mask: Optional[np.ndarray] = None,
        vertex_types: Optional[np.ndarray] = None,
        fallback_U_bar: Optional[np.ndarray] = None,
        fallback_X_bar: Optional[np.ndarray] = None,
    ) -> PlanResult:
        cfg = self.cfg
        N = int(cfg.N)
        if fallback_U_bar is None:
            U_aug = self._make_bias_control(order)
        else:
            U_aug = np.asarray(fallback_U_bar, dtype=float).reshape(6, N).copy()

        if fallback_X_bar is None:
            X_aug = self._make_augmented_warm_start(x0, U_aug)
        else:
            state_dim = self._solver_state_dim()
            X_in = np.asarray(fallback_X_bar, dtype=float).reshape(-1, N + 1)
            X_aug = np.zeros((state_dim, N + 1), dtype=float)
            rows = min(state_dim, int(X_in.shape[0]))
            X_aug[:rows, :] = X_in[:rows, :]
            X_aug[:, 0] = augment_ego_state(x0, cfg)
            X_aug[3, :] = np.arctan2(np.sin(X_aug[3, :]), np.cos(X_aug[3, :]))
        return PlanResult(
            U=U_aug[:2, :].copy(),
            X=X_aug[:4, :].copy(),
            dt=float(cfg.dt_init),
            exitflag=-1,
            cost=float("nan"),
            vertex_set=vertex_set_arr,
            halfspaces=halfspaces_arr,
            slack=U_aug[2, :].copy(),
            normal_slack=U_aug[3, :].copy(),
            surface_slack=U_aug[2:6, :].copy(),
            boundary_slack=0.0,
            vertex_types=vertex_types,
            dynamic_halfspace_mask=dynamic_halfspace_mask,
        )



    def _delta_warm_start_from_nominal(self, U_bar: np.ndarray) -> np.ndarray:
        """Return solver-sized delta-control warm start."""
        cfg = self.cfg
        N = int(cfg.N)
        U_bar = np.asarray(U_bar, dtype=float).reshape(6, N)
        nu = self._solver_input_dim()
        U_delta = np.zeros((nu, N), dtype=float)
        if mpc_formulation_flag(cfg) == 4:
            active_slack_count = proposed_active_slack_count(cfg)
            U_delta[2:2 + active_slack_count, :] = np.maximum(
                U_bar[2:2 + active_slack_count, :], 0.0
            )
        else:
            U_delta[2:6, :] = np.maximum(U_bar[2:6, :], 0.0)
        return U_delta




    def _set_reference_and_params(
        self,
        goal: np.ndarray,
        halfspaces: np.ndarray,
        x0: np.ndarray,
        X_bar: np.ndarray,
        U_bar: np.ndarray,
        stage_halfspaces: Optional[np.ndarray] = None,
        dynamic_halfspace_mask: Optional[np.ndarray] = None,
        sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
        vertex_set: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        vertex_types: Optional[np.ndarray] = None,
        feature_obs_indices: Optional[np.ndarray] = None,
        reference_states: Optional[np.ndarray] = None,
        global_goal: Optional[np.ndarray] = None,
        set_yref: bool = True,
    ) -> None:
        """Set one outer-refinement subproblem and its frozen linearizations."""
        cfg = self.cfg
        N = int(cfg.N)
        M = int(cfg.vertex_capacity)
        compact_direct_params = use_external_collision_linearization(cfg)
        goal = np.asarray(goal, dtype=float).reshape(2)
        global_goal_arr = np.asarray(
            global_goal if global_goal is not None else goal, dtype=float
        ).reshape(2)
        raw_global_yaw = math.atan2(
            global_goal_arr[1] - x0[1], global_goal_arr[0] - x0[0]
        )
        global_yaw = float(x0[3]) + wrap_to_pi(raw_global_yaw - float(x0[3]))

        if compact_direct_params:
            halfspaces = None
        else:
            halfspaces = np.asarray(halfspaces, dtype=float).reshape(M, 3)

        solver_nu = self._solver_input_dim()
        X_bar = np.asarray(X_bar, dtype=float).reshape(self._solver_state_dim(), N + 1)
        U_bar = np.asarray(U_bar, dtype=float).reshape(6, N)

        stage_hs_arr = None
        if (not compact_direct_params) and stage_halfspaces is not None:
            stage_hs_arr = np.asarray(stage_halfspaces, dtype=float).reshape(N + 1, M, 3)

        vertex_set_arr = _prepare_collision_feature_tensor(vertex_set, N)
        vertex_normals_arr = _prepare_collision_feature_tensor(vertex_normals, N)
        vertex_types_arr = None if vertex_types is None else np.asarray(vertex_types, dtype=np.int32).reshape(-1)
        feature_obs_indices_arr = None if feature_obs_indices is None else np.asarray(feature_obs_indices, dtype=np.int32).reshape(-1)

        formulation_flag = mpc_formulation_flag(cfg)
        proposed = formulation_flag == 4
        collision_constraint_enabled = collision_avoidance_constraints_enabled(cfg)
        active_slack_count = proposed_active_slack_count(cfg) if proposed else 4
        collision_block_count = active_slack_count

        # Dynamics are linearized once around the current nominal whenever they
        # are needed either by the affine shooting mode or by collision-row
        # composition.  This avoids duplicate RK4 Jacobian work.
        nx = self._solver_state_dim()
        collision_nominal = X_bar[:4, 1:N + 1].T.copy()
        direct_collision = collision_constraint_enabled and compact_direct_params
        linearized_dynamics = not nonlinear_dynamics_enabled(cfg)
        raw_collision_lin_seq = None
        collision_lin_seq = None
        dyn_next_bar = None
        dyn_A = None
        dyn_B = None
        dt_seq = np.full(N, float(cfg.dt_init), dtype=float)
        if linearized_dynamics or direct_collision:
            dyn_next_bar, dyn_A, dyn_B, _ = build_linearized_dynamics_sequence(
                X_bar[:, :N].T,
                U_bar[:2, :].T,
                dt_seq,
                cfg,
            )
        if direct_collision:
            raw_collision_lin_seq = build_selected_mpc_collision_linearization_sequence(
                collision_nominal, vertex_set_arr, vertex_normals_arr, vertex_types_arr,
                feature_obs_indices_arr, np.arange(1, N + 1, dtype=float), dt_seq, cfg,
                sector_rays_body=sector_rays_body,
            )
            collision_lin_seq = compose_collision_linearization_through_dynamics(
                raw_collision_lin_seq,
                collision_nominal,
                dyn_next_bar,
                dyn_A,
                dyn_B,
            )

        self._active_collision_nominal = collision_nominal.copy()
        # Keep the raw x_{k+1}-space linearization for diagnostics/RViz so its
        # historical interpretation does not change.  The composed sequence is
        # the one actually packed into the acados parameters below.
        self._active_collision_lin_seq = (
            None if raw_collision_lin_seq is None else
            np.asarray(raw_collision_lin_seq, dtype=float).reshape(N, 4, 7).copy()
        )
        self._active_solver_collision_lin_seq = (
            None if collision_lin_seq is None else
            np.asarray(collision_lin_seq, dtype=float).reshape(N, 4, 1 + nx + 2).copy()
        )

        # Stage-wise position reference p_{r,k}:
        #   - with actual initial guidance: use the guidance positions for k=1..N;
        #   - without guidance: use the global goal at every stage.
        # The ROS wrappers pass reference_states only when a valid
        # /initial_guidance nominal is available, so a fixed-bias initial_X used
        # purely for SQP warm-start/linearization never becomes p_{r,k}.
        pos_ref_seq = np.tile(global_goal_arr.reshape(1, 2), (N, 1))
        terminal_pos_ref = global_goal_arr.copy()
        if reference_states is not None:
            ref = np.asarray(reference_states, dtype=float)
            if ref.ndim != 2:
                raise ValueError("reference_states must be a two-dimensional state sequence")
            if ref.shape[0] < 2 and ref.shape[1] >= 2:
                ref = ref.T
            if ref.shape[0] < 2 or ref.shape[1] < N + 1:
                raise ValueError(
                    "reference_states must contain at least x/y rows and N+1 samples"
                )
            guide_xy = np.asarray(ref[0:2, :N + 1], dtype=float)
            if not np.all(np.isfinite(guide_xy)):
                raise ValueError("reference_states contains non-finite x/y guidance values")
            # yref_all[k] represents the k-th future tracking term, hence it
            # tracks guidance state k+1 as requested: p_{r,k}=p_guidance,k.
            pos_ref_seq[:, :] = guide_xy[:, 1:N + 1].T
            terminal_pos_ref = guide_xy[:, N].copy()

        speed_ref = float(cfg.v_des)
        if set_yref:
            yref_e = np.array([
                terminal_pos_ref[0], terminal_pos_ref[1], speed_ref, global_yaw
            ], dtype=float)

        sector_np = 6
        param_M = 0 if compact_direct_params else M
        base_np = 3 * param_M + sector_np
        col_block_np = 1 + nx + 2
        x_off = base_np
        u_off = x_off + nx
        col_state_off = u_off + solver_nu
        col_off = col_state_off + 4
        collision_np = collision_block_count * col_block_np
        dyn_A_np = nx * nx if linearized_dynamics else 0
        dyn_B_np = nx * 2 if linearized_dynamics else 0
        dyn_F_np = nx if linearized_dynamics else 0
        dyn_A_off = col_off + collision_np
        dyn_B_off = dyn_A_off + dyn_A_np
        dyn_F_off = dyn_B_off + dyn_B_np
        np_param = dyn_F_off + dyn_F_np
        p_mat = np.zeros((N + 1, np_param), dtype=float)

        sector_vec = sector_rays_to_parameter_vector(sector_rays_body, cfg)
        sector_off = 3 * param_M
        p_mat[:, sector_off:sector_off + sector_np] = sector_vec.reshape(1, sector_np)
        if param_M > 0:
            if stage_hs_arr is not None:
                hs_seq = stage_hs_arr[:, :param_M, :]
            else:
                hs_seq = np.tile(halfspaces[:param_M].reshape(1, param_M, 3), (N + 1, 1, 1))
            p_mat[:, 0:3 * param_M] = hs_seq.reshape(N + 1, 3 * param_M)

        X_stage = X_bar[:, :].T.astype(float, copy=True)
        X_stage[:, 3] = np.arctan2(np.sin(X_stage[:, 3]), np.cos(X_stage[:, 3]))

        U_stage_full = np.empty((N + 1, 6), dtype=float)
        U_stage_full[:N, :] = U_bar.T
        U_stage_full[N, :] = U_bar[:, N - 1] if N > 0 else 0.0
        U_stage_full[:, 0] = np.clip(U_stage_full[:, 0], float(cfg.yawrate_min), float(cfg.yawrate_max))
        U_stage_full[:, 1] = np.clip(U_stage_full[:, 1], float(cfg.a_min), float(cfg.a_max))
        U_stage_full[:, 2:6] = np.maximum(U_stage_full[:, 2:6], 0.0)
        U_stage = U_stage_full[:, 0:2 + active_slack_count] if proposed else U_stage_full

        p_mat[:, x_off:u_off] = X_stage
        p_mat[:, u_off:col_state_off] = U_stage

        X_col_stage = np.empty((N + 1, 4), dtype=float)
        X_col_stage[:N, :] = collision_nominal
        X_col_stage[N, :] = collision_nominal[-1, :] if N > 0 else X_stage[N, :4]
        p_mat[:, col_state_off:col_off] = X_col_stage

        disabled = -float(getattr(cfg, "h_upper_bound", 1.0e15))
        compact_blocks = np.zeros((N + 1, collision_block_count, col_block_np), dtype=float)
        compact_blocks[:, :, 0] = disabled
        if direct_collision and collision_lin_seq is not None:
            raw_blocks = np.asarray(collision_lin_seq, dtype=float).reshape(N, 4, col_block_np)
            compact_blocks[:N, :, :] = raw_blocks[:, :collision_block_count, :]
        p_mat[:, col_off:col_off + collision_np] = compact_blocks.reshape(N + 1, -1)

        if linearized_dynamics:
            if dyn_next_bar is None or dyn_A is None or dyn_B is None:
                raise RuntimeError("linearized dynamics coefficients were not constructed")
            # CasADi reshape() consumes vectors in column-major order.
            p_mat[:N, dyn_A_off:dyn_A_off + dyn_A_np] = np.transpose(
                np.asarray(dyn_A, dtype=float), (0, 2, 1)
            ).reshape(N, nx * nx)
            p_mat[:N, dyn_B_off:dyn_B_off + dyn_B_np] = np.transpose(
                np.asarray(dyn_B, dtype=float), (0, 2, 1)
            ).reshape(N, nx * 2)
            p_mat[:N, dyn_F_off:dyn_F_off + dyn_F_np] = np.asarray(dyn_next_bar, dtype=float).reshape(N, nx)
            # Terminal-stage p is not used by dynamics, but keep it finite and
            # deterministic for flat parameter I/O.
            p_mat[N, dyn_A_off:dyn_A_off + dyn_A_np] = np.eye(nx, dtype=float).reshape(-1, order="F")
            p_mat[N, dyn_B_off:dyn_B_off + dyn_B_np] = 0.0
            p_mat[N, dyn_F_off:dyn_F_off + dyn_F_np] = X_stage[N, :]

        p_mat[~np.isfinite(p_mat)] = 0.0

        yaw_nom = np.clip(U_bar[0, :], float(cfg.yawrate_min), float(cfg.yawrate_max))
        acc_nom = np.clip(U_bar[1, :], float(cfg.a_min), float(cfg.a_max))
        slack_ub_vec = surface_slack_upper_bounds(cfg)
        lbu_all = np.zeros((N, solver_nu), dtype=float)
        ubu_all = np.zeros((N, solver_nu), dtype=float)
        lbu_all[:, 0] = float(cfg.yawrate_min) - yaw_nom
        lbu_all[:, 1] = float(cfg.a_min) - acc_nom
        ubu_all[:, 0] = float(cfg.yawrate_max) - yaw_nom
        ubu_all[:, 1] = float(cfg.a_max) - acc_nom
        ubu_all[:, 2:] = slack_ub_vec[:active_slack_count].reshape(1, active_slack_count)

        yref_all = None
        if set_yref:
            yref_all = np.zeros((N, 4 + solver_nu), dtype=float)
            yref_all[:, 0:2] = pos_ref_seq
            yref_all[:, 2] = speed_ref
            yref_all[:, 3] = global_yaw
            # delta controls reference -u_bar => absolute control regularization.
            yref_all[:, 4] = -yaw_nom
            yref_all[:, 5] = -acc_nom

        p_is_flat = self._set_flat_if_supported("p", p_mat)
        yref_is_flat = yref_all is not None and self._set_flat_if_supported("yref", yref_all)
        lbu_is_flat = self._set_flat_if_supported("lbu", lbu_all)
        ubu_is_flat = self._set_flat_if_supported("ubu", ubu_all)
        for k in range(N):
            if not p_is_flat:
                self.solver.set(k, "p", p_mat[k])
            if yref_all is not None and not yref_is_flat:
                self.solver.set(k, "yref", yref_all[k])
            if not lbu_is_flat:
                self.solver.set(k, "lbu", lbu_all[k])
            if not ubu_is_flat:
                self.solver.set(k, "ubu", ubu_all[k])
        if not p_is_flat:
            self.solver.set(N, "p", p_mat[N])
        if set_yref:
            self.solver.set(N, "yref", yref_e)


    def _warm_start(self, X_bar: np.ndarray, U_bar: np.ndarray) -> None:
        cfg = self.cfg
        N = int(cfg.N)
        state_dim = self._solver_state_dim()
        X_bar = np.asarray(X_bar, dtype=float).reshape(state_dim, N + 1).copy()
        U_bar = np.asarray(U_bar, dtype=float).reshape(6, N).copy()
        X_bar[3, :] = np.arctan2(np.sin(X_bar[3, :]), np.cos(X_bar[3, :]))
        U_bar[0, :] = np.clip(U_bar[0, :], float(cfg.yawrate_min), float(cfg.yawrate_max))
        U_bar[1, :] = np.clip(U_bar[1, :], float(cfg.a_min), float(cfg.a_max))
        U_bar[2:6, :] = np.maximum(U_bar[2:6, :], 0.0)
        U_delta = self._delta_warm_start_from_nominal(U_bar)
        x_is_flat = self._set_flat_if_supported("x", X_bar.T)
        u_is_flat = self._set_flat_if_supported("u", U_delta.T)
        if not x_is_flat:
            for k in range(N + 1):
                self.solver.set(k, "x", X_bar[:, k])
        if not u_is_flat:
            for k in range(N):
                self.solver.set(k, "u", U_delta[:, k])




    def _solver_iterate_vector(self) -> np.ndarray:
        """Return the current acados primal iterate as one flat vector.

        This is used only for the inner SQP_RTI fixed-point convergence test.
        Reading the iterate does not rebuild the externally supplied collision
        linearization or alter the OCP parameters.
        """
        N = int(self.cfg.N)
        nx = self._solver_state_dim()
        nu = self._solver_input_dim()

        x_flat = self._get_flat_if_supported("x", (N + 1) * nx)
        if x_flat is None:
            X = np.zeros((N + 1, nx), dtype=float)
            for k in range(N + 1):
                X[k, :] = np.asarray(self.solver.get(k, "x"), dtype=float).reshape(nx)
            x_flat = X.reshape(-1)
        else:
            x_flat = np.asarray(x_flat, dtype=float).reshape(-1)

        u_flat = self._get_flat_if_supported("u", N * nu)
        if u_flat is None:
            U = np.zeros((N, nu), dtype=float)
            for k in range(N):
                U[k, :] = np.asarray(self.solver.get(k, "u"), dtype=float).reshape(nu)
            u_flat = U.reshape(-1)
        else:
            u_flat = np.asarray(u_flat, dtype=float).reshape(-1)

        return np.concatenate((x_flat, u_flat))


    def _run_rti_steps(
        self,
        max_steps: Optional[int] = None,
        tol: Optional[float] = None,
        min_steps: int = 1,
    ) -> Tuple[int, int, bool, float]:
        """Run repeated SQP_RTI steps under the current frozen linearization.

        ``status == 0`` means that one RTI/QP step succeeded; it is NOT used as
        a convergence test.  Convergence is detected from the infinity norm of
        the change in the full primal iterate (x,u) between consecutive RTI
        iterates.  The first comparison is against the warm start installed
        immediately before this routine is called.

        Returns:
            (status, steps_used, converged, last_step_inf)
        """
        cfg = self.cfg
        linearized_dynamics = not nonlinear_dynamics_enabled(cfg)
        solver_type = (
            "SQP_RTI" if linearized_dynamics else
            str(getattr(cfg, "acados_nlp_solver_type", "SQP_RTI")).strip().upper()
        )
        if max_steps is None:
            max_steps = int(getattr(cfg, "acados_nlp_solver_max_iter", 1))
        if linearized_dynamics:
            # The frozen subproblem is already a convex QP; one HPIPM solve is
            # sufficient. Further refinement belongs to the outer relinearization.
            max_steps = 1
        max_steps = max(1, int(max_steps))
        min_steps = max(1, min(int(min_steps), max_steps))
        if tol is None:
            tol = float(getattr(cfg, "acados_tol", 0.0))
        tol = max(float(tol), 0.0)

        # Non-RTI solvers already own their internal NLP iteration loop.
        if solver_type != "SQP_RTI":
            status = int(self.solver.solve())
            return status, 1, bool(status == 0), 0.0 if status == 0 else float("inf")

        try:
            previous = self._solver_iterate_vector()
        except Exception:
            previous = None

        status = -1
        steps_used = 0
        converged = False
        last_step_inf = float("inf")

        for _ in range(max_steps):
            status = int(self.solver.solve())
            steps_used += 1
            if status != 0:
                break

            try:
                current = self._solver_iterate_vector()
            except Exception:
                current = None

            if (previous is not None and current is not None
                    and previous.shape == current.shape
                    and current.size > 0):
                diff = np.abs(current - previous)
                if np.all(np.isfinite(diff)):
                    last_step_inf = float(np.max(diff))
                    if tol > 0.0 and steps_used >= min_steps and last_step_inf <= tol:
                        converged = True
                        break
            previous = current

        return int(status), int(steps_used), bool(converged), float(last_step_inf)


    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        vertex_set: Optional[np.ndarray] = None,
        halfspaces: Optional[np.ndarray] = None,
        stage_halfspaces: Optional[np.ndarray] = None,
        dynamic_halfspace_mask: Optional[np.ndarray] = None,
        vertex_types: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        feature_obs_indices: Optional[np.ndarray] = None,
        order: Optional[int] = None,
        sqp_iter: Optional[int] = None,
        sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
        initial_U: Optional[np.ndarray] = None,
        initial_X: Optional[np.ndarray] = None,
        initial_dt: Optional[float] = None,
        reference_states: Optional[np.ndarray] = None,
        original_goal: Optional[np.ndarray] = None,
        stage_vertex_set: Optional[np.ndarray] = None,
        stage_vertex_normals: Optional[np.ndarray] = None,
        fallback_U: Optional[np.ndarray] = None,
        fallback_X: Optional[np.ndarray] = None,
    ) -> PlanResult:
        cfg = self.cfg
        N = int(cfg.N)

        # Match vertex_selector.py's timing style: use perf_counter() and keep
        # one aggregated phase breakdown for the complete solve call.
        solve_wall_t0 = time.perf_counter()
        solve_timing_ms = {
            "input_prep": 0.0,
            "halfspace": 0.0,
            "stage_prep": 0.0,
            "nominal_prep": 0.0,
            "construct_lin": 0.0,
            "warm_start": 0.0,
            "phase0": 0.0,
            "phase1": 0.0,
            "phase2": 0.0,
            "main": 0.0,
            "extract": 0.0,
            "rollout": 0.0,
            "candidate_finalize": 0.0,
            "jglobal": 0.0,
            "warm_update": 0.0,
            "finalize": 0.0,
            "fallback": 0.0,
        }
        iteration_timing_ms = []

        def _acc_timing(key: str, start: float) -> float:
            elapsed = (time.perf_counter() - float(start)) * 1000.0
            solve_timing_ms[key] = float(solve_timing_ms.get(key, 0.0)) + float(elapsed)
            return float(elapsed)

        def _attach_timing(result_obj: PlanResult) -> PlanResult:
            total_ms = (time.perf_counter() - solve_wall_t0) * 1000.0
            accounted_ms = sum(float(v) for v in solve_timing_ms.values())
            timing = dict(solve_timing_ms)
            timing["total"] = float(total_ms)
            timing["other"] = max(0.0, float(total_ms) - float(accounted_ms))
            result_obj.timing_ms = timing
            result_obj.iteration_timing_ms = [dict(row) for row in iteration_timing_ms]
            return result_obj

        # Reset candidate state for this planning cycle.
        self.last_result = None
        self.last_iteration_results = []
        self.last_feasible_candidates = []

        input_prep_t0 = time.perf_counter()
        # x0 may be [px,py,v,yaw] (actual yaw rate from cfg.ego_yawrate0) or
        # [px,py,v,yaw,yawrate].  ``x0`` keeps the four-state pose used by the
        # planners/messages; ``x0_full`` is the five-state yaw-rate-lag solver
        # initial condition.
        x0_full = augment_ego_state(x0, cfg)
        x0_full[3] = wrap_to_pi(float(x0_full[3]))
        x0 = x0_full[:4].copy()
        goal = np.asarray(goal, dtype=float).reshape(2)
        original_goal_arr = np.asarray(original_goal if original_goal is not None else goal, dtype=float).reshape(2)
        order_i = int(order if order is not None else getattr(cfg, "ORDER", 1))
        outer_iter = int(sqp_iter if sqp_iter is not None else getattr(cfg, "SQP_iter", 3))
        outer_iter = max(1, outer_iter)
        if sqp_iter is None:
            realtime_cap = int(getattr(cfg, "realtime_max_sqp_iter", 2))
            if realtime_cap > 0:
                outer_iter = min(outer_iter, realtime_cap)

        vertex_normals_arr = None
        if vertex_normals is not None:
            vertex_normals_arr = np.asarray(vertex_normals, dtype=float).reshape(-1, 2)
        feature_obs_indices_arr = None
        if feature_obs_indices is not None:
            feature_obs_indices_arr = np.asarray(feature_obs_indices, dtype=np.int32).reshape(-1)

        direct_collision_mode = (
            collision_avoidance_constraints_enabled(cfg)
            and use_external_collision_linearization(self.cfg)
        )
        _acc_timing("input_prep", input_prep_t0)

        halfspace_t0 = time.perf_counter()
        if halfspaces is None:
            vertex_set_arr = np.asarray(vertex_set if vertex_set is not None else np.zeros((0, 2)), dtype=float).reshape(-1, 2)
            if direct_collision_mode:
                # Halfspaces are unused by the compact direct-collision acados model.
                halfspaces_arr = np.zeros((int(cfg.vertex_capacity), 3), dtype=float)
                halfspaces_arr[:, 2] = float(cfg.disabled_c_value)
            elif vertex_normals_arr is not None and vertex_normals_arr.size > 0:
                halfspaces_arr = selected_vertices_normals_to_halfspaces(
                    vertex_set_arr,
                    vertex_normals_arr,
                    x0,
                    cfg.vertex_capacity,
                    cfg,
                    disabled_c_value=cfg.disabled_c_value,
                )
            else:
                halfspaces_arr = vertex_set_to_halfspaces(
                    vertex_set_arr,
                    x0[:2],
                    cfg.vertex_capacity,
                    buffer_m=0.0,
                    disabled_c_value=cfg.disabled_c_value,
                )
        else:
            halfspaces_arr = np.asarray(halfspaces, dtype=float).reshape(-1, 3)
            if halfspaces_arr.shape[0] < cfg.vertex_capacity:
                pad = np.zeros((cfg.vertex_capacity - halfspaces_arr.shape[0], 3), dtype=float)
                pad[:, 2] = cfg.disabled_c_value
                halfspaces_arr = np.vstack((halfspaces_arr, pad))
            halfspaces_arr = halfspaces_arr[: cfg.vertex_capacity, :]
            vertex_set_arr = np.asarray(vertex_set if vertex_set is not None else np.zeros((0, 2)), dtype=float).reshape(-1, 2)
        _acc_timing("halfspace", halfspace_t0)

        stage_prep_t0 = time.perf_counter()
        # Keep the legacy stage-0 set for result/halfspace bookkeeping, while
        # direct collision kernels consume the N-stage predicted/inflated set.
        collision_vertex_set_arr = _prepare_collision_feature_tensor(
            stage_vertex_set if stage_vertex_set is not None else vertex_set_arr, N
        )
        collision_vertex_normals_arr = _prepare_collision_feature_tensor(
            stage_vertex_normals if stage_vertex_normals is not None else vertex_normals_arr, N
        )

        stage_halfspaces_arr = None
        if (not direct_collision_mode) and stage_halfspaces is not None:
            arr = np.asarray(stage_halfspaces, dtype=float)
            if arr.ndim == 2:
                arr = np.repeat(arr.reshape(1, arr.shape[0], arr.shape[1]), N + 1, axis=0)
            arr = arr.reshape(-1, arr.shape[-2], 3)
            if arr.shape[0] < N + 1:
                last = arr[-1:, :, :]
                arr = np.concatenate((arr, np.repeat(last, N + 1 - arr.shape[0], axis=0)), axis=0)
            arr = arr[: N + 1, :, :]
            fixed = np.zeros((N + 1, int(cfg.vertex_capacity), 3), dtype=float)
            fixed[:, :, 2] = float(cfg.disabled_c_value)
            cols = min(arr.shape[1], int(cfg.vertex_capacity))
            fixed[:, :cols, :] = arr[:, :cols, :]
            stage_halfspaces_arr = fixed
        vertex_types_arr = None
        if vertex_types is not None:
            vertex_types_arr = np.asarray(vertex_types, dtype=np.int32).reshape(-1)
        dyn_mask_arr = None
        if dynamic_halfspace_mask is not None:
            dyn_mask_arr = np.asarray(dynamic_halfspace_mask, dtype=bool).reshape(-1)
        elif vertex_types_arr is not None:
            dyn_mask_arr = vertex_types_to_dynamic_mask(vertex_types_arr, int(cfg.vertex_capacity))
        if dyn_mask_arr is not None:
            if dyn_mask_arr.size < int(cfg.vertex_capacity):
                padded = np.zeros(int(cfg.vertex_capacity), dtype=bool)
                padded[:dyn_mask_arr.size] = dyn_mask_arr
                dyn_mask_arr = padded
            dyn_mask_arr = dyn_mask_arr[: int(cfg.vertex_capacity)]

        # Sector rays are stage parameters.  MPC_planner normally computes
        # them from JSON and passes them in; this fallback keeps direct calls
        # to AcadosMPCPlanner.solve() compatible.
        sector_rays_body_arr = sector_rays_body
        if sector_rays_body_arr is None:
            sector_rays_body_arr = sector_rays_body_from_cfg(cfg)
        else:
            sector_rays_body_arr = np.asarray(sector_rays_body_arr, dtype=float).reshape(3, 2)
        _acc_timing("stage_prep", stage_prep_t0)

        nominal_prep_t0 = time.perf_counter()
        # Initial nominal trajectory for this planning cycle.  Do not reuse
        # the previous cycle's X_bar/U_bar unless an explicit
        # /initial_guidance nominal was supplied by the ROS wrapper for this
        # exact recycle.
        self.reset_cycle_nominal(x0_full, order_i)
        U_bar = self.U_warm.copy()
        X_bar = self.X_warm.copy()
        flag = 0
        # print("U_bar: ", U_bar)
        # print("X_bar: ", X_bar)
        # print("N:", N)
        # print("order: ", order_i)
        
        # External guidance initializes U_bar.  X_bar is always reconstructed
        # by rolling U_bar out from x0; guidance states are retained only through
        # reference_states for the stage-wise tracking objective.
        external_nominal = self._coerce_initial_nominal(
            x0_full, initial_U, initial_X, initial_dt, order_i
        )
        if external_nominal is not None:
            U_bar, X_bar = external_nominal

        # Keep a dedicated deterministic fallback nominal for this planning cycle.
        # The ROS wrappers pass the ORDER -> (acc_order, yawrate_order) seed here,
        # so a solver failure publishes exactly that X_nom/U_nom rather than the
        # legacy hard-coded 3-way ORDER bias in _make_bias_control().
        explicit_fallback_nominal = self._coerce_initial_nominal(
            x0_full, fallback_U, fallback_X, float(cfg.dt_init), order_i
        )
        if explicit_fallback_nominal is None:
            fallback_U_bar = np.asarray(U_bar, dtype=float).copy()
            fallback_X_bar = np.asarray(X_bar, dtype=float).copy()
        else:
            fallback_U_bar, fallback_X_bar = explicit_fallback_nominal
            fallback_U_bar = np.asarray(fallback_U_bar, dtype=float).copy()
            fallback_X_bar = np.asarray(fallback_X_bar, dtype=float).copy()
        _acc_timing("nominal_prep", nominal_prep_t0)

        def _last_stat_value(name: str):
            """Best-effort read of the latest scalar acados statistic."""
            getter = getattr(self.solver, "get_stats", None)
            if getter is None:
                return None
            try:
                arr = np.asarray(getter(str(name)), dtype=float).reshape(-1)
                if arr.size == 0:
                    return None
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return None
                return float(finite[-1])
            except Exception:
                return None

        def _solver_hit_iteration_limit(status_code: int) -> Tuple[bool, Optional[float], Optional[float]]:
            """Return whether a nonzero solver status is only an iteration-limit exit.

            acados status 2 is ACADOS_MAXITER.  QP solver statuses are mapped to
            the same acados status space, so qp_status == 2 is also treated as a
            max-iteration exit.  Recovery is allowed only when the current primal
            iterate can be read and is finite.
            """
            qp_status = _last_stat_value("qp_status")
            qp_iter = _last_stat_value("qp_iter")
            # acados status 2 is ACADOS_MAXITER. QP solver statuses are mapped
            # to the same acados status space, so qp_status == 2 is likewise a
            # QP iteration-limit exit (including HPIPM MAXIT).
            nlp_maxiter = int(status_code) == 2
            qp_maxiter = bool(
                qp_status is not None
                and np.isfinite(float(qp_status))
                and int(round(float(qp_status))) == 2
            )
            if not (nlp_maxiter or qp_maxiter):
                return False, qp_status, qp_iter

            try:
                iterate = np.asarray(self._solver_iterate_vector(), dtype=float).reshape(-1)
                finite_iterate = bool(iterate.size > 0 and np.all(np.isfinite(iterate)))
            except Exception:
                finite_iterate = False
            return bool(finite_iterate), qp_status, qp_iter

        def _solver_failure_diagnostics(status_code: int, phase_label: str, sqp_index: int) -> None:
            """Print the QP/NLP status immediately before solver fallback."""
            qp_status = _last_stat_value("qp_status")
            qp_iter = _last_stat_value("qp_iter")
            res_eq = _last_stat_value("res_eq")
            res_ineq = _last_stat_value("res_ineq")

            # Some acados versions expose NLP residuals through get_residuals()
            # rather than get_stats("res_eq") / get_stats("res_ineq").
            if res_eq is None or res_ineq is None:
                get_residuals = getattr(self.solver, "get_residuals", None)
                if get_residuals is not None:
                    try:
                        residuals = np.asarray(get_residuals(), dtype=float).reshape(-1)
                        # acados ordering: [res_stat, res_eq, res_ineq, res_comp]
                        if res_eq is None and residuals.size > 1 and np.isfinite(residuals[1]):
                            res_eq = float(residuals[1])
                        if res_ineq is None and residuals.size > 2 and np.isfinite(residuals[2]):
                            res_ineq = float(residuals[2])
                    except Exception:
                        pass

            def _fmt(v, integer=False):
                if v is None:
                    return "N/A"
                if integer:
                    return str(int(round(float(v))))
                return "{:.6e}".format(float(v))

            print(
                "[ACADOS-FAIL] sqp_iter={} phase={} status={} qp_status={} qp_iter={} "
                "res_eq={} res_ineq={}".format(
                    int(sqp_index), str(phase_label), int(status_code),
                    _fmt(qp_status, integer=True), _fmt(qp_iter, integer=True),
                    _fmt(res_eq), _fmt(res_ineq),
                )
            )

        def _return_infeasible_fallback(
            status_code: int, phase_label: str = "unknown", sqp_index: int = -1
        ) -> PlanResult:
            """Return the deterministic nominal immediately after solver failure."""
            fallback_t0 = time.perf_counter()
            _solver_failure_diagnostics(status_code, phase_label, sqp_index)
            print("###################")
            print("# fallback #")
            print("###################")
            fallback = self._make_fallback_result(
                x0_full, order_i, vertex_set_arr, halfspaces_arr,
                dynamic_halfspace_mask=dyn_mask_arr, vertex_types=vertex_types_arr,
                fallback_U_bar=fallback_U_bar, fallback_X_bar=fallback_X_bar,
            )
            fallback.exitflag = -abs(int(status_code)) if int(status_code) != 0 else -1
            fallback.trajectory_type = 0
            fallback.trajectory_stage = "single_dual_constraint_fallback"
            fallback.is_order_fallback = True

            # A solver fallback is never a feasible MPC candidate.  Do not run any
            # collision/field feasibility or J_global evaluation on it.
            fallback.ocp_feasible = False
            fallback.original_ocp_feasible = False
            fallback.paper_feasible = False
            fallback.zero_slack_feasible = False
            fallback.s_r_feasible = False
            fallback.selection_eligible = False
            fallback.J_global = 1.0e15
            fallback.global_cost = 1.0e15
            fallback.phi_min = float("-inf")
            fallback.min_clearance = float("-inf")
            fallback.intermediate_results = []
            fallback.successful_iteration_count = 0
            fallback.returned_candidate_count = 0
            fallback.returned_candidate_costs = []
            fallback.returned_candidate_J_globals = []
            fallback.feasible_candidates = []
            fallback.feasible_candidate_count = 0
            fallback.feasible_candidate_costs = []
            fallback.feasible_candidate_J_globals = []
            fallback.zero_slack_feasible_mask = np.zeros(0, dtype=bool)
            fallback.nonsmooth_feasible_mask = np.zeros(0, dtype=bool)

            self.last_iteration_results = []
            self.last_feasible_candidates = []
            self.last_result = fallback
            _acc_timing("fallback", fallback_t0)
            return _attach_timing(fallback)

        # Position reference is supplied only by actual initial guidance.
        # When guidance is absent, initial_X remains only the SQP warm-start /
        # linearization nominal and must not become p_r,k.
        stage2_reference_states = reference_states

        last_success = None
        last_status = -1
        iteration_results = []
        last_failure_phase = "unknown"
        last_failure_sqp = -1

        for sqp_i in range(outer_iter):
            iter_wall_t0 = time.perf_counter()
            iter_timing = {
                "sqp": int(sqp_i),
                "construct_lin": 0.0,
                "warm_start": 0.0,
                "phase0": 0.0,
                "phase1": 0.0,
                "phase2": 0.0,
                "main": 0.0,
                "extract": 0.0,
                "rollout": 0.0,
                "candidate_finalize": 0.0,
                "jglobal": 0.0,
                "warm_update": 0.0,
            }

            construct_t0 = time.perf_counter()
            self._set_reference_and_params(
                goal, halfspaces_arr, x0, X_bar, U_bar, stage_halfspaces_arr,
                dynamic_halfspace_mask=dyn_mask_arr,
                sector_rays_body=sector_rays_body_arr,
                vertex_set=collision_vertex_set_arr,
                vertex_normals=collision_vertex_normals_arr,
                vertex_types=vertex_types_arr,
                feature_obs_indices=feature_obs_indices_arr,
                reference_states=stage2_reference_states,
                global_goal=original_goal_arr,
                set_yref=True,
            )
            iter_timing["construct_lin"] = _acc_timing("construct_lin", construct_t0)

            warm_start_t0 = time.perf_counter()
            self._warm_start(X_bar, U_bar)
            x0_solver = np.asarray(x0_full, dtype=float)
            self.solver.set(0, "lbx", x0_solver)
            self.solver.set(0, "ubx", x0_solver)
            iter_timing["warm_start"] = _acc_timing("warm_start", warm_start_t0)

            formulation_flag = mpc_formulation_flag(cfg)
            inner_budget = max(1, int(getattr(cfg, "acados_nlp_solver_max_iter", 1)))
            inner_tol = max(float(getattr(cfg, "acados_tol", 0.0)), 0.0)
            outer_rti_steps = 0
            outer_rti_converged = False
            outer_rti_step_inf = float("inf")
            outer_rti_phase_steps = {}
            failure_phase = "unknown"

            if formulation_flag == 4:
                # Proposed OCP: every outer collision-refinement iteration uses
                # the same phase-0 scalarized objective.  No final phase switch
                # and no slack-budget constraint are applied.
                phase_t0 = time.perf_counter()
                self._set_runtime_cost_profile(lexicographic_phase=0)
                status, used, conv, step_inf = self._run_rti_steps(
                    max_steps=inner_budget, tol=inner_tol
                )
                iter_timing["phase0"] = _acc_timing("phase0", phase_t0)
                failure_phase = "phase0"
                outer_rti_steps = int(used)
                outer_rti_converged = bool(conv)
                outer_rti_step_inf = float(step_inf)
                outer_rti_phase_steps["phase0"] = int(used)
            elif formulation_flag == 3:
                # Existing surface-wise lexicographic baseline.  As above, share
                # one outer RTI budget across phase 1 and phase 2.
                phase_t0 = time.perf_counter()
                self._set_runtime_cost_profile(lexicographic_phase=1)
                phase1_budget = max(1, inner_budget - 1) if inner_budget >= 2 else 1
                status_phase1, phase1_steps, phase1_conv, phase1_inf = self._run_rti_steps(
                    max_steps=phase1_budget, tol=inner_tol
                )
                iter_timing["phase1"] = _acc_timing("phase1", phase_t0)
                failure_phase = "phase1"
                outer_rti_steps += phase1_steps
                outer_rti_phase_steps["phase1"] = int(phase1_steps)
                outer_rti_step_inf = phase1_inf
                if status_phase1 == 0:
                    phase1_result = self._extract_solution(
                        status_phase1, vertex_set_arr, halfspaces_arr, U_bar,
                        dynamic_halfspace_mask=dyn_mask_arr, vertex_types=vertex_types_arr,
                    )
                    v_idx = vulnerable_surface_index(cfg)
                    s_v_star = np.asarray(phase1_result.surface_slack[v_idx, :], dtype=float).copy()
                    self._fix_lexicographic_vulnerable_slack(U_bar, s_v_star)
                    phase_t0 = time.perf_counter()
                    self._set_runtime_cost_profile(lexicographic_phase=2)
                    phase2_budget = max(1, inner_budget - outer_rti_steps) if inner_budget >= 2 else 1
                    status, phase2_steps, phase2_conv, phase2_inf = self._run_rti_steps(
                        max_steps=phase2_budget, tol=inner_tol
                    )
                    iter_timing["phase2"] = _acc_timing("phase2", phase_t0)
                    failure_phase = "phase2"
                    outer_rti_steps += phase2_steps
                    outer_rti_phase_steps["phase2"] = int(phase2_steps)
                    outer_rti_converged = bool(phase2_conv)
                    outer_rti_step_inf = phase2_inf
                else:
                    status = status_phase1
                    outer_rti_converged = bool(phase1_conv)
            else:
                phase_t0 = time.perf_counter()
                status, used, conv, step_inf = self._run_rti_steps(
                    max_steps=inner_budget, tol=inner_tol
                )
                iter_timing["main"] = _acc_timing("main", phase_t0)
                failure_phase = "main"
                outer_rti_steps = int(used)
                outer_rti_converged = bool(conv)
                outer_rti_step_inf = float(step_inf)
                outer_rti_phase_steps["main"] = int(used)
            last_status = status
            maxiter_recovered = False
            maxiter_qp_status = None
            maxiter_qp_iter = None
            if status != 0:
                maxiter_recovered, maxiter_qp_status, maxiter_qp_iter = _solver_hit_iteration_limit(status)
                if not maxiter_recovered:
                    last_failure_phase = str(failure_phase)
                    last_failure_sqp = int(sqp_i)

            if status == 0 or maxiter_recovered:
                if maxiter_recovered:
                    print(
                        "[ACADOS-MAXITER-USE] sqp_iter={} phase={} status={} "
                        "qp_status={} qp_iter={} -> use current iterate as nominal".format(
                            int(sqp_i), str(failure_phase), int(status),
                            "N/A" if maxiter_qp_status is None else int(round(float(maxiter_qp_status))),
                            "N/A" if maxiter_qp_iter is None else int(round(float(maxiter_qp_iter))),
                        )
                    )
                extract_t0 = time.perf_counter()
                result = self._extract_solution(
                    status, vertex_set_arr, halfspaces_arr, U_bar,
                    dynamic_halfspace_mask=dyn_mask_arr, vertex_types=vertex_types_arr,
                )
                # Keep the raw nonzero solver exit visible, but allow an iteration-limit
                # iterate to proceed through candidate finalization and, crucially, to
                # update U_bar/X_bar for the next outer collision re-linearization.
                result.raw_solver_status = int(status)
                result.maxiter_recovered = bool(maxiter_recovered)
                iter_timing["extract"] = _acc_timing("extract", extract_t0)
                du_inf = float(np.max(np.abs(result.U - U_bar[:2, :]))) if result.U.size else 0.0
                # Never use the solver-returned state as X_nominal.  Build the
                # physical trajectory only from the current x0 and optimized
                # nominal input sequence, then use that rollout consistently for
                # candidate evaluation, output, and the next SQP linearization.
                rollout_t0 = time.perf_counter()
                X_nl = rollout_ego(x0_full, result.U, float(cfg.dt_init), cfg)
                iter_timing["rollout"] = _acc_timing("rollout", rollout_t0)
                dx_inf = float(np.max(np.abs(X_nl - X_bar[:4, :]))) if X_nl.size else 0.0
                result.X = X_nl
                result.sqp_iteration = int(sqp_i)
                result.rti_steps = int(outer_rti_steps)
                result.rti_converged = bool(outer_rti_converged)
                result.rti_step_inf = float(outer_rti_step_inf)
                result.rti_phase_steps = dict(outer_rti_phase_steps)

                # Candidate metrics are owned by utils_linearized.  Proposed
                # candidates always come from phase 0, so the solver objective
                # already includes the configured scalarized slack penalty.
                candidate_t0 = time.perf_counter()
                result = self._finalize_returned_candidate(result)
                iter_timing["candidate_finalize"] = _acc_timing("candidate_finalize", candidate_t0)

                warm_update_t0 = time.perf_counter()
                iteration_results.append(result)

                last_success = result
                ss_result = np.asarray(
                    getattr(result, "surface_slack", np.zeros((4, N))), dtype=float
                ).reshape(4, N)
                ds_inf = float(np.max(np.abs(ss_result - U_bar[2:6, :]))) if ss_result.size else 0.0

                U_bar = np.zeros((6, N), dtype=float)
                U_bar[:2, :] = result.U
                U_bar[2:6, :] = ss_result
                X_bar = self._make_augmented_warm_start(x0_full, U_bar)
                iter_timing["warm_update"] = _acc_timing("warm_update", warm_update_t0)
                iter_timing["total"] = (time.perf_counter() - iter_wall_t0) * 1000.0
                iteration_timing_ms.append(iter_timing)

                early_tol = float(getattr(cfg, "sqp_early_stop_tol", 0.0))
                min_iter = max(1, int(getattr(cfg, "sqp_min_iter", 1)))
                if (early_tol > 0.0 and (sqp_i + 1) >= min_iter
                        and max(du_inf, dx_inf, ds_inf) <= early_tol):
                    break
            else:
                # Only genuine solver/numerical failures reach this branch.
                # Iteration-limit exits with a finite iterate are handled above and
                # become the nominal trajectory for the next outer SQP iteration.
                iter_timing["total"] = (time.perf_counter() - iter_wall_t0) * 1000.0
                iteration_timing_ms.append(iter_timing)
                return _return_infeasible_fallback(status, failure_phase, sqp_i)



        if last_success is not None:
            result = last_success

            # J_global is needed only for the final successful outer-SQP iterate,
            # because MPC_planner0..6 always publish that iterate and do not rank
            # intermediate iterates by J_global.  Deferring this nonlinear field
            # evaluation avoids one J_global pass for every earlier SQP iterate.
            jglobal_t0 = time.perf_counter()
            precomputed_dual = None
            precomputed_impact_alignment = None
            if formulation_flag == 4:
                try:
                    precomputed_dual = _proposed_dual_result_cache(
                        result,
                        collision_vertex_set_arr,
                        collision_vertex_normals_arr,
                        vertex_types_arr,
                        feature_obs_indices_arr,
                        cfg,
                        sector_rays_body=sector_rays_body,
                    )
                    if precomputed_dual is not None:
                        precomputed_impact_alignment = getattr(
                            result, "_proposed_impact_alignment_cache", None
                        )
                except Exception:
                    precomputed_dual = None
                    precomputed_impact_alignment = None
            try:
                j_global = evaluate_matlab_global_candidate_objective(
                    X=result.X,
                    slack=result.slack,
                    goal=original_goal_arr,
                    vertex_set=collision_vertex_set_arr,
                    vertex_normals=collision_vertex_normals_arr,
                    vertex_types=vertex_types_arr,
                    feature_obs_indices=feature_obs_indices_arr,
                    dt=float(cfg.dt_init),
                    cfg=cfg,
                    boundary_slack=float(getattr(result, "boundary_slack", 0.0)),
                    normal_slack=getattr(result, "normal_slack", None),
                    surface_slack=getattr(result, "surface_slack", None),
                    precomputed_proposed_dual=precomputed_dual,
                    precomputed_proposed_alignment=precomputed_impact_alignment,
                    sector_rays_body=sector_rays_body,
                )
            except Exception:
                j_global = float("inf")
            result.J_global = float(j_global) if np.isfinite(float(j_global)) else float("inf")
            result.global_cost = float(result.J_global)
            jglobal_ms = _acc_timing("jglobal", jglobal_t0)
            if iteration_timing_ms:
                iteration_timing_ms[-1]["jglobal"] = float(jglobal_ms)
                iteration_timing_ms[-1]["total"] = (
                    float(iteration_timing_ms[-1].get("total", 0.0)) + float(jglobal_ms)
                )

            finalize_t0 = time.perf_counter()
            result.dt = float(cfg.dt_init)
            result.trajectory_type = 0
            result.trajectory_stage = "single_dual_constraint_traj"
            result.selection_tier = 0
            result.stage1_U = None
            result.stage1_X = None

            # Explicitly return all successful outer-SQP iterates with cost and
            # zero-regulation-slack feasibility already finalized above.
            self.last_iteration_results = list(iteration_results)
            result.intermediate_results = list(iteration_results[:-1])
            result.successful_iteration_count = int(len(iteration_results))
            result.returned_candidate_count = int(len(iteration_results))
            result.returned_candidate_costs = [float(c.cost) for c in iteration_results]
            result.returned_candidate_J_globals = [
                float(getattr(c, "J_global", float("inf"))) for c in iteration_results
            ]
            zero_mask = np.asarray(
                [bool(getattr(c, "zero_slack_feasible", False)) for c in iteration_results],
                dtype=bool,
            )
            result.zero_slack_feasible_mask = zero_mask.copy()
            result.nonsmooth_feasible_mask = zero_mask.copy()  # compatibility alias
            result.feasible_candidates = [
                c for c, keep in zip(iteration_results, zero_mask.tolist()) if keep
            ]
            self.last_feasible_candidates = list(result.feasible_candidates)
            result.feasible_candidate_count = int(len(result.feasible_candidates))
            result.feasible_candidate_costs = [float(c.cost) for c in result.feasible_candidates]
            result.feasible_candidate_J_globals = [
                float(getattr(c, "J_global", float("inf"))) for c in result.feasible_candidates
            ]
            self.last_result = result
            self.U_warm = U_bar.copy()
            self.X_warm = X_bar.copy()
            self.dt = float(result.dt)
            _acc_timing("finalize", finalize_t0)
            return _attach_timing(result)

        finalize_t0 = time.perf_counter()
        _acc_timing("finalize", finalize_t0)
        return _return_infeasible_fallback(last_status, last_failure_phase, last_failure_sqp)


def evaluate_matlab_global_candidate_objective(
    X: np.ndarray,
    slack: np.ndarray,
    goal: np.ndarray,
    vertex_set: np.ndarray,
    vertex_normals: Optional[np.ndarray],
    vertex_types: Optional[np.ndarray],
    feature_obs_indices: Optional[np.ndarray],
    dt: float,
    cfg: MPCConfig,
    boundary_slack: float = 0.0,
    normal_slack: Optional[np.ndarray] = None,
    surface_slack: Optional[np.ndarray] = None,
    precomputed_surface_distance: Optional[np.ndarray] = None,
    precomputed_proposed_dual: Optional[np.ndarray] = None,
    precomputed_proposed_alignment: Optional[np.ndarray] = None,
    sector_rays_body: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """Return the paper global candidate-ranking cost.

    For ``mpc_flag == 4`` this is the planner-evaluable part of the global
    candidate-ranking objective:

        sum_{k=1}^H ||x_k-x_G||_Q^2
        - rho_Phi sum_{k=1}^H Phi_k
        + rho_s * sum_{k=1}^H (s_k^a)^2
        + rho_c * sum_{k=1}^H (s_k^r)^2
        + rho_impact * sum_{k=1}^H (-grad Phi_k(0))^T n(phi_c).

    When impact-normal regulation is disabled, the s_k^r term is omitted.
    The previous-trajectory similarity term is deliberately not included here;
    ego_mover adds

        sum_{k=0}^{H-1} ||x_k-tilde{x}^{prev}_k||_{Q_sim}^2

    because it owns the previously selected trajectory. No extra yaw,
    terminal-only, or boundary-slack ranking term is added.

    Surface-wise baselines (mpc_flag 0..3) retain their surface-distance/slack
    ranking branch for compatibility, but share the same stage-wise goal term
    and do not receive the removed yaw ranking term.
    """
    X = np.asarray(X, dtype=float).reshape(4, -1)
    slack_arr = np.asarray(slack if slack is not None else np.zeros(0), dtype=float).reshape(-1)
    normal_slack_arr = np.asarray(
        normal_slack if normal_slack is not None else np.zeros(0), dtype=float
    ).reshape(-1)
    H = min(int(cfg.N), X.shape[1] - 1)
    if H <= 0:
        return 0.0

    dt_seq = np.full(H, float(dt), dtype=float)
    flag = mpc_formulation_flag(cfg)
    if flag == 4:
        impact_alignment = None
        if precomputed_proposed_dual is None:
            try:
                dual, impact_block = build_external_dual_collision_linearization_sequence(
                    X[:, 1:H + 1].T, vertex_set, vertex_normals, vertex_types,
                    feature_obs_indices, np.arange(1, H + 1, dtype=float),
                    dt_seq, cfg, sector_rays_body=sector_rays_body, return_alignment=True,
                )
                dual = np.asarray(dual, dtype=float).reshape(H, 14)
                impact_alignment = np.asarray(impact_block, dtype=float).reshape(H, 7)[:, 0]
            except Exception:
                return float("inf")
        else:
            try:
                dual = np.asarray(precomputed_proposed_dual, dtype=float).reshape(H, 14)
                if precomputed_proposed_alignment is not None:
                    impact_alignment = np.asarray(
                        precomputed_proposed_alignment, dtype=float
                    ).reshape(H)
            except Exception:
                return float("inf")

        # Older/external callers may provide only the Hx14 dual block. In that
        # case, evaluate the third native block once so J_global still contains
        # the requested impact-aligned term.
        if impact_alignment is None:
            try:
                _, impact_block = build_external_dual_collision_linearization_sequence(
                    X[:, 1:H + 1].T, vertex_set, vertex_normals, vertex_types,
                    feature_obs_indices, np.arange(1, H + 1, dtype=float),
                    dt_seq, cfg, sector_rays_body=sector_rays_body, return_alignment=True,
                )
                impact_alignment = np.asarray(impact_block, dtype=float).reshape(H, 7)[:, 0]
            except Exception:
                return float("inf")

        clearance_score = _proposed_phi_from_dual(dual, cfg)
        if not (np.all(np.isfinite(clearance_score)) and np.all(np.isfinite(impact_alignment))):
            return float("inf")

        # Proposed global slack ranking uses the same OCP weights:
        #   rho_s * sum_k (s_k^a)^2 + rho_c * sum_k (s_k^r)^2.
        # The regulation term is omitted when the impact-normal constraint is disabled.
        s_a = np.zeros(H, dtype=float)
        n_slack = min(H, slack_arr.size)
        if n_slack > 0:
            vals = np.where(np.isfinite(slack_arr[:n_slack]), slack_arr[:n_slack], 0.0)
            s_a[:n_slack] = np.maximum(vals, 0.0)

        s_r = np.zeros(H, dtype=float)
        if impact_normal_constraint_enabled(cfg):
            n_reg = min(H, normal_slack_arr.size)
            if n_reg > 0:
                vals_r = np.where(
                    np.isfinite(normal_slack_arr[:n_reg]),
                    normal_slack_arr[:n_reg],
                    0.0,
                )
                s_r[:n_reg] = np.maximum(vals_r, 0.0)

        rho_s = max(_finite_or_default(getattr(cfg, "rho_s", 0.0), 0.0), 0.0)
        rho_c = max(_finite_or_default(getattr(cfg, "rho_c", 0.0), 0.0), 0.0)
        slack_cost = rho_s * float(np.sum(s_a * s_a))
        if impact_normal_constraint_enabled(cfg):
            slack_cost += rho_c * float(np.sum(s_r * s_r))
        rho_impact = global_impact_alignment_weight(cfg)
        impact_alignment_cost = rho_impact * float(np.sum(impact_alignment))
    else:
        if precomputed_surface_distance is None:
            d_surf = evaluate_surface_distance_sequence(
                X[:, 1:H + 1], vertex_set, vertex_types, feature_obs_indices,
                np.arange(1, H + 1, dtype=float), dt_seq, cfg,
            )
            d_surf = np.asarray(d_surf, dtype=float).reshape(H, 4)
        else:
            try:
                d_surf = np.asarray(precomputed_surface_distance, dtype=float).reshape(H, 4)
            except Exception:
                return float("inf")
        if not np.all(np.isfinite(d_surf)):
            return float("inf")
        clearance_score = np.min(d_surf, axis=1)

        ss = np.zeros((4, H), dtype=float)
        if surface_slack is not None:
            raw = np.asarray(surface_slack, dtype=float)
            try:
                raw = raw.reshape(4, -1)
                n = min(H, raw.shape[1])
                ss[:, :n] = raw[:, :n]
            except Exception:
                return float("inf")
        else:
            n0 = min(H, slack_arr.size)
            n1 = min(H, normal_slack_arr.size)
            if n0:
                ss[0, :n0] = slack_arr[:n0]
            if n1:
                ss[1, :n1] = normal_slack_arr[:n1]
        if not np.all(np.isfinite(ss)):
            return float("inf")

        if flag == 3:
            weights = np.full(
                4, max(float(getattr(cfg, "q_nonvulnerable_slack", 1.0e8)), 0.0), dtype=float
            )
            weights[vulnerable_surface_index(cfg)] = max(
                float(getattr(cfg, "q_vulnerable_slack", 1.0e10)), 0.0
            )
        else:
            weights = surface_slack_cost_weights(cfg, lexicographic_phase=0)
        slack_cost = float(np.sum(weights.reshape(4, 1) * (ss * ss)))
        impact_alignment_cost = 0.0

    # sum_{k=1}^H ||x_k-x_G||_Q^2.  The current global goal is positional, so
    # Q = q_global_goal I_2 on [x,y]; local MPC q_yaw/q_v are not ranking terms.
    goal_xy = np.asarray(goal, dtype=float).reshape(2)
    goal_err = X[0:2, 1:H + 1].T - goal_xy.reshape(1, 2)
    q_goal = float(getattr(cfg, "q_global_goal", getattr(cfg, "q_pos", 1.0)))
    goal_cost = q_goal * float(np.sum(goal_err * goal_err))

    J = (
        goal_cost
        - float(getattr(cfg, "rho_Phi_global", 1.0)) * float(np.sum(clearance_score))
        + slack_cost
        + impact_alignment_cost
    )
    return float(J) if math.isfinite(float(J)) else float("inf")

def trajectory_path_length(X: np.ndarray) -> float:
    """Return the xy path length of a planned trajectory."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[1] < 2:
        return 0.0

    xy = X_arr[:2, :]
    diffs = np.diff(xy, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)

    finite_segments = segment_lengths[np.isfinite(segment_lengths)]
    if finite_segments.size == 0:
        return 0.0

    return float(np.sum(finite_segments))
