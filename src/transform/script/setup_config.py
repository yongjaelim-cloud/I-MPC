#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS /setup transport helpers.

The setter is the only JSON file reader. Every other ROS node obtains the
configuration from the latched /setup topic and reconstructs the same nested
dictionary in memory.
"""
import copy
import math
import threading
import time

import rospy
from drone_msgs.msg import Setup


# -----------------------------------------------------------------------------
# Flag-organized JSON resolver
# -----------------------------------------------------------------------------
#
# New JSON layout:
#   mpc_flag.selected / mpc_flag[0..4].variables
#   field_flag.selected / field_flag[0..4].variables
#   common
#
# Only setter.py / setter_gazebo.py read that JSON.  Before filling the typed Setup message we
# resolve it back to the legacy nested runtime tree (config/ego/mpc/normal_cfg,
# ...).  Therefore every subscriber can keep using setup_msg_to_raw() unchanged.
# Legacy flat/nested JSON files are still accepted for backward compatibility.

_FIELD_PROFILE_NORMAL_CFG_KEYS = {
    "N_n",
    "circle_union_lambda",
    "dcsat_alpha",
    "dcsat_union_lambda",
    "wu_kappa",
    "wu_union_lambda",
    "wu_buffer_offset",
    "bp_order",
    "bp_training_n_x",
    "bp_training_n_y",
    "bp_safety_margin_l",
    "bp_domain_padding",
}


def _deep_merge_dict(base, override):
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _selected_flag(group, name, default=4):
    if not isinstance(group, dict):
        return int(default)
    try:
        flag = int(round(float(group.get("selected", default))))
    except Exception:
        flag = int(default)
    if flag not in (0, 1, 2, 3, 4):
        raise ValueError("{}.selected must be one of 0,1,2,3,4; got {!r}".format(name, flag))
    return flag


def _profile_variables(group, flag):
    if not isinstance(group, dict):
        return {}
    entry = group.get(str(int(flag)), {})
    if not isinstance(entry, dict):
        return {}
    variables = entry.get("variables", {})
    return variables if isinstance(variables, dict) else {}


def _apply_mpc_profile(resolved, variables):
    resolved.setdefault("mpc", {})
    if not isinstance(resolved["mpc"], dict):
        resolved["mpc"] = {}
    for key, value in variables.items():
        resolved["mpc"][key] = copy.deepcopy(value)


def _apply_field_profile(resolved, variables):
    resolved.setdefault("mpc", {})
    resolved.setdefault("normal_cfg", {})
    if not isinstance(resolved["mpc"], dict):
        resolved["mpc"] = {}
    if not isinstance(resolved["normal_cfg"], dict):
        resolved["normal_cfg"] = {}
    for key, value in variables.items():
        # Most collision-field parameters historically live in normal_cfg.
        # lambda_o is retained in mpc because the typed Setup.msg already has
        # the corresponding mpc_lambda_o field and downstream flattening expects
        # that legacy transport layout.
        target = resolved["normal_cfg"] if key in _FIELD_PROFILE_NORMAL_CFG_KEYS else resolved["mpc"]
        target[key] = copy.deepcopy(value)



def _auto_resolve_proposed_parameters(resolved):
    """Derive theorem parameters from theta, delta_o, and delta_f.

    Fixed design choice:
        epsilon_o = 0.999

    For each N_n, let
        Delta = pi / (2 (N_n + 1))
        theta_err = atan2(1-epsilon_tot,
                          epsilon_tot*cos(Delta) - (1-epsilon_tot)).

    We choose the smallest N_n for which there exists
        1/2 < epsilon_tot < epsilon_o
    satisfying theta_err + Delta <= theta.  For that N_n, epsilon_tot is the
    smallest value satisfying the angular inequality (plus a tiny numerical
    guard), and epsilon_f = epsilon_tot / epsilon_o.

    The common smoothing value is chosen conservatively from the dominant-weight
    bounds
        lambda_o <= delta_o / log((M_max-1) epsilon_o/(1-epsilon_o)),
        lambda_f <= delta_f / log((F_max-1) epsilon_f/(1-epsilon_f)),
    where M_max and F_max are explicit JSON parameters under common.mpc.
    They are not inferred from config.max_obs or obstacle vertex-count settings.

    Finally lambda_f = lambda_o = lambda_r = regulation_lambda = lambda_common.
    """
    resolved.setdefault("mpc", {})
    resolved.setdefault("normal_cfg", {})
    mpc = resolved["mpc"]
    normal = resolved["normal_cfg"]
    config = resolved.get("config", {}) if isinstance(resolved.get("config"), dict) else {}

    def _finite_positive(name, default):
        try:
            value = float(mpc.get(name, default))
        except Exception:
            value = float(default)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and > 0; got {!r}".format(name, mpc.get(name)))
        return value

    theta = _finite_positive("theta", math.pi / 4.0)
    if theta > math.pi / 2.0:
        raise ValueError("theta must satisfy 0 < theta <= pi/2; got {!r}".format(theta))
    delta_o = _finite_positive("delta_o", 0.008)
    delta_f = _finite_positive("delta_f", 0.005)

    epsilon_o = 0.999
    numerical_guard = 1.0e-10
    chosen_n = None
    epsilon_tot = None

    for n in range(0, 100000):
        delta_theta_n = math.pi / (2.0 * float(n + 1))
        # With epsilon_tot < 1, theta_err is strictly positive, so Delta must be < theta.
        if delta_theta_n >= theta:
            continue
        gamma = theta - delta_theta_n
        tan_gamma = math.tan(gamma)
        denom = 1.0 + tan_gamma * (1.0 + math.cos(delta_theta_n))
        if not math.isfinite(denom) or denom <= 0.0:
            continue
        eps_required = (1.0 + tan_gamma) / denom
        eps_required = max(float(eps_required), 0.5)
        eps_candidate = max(eps_required + numerical_guard, 0.5 + numerical_guard)
        # epsilon_f=epsilon_tot/epsilon_o must remain strictly below one so that
        # a positive finite feature-smoothing lambda exists.
        if eps_candidate < epsilon_o:
            chosen_n = int(n)
            epsilon_tot = float(eps_candidate)
            break

    if chosen_n is None or epsilon_tot is None:
        raise ValueError(
            "no finite N_n can satisfy theta={!r} with epsilon_o=0.999 and positive lambda_f".format(theta)
        )

    epsilon_f = epsilon_tot / epsilon_o
    if not (0.5 < epsilon_tot < epsilon_o < 1.0) or not (0.0 < epsilon_f < 1.0):
        raise ValueError(
            "invalid derived epsilons: epsilon_tot={!r}, epsilon_o={!r}, epsilon_f={!r}".format(
                epsilon_tot, epsilon_o, epsilon_f
            )
        )

    def _positive_int(name, default):
        try:
            value = int(mpc.get(name, default))
        except Exception:
            value = int(default)
        if value < 1:
            raise ValueError("{} must be an integer >= 1; got {!r}".format(name, mpc.get(name)))
        return value

    # The theorem cardinality bounds are explicit design parameters.
    # Do not infer them from runtime capacities such as config.max_obs,
    # obs_max_vertex_num, or msg_max_vertices.
    M_max = _positive_int("M_max", 200)
    F_max = _positive_int("F_max", 36)

    def _lambda_upper(delta, epsilon, count):
        if count <= 1:
            return float("inf")
        ratio = (float(count) - 1.0) * float(epsilon) / (1.0 - float(epsilon))
        if ratio <= 1.0:
            return float("inf")
        return float(delta) / math.log(ratio)

    lambda_o_max = _lambda_upper(delta_o, epsilon_o, M_max)
    lambda_f_max = _lambda_upper(delta_f, epsilon_f, F_max)
    lambda_bound = min(lambda_o_max, lambda_f_max)
    if not math.isfinite(lambda_bound) or lambda_bound <= 0.0:
        raise ValueError(
            "failed to derive positive smoothing lambda: lambda_o_max={!r}, lambda_f_max={!r}".format(
                lambda_o_max, lambda_f_max
            )
        )
    # Stay just inside both bounds rather than relying on equality at floating-point precision.
    lambda_common = max(1.0e-12, 0.999 * lambda_bound)

    mpc["delta_o"] = float(delta_o)
    mpc["delta_f"] = float(delta_f)
    mpc["M_max"] = int(M_max)
    mpc["F_max"] = int(F_max)
    mpc["epsilon_o"] = float(epsilon_o)
    mpc["epsilon_f"] = float(epsilon_f)
    mpc["epsilon_tot"] = float(epsilon_tot)
    mpc["lambda_f"] = float(lambda_common)
    mpc["lambda_o"] = float(lambda_common)
    mpc["lambda_r"] = float(lambda_common)
    mpc["regulation_lambda"] = float(lambda_common)
    normal["N_n"] = int(chosen_n)
    return resolved

def resolve_flagged_config(raw):
    """Resolve the flag-organized JSON into the legacy runtime dictionary.

    All profiles are first merged to preserve the complete typed /setup payload
    used by older nodes/tools.  The selected profile is then merged once more so
    any duplicated parameter (for example surface_slack_max) takes the value of
    the active mode.  The selected values themselves are restored as
    mpc.mpc_flag and normal_cfg.field_flag.
    """
    if not isinstance(raw, dict):
        raise ValueError("setup configuration must be a dictionary")

    common = raw.get("common")
    mpc_group = raw.get("mpc_flag")
    field_group = raw.get("field_flag")
    if not isinstance(common, dict) or not isinstance(mpc_group, dict) or not isinstance(field_group, dict):
        return copy.deepcopy(raw)

    resolved = copy.deepcopy(common)
    if "description" in raw:
        resolved["description"] = copy.deepcopy(raw["description"])

    mpc_selected = _selected_flag(mpc_group, "mpc_flag", 4)
    field_selected = _selected_flag(field_group, "field_flag", 4)

    # Populate the full compatibility payload, then let the active profile win.
    for flag in range(5):
        _apply_mpc_profile(resolved, _profile_variables(mpc_group, flag))
    _apply_mpc_profile(resolved, _profile_variables(mpc_group, mpc_selected))
    resolved.setdefault("mpc", {})["mpc_flag"] = int(mpc_selected)

    for flag in range(5):
        _apply_field_profile(resolved, _profile_variables(field_group, flag))
    _apply_field_profile(resolved, _profile_variables(field_group, field_selected))
    resolved.setdefault("normal_cfg", {})["field_flag"] = int(field_selected)

    # Preserve the old informational mpc_variants block in /setup without
    # duplicating it in the JSON source.
    variants = {}
    for flag in range(5):
        entry = mpc_group.get(str(flag), {})
        if isinstance(entry, dict):
            text = entry.get("description", entry.get("name", ""))
            variants[str(flag)] = str(text)
    resolved["mpc_variants"] = variants

    # initial_guidance_count is derived from the single-source seed ordering.
    # This avoids maintaining a redundant count in JSON.
    mpc_cfg = resolved.get("mpc", {})
    if isinstance(mpc_cfg, dict):
        acc_order = mpc_cfg.get("acc_order", [])
        yawrate_order = mpc_cfg.get("yawrate_order", [])
        if isinstance(acc_order, (list, tuple)) and isinstance(yawrate_order, (list, tuple)):
            if len(acc_order) > 0 and len(yawrate_order) > 0:
                mpc_cfg["initial_guidance_count"] = int(len(acc_order) * len(yawrate_order))

    # Backward-compatible transport for warehouse dynamic obstacle count.
    # Older generated Setup.msg files already contain config_obs_num_dyn but
    # do not contain config_warehouse_dynamic_obstacle_count.  In scenario 2,
    # copy the warehouse count into obs_num_dyn before serializing /setup.
    # The JSON remains single-source: warehouse_dynamic_obstacle_count is the
    # only warehouse-count parameter the user needs to edit.
    cfg = resolved.get("config", {})
    if isinstance(cfg, dict):
        try:
            scenario_flag = int(cfg.get("scenario_flag", 0))
        except Exception:
            scenario_flag = 0
        if scenario_flag == 2:
            if "warehouse_static_obstacle_count" in cfg:
                cfg["obs_num_static"] = int(cfg["warehouse_static_obstacle_count"])
            if "warehouse_dynamic_obstacle_count" in cfg:
                cfg["obs_num_dyn"] = int(cfg["warehouse_dynamic_obstacle_count"])

    _auto_resolve_proposed_parameters(resolved)
    return resolved

FIELD_MAP = {'config_fov': {'kind': 'scalar', 'path': ['config', 'FOV'], 'type': 'float'},
 'config_goal_x': {'kind': 'scalar', 'path': ['config', 'goal_x'], 'type': 'float'},
 'config_goal_y': {'kind': 'scalar', 'path': ['config', 'goal_y'], 'type': 'float'},
 'config_max_obs': {'kind': 'scalar', 'path': ['config', 'max_obs'], 'type': 'int'},
 'config_max_velocity': {'kind': 'scalar', 'path': ['config', 'max_velocity'], 'type': 'float'},
 'config_scenario_flag': {'kind': 'scalar', 'path': ['config', 'scenario_flag'], 'type': 'int'},
 'config_warehouse_static_obstacle_count': {'kind': 'scalar', 'path': ['config', 'warehouse_static_obstacle_count'], 'type': 'int'},
 'config_warehouse_static_min_node_gap': {'kind': 'scalar', 'path': ['config', 'warehouse_static_min_node_gap'], 'type': 'int'},
 'config_warehouse_graph_spacing_m': {'kind': 'scalar', 'path': ['config', 'warehouse_graph_spacing_m'], 'type': 'float'},
 'config_warehouse_rack_station_start_m': {'kind': 'scalar', 'path': ['config', 'warehouse_rack_station_start_m'], 'type': 'float'},
 'config_warehouse_rack_station_spacing_m': {'kind': 'scalar', 'path': ['config', 'warehouse_rack_station_spacing_m'], 'type': 'float'},
 'config_warehouse_rack_length_m': {'kind': 'scalar', 'path': ['config', 'warehouse_rack_length_m'], 'type': 'float'},
 'config_warehouse_rack_width_m': {'kind': 'scalar', 'path': ['config', 'warehouse_rack_width_m'], 'type': 'float'},
 'config_warehouse_rack_lateral_offset_m': {'kind': 'scalar', 'path': ['config', 'warehouse_rack_lateral_offset_m'], 'type': 'float'},
 'config_warehouse_dynamic_obstacle_count': {'kind': 'scalar', 'path': ['config', 'warehouse_dynamic_obstacle_count'], 'type': 'int'},
 'config_warehouse_forklift_length_m': {'kind': 'scalar', 'path': ['config', 'warehouse_forklift_length_m'], 'type': 'float'},
 'config_warehouse_forklift_width_m': {'kind': 'scalar', 'path': ['config', 'warehouse_forklift_width_m'], 'type': 'float'},
 'config_warehouse_forklift_speed_mps': {'kind': 'scalar', 'path': ['config', 'warehouse_forklift_speed_mps'], 'type': 'float'},
 'config_warehouse_forklift_yawrate_rps': {'kind': 'scalar', 'path': ['config', 'warehouse_forklift_yawrate_rps'], 'type': 'float'},
 'config_obs_fov_deg': {'kind': 'scalar', 'path': ['config', 'obs_fov_deg'], 'type': 'float'},
 'config_obs_max_radius': {'kind': 'scalar', 'path': ['config', 'obs_max_radius'], 'type': 'float'},
 'config_obs_max_vertex_num': {'kind': 'scalar', 'path': ['config', 'obs_max_vertex_num'], 'type': 'int'},
 'config_obs_min_dist_m': {'kind': 'scalar', 'path': ['config', 'obs_min_dist_m'], 'type': 'float'},
 'config_obs_min_radius': {'kind': 'scalar', 'path': ['config', 'obs_min_radius'], 'type': 'float'},
 'config_obs_min_vertex_num': {'kind': 'scalar', 'path': ['config', 'obs_min_vertex_num'], 'type': 'int'},
 'config_obs_num_dyn': {'kind': 'scalar', 'path': ['config', 'obs_num_dyn'], 'type': 'int'},
 'config_obs_num_static': {'kind': 'scalar', 'path': ['config', 'obs_num_static'], 'type': 'int'},
 'config_obstacle_max_trial': {'kind': 'scalar', 'path': ['config', 'obstacle_max_trial'], 'type': 'int'},
 'config_obstacle_save_path': {'kind': 'scalar', 'path': ['config', 'obstacle_save_path'], 'type': 'str'},
 'config_obstacle_seed': {'kind': 'scalar', 'path': ['config', 'obstacle_seed'], 'type': 'int'},
 'config_publish_rate': {'kind': 'scalar', 'path': ['config', 'publish_rate'], 'type': 'float'},
 'config_random_force_strength': {'kind': 'scalar', 'path': ['config', 'random_force_strength'], 'type': 'float'},
 'config_regenerate_obstacles': {'kind': 'scalar', 'path': ['config', 'regenerate_obstacles'], 'type': 'bool'},
 'config_repulsion_extra_margin': {'kind': 'scalar', 'path': ['config', 'repulsion_extra_margin'], 'type': 'float'},
 'config_repulsion_strength': {'kind': 'scalar', 'path': ['config', 'repulsion_strength'], 'type': 'float'},
 'config_sector_dir': {'kind': 'scalar', 'path': ['config', 'sector_dir'], 'type': 'str'},
 'config_sector_id': {'kind': 'scalar', 'path': ['config', 'sector_id'], 'type': 'int'},
 'config_start_goal_clearance_m': {'kind': 'scalar', 'path': ['config', 'start_goal_clearance_m'], 'type': 'float'},
 'config_time_interval': {'kind': 'scalar', 'path': ['config', 'time_interval'], 'type': 'float'},
 'config_topics_ego_goal_set': {'kind': 'scalar', 'path': ['config', 'topics', 'ego_goal_set'], 'type': 'str'},
 'config_topics_ego_state': {'kind': 'scalar', 'path': ['config', 'topics', 'ego_state'], 'type': 'str'},
 'config_topics_obstacle': {'kind': 'scalar', 'path': ['config', 'topics', 'obstacle'], 'type': 'str'},
 'config_collision_push_static': {'kind': 'scalar', 'path': ['config', 'collision_push_static'], 'type': 'bool'},
 'config_v_max': {'kind': 'scalar', 'path': ['config', 'V_max'], 'type': 'float'},
 'config_x_end': {'kind': 'scalar', 'path': ['config', 'x_end'], 'type': 'float'},
 'config_x_max': {'kind': 'scalar', 'path': ['config', 'x_max'], 'type': 'float'},
 'config_x_min': {'kind': 'scalar', 'path': ['config', 'x_min'], 'type': 'float'},
 'config_y_end': {'kind': 'scalar', 'path': ['config', 'y_end'], 'type': 'float'},
 'config_y_max': {'kind': 'scalar', 'path': ['config', 'y_max'], 'type': 'float'},
 'config_y_min': {'kind': 'scalar', 'path': ['config', 'y_min'], 'type': 'float'},
 'description': {'kind': 'scalar', 'path': ['description'], 'type': 'str'},
 'ego_control_candidates_per_seq': {'kind': 'scalar', 'path': ['ego', 'control_candidates_per_seq'], 'type': 'int'},
 'ego_ego_l': {'kind': 'scalar', 'path': ['ego', 'ego_L'], 'type': 'float'},
 'ego_ego_v': {'kind': 'scalar', 'path': ['ego', 'ego_v'], 'type': 'float'},
 'ego_ego_w': {'kind': 'scalar', 'path': ['ego', 'ego_W'], 'type': 'float'},
 'ego_ego_x': {'kind': 'scalar', 'path': ['ego', 'ego_x'], 'type': 'float'},
 'ego_ego_y': {'kind': 'scalar', 'path': ['ego', 'ego_y'], 'type': 'float'},
 'ego_ego_yaw': {'kind': 'scalar', 'path': ['ego', 'ego_yaw'], 'type': 'float'},
 'ego_frame_id': {'kind': 'scalar', 'path': ['ego', 'frame_id'], 'type': 'str'},
 'ego_goal_corner_margin': {'kind': 'scalar', 'path': ['ego', 'goal_corner_margin'], 'type': 'float'},
 'ego_goal_switch_enabled': {'kind': 'scalar', 'path': ['ego', 'goal_switch_enabled'], 'type': 'bool'},
 'ego_goal_switch_timeout_s': {'kind': 'scalar', 'path': ['ego', 'goal_switch_timeout_s'], 'type': 'float'},
 'ego_goal_tolerance': {'kind': 'scalar', 'path': ['ego', 'goal_tolerance'], 'type': 'float'},
 'ego_hz': {'kind': 'scalar', 'path': ['ego', 'hz'], 'type': 'float'},
 'ego_input_topics_ego_goal_set': {'kind': 'scalar', 'path': ['ego', 'input_topics', 'ego_goal_set'], 'type': 'str'},
 'ego_input_topics_mpc_control_seq': {'kind': 'scalar', 'path': ['ego', 'input_topics', 'mpc_control_seq'], 'type': 'str'},
 'ego_input_topics_mpc_control_seq_topics': {'kind': 'array', 'path': ['ego', 'input_topics', 'mpc_control_seq_topics'], 'type': 'str'},
 'ego_publish_initial_cycle_goal': {'kind': 'scalar', 'path': ['ego', 'publish_initial_cycle_goal'], 'type': 'bool'},
 'ego_publishers_ego_goal_setting': {'kind': 'scalar', 'path': ['ego', 'publishers', 'ego_goal_setting'], 'type': 'str'},
 'ego_publishers_ego_state': {'kind': 'scalar', 'path': ['ego', 'publishers', 'ego_state'], 'type': 'str'},
 'ego_reset_from_goal_once': {'kind': 'scalar', 'path': ['ego', 'reset_from_goal_once'], 'type': 'bool'},
 'ego_reset_on_every_goal': {'kind': 'scalar', 'path': ['ego', 'reset_on_every_goal'], 'type': 'bool'},
 'ego_traj_similarity_pos_weight': {'kind': 'scalar', 'path': ['ego', 'traj_similarity_pos_weight'], 'type': 'float'},
 'ego_traj_similarity_v_weight': {'kind': 'scalar', 'path': ['ego', 'traj_similarity_v_weight'], 'type': 'float'},
 'ego_traj_similarity_weight': {'kind': 'scalar', 'path': ['ego', 'traj_similarity_weight'], 'type': 'float'},
 'ego_traj_similarity_yaw_weight': {'kind': 'scalar', 'path': ['ego', 'traj_similarity_yaw_weight'], 'type': 'float'},
 'ego_acc_max': {'kind': 'scalar', 'path': ['ego', 'acc_max'], 'type': 'float'},
 'ego_v_max': {'kind': 'scalar', 'path': ['ego', 'v_max'], 'type': 'float'},
 'guidance_dt_init': {'kind': 'scalar', 'path': ['guidance', 'dt_init'], 'type': 'float'},
 'guidance_radius_interval': {'kind': 'scalar', 'path': ['guidance', 'radius_interval'], 'type': 'float'},
 'guidance_radius_max': {'kind': 'scalar', 'path': ['guidance', 'radius_max'], 'type': 'float'},
 'guidance_ego_v_min': {'kind': 'scalar', 'path': ['guidance', 'ego_v_min'], 'type': 'float'},
 'guidance_n': {'kind': 'scalar', 'path': ['guidance', 'N'], 'type': 'int'},
 'guidance_theta_sampling_num': {'kind': 'scalar', 'path': ['guidance', 'theta_sampling_num'], 'type': 'int'},
 'guidance_theta_min_deg': {'kind': 'scalar', 'path': ['guidance', 'theta_min_deg'], 'type': 'float'},
 'guidance_theta_max_deg': {'kind': 'scalar', 'path': ['guidance', 'theta_max_deg'], 'type': 'float'},
 'guidance_yawrate_min': {'kind': 'scalar', 'path': ['guidance', 'yawrate_min'], 'type': 'float'},
 'guidance_yawrate_max': {'kind': 'scalar', 'path': ['guidance', 'yawrate_max'], 'type': 'float'},
 'mpc_a_max': {'kind': 'scalar', 'path': ['mpc', 'a_max'], 'type': 'float'},
 'mpc_a_min': {'kind': 'scalar', 'path': ['mpc', 'a_min'], 'type': 'float'},
 'mpc_acados_code_export_dir': {'kind': 'scalar', 'path': ['mpc', 'acados_code_export_dir'], 'type': 'str'},
 'mpc_acados_fixed_hess': {'kind': 'scalar', 'path': ['mpc', 'acados_fixed_hess'], 'type': 'bool'},
 'mpc_acados_hessian_approx': {'kind': 'scalar', 'path': ['mpc', 'acados_hessian_approx'], 'type': 'str'},
 'mpc_acados_integrator_type': {'kind': 'scalar', 'path': ['mpc', 'acados_integrator_type'], 'type': 'str'},
 'mpc_acados_json_file': {'kind': 'scalar', 'path': ['mpc', 'acados_json_file'], 'type': 'str'},
 'mpc_acados_model_name': {'kind': 'scalar', 'path': ['mpc', 'acados_model_name'], 'type': 'str'},
 'mpc_acados_nlp_solver_max_iter': {'kind': 'scalar', 'path': ['mpc', 'acados_nlp_solver_max_iter'], 'type': 'int'},
 'mpc_acados_nlp_solver_type': {'kind': 'scalar', 'path': ['mpc', 'acados_nlp_solver_type'], 'type': 'str'},
 'mpc_acados_qp_solver': {'kind': 'scalar', 'path': ['mpc', 'acados_qp_solver'], 'type': 'str'},
 'mpc_acados_qp_solver_iter_max': {'kind': 'scalar', 'path': ['mpc', 'acados_qp_solver_iter_max'], 'type': 'int'},
 'mpc_acados_rebuild_solver': {'kind': 'scalar', 'path': ['mpc', 'acados_rebuild_solver'], 'type': 'bool'},
 'mpc_acados_sim_method_num_stages': {'kind': 'scalar', 'path': ['mpc', 'acados_sim_method_num_stages'], 'type': 'int'},
 'mpc_acados_sim_method_num_steps': {'kind': 'scalar', 'path': ['mpc', 'acados_sim_method_num_steps'], 'type': 'int'},
 'mpc_acados_tol': {'kind': 'scalar', 'path': ['mpc', 'acados_tol'], 'type': 'float'},
 'mpc_alpha': {'kind': 'scalar', 'path': ['mpc', 'alpha'], 'type': 'float'},
 'mpc_angle_margin': {'kind': 'scalar', 'path': ['mpc', 'angle_margin'], 'type': 'float'},
 'mpc_delta_o': {'kind': 'scalar', 'path': ['mpc', 'delta_o'], 'type': 'float'},
 'mpc_delta_f': {'kind': 'scalar', 'path': ['mpc', 'delta_f'], 'type': 'float'},
 'mpc_m_max': {'kind': 'scalar', 'path': ['mpc', 'M_max'], 'type': 'int'},
 'mpc_f_max': {'kind': 'scalar', 'path': ['mpc', 'F_max'], 'type': 'int'},
 'mpc_epsilon_o': {'kind': 'scalar', 'path': ['mpc', 'epsilon_o'], 'type': 'float'},
 'mpc_epsilon_f': {'kind': 'scalar', 'path': ['mpc', 'epsilon_f'], 'type': 'float'},
 'mpc_boundary_slack_max': {'kind': 'scalar', 'path': ['mpc', 'boundary_slack_max'], 'type': 'float'},
 'mpc_collision_slack_max': {'kind': 'scalar', 'path': ['mpc', 'collision_slack_max'], 'type': 'float'},
 'mpc_collsion_avoidance_off_over_20': {'kind': 'scalar', 'path': ['mpc', 'collsion_avoidance_off_over_20'], 'type': 'bool'},
 'mpc_cpp_omp_threads': {'kind': 'scalar', 'path': ['mpc', 'cpp_omp_threads'], 'type': 'int'},
 'mpc_d_safe': {'kind': 'scalar', 'path': ['mpc', 'd_safe'], 'type': 'float'},
 'mpc_disabled_c_value': {'kind': 'scalar', 'path': ['mpc', 'disabled_c_value'], 'type': 'float'},
 'mpc_dt_init': {'kind': 'scalar', 'path': ['mpc', 'dt_init'], 'type': 'float'},
 'mpc_ego_trail_max': {'kind': 'scalar', 'path': ['mpc', 'ego_trail_max'], 'type': 'int'},
 'mpc_enable_boundary_constraint': {'kind': 'scalar', 'path': ['mpc', 'enable_boundary_constraint'], 'type': 'bool'},
 'mpc_enable_impact_normal_constraint': {'kind': 'scalar', 'path': ['mpc', 'enable_impact_normal_constraint'], 'type': 'bool'},
 'mpc_enable_normal_mag_avoidance_constraint': {'kind': 'scalar',
                                                'path': ['mpc', 'enable_normal_mag_avoidance_constraint'],
                                                'type': 'bool'},
 'mpc_fallback_polygon_vertices': {'kind': 'scalar', 'path': ['mpc', 'fallback_polygon_vertices'], 'type': 'int'},
 'mpc_global_candidate_objective': {'kind': 'scalar', 'path': ['mpc', 'global_candidate_objective'], 'type': 'str'},
 'mpc_h_upper_bound': {'kind': 'scalar', 'path': ['mpc', 'h_upper_bound'], 'type': 'float'},
 'mpc_hz': {'kind': 'scalar', 'path': ['mpc', 'hz'], 'type': 'float'},
 'mpc_initial_guidance_count': {'kind': 'scalar', 'path': ['mpc', 'initial_guidance_count'], 'type': 'int'},
 'mpc_acc_order': {'kind': 'array', 'path': ['mpc', 'acc_order'], 'type': 'int'},
 'mpc_yawrate_order': {'kind': 'array', 'path': ['mpc', 'yawrate_order'], 'type': 'int'},
 'mpc_initial_guidance_topic': {'kind': 'scalar', 'path': ['mpc', 'initial_guidance_topic'], 'type': 'str'},
 'mpc_initial_traj_topic': {'kind': 'scalar', 'path': ['mpc', 'initial_traj_topic'], 'type': 'str'},
 'mpc_lambda_f': {'kind': 'scalar', 'path': ['mpc', 'lambda_f'], 'type': 'float'},
 'mpc_lambda_o': {'kind': 'scalar', 'path': ['mpc', 'lambda_o'], 'type': 'float'},
 'mpc_lambda_r': {'kind': 'scalar', 'path': ['mpc', 'lambda_r'], 'type': 'float'},
 'mpc_lexicographic_fix_tolerance': {'kind': 'scalar', 'path': ['mpc', 'lexicographic_fix_tolerance'], 'type': 'float'},
 'mpc_theta': {'kind': 'scalar', 'path': ['mpc', 'theta'], 'type': 'float'},
 'mpc_reg_margin': {'kind': 'scalar', 'path': ['mpc', 'reg_margin'], 'type': 'float'},
 'mpc_comment': {'kind': 'scalar', 'path': ['mpc', 'comment'], 'type': 'str'},
 'mpc_epsilon_tot': {'kind': 'scalar', 'path': ['mpc', 'epsilon_tot'], 'type': 'float'},
 'mpc_marker_width': {'kind': 'scalar', 'path': ['mpc', 'marker_width'], 'type': 'float'},
 'mpc_moving_obstacle_r': {'kind': 'scalar', 'path': ['mpc', 'moving_obstacle_R'], 'type': 'float'},
 'mpc_mpc_flag': {'kind': 'scalar', 'path': ['mpc', 'mpc_flag'], 'type': 'int'},
 'mpc_mppi_sampling_cnt': {'kind': 'scalar', 'path': ['mpc', 'mppi_sampling_cnt'], 'type': 'int'},
 'mpc_msg_max_vertices': {'kind': 'scalar', 'path': ['mpc', 'msg_max_vertices'], 'type': 'int'},
 'mpc_n': {'kind': 'scalar', 'path': ['mpc', 'N'], 'type': 'int'},
 'mpc_no_acc_flag': {'kind': 'scalar', 'path': ['mpc', 'no_acc_flag'], 'type': 'bool'},
 'mpc_nonlinear_dyn_flag': {'kind': 'scalar', 'path': ['mpc', 'nonlinear_dyn_flag'], 'type': 'bool'},
 'mpc_debug_mode': {'kind': 'scalar', 'path': ['mpc', 'debug_mode'], 'type': 'bool'},
 'mpc_cal_time_log_flag': {'kind': 'scalar', 'path': ['mpc', 'cal_time_log_flag'], 'type': 'bool'},
 'mpc_normal_alignment_slack_max': {'kind': 'scalar', 'path': ['mpc', 'normal_alignment_slack_max'], 'type': 'float'},
 'mpc_normal_eps': {'kind': 'scalar', 'path': ['mpc', 'normal_eps'], 'type': 'float'},
 'mpc_normal_orientation': {'kind': 'scalar', 'path': ['mpc', 'normal_orientation'], 'type': 'str'},
 'mpc_np_max': {'kind': 'scalar', 'path': ['mpc', 'Np_max'], 'type': 'int'},
 'mpc_np_min': {'kind': 'scalar', 'path': ['mpc', 'Np_min'], 'type': 'int'},
 'mpc_order': {'kind': 'scalar', 'path': ['mpc', 'ORDER'], 'type': 'int'},
 'mpc_perception_rate': {'kind': 'scalar', 'path': ['mpc', 'perception_rate'], 'type': 'float'},
 'mpc_planner_id': {'kind': 'scalar', 'path': ['mpc', 'planner_id'], 'type': 'int'},
 'mpc_q_boundary_slack': {'kind': 'scalar', 'path': ['mpc', 'q_boundary_slack'], 'type': 'float'},
 'mpc_q_bspline_tracking': {'kind': 'scalar', 'path': ['mpc', 'q_bspline_tracking'], 'type': 'float'},
 'mpc_q_collision_slack': {'kind': 'scalar', 'path': ['mpc', 'q_collision_slack'], 'type': 'float'},
 'mpc_q_global_goal': {'kind': 'scalar', 'path': ['mpc', 'q_global_goal'], 'type': 'float'},
 'mpc_q_nonvulnerable_slack': {'kind': 'scalar', 'path': ['mpc', 'q_nonvulnerable_slack'], 'type': 'float'},
 'mpc_q_normal_alignment_slack': {'kind': 'scalar', 'path': ['mpc', 'q_normal_alignment_slack'], 'type': 'float'},
 'mpc_q_pos': {'kind': 'scalar', 'path': ['mpc', 'q_pos'], 'type': 'float'},
 'mpc_q_seq': {'kind': 'scalar', 'path': ['mpc', 'q_seq'], 'type': 'float'},
 'mpc_q_pos_terminal': {'kind': 'scalar', 'path': ['mpc', 'q_pos_terminal'], 'type': 'float'},
 'mpc_q_surface_slack': {'kind': 'scalar', 'path': ['mpc', 'q_surface_slack'], 'type': 'float'},
 'mpc_q_v': {'kind': 'scalar', 'path': ['mpc', 'q_v'], 'type': 'float'},
 'mpc_q_vulnerable_slack': {'kind': 'scalar', 'path': ['mpc', 'q_vulnerable_slack'], 'type': 'float'},
 'mpc_q_yaw': {'kind': 'scalar', 'path': ['mpc', 'q_yaw'], 'type': 'float'},
 'mpc_r_a': {'kind': 'scalar', 'path': ['mpc', 'r_a'], 'type': 'float'},
 'mpc_r_du': {'kind': 'scalar', 'path': ['mpc', 'r_du'], 'type': 'float'},
 'mpc_r_yawrate': {'kind': 'scalar', 'path': ['mpc', 'r_yawrate'], 'type': 'float'},
 'mpc_realtime_max_sqp_iter': {'kind': 'scalar', 'path': ['mpc', 'realtime_max_sqp_iter'], 'type': 'int'},
 'mpc_regulation_lambda': {'kind': 'scalar', 'path': ['mpc', 'regulation_lambda'], 'type': 'float'},
 'mpc_rho_s': {'kind': 'scalar', 'path': ['mpc', 'rho_s'], 'type': 'float'},
 'mpc_rho_c': {'kind': 'scalar', 'path': ['mpc', 'rho_c'], 'type': 'float'},
 'mpc_rho_impact': {'kind': 'scalar', 'path': ['mpc', 'rho_impact'], 'type': 'float'},
 'mpc_rho_phi_global': {'kind': 'scalar', 'path': ['mpc', 'rho_Phi_global'], 'type': 'float'},
 'mpc_rho_terminal_goal_global': {'kind': 'scalar', 'path': ['mpc', 'rho_terminal_goal_global'], 'type': 'float'},
 'mpc_selected_vertex_set_timeout_s': {'kind': 'scalar', 'path': ['mpc', 'selected_vertex_set_timeout_s'], 'type': 'float'},
 'mpc_selected_vertex_set_topic': {'kind': 'scalar', 'path': ['mpc', 'selected_vertex_set_topic'], 'type': 'str'},
 'mpc_sqp_iter': {'kind': 'scalar', 'path': ['mpc', 'SQP_iter'], 'type': 'int'},
 'mpc_surface_slack_max': {'kind': 'scalar', 'path': ['mpc', 'surface_slack_max'], 'type': 'float'},
 'mpc_t_scale': {'kind': 'scalar', 'path': ['mpc', 't_scale'], 'type': 'float'},
 'mpc_use_cupy_local_frame_batch': {'kind': 'scalar', 'path': ['mpc', 'use_cupy_local_frame_batch'], 'type': 'bool'},
 'mpc_use_external_collision_linearization': {'kind': 'scalar', 'path': ['mpc', 'use_external_collision_linearization'], 'type': 'bool'},
 'mpc_use_native_cuda_proposed': {'kind': 'scalar', 'path': ['mpc', 'use_native_cuda_proposed'], 'type': 'bool'},
 'mpc_native_cuda_min_hf': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_min_hf'], 'type': 'int'},
 'mpc_native_cuda_threads': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_threads'], 'type': 'int'},
 'mpc_native_cuda_fallback_to_cpu': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_fallback_to_cpu'], 'type': 'bool'},
 'mpc_native_cuda_profile': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_profile'], 'type': 'bool'},
 'mpc_native_cuda_compare_cpu': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_compare_cpu'], 'type': 'bool'},
 'mpc_native_cuda_profile_every': {'kind': 'scalar', 'path': ['mpc', 'native_cuda_profile_every'], 'type': 'int'},
 'mpc_v_des': {'kind': 'scalar', 'path': ['mpc', 'v_des'], 'type': 'float'},
 'mpc_v_max': {'kind': 'scalar', 'path': ['mpc', 'v_max'], 'type': 'float'},
 'mpc_v_min': {'kind': 'scalar', 'path': ['mpc', 'v_min'], 'type': 'float'},
 'mpc_v_slack_weight': {'kind': 'scalar', 'path': ['mpc', 'v_slack_weight'], 'type': 'float'},
 'mpc_variants_v0': {'kind': 'scalar', 'path': ['mpc_variants', '0'], 'type': 'str'},
 'mpc_variants_v1': {'kind': 'scalar', 'path': ['mpc_variants', '1'], 'type': 'str'},
 'mpc_variants_v2': {'kind': 'scalar', 'path': ['mpc_variants', '2'], 'type': 'str'},
 'mpc_variants_v3': {'kind': 'scalar', 'path': ['mpc_variants', '3'], 'type': 'str'},
 'mpc_variants_v4': {'kind': 'scalar', 'path': ['mpc_variants', '4'], 'type': 'str'},
 'mpc_yawrate_max': {'kind': 'scalar', 'path': ['mpc', 'yawrate_max'], 'type': 'float'},
 'mpc_yawrate_min': {'kind': 'scalar', 'path': ['mpc', 'yawrate_min'], 'type': 'float'},
 'mppi_publishers_initial_guidance': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_guidance'], 'type': 'str'},
 'mppi_publishers_initial_guidance1': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_guidance1'], 'type': 'str'},
 'mppi_publishers_initial_guidance2': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_guidance2'], 'type': 'str'},
 'mppi_publishers_initial_guidance3': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_guidance3'], 'type': 'str'},
 'mppi_publishers_initial_traj': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_traj'], 'type': 'str'},
 'mppi_publishers_initial_traj1': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_traj1'], 'type': 'str'},
 'mppi_publishers_initial_traj2': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_traj2'], 'type': 'str'},
 'mppi_publishers_initial_traj3': {'kind': 'scalar', 'path': ['mppi', 'publishers', 'initial_traj3'], 'type': 'str'},
 'normal_cfg_bp_domain_padding': {'kind': 'scalar', 'path': ['normal_cfg', 'bp_domain_padding'], 'type': 'float'},
 'normal_cfg_bp_order': {'kind': 'scalar', 'path': ['normal_cfg', 'bp_order'], 'type': 'int'},
 'normal_cfg_bp_safety_margin_l': {'kind': 'scalar', 'path': ['normal_cfg', 'bp_safety_margin_l'], 'type': 'float'},
 'normal_cfg_bp_training_n_x': {'kind': 'scalar', 'path': ['normal_cfg', 'bp_training_n_x'], 'type': 'int'},
 'normal_cfg_bp_training_n_y': {'kind': 'scalar', 'path': ['normal_cfg', 'bp_training_n_y'], 'type': 'int'},
 'normal_cfg_circle_union_lambda': {'kind': 'scalar', 'path': ['normal_cfg', 'circle_union_lambda'], 'type': 'float'},
 'normal_cfg_dcsat_alpha': {'kind': 'scalar', 'path': ['normal_cfg', 'dcsat_alpha'], 'type': 'float'},
 'normal_cfg_dcsat_union_lambda': {'kind': 'scalar', 'path': ['normal_cfg', 'dcsat_union_lambda'], 'type': 'float'},
 'normal_cfg_fd_position_eps_m': {'kind': 'scalar', 'path': ['normal_cfg', 'fd_position_eps_m'], 'type': 'float'},
 'normal_cfg_fd_yaw_eps_rad': {'kind': 'scalar', 'path': ['normal_cfg', 'fd_yaw_eps_rad'], 'type': 'float'},
 'normal_cfg_field_flag': {'kind': 'scalar', 'path': ['normal_cfg', 'field_flag'], 'type': 'int'},
 'normal_cfg_n_n': {'kind': 'scalar', 'path': ['normal_cfg', 'N_n'], 'type': 'int'},
 'normal_cfg_probability_collision': {'kind': 'scalar', 'path': ['normal_cfg', 'probability_collision'], 'type': 'float'},
 'normal_cfg_sigma_vx': {'kind': 'scalar', 'path': ['normal_cfg', 'sigma_vx'], 'type': 'float'},
 'normal_cfg_sigma_vy': {'kind': 'scalar', 'path': ['normal_cfg', 'sigma_vy'], 'type': 'float'},
 'normal_cfg_use_ego_scaled_local_frame': {'kind': 'scalar', 'path': ['normal_cfg', 'use_ego_scaled_local_frame'], 'type': 'bool'},
 'normal_cfg_use_probability_propagation_margin': {'kind': 'scalar',
                                                   'path': ['normal_cfg', 'use_probability_propagation_margin'],
                                                   'type': 'bool'},
 'normal_cfg_wu_buffer_offset': {'kind': 'scalar', 'path': ['normal_cfg', 'wu_buffer_offset'], 'type': 'float'},
 'normal_cfg_wu_kappa': {'kind': 'scalar', 'path': ['normal_cfg', 'wu_kappa'], 'type': 'float'},
 'normal_cfg_wu_union_lambda': {'kind': 'scalar', 'path': ['normal_cfg', 'wu_union_lambda'], 'type': 'float'},
 'planners_mpc_planner1_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner1', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner1_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner1', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner1_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner1', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner1_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner1',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner1_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner1', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner1_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner1',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner1_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner1', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner1_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner1', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner1_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner1', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner1_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner1', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner1_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner1', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner1_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner1', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner1_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner1', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner1_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner1', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner1_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner1', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner1_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner1', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner1_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner1', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner2_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner2', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner2_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner2', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner2_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner2', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner2_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner2',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner2_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner2', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner2_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner2',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner2_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner2', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner2_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner2', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner2_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner2', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner2_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner2', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner2_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner2', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner2_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner2', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner2_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner2', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner2_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner2', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner2_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner2', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner2_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner2', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner2_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner2', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner3_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner3', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner3_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner3', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner3_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner3', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner3_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner3',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner3_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner3', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner3_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner3',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner3_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner3', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner3_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner3', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner3_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner3', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner3_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner3', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner3_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner3', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner3_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner3', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner3_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner3', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner3_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner3', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner3_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner3', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner3_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner3', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner3_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner3', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner4_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner4', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner4_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner4', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner4_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner4', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner4_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner4',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner4_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner4', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner4_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner4',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner4_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner4', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner4_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner4', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner4_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner4', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner4_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner4', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner4_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner4', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner4_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner4', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner4_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner4', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner4_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner4', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner4_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner4', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner4_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner4', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner4_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner4', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner5_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner5', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner5_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner5', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner5_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner5', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner5_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner5',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner5_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner5', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner5_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner5',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner5_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner5', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner5_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner5', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner5_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner5', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner5_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner5', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner5_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner5', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner5_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner5', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner5_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner5', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner5_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner5', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner5_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner5', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner5_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner5', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner5_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner5', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner6_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner6', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner6_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner6', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner6_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner6', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner6_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner6',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner6_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner6', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner6_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner6',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner6_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner6', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner6_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner6', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner6_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner6', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner6_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner6', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner6_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner6', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner6_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner6', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner6_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner6', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner6_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner6', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner6_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner6', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner6_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner6', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner6_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner6', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner7_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner7', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner7_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner7', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner7_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner7', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner7_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner7',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner7_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner7', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner7_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner7',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner7_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner7', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner7_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner7', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner7_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner7', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner7_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner7', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner7_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner7', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner7_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner7', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner7_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner7', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner7_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner7', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner7_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner7', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner7_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner7', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner7_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner7', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner8_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner8', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner8_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner8', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner8_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner8', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner8_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner8',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner8_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner8', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner8_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner8',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner8_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner8', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner8_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner8', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner8_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner8', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner8_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner8', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner8_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner8', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner8_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner8', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner8_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner8', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner8_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner8', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner8_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner8', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner8_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner8', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner8_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner8', 'selected_vertex_color'],
                                                 'type': 'float'},
 'planners_mpc_planner9_ego_traj_color': {'kind': 'array', 'path': ['planners', 'MPC_planner9', 'ego_traj_color'], 'type': 'float'},
 'planners_mpc_planner9_initial_guidance_topic': {'kind': 'scalar',
                                                  'path': ['planners', 'MPC_planner9', 'initial_guidance_topic'],
                                                  'type': 'str'},
 'planners_mpc_planner9_input_topics_initial_guidance': {'kind': 'scalar',
                                                         'path': ['planners', 'MPC_planner9', 'input_topics', 'initial_guidance'],
                                                         'type': 'str'},
 'planners_mpc_planner9_input_topics_mpc_control_seq_topics': {'kind': 'array',
                                                               'path': ['planners',
                                                                        'MPC_planner9',
                                                                        'input_topics',
                                                                        'mpc_control_seq_topics'],
                                                               'type': 'str'},
 'planners_mpc_planner9_input_topics_selected_vertex_data': {'kind': 'scalar',
                                                             'path': ['planners', 'MPC_planner9', 'input_topics', 'selected_vertex_data'],
                                                             'type': 'str'},
 'planners_mpc_planner9_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                                     'path': ['planners',
                                                                              'MPC_planner9',
                                                                              'input_topics',
                                                                              'selected_vertex_horizon_data'],
                                                                     'type': 'str'},
 'planners_mpc_planner9_mpc_pred_color': {'kind': 'array', 'path': ['planners', 'MPC_planner9', 'mpc_pred_color'], 'type': 'float'},
 'planners_mpc_planner9_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner9', 'ORDER'], 'type': 'int'},
 'planners_mpc_planner9_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner9', 'planner_id'], 'type': 'int'},
 'planners_mpc_planner9_publish_percepted_obstacle': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner9', 'publish_percepted_obstacle'],
                                                      'type': 'bool'},
 'planners_mpc_planner9_publish_selected_vertices': {'kind': 'scalar',
                                                     'path': ['planners', 'MPC_planner9', 'publish_selected_vertices'],
                                                     'type': 'bool'},
 'planners_mpc_planner9_publishers_ego_traj': {'kind': 'scalar',
                                               'path': ['planners', 'MPC_planner9', 'publishers', 'ego_traj'],
                                               'type': 'str'},
 'planners_mpc_planner9_publishers_initial_guidance': {'kind': 'scalar',
                                                       'path': ['planners', 'MPC_planner9', 'publishers', 'initial_guidance'],
                                                       'type': 'str'},
 'planners_mpc_planner9_publishers_initial_traj': {'kind': 'scalar',
                                                   'path': ['planners', 'MPC_planner9', 'publishers', 'initial_traj'],
                                                   'type': 'str'},
 'planners_mpc_planner9_publishers_mpc_control_seq': {'kind': 'scalar',
                                                      'path': ['planners', 'MPC_planner9', 'publishers', 'mpc_control_seq'],
                                                      'type': 'str'},
 'planners_mpc_planner9_publishers_mpc_pred_traj': {'kind': 'scalar',
                                                    'path': ['planners', 'MPC_planner9', 'publishers', 'mpc_pred_traj'],
                                                    'type': 'str'},
 'planners_mpc_planner9_selected_vertex_color': {'kind': 'array',
                                                 'path': ['planners', 'MPC_planner9', 'selected_vertex_color'],
                                                 'type': 'float'},
 'plotter_ego_sector_arc_steps': {'kind': 'scalar', 'path': ['plotter', 'ego_sector_arc_steps'], 'type': 'int'},
 'plotter_ego_sector_fill_color': {'kind': 'array', 'path': ['plotter', 'ego_sector_fill_color'], 'type': 'float'},
 'plotter_ego_sector_outline_color': {'kind': 'array', 'path': ['plotter', 'ego_sector_outline_color'], 'type': 'float'},
 'plotter_ego_sector_radius_m': {'kind': 'scalar', 'path': ['plotter', 'ego_sector_radius_m'], 'type': 'float'},
 'plotter_ego_sector_topic': {'kind': 'scalar', 'path': ['plotter', 'ego_sector_topic'], 'type': 'str'},
 'plotter_initial_traj_topic': {'kind': 'scalar', 'path': ['plotter', 'initial_traj_topic'], 'type': 'str'},
 'plotter_initial_traj_topics': {'kind': 'array', 'path': ['plotter', 'initial_traj_topics'], 'type': 'str'},
 'plotter_mpc_control_seq_topics': {'kind': 'array', 'path': ['plotter', 'mpc_control_seq_topics'], 'type': 'str'},
 'plotter_mpc_pred_marker_width': {'kind': 'scalar', 'path': ['plotter', 'mpc_pred_marker_width'], 'type': 'float'},
 'plotter_mpc_pred_traj_colors': {'kind': 'matrix',
                                  'path': ['plotter', 'mpc_pred_traj_colors'],
                                  'shape_field': 'plotter_mpc_pred_traj_colors_shape',
                                  'type': 'float'},
 'plotter_mpc_pred_traj_topics': {'kind': 'array', 'path': ['plotter', 'mpc_pred_traj_topics'], 'type': 'str'},
 'plotter_pred_horizon': {'kind': 'scalar', 'path': ['plotter', 'pred_horizon'], 'type': 'int'},
 'plotter_pred_vis_step': {'kind': 'scalar', 'path': ['plotter', 'pred_vis_step'], 'type': 'int'},
 'plotter_publish_sdf': {'kind': 'scalar', 'path': ['plotter', 'publish_sdf'], 'type': 'bool'},
 'plotter_safe_margin_text_enabled': {'kind': 'scalar', 'path': ['plotter', 'safe_margin_text_enabled'], 'type': 'bool'},
 'plotter_sdf_grid_batch_size': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_batch_size'], 'type': 'int'},
 'plotter_sdf_grid_max': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_max'], 'type': 'float'},
 'plotter_sdf_grid_min': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_min'], 'type': 'float'},
 'plotter_sdf_grid_pub_hz': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_pub_hz'], 'type': 'float'},
 'plotter_sdf_grid_resolution': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_resolution'], 'type': 'float'},
 'plotter_sdf_grid_x_topic': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_x_topic'], 'type': 'str'},
 'plotter_sdf_grid_y_topic': {'kind': 'scalar', 'path': ['plotter', 'sdf_grid_y_topic'], 'type': 'str'},
 'plotter_sdf_topic': {'kind': 'scalar', 'path': ['plotter', 'sdf_topic'], 'type': 'str'},
 'plotter_selected_vertex_normal_topic': {'kind': 'scalar', 'path': ['plotter', 'selected_vertex_normal_topic'], 'type': 'str'},
 'plotter_selected_vertex_source': {'kind': 'scalar', 'path': ['plotter', 'selected_vertex_source'], 'type': 'str'},
 'plotter_show_ego_sector': {'kind': 'scalar', 'path': ['plotter', 'show_ego_sector'], 'type': 'bool'},
 'ros_input_topics_ego_goal_set': {'kind': 'scalar', 'path': ['ros', 'input_topics', 'ego_goal_set'], 'type': 'str'},
 'ros_input_topics_ego_state': {'kind': 'scalar', 'path': ['ros', 'input_topics', 'ego_state'], 'type': 'str'},
 'ros_input_topics_obstacle': {'kind': 'scalar', 'path': ['ros', 'input_topics', 'obstacle'], 'type': 'str'},
 'ros_input_topics_selected_vertex_data': {'kind': 'scalar', 'path': ['ros', 'input_topics', 'selected_vertex_data'], 'type': 'str'},
 'ros_input_topics_selected_vertex_horizon_data': {'kind': 'scalar',
                                                   'path': ['ros', 'input_topics', 'selected_vertex_horizon_data'],
                                                   'type': 'str'},
 'ros_input_topics_selected_vertex_set': {'kind': 'scalar', 'path': ['ros', 'input_topics', 'selected_vertex_set'], 'type': 'str'},
 'vertex_selector_hz': {'kind': 'scalar', 'path': ['vertex_selector', 'hz'], 'type': 'float'},
 'vertex_selector_input_topics_ego_state': {'kind': 'scalar', 'path': ['vertex_selector', 'input_topics', 'ego_state'], 'type': 'str'},
 'vertex_selector_input_topics_obstacle': {'kind': 'scalar', 'path': ['vertex_selector', 'input_topics', 'obstacle'], 'type': 'str'},
 'vertex_selector_publishers_percepted_obstacle': {'kind': 'scalar',
                                                   'path': ['vertex_selector', 'publishers', 'percepted_obstacle'],
                                                   'type': 'str'},
 'vertex_selector_publishers_selected_vertex_data': {'kind': 'scalar',
                                                     'path': ['vertex_selector', 'publishers', 'selected_vertex_data'],
                                                     'type': 'str'},
 'vertex_selector_publishers_selected_vertex_horizon_data': {'kind': 'scalar',
                                                             'path': ['vertex_selector', 'publishers', 'selected_vertex_horizon_data'],
                                                             'type': 'str'},
 'vertex_selector_publishers_selected_vertex_normal': {'kind': 'scalar',
                                                       'path': ['vertex_selector', 'publishers', 'selected_vertex_normal'],
                                                       'type': 'str'},
 'vertex_selector_publishers_selected_vertex_set': {'kind': 'scalar',
                                                    'path': ['vertex_selector', 'publishers', 'selected_vertex_set'],
                                                    'type': 'str'},
 'vertex_selector_selected_vertex_color': {'kind': 'array', 'path': ['vertex_selector', 'selected_vertex_color'], 'type': 'float'},
 'vertex_selector_selected_vertex_marker_width': {'kind': 'scalar',
                                                  'path': ['vertex_selector', 'selected_vertex_marker_width'],
                                                  'type': 'float'},
 'vertex_selector_selected_vertex_normal_arrow_length': {'kind': 'scalar',
                                                         'path': ['vertex_selector', 'selected_vertex_normal_arrow_length'],
                                                         'type': 'float'},
 'vertex_selector_selected_vertex_normal_color': {'kind': 'array',
                                                  'path': ['vertex_selector', 'selected_vertex_normal_color'],
                                                  'type': 'float'},
 'vertex_selector_selected_vertex_normal_head_diameter': {'kind': 'scalar',
                                                          'path': ['vertex_selector', 'selected_vertex_normal_head_diameter'],
                                                          'type': 'float'},
 'vertex_selector_selected_vertex_normal_head_length': {'kind': 'scalar',
                                                        'path': ['vertex_selector', 'selected_vertex_normal_head_length'],
                                                        'type': 'float'},
 'vertex_selector_selected_vertex_normal_shaft_diameter': {'kind': 'scalar',
                                                           'path': ['vertex_selector', 'selected_vertex_normal_shaft_diameter'],
                                                           'type': 'float'}}

# MPC_planner0: previous-selected-trajectory nominal planner.
# Keep this explicit in the typed /setup transport so downstream nodes can
# discover /mpc_control_seq0 and /selected_traj without reading JSON directly.
FIELD_MAP.update({
    'planners_mpc_planner0_planner_id': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'planner_id'], 'type': 'int'},
    'planners_mpc_planner0_order': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'ORDER'], 'type': 'int'},
    'planners_mpc_planner0_use_initial_guidance': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'use_initial_guidance'], 'type': 'bool'},
    'planners_mpc_planner0_publish_percepted_obstacle': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'publish_percepted_obstacle'], 'type': 'bool'},
    'planners_mpc_planner0_publish_selected_vertices': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'publish_selected_vertices'], 'type': 'bool'},
    'planners_mpc_planner0_publishers_mpc_control_seq': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'publishers', 'mpc_control_seq'], 'type': 'str'},
    'planners_mpc_planner0_publishers_ego_traj': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'publishers', 'ego_traj'], 'type': 'str'},
    'planners_mpc_planner0_input_topics_selected_vertex_data': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'input_topics', 'selected_vertex_data'], 'type': 'str'},
    'planners_mpc_planner0_input_topics_selected_vertex_horizon_data': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'input_topics', 'selected_vertex_horizon_data'], 'type': 'str'},
    'planners_mpc_planner0_input_topics_selected_traj': {'kind': 'scalar', 'path': ['planners', 'MPC_planner0', 'input_topics', 'selected_traj'], 'type': 'str'},
})

def _deep_set(root, path, value):
    cur = root
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value

def raw_to_setup_msg(raw, source_file=""):
    raw = resolve_flagged_config(raw)
    msg = Setup()
    if hasattr(msg, "header"):
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "setup"
    msg.source_file = str(source_file)
    for field, spec in FIELD_MAP.items():
        # Keep compatibility with an older generated Setup.msg until catkin is rebuilt.
        if not hasattr(msg, field):
            continue
        cur = raw
        found = True
        for key in spec["path"]:
            if not isinstance(cur, dict) or key not in cur:
                found = False
                break
            cur = cur[key]
        if not found:
            continue
        kind = spec["kind"]
        if kind == "matrix":
            rows = list(cur)
            cols = max([len(r) for r in rows] or [0])
            flat = []
            for row in rows:
                row = list(row)
                flat.extend(float(x) for x in row)
                if len(row) < cols:
                    flat.extend([0.0] * (cols - len(row)))
            setattr(msg, field, flat)
            setattr(msg, spec["shape_field"], [len(rows), cols])
        elif kind == "array":
            setattr(msg, field, list(cur))
        else:
            setattr(msg, field, cur)
    return msg

def setup_msg_to_raw(msg):
    raw = {}
    for field, spec in FIELD_MAP.items():
        # Older cached ROS message classes may not contain newly added setup fields.
        if not hasattr(msg, field):
            continue
        value = getattr(msg, field)
        kind = spec["kind"]
        if kind == "matrix":
            shape = list(getattr(msg, spec["shape_field"]))
            flat = list(value)
            if len(shape) >= 2 and shape[0] > 0 and shape[1] > 0:
                rows, cols = int(shape[0]), int(shape[1])
                value = [flat[r*cols:(r+1)*cols] for r in range(rows)]
            else:
                value = []
        elif kind == "array":
            value = list(value)
        _deep_set(raw, spec["path"], value)
    return raw

class SetupClient(object):
    def __init__(self, topic="/setup"):
        self.topic = str(topic)
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._raw = None
        self._msg = None
        self.subscriber = rospy.Subscriber(self.topic, Setup, self._callback, queue_size=1)

    def _callback(self, msg):
        raw = setup_msg_to_raw(msg)
        with self._lock:
            self._msg = msg
            self._raw = raw
        self._event.set()

    def wait_raw(self, timeout=None):
        t0 = time.monotonic()
        while not rospy.is_shutdown():
            if self._event.wait(0.05):
                with self._lock:
                    return copy.deepcopy(self._raw)
            if timeout is not None and timeout > 0.0 and (time.monotonic() - t0) >= float(timeout):
                raise rospy.ROSException("timeout waiting for %s" % self.topic)
        raise rospy.ROSInterruptException("shutdown while waiting for %s" % self.topic)

    def latest_raw(self):
        with self._lock:
            return copy.deepcopy(self._raw) if self._raw is not None else None

_GLOBAL_CLIENT = None
_GLOBAL_LOCK = threading.Lock()

def get_setup_client(topic="/setup"):
    global _GLOBAL_CLIENT
    with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT is None:
            _GLOBAL_CLIENT = SetupClient(topic)
        return _GLOBAL_CLIENT

def get_setup_raw(topic="/setup", timeout=None):
    return get_setup_client(topic).wait_raw(timeout=timeout)
