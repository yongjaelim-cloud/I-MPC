#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS wrapper for the linearized acados vertex-set MPC planner implemented in utils_linearized.py.

Decoupled data flow
-------------------
1) Subscribe /tracked_pose (PoseStamped), estimate velocity, and build MPC x0 directly.
2) Subscribe /obstacle and parse obstacle_polygon vertices.
3) Subscribe /selected_vertex_data from vertex_selector.py.  This planner
   no longer performs FOV/vertex selection internally.
4) Read the designated ego sector from setter.py's /setup message, compute body-frame l_a/l_b/l_c, and call utils_linearized.AcadosMPCPlanner.solve(..., sector_rays_body=...).
5) Publish:
      - /mpc_control_seq : acc/yawrate sequence with fixed dt_init
      - /percepted_obstacle obstacle_polygon and ego_traj markers

Selected-vertex RViz visualization is intentionally handled only by plotter.py.

The MPC initial state is taken from /tracked_pose; no /ego_state subscriber is required.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import re
import time
import threading

import numpy as np
import rospy
from setup_config import setup_msg_to_raw

from drone_msgs.msg import obstacle_polygon, Ego_state, MPCControlSeq, Setup, Ego_goal_setting
from geometry_msgs.msg import Point, PoseStamped, Twist
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Int32

try:
    from utils_linearized import (
        AcadosMPCPlanner,
        PlanResult,
        build_obstacle_from_polygon,
        build_external_avoidance_linearization_sequence,
        circle_fallback_polygon,
        load_planner_config,
        normalize_sector_config,
        normalize_ellipse_local_frame_config,
        sector_rays_body_from_id,
        rollout_ego,
        vertex_types_to_dynamic_mask,
        selected_vertices_normals_to_halfspaces,
        wrap_to_pi,
    )
except ImportError:
    # If you rename utils_linearized.py to utils.py in the ROS package, this
    # fallback keeps the planner script usable.  The imported utils.py must be
    # the linearized version that accepts solve(..., order=..., sqp_iter=...).
    from utils import (
        AcadosMPCPlanner,
        PlanResult,
        build_obstacle_from_polygon,
        build_external_avoidance_linearization_sequence,
        circle_fallback_polygon,
        load_planner_config,
        normalize_sector_config,
        normalize_ellipse_local_frame_config,
        sector_rays_body_from_id,
        rollout_ego,
        vertex_types_to_dynamic_mask,
        selected_vertices_normals_to_halfspaces,
        wrap_to_pi,
    )


# ------------------------------------------------------------------
# /setup/ROS runtime configuration helpers
# ------------------------------------------------------------------


def infer_planner_key(default: str = "MPC_planner1") -> str:
    """Infer planner key from script name, e.g. MPC_planner2.py -> MPC_planner2."""
    stem = Path(__file__).stem
    match = re.search(r"MPC_planner\d+", stem)
    if match:
        return match.group(0)
    return default

def infer_planner_id_from_key(planner_key: str, default: int = 1) -> int:
    """Infer numeric planner id from planner key/name, e.g. MPC_planner9 -> 9."""
    match = re.search(r"MPC_planner(\d+)", str(planner_key))
    if match:
        return int(match.group(1))
    return int(default)


def _merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_nested_dict(out[key], value)
        else:
            out[key] = value
    return out


def get_planner_runtime_config(raw: Dict[str, Any], planner_key: str) -> Dict[str, Any]:
    """Return merged runtime config = common ros block + planners[planner_key]."""
    runtime: Dict[str, Any] = {}
    if isinstance(raw.get("ros"), dict):
        runtime = _merge_nested_dict(runtime, raw["ros"])

    planners = raw.get("planners", {})
    if isinstance(planners, dict):
        selected = planners.get(planner_key, {})
        if not selected and planner_key.lower() in planners:
            selected = planners[planner_key.lower()]
        if isinstance(selected, dict):
            runtime = _merge_nested_dict(runtime, selected)
    return runtime


def _apply_config_overrides(cfg: Any, runtime: Dict[str, Any]) -> None:
    """Apply planner-specific /setup values to cfg before ROS overrides."""
    # Allow both direct planner fields {"ORDER": 2, ...} and nested {"mpc": {...}}.
    for source in (runtime, runtime.get("mpc", {}) if isinstance(runtime.get("mpc"), dict) else {}):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key in getattr(cfg, "__dataclass_fields__", {}):
                setattr(cfg, key, value)


# def _normalize_cfg_after_overrides(cfg: Any) -> None:
#     """Enforce the same horizon, fixed dt, bounds, and unit-disk frame as MATLAB."""
#     cfg.N = 20
#     cfg.dt_init = 0.05
#     cfg.yawrate_min = -1.0
#     cfg.yawrate_max = 1.0
#     cfg.a_min = -2.0
#     cfg.a_max = 2.0
#     if bool(getattr(cfg, "no_acc_flag", False)):
#         cfg.a_min = 0.0
#         cfg.a_max = 0.0
#     cfg.v_min = 0.0
#     cfg.v_max = 2.0
#     cfg.q_pos = 1.0
#     cfg.q_pos_terminal = 10.0
#     cfg.q_bspline_tracking = 40.0
#     cfg.q_yaw = 0.0
#     cfg.q_v = 0.0
#     cfg.v_des = 1.2
#     cfg.r_yawrate = 0.0
#     cfg.r_a = 0.0
#     cfg.r_du = 0.0
#     cfg.q_collision_slack = 1.0e8
#     cfg.q_normal_alignment_slack = 1.0e8
#     cfg.collision_slack_max = 1.0e6
#     cfg.normal_alignment_slack_max = 1.0e6
#     cfg.rho_Phi_global = 1.0
#     cfg.rho_a_global = 1.0e8
#     cfg.rho_c_global = 1.0e8
#     cfg.rho_terminal_goal_global = 1.0e3
#     cfg.enable_boundary_constraint = False
#     cfg.enable_normal_mag_avoidance_constraint = True
#     cfg.designated_surface_id = int(getattr(cfg, "sector_id", getattr(cfg, "designated_surface_id", 1)))
#     cfg.sector_dir = "outward"
#     normalize_sector_config(cfg)

#     normalize_ellipse_local_frame_config(cfg)



def _normalize_cfg_after_overrides(cfg: Any) -> None:
    """Normalize dependent configuration values without overriding /setup values."""

    # Horizon
    cfg.N = int(cfg.N)

    # Timing
    cfg.dt_init = float(cfg.dt_init)

    # no-acc mode only
    if bool(getattr(cfg, "no_acc_flag", False)):
        cfg.a_min = 0.0
        cfg.a_max = 0.0

    cfg.designated_surface_id = int(
        getattr(
            cfg,
            "sector_id",
            getattr(cfg, "designated_surface_id", 1),
        )
    )

    cfg.sector_dir = str(
        getattr(cfg, "sector_dir", "outward")
    )

    normalize_sector_config(cfg)
    normalize_ellipse_local_frame_config(cfg)



def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def _get_nested(runtime: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    block = runtime.get(section, {})
    if isinstance(block, dict) and key in block:
        return block[key]
    return default


def _rgba(value: Any, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    try:
        arr = list(value)
        if len(arr) == 3:
            arr.append(default[3])
        if len(arr) >= 4:
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    except Exception:
        pass
    return tuple(float(v) for v in default)


def _set_marker_color(marker: Marker, rgba: Tuple[float, float, float, float]) -> None:
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


class MPCPlannerTrajectoryNode:
    def _loginfo(self, *args, **kwargs) -> None:
        if bool(getattr(self, "cal_time_log_flag", False)):
            rospy.loginfo(*args, **kwargs)

    def _loginfo_throttle(self, *args, **kwargs) -> None:
        if bool(getattr(self, "cal_time_log_flag", False)):
            rospy.loginfo_throttle(*args, **kwargs)

    def _runtime_goal_callback(self, msg):
        """Goals published by setter are already expressed in map."""
        gx, gy = float(msg.x_end), float(msg.y_end)
        if not (math.isfinite(gx) and math.isfinite(gy)):
            return
        self.goal_x, self.goal_y = gx, gy
        self.cfg.goal_x, self.cfg.goal_y = gx, gy
        if hasattr(self, "planner") and hasattr(self.planner, "cfg"):
            self.planner.cfg.goal_x, self.planner.cfg.goal_y = gx, gy

    def __init__(self):
        self.cal_time_log_flag = False
        self.planner_key = infer_planner_key()

        # Persistent /setup subscription. setter.py is the only JSON reader and
        # publishes this message latched, so a planner may start before or after
        # setter.py and still receives the same typed configuration.
        self.setup_topic = "/setup"
        self._setup_event = threading.Event()
        self._setup_raw = None
        self.setup_subscriber = rospy.Subscriber(
            self.setup_topic, Setup, self._setup_callback, queue_size=1
        )
        self._loginfo("[%s] waiting for latched %s", self.planner_key, self.setup_topic)
        raw_cfg = self._wait_for_setup_raw()
        _mpc_log_cfg = raw_cfg.get("mpc", {}) if isinstance(raw_cfg.get("mpc", {}), dict) else {}
        self.cal_time_log_flag = _as_bool(_mpc_log_cfg.get("cal_time_log_flag", True), True)
        self._loginfo("[%s] configuration received from %s", self.planner_key, self.setup_topic)

        self.cfg, _, raw_cfg = load_planner_config(raw_config=raw_cfg)
        # load_planner_config applies the /setup-derived raw["config"] to overlapping
        # MPC fields such as x_min/x_max/y_min/y_max, goal_x/goal_y, FOV,
        # ego_R, and moving_obstacle_R.
        self.runtime_cfg = get_planner_runtime_config(raw_cfg, self.planner_key)
        _apply_config_overrides(self.cfg, self.runtime_cfg)
        _normalize_cfg_after_overrides(self.cfg)

        # /setup is the only configuration source. Do not apply ROS private-parameter overrides.

        self.guidance_topic_id = infer_planner_id_from_key(
            self.planner_key, int(getattr(self.cfg, "planner_id", 1))
        )

        # Latched acknowledgement used by repeat_scenario_OCP to guarantee that
        # every parallel planner has applied the requested vulnerable sector
        # before the next scenario starts.
        self.sector_ack_topic = "/mpc_sector_ack%d" % int(self.guidance_topic_id)
        self.sector_ack_pub = rospy.Publisher(
            self.sector_ack_topic, Int32, queue_size=1, latch=True
        )

        # External linearization-loop settings.
        # ORDER indexes the Cartesian product provided by /setup.mpc:
        # acc_order is the major index and yawrate_order is the minor index.
        self.ORDER = int(getattr(self.cfg, "ORDER", 1))
        self.SQP_iter = max(1, int(getattr(self.cfg, "SQP_iter", 3)))
        self.cfg.ORDER = self.ORDER
        self.cfg.SQP_iter = self.SQP_iter

        # A successful SQP candidate is treated as feasible when every
        # regulation slack s_r^k (= PlanResult.normal_slack[k]) is <= 1e-3.
        self.s_r_feasibility_tol = 1.0e-3

        # Seed ordering is owned by setter.py -> /setup; this planner never reads JSON.
        setup_mpc_cfg = raw_cfg.get("mpc", {}) if isinstance(raw_cfg.get("mpc", {}), dict) else {}
        self.acc_order = [int(v) for v in setup_mpc_cfg.get("acc_order", [])]
        self.yawrate_order = [int(v) for v in setup_mpc_cfg.get("yawrate_order", [])]
        if not self.acc_order or not self.yawrate_order:
            raise ValueError("/setup.mpc must contain non-empty acc_order and yawrate_order")
        if any(v not in (-1, 0, 1) for v in self.acc_order):
            raise ValueError("/setup.mpc.acc_order entries must be -1, 0, or 1")
        if any(v not in (-1, 0, 1) for v in self.yawrate_order):
            raise ValueError("/setup.mpc.yawrate_order entries must be -1, 0, or 1")
        order_count = len(self.acc_order) * len(self.yawrate_order)
        if self.ORDER < 1 or self.ORDER > order_count:
            raise ValueError("ORDER={} is outside 1..{} for acc_order x yawrate_order".format(self.ORDER, order_count))


        # /setup sector selection.  Surface id mapping is MATLAB-compatible:
        #   1: left, 2: rear, 3: right, 4: front.
        self.sector_id = normalize_sector_config(self.cfg)
        self.sector_rays_body = sector_rays_body_from_id(
            float(self.cfg.ego_L), float(self.cfg.ego_W), int(self.sector_id)
        )
        # Publish the startup sector as the initial sector ACK.
        self.sector_ack_pub.publish(Int32(data=int(self.sector_id)))

        self.planner = AcadosMPCPlanner(self.cfg)
        self.hz = float(self.cfg.hz)
        self.frame_id = "map"  # Planning/RViz reference frame
        self.cfg.frame_id = "map"
        self.marker_width = float(self.cfg.marker_width)
        self.t_scale = float(self.cfg.t_scale)

        self.mpc_pred_color = _rgba(self.runtime_cfg.get("mpc_pred_color"), (0.0, 1.0, 0.0, 1.0))
        self.ego_traj_color = _rgba(self.runtime_cfg.get("ego_traj_color"), (1.0, 0.0, 0.0, 1.0))
        self.publish_percepted_obstacle_enabled = _as_bool(self.runtime_cfg.get("publish_percepted_obstacle"), True)
        # RViz visualization of selected vertices/normals is owned by vertex_selector.py.
        # This planner consumes /selected_vertex_data and must not publish /selected_vertex_set or /selected_vertex_normal.
        self.publish_selected_vertices_enabled = False

        # Local copy of the latest /ego_state. This node never integrates or
        # publishes ego state; ego_mover.py owns /ego_state.
        self.ego_x = float(self.cfg.ego_x)
        self.ego_y = float(self.cfg.ego_y)
        self.ego_z = float(self.cfg.ego_z)
        self.ego_v = float(self.cfg.ego_v0)
        self.ego_yaw = float(self.cfg.ego_yaw0)

        # Global terminal goal is owned by /setup.config.x_end/y_end.
        # With valid initial guidance, solve() may temporarily use the guidance
        # endpoint as a local refinement goal; otherwise it must target x_end/y_end.
        setup_config = raw_cfg.get("config", {}) if isinstance(raw_cfg.get("config", {}), dict) else {}
        if "x_end" not in setup_config or "y_end" not in setup_config:
            raise ValueError("/setup.config must contain x_end and y_end")
        self.goal_x = float(setup_config["x_end"])
        self.goal_y = float(setup_config["y_end"])
        self.goal_z = float(self.cfg.goal_z)
        self.goal_subscriber = rospy.Subscriber("/ego_goal_set", Ego_goal_setting, self._runtime_goal_callback, queue_size=1)

        self.x_min = float(self.cfg.x_min)
        self.x_max = float(self.cfg.x_max)
        self.y_min = float(self.cfg.y_min)
        self.y_max = float(self.cfg.y_max)
        self.moving_obstacle_R = float(self.cfg.moving_obstacle_R)
        self.ego_R = float(self.cfg.ego_R)

        self.obstacles: List[Dict[str, Any]] = []
        self.latest_vertex_set = np.zeros((0, 2), dtype=float)
        self.latest_vertex_normals = np.zeros((0, 2), dtype=float)
        self.latest_vertex_set_horizon = np.zeros((int(self.cfg.N), 0, 2), dtype=float)
        self.latest_vertex_normals_horizon = np.zeros((int(self.cfg.N), 0, 2), dtype=float)
        self.latest_vertex_types = np.zeros(0, dtype=np.int32)
        self.latest_selected_obs_indices = np.zeros(0, dtype=np.int32)
        # Atomic h=0 selector snapshot: (vertices, normals, types, group_ids).
        # /selected_vertex_horizon_data starts at h=1, so it must never
        # overwrite the geometry used for k=0 impact-normal diagnostics.
        self.current_vertex_snapshot = (
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
        self.have_horizon_vertices = False
        self.latest_halfspaces = np.zeros((0, 3), dtype=float)
        self.latest_dynamic_halfspace_mask = np.zeros(int(self.cfg.vertex_capacity), dtype=bool)
        self.latest_plan: Optional[PlanResult] = None
        self.ego_trail: List[Tuple[float, float]] = []
        # Common planning-round ID and matching ego snapshot from /ego_state.
        # All planners receive the same round ID; no planner-local sequence counter is used.
        self.latest_ego_packet = (-1, self.build_x0().copy())
        self.last_planned_round_id = -1
        self.planning_cycle_id = -1  # One sequence ID per MPC solve, independent of pose arrival.
        self.tracked_pose_lock = threading.Lock()
        self.tracked_pose_condition = threading.Condition(self.tracked_pose_lock)
        self.last_consumed_pose_round = -1
        self.tracked_pose_round = -1
        # /cmd_vel is the only MPC velocity source. Never derive v from pose differences.
        self.cmd_vel_topic = "/cmd_vel"
        self.latest_cmd_vel = 0.0
        self.last_pose_stamp = None
        self.tracked_pose_timeout_s = 0.5
        self.latest_ego_snapshot_stamp = rospy.Time(0)

        self.latest_initial_guidance: Dict[str, Any] = {}
        self.latest_seed_info: Dict[str, Any] = {}
        self.initial_guidance_recv_id = 0
        self.last_used_initial_guidance_recv_id = 0
        default_initial_guidance_topic = _get_nested(
            self.runtime_cfg,
            "input_topics",
            "initial_guidance",
            "/initial_guidance%d" % int(self.guidance_topic_id),
        )
        default_initial_guidance_topic = str(default_initial_guidance_topic)
        if "{planner_id}" in default_initial_guidance_topic:
            default_initial_guidance_topic = default_initial_guidance_topic.format(
                planner_id=int(self.guidance_topic_id)
            )
        elif "%d" in default_initial_guidance_topic:
            try:
                default_initial_guidance_topic = default_initial_guidance_topic % int(self.guidance_topic_id)
            except TypeError:
                pass
        elif default_initial_guidance_topic.strip() in ("", "/initial_guidance", "/initial_guidanceN"):
            default_initial_guidance_topic = "/initial_guidance%d" % int(self.guidance_topic_id)
        self.initial_guidance_topic = str(default_initial_guidance_topic)
        self.initial_guidance_timeout_s = max(
            0.0,
            float(self.runtime_cfg.get(
                "initial_guidance_timeout_s",
                max(0.25, 4.0 / max(float(self.cfg.hz), 1.0)),
            )),
        )
        self.initial_guidance_invalid_cost_threshold = float(
            self.runtime_cfg.get("initial_guidance_invalid_cost_threshold", 1.0e99)
        )
        self.use_initial_guidance = _as_bool(
            self.runtime_cfg.get("use_initial_guidance", True), True
        )
        self.fallback_acc_bias = float(
            self.runtime_cfg.get("fallback_acc_bias", float("nan"))
        )
        self.fallback_yawrate_bias = float(
            self.runtime_cfg.get("fallback_yawrate_bias", float("nan"))
        )

        # Bounds and goal are initialized only from the setter-owned /setup message.
        self.have_goal = True
        self.have_bounds = True
        # vertex_selector.py owns FOV filtering, pseudo-vertex generation,
        # per-vertex obstacle/type bookkeeping, and selected-vertex RViz output.
        # The planner only consumes the selected vertex topic and converts the
        # received points into the fixed-size halfspace matrix used by acados.
        self.have_obs = False
        self.have_selected_vertices = False
        self.have_ego = False
        self.have_initial_guidance = False

        ego_state_topic = str(_get_nested(self.runtime_cfg, "input_topics", "ego_state", "/ego_state"))
        selected_vertex_topic = str(_get_nested(
            self.runtime_cfg, "input_topics", "selected_vertex_data", "/selected_vertex_data"
        ))
        selected_vertex_horizon_topic = str(_get_nested(
            self.runtime_cfg, "input_topics", "selected_vertex_horizon_data", "/selected_vertex_horizon_data"
        ))
        self.tracked_pose_subscriber = rospy.Subscriber("/tracked_pose", PoseStamped, self.tracked_pose_callback, queue_size=1)
        self.cmd_vel_subscriber = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
        rospy.Subscriber(selected_vertex_topic, obstacle_polygon, self.selected_vertex_callback, queue_size=1)
        rospy.Subscriber(selected_vertex_horizon_topic, Float64MultiArray, self.selected_vertex_horizon_callback, queue_size=1)
        self.initial_guidance_subscriber = None
        if self.use_initial_guidance:
            self.initial_guidance_subscriber = rospy.Subscriber(
                self.initial_guidance_topic,
                MPCControlSeq,
                self.initial_guidance_callback,
                queue_size=1,
            )
        else:
            self._loginfo(
                "[MPC_planner%d] initial guidance disabled; fixed fallback (acc_bias,yawrate_bias)=(%.3f,%.3f)",
                int(self.cfg.planner_id),
                float(self.fallback_acc_bias),
                float(self.fallback_yawrate_bias),
            )
        self.selected_vertex_topic = selected_vertex_topic
        self.selected_vertex_horizon_topic = selected_vertex_horizon_topic

        self.mpc_candidate_queue_size = max(
            10, int(self.runtime_cfg.get("mpc_candidate_queue_size", 100))
        )
        self.pub_mpc_control_seq = rospy.Publisher(
            str(_get_nested(self.runtime_cfg, "publishers", "mpc_control_seq", "/mpc_control_seq")),
            MPCControlSeq,
            queue_size=self.mpc_candidate_queue_size,
        )

        # Planner-specific predicted impact-normal candidate.  The payload also
        # carries this candidate's rolled-out state trajectory so repeat_scenario
        # can match it against ego_mover's /selected_traj and keep ONLY the normal
        # belonging to the trajectory that was actually executed.
        self.predicted_impact_normal_candidate_topic = (
            "/predicted_impact_normal_candidate%d" % int(self.guidance_topic_id)
        )
        self.pub_predicted_impact_normal_candidate = rospy.Publisher(
            self.predicted_impact_normal_candidate_topic,
            Float64MultiArray,
            queue_size=10,
        )

        # Separating-hyperplane RViz/debug output is disabled here; MPC_planner
        # publishes planning data only.
        self.separating_hyperplane_data_topic = None
        self.pub_separating_hyperplane_data = None
        # RViz output is centralized in plotter.py. MPC_planner publishes the
        # control/result stream; plotter rolls it out to /mpc_pred_trajs and also
        # owns /obs_polygons_horizon from selector horizon data.
        self.pub_mpc_pred = None
        # No RViz trajectory publisher in MPC_planner.  plotter.py owns
        # /mpc_pred_trajs and reconstructs it from /mpc_control_seqN.
        self.pub_ego_traj = None
        # Obstacle visualization/publishing is centralized in vertex_selector + plotter.
        self.pub_percepted_obs = None
        # No selected-vertex visualization publisher here.  vertex_selector.py computes
        # and publishes /selected_vertex_set and /selected_vertex_normal from
        # obstacle_polygon messages.
        self.pub_selected_vertices = None

        self._loginfo(
            "[MPC_planner%d] MATLAB-aligned fixed-dt linearized MPC initialized, N=%d, dt_init=%.4f, vertex_capacity=%d, ORDER=%d, SQP_iter=%d, mpc_flag=%d, sector_id=%d, boundary_constraint=%s, no_acc_flag=%s, selected_vertex_topic=%s, initial_guidance_topic=%s",
            int(self.cfg.planner_id), int(self.cfg.N), float(self.cfg.dt_init),
            int(self.cfg.vertex_capacity), int(self.ORDER), int(self.SQP_iter), int(getattr(self.cfg, "mpc_flag", 4)), int(self.sector_id),
            str(bool(getattr(self.cfg, "enable_boundary_constraint", True))),
            str(bool(getattr(self.cfg, "no_acc_flag", False))),
            str(self.selected_vertex_topic), str(self.initial_guidance_topic),
        )
        self._loginfo(
            "[MPC_planner%d] /setup seed ordering acc_order=%s yawrate_order=%s",
            int(self.cfg.planner_id), str(self.acc_order), str(self.yawrate_order),
        )

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def cmd_vel_callback(self, msg: Twist) -> None:
        """Store commanded forward speed; /cmd_vel is expressed in the robot body frame."""
        speed = float(msg.linear.x)
        if not math.isfinite(speed):
            return
        with self.tracked_pose_lock:
            self.latest_cmd_vel = max(float(self.cfg.v_min), min(float(self.cfg.v_max), speed))

    def tracked_pose_callback(self, msg: PoseStamped) -> None:
        """Build map-frame [x, y, cmd_vel.linear.x, yaw] for the next MPC solve."""
        frame = str(msg.header.frame_id).lstrip("/")
        if frame and frame != self.frame_id:
            return
        p = msg.pose.position
        q = msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(float(v)) for v in values):
            return
        qnorm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if qnorm < 1e-9:
            return
        qx, qy, qz, qw = q.x/qnorm, q.y/qnorm, q.z/qnorm, q.w/qnorm
        yaw = wrap_to_pi(math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))
        now = rospy.Time.now()
        with self.tracked_pose_lock:
            self.ego_v = float(self.latest_cmd_vel)
            self.ego_x = float(p.x)
            self.ego_y = float(p.y)
            self.ego_z = float(p.z)
            self.ego_yaw = float(yaw)
            self.tracked_pose_round += 1
            self.latest_ego_packet = (self.tracked_pose_round, self.build_x0().copy())
            self.latest_ego_snapshot_stamp = now
            self.have_ego = True
            self.tracked_pose_condition.notify_all()

    def extract_polygon_vertices(self, msg: obstacle_polygon, obs_index: int, cx: float, cy: float) -> np.ndarray:
        nv = 0
        if hasattr(msg, "obs_vertex_counts") and len(msg.obs_vertex_counts) > obs_index:
            nv = int(msg.obs_vertex_counts[obs_index])
        nv = max(0, min(nv, int(self.cfg.msg_max_vertices)))

        verts: List[Tuple[float, float]] = []
        if nv >= 3 and hasattr(msg, "obs_poly_xs") and hasattr(msg, "obs_poly_ys"):
            for j in range(nv):
                flat_idx = obs_index * int(self.cfg.msg_max_vertices) + j
                if flat_idx >= len(msg.obs_poly_xs) or flat_idx >= len(msg.obs_poly_ys):
                    break
                xj = float(msg.obs_poly_xs[flat_idx])
                yj = float(msg.obs_poly_ys[flat_idx])
                if np.isfinite(xj) and np.isfinite(yj):
                    verts.append((xj, yj))

        if len(verts) < 3:
            return circle_fallback_polygon(
                cx,
                cy,
                max(float(self.moving_obstacle_R), 1.0e-3),
                int(self.cfg.fallback_polygon_vertices),
            )
        return np.asarray(verts, dtype=float).reshape(-1, 2)

    def obstacle_callback(self, msg: obstacle_polygon) -> None:
        self.have_obs = True
        total_obs_cnt = int(msg.obs_cnt) if hasattr(msg, "obs_cnt") else 0
        new_obstacles: List[Dict[str, Any]] = []

        for i in range(total_obs_cnt):
            if i >= len(msg.obs_xs) or i >= len(msg.obs_ys):
                break
            if self.cfg.perception_rate < 1.0 and np.random.rand() > float(self.cfg.perception_rate):
                continue

            cx = float(msg.obs_xs[i])
            cy = float(msg.obs_ys[i])
            vx = float(msg.obs_vxs[i]) if hasattr(msg, "obs_vxs") and i < len(msg.obs_vxs) else 0.0
            vy = float(msg.obs_vys[i]) if hasattr(msg, "obs_vys") and i < len(msg.obs_vys) else 0.0

            if hasattr(msg, "obs_types") and i < len(msg.obs_types):
                obs_type = "dyn" if int(msg.obs_types[i]) == 1 else "static"
            else:
                obs_type = "dyn" if abs(vx) + abs(vy) > 1.0e-12 else "static"

            V = self.extract_polygon_vertices(msg, i, cx, cy)
            try:
                obs = build_obstacle_from_polygon(
                    V,
                    vel=(vx, vy),
                    obs_type=obs_type,
                    center=(cx, cy),
                )
                new_obstacles.append(obs)
            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0,
                    "[MPC_planner%d] obstacle %d skipped: %s",
                    int(self.cfg.planner_id), i, str(exc)
                )

        self.obstacles = new_obstacles

    def selected_vertex_callback(self, msg: obstacle_polygon) -> None:
        """Receive selector-owned pseudo vertices for MPC constraints.

        vertex_selector.py encodes each selected feature as one entry of
        obstacle_polygon:
          - obs_xs / obs_ys: selected pseudo/original feature position,
          - obs_types: source obstacle type, 0=static, 1=dynamic,
          - obs_rs: convex-partition group id assigned by vertex_selector.py,
          - obs_vxs / obs_vys: feature normal for debug/visualization.

        The solver uses positions, feature normals, dynamic/static types,
        and convex-partition group ids for MATLAB-style feature grouping.
        """
        total = int(msg.obs_cnt) if hasattr(msg, "obs_cnt") else 0
        max_n = int(self.cfg.vertex_capacity)
        if max_n <= 0 or total <= 0:
            vertex_set = np.zeros((0, 2), dtype=float)
            vertex_normals = np.zeros((0, 2), dtype=float)
            vertex_types = np.zeros(0, dtype=np.int32)
            source_obs_indices = np.zeros(0, dtype=np.int32)
        else:
            xs = msg.obs_xs if hasattr(msg, "obs_xs") else []
            ys = msg.obs_ys if hasattr(msg, "obs_ys") else []
            nxs = msg.obs_vxs if hasattr(msg, "obs_vxs") else []
            nys = msg.obs_vys if hasattr(msg, "obs_vys") else []
            limit = min(total, max_n, len(xs), len(ys))
            vertices: List[Tuple[float, float]] = []
            normals: List[Tuple[float, float]] = []
            types: List[int] = []
            source_indices: List[int] = []
            for i in range(limit):
                x = float(xs[i])
                y = float(ys[i])
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                vertices.append((x, y))
                nx = float(nxs[i]) if i < len(nxs) else 0.0
                ny = float(nys[i]) if i < len(nys) else 0.0
                if not (np.isfinite(nx) and np.isfinite(ny)) or math.hypot(nx, ny) <= 1.0e-12:
                    # Fallback direction only for malformed old messages.
                    rel = np.array([x - self.ego_x, y - self.ego_y], dtype=float)
                    rel_norm = float(np.linalg.norm(rel))
                    if rel_norm > 1.0e-12:
                        if str(getattr(self.cfg, "normal_orientation", "inward")).lower() == "inward":
                            nx, ny = (-rel / rel_norm).tolist()
                        else:
                            nx, ny = (rel / rel_norm).tolist()
                    else:
                        nx, ny = 0.0, 0.0
                normals.append((nx, ny))
                if hasattr(msg, "obs_types") and i < len(msg.obs_types):
                    types.append(1 if int(msg.obs_types[i]) == 1 else 0)
                else:
                    types.append(0)
                if hasattr(msg, "obs_rs") and i < len(msg.obs_rs):
                    source_indices.append(int(round(float(msg.obs_rs[i]))))
                else:
                    source_indices.append(-1)

            vertex_set = np.asarray(vertices, dtype=float).reshape(-1, 2) if vertices else np.zeros((0, 2), dtype=float)
            vertex_normals = np.asarray(normals, dtype=float).reshape(-1, 2) if normals else np.zeros((0, 2), dtype=float)
            vertex_types = np.asarray(types, dtype=np.int32) if types else np.zeros(0, dtype=np.int32)
            source_obs_indices = np.asarray(source_indices, dtype=np.int32) if source_indices else np.zeros(0, dtype=np.int32)

        halfspaces = selected_vertices_normals_to_halfspaces(
            vertex_set,
            vertex_normals,
            self.build_x0(),
            int(self.cfg.vertex_capacity),
            self.cfg,
            disabled_c_value=float(self.cfg.disabled_c_value),
        )
        dynamic_halfspace_mask = vertex_types_to_dynamic_mask(vertex_types, int(self.cfg.vertex_capacity))

        self.latest_vertex_set = vertex_set
        self.latest_vertex_normals = vertex_normals
        self.latest_vertex_types = vertex_types
        self.latest_selected_obs_indices = source_obs_indices
        # Preserve one coherent h=0 selector generation.  Assigning the
        # tuple atomically prevents a planning thread from mixing arrays
        # from two adjacent selector callbacks.
        self.current_vertex_snapshot = (
            np.ascontiguousarray(vertex_set, dtype=float).copy(),
            np.ascontiguousarray(vertex_normals, dtype=float).copy(),
            np.ascontiguousarray(vertex_types, dtype=np.int32).copy(),
            np.ascontiguousarray(source_obs_indices, dtype=np.int32).copy(),
        )
        self.latest_halfspaces = halfspaces
        self.latest_dynamic_halfspace_mask = dynamic_halfspace_mask
        self.have_selected_vertices = True
        self.have_obs = True

    def selected_vertex_horizon_callback(self, msg: Float64MultiArray) -> None:
        """Receive the selector-owned N x K predicted/inflated feature tensor.

        The flattened payload fields are [x, y, nx, ny, type, group_id].
        Types and group ids are stage-invariant; geometry is stage-specific.
        """
        dims = list(getattr(getattr(msg, "layout", None), "dim", []))
        if len(dims) < 3:
            rospy.logwarn_throttle(1.0, "[MPC_planner%d] malformed horizon vertex layout", int(self.cfg.planner_id))
            return
        H_msg = max(0, int(dims[0].size))
        K_msg = max(0, int(dims[1].size))
        F_msg = max(0, int(dims[2].size))
        if H_msg <= 0 or F_msg < 6:
            return
        raw = np.asarray(msg.data, dtype=float).reshape(-1)
        needed = H_msg * K_msg * F_msg
        if raw.size < needed:
            rospy.logwarn_throttle(1.0, "[MPC_planner%d] truncated horizon vertex payload: got=%d need=%d", int(self.cfg.planner_id), int(raw.size), int(needed))
            return
        tensor = raw[:needed].reshape(H_msg, K_msg, F_msg)
        H = int(self.cfg.N)
        K = min(K_msg, int(self.cfg.vertex_capacity))
        if K <= 0:
            Vh = np.zeros((H, 0, 2), dtype=float)
            Nh = np.zeros((H, 0, 2), dtype=float)
            types = np.zeros(0, dtype=np.int32)
            groups = np.zeros(0, dtype=np.int32)
        else:
            used = tensor[:min(H_msg, H), :K, :]
            Vh = np.asarray(used[:, :, 0:2], dtype=float)
            Nh = np.asarray(used[:, :, 2:4], dtype=float)
            if used.shape[0] < H:
                Vh = np.concatenate((Vh, np.repeat(Vh[-1:, :, :], H - Vh.shape[0], axis=0)), axis=0)
                Nh = np.concatenate((Nh, np.repeat(Nh[-1:, :, :], H - Nh.shape[0], axis=0)), axis=0)
            types = np.rint(used[0, :, 4]).astype(np.int32)
            groups = np.rint(used[0, :, 5]).astype(np.int32)

        self.latest_vertex_set_horizon = np.ascontiguousarray(Vh, dtype=float)
        self.latest_vertex_normals_horizon = np.ascontiguousarray(Nh, dtype=float)
        self.latest_vertex_set = self.latest_vertex_set_horizon[0].copy() if H > 0 else np.zeros((0, 2), dtype=float)
        self.latest_vertex_normals = self.latest_vertex_normals_horizon[0].copy() if H > 0 else np.zeros((0, 2), dtype=float)
        self.latest_vertex_types = types
        self.latest_selected_obs_indices = groups
        self.latest_halfspaces = selected_vertices_normals_to_halfspaces(
            self.latest_vertex_set, self.latest_vertex_normals, self.build_x0(),
            int(self.cfg.vertex_capacity), self.cfg,
            disabled_c_value=float(self.cfg.disabled_c_value),
        )
        self.latest_dynamic_halfspace_mask = vertex_types_to_dynamic_mask(types, int(self.cfg.vertex_capacity))
        self.have_horizon_vertices = True
        self.have_selected_vertices = True
        self.have_obs = True



    def _first_numeric_array_field(self, msg: MPCControlSeq, names: Tuple[str, ...]) -> Optional[np.ndarray]:
        for name in names:
            if not hasattr(msg, name):
                continue
            try:
                arr = np.asarray(list(getattr(msg, name)), dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.size > 0:
                return arr
        return None

    def _initial_guidance_cost_is_valid(self, msg: MPCControlSeq) -> Tuple[bool, str]:
        """Return False for placeholder/failed guidance messages.

        visibility_prm publishes an MPCControlSeq on every /initial_guidanceN
        topic even when fewer than N valid guidances exist.  Those placeholders
        use a very large cost, so the MPC node must reject them and fall back to
        its ORDER-based nominal seed.
        """
        if not hasattr(msg, "cost"):
            return True, "no_cost_field"
        try:
            cost = float(getattr(msg, "cost"))
        except Exception:
            return False, "non_numeric_cost"
        threshold = float(getattr(self, "initial_guidance_invalid_cost_threshold", 1.0e99))
        if (not np.isfinite(cost)) or cost >= threshold:
            return False, "invalid_cost=%.3e threshold=%.3e" % (float(cost), float(threshold))
        return True, "cost=%.3e" % float(cost)

    @staticmethod
    def _array_is_finite(arr: Any) -> bool:
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            return False
        return bool(a.size > 0 and np.all(np.isfinite(a)))

    def _clear_invalid_initial_guidance(self, reason: str, seq_id: int = -1) -> None:
        """Drop the cached external seed so the next solve uses ORDER seed."""
        self.latest_initial_guidance = {}
        self.have_initial_guidance = False
        self.latest_seed_info = {
            "used": False,
            "reason": "invalid_initial_guidance",
            "detail": str(reason),
            "seq_id": int(seq_id),
        }

    def _extract_guidance_state_nominal(self, msg: MPCControlSeq, U: np.ndarray, dt: float) -> Optional[np.ndarray]:
        x_arr = self._first_numeric_array_field(msg, ("xs", "x_seq", "x_seqs", "traj_xs", "traj_x", "ref_xs", "ref_x", "mpc_xs", "path_xs", "pos_xs", "pxs"))
        y_arr = self._first_numeric_array_field(msg, ("ys", "y_seq", "y_seqs", "traj_ys", "traj_y", "ref_ys", "ref_y", "mpc_ys", "path_ys", "pos_ys", "pys"))
        if x_arr is None or y_arr is None:
            return None

        N = int(self.cfg.N)
        count = min(int(x_arr.size), int(y_arr.size), N + 1)
        if count <= 0:
            return None

        try:
            X = rollout_ego(self.build_x0(), U, float(dt), self.cfg)
        except Exception:
            X = np.repeat(self.build_x0().reshape(4, 1), N + 1, axis=1)

        X[0, :count] = x_arr[:count]
        X[1, :count] = y_arr[:count]
        if count < N + 1:
            X[0, count:] = X[0, count - 1]
            X[1, count:] = X[1, count - 1]

        yaw_arr = self._first_numeric_array_field(msg, ("yaws", "yaw_seq", "yaw_seqs", "traj_yaws", "traj_yaw", "ref_yaws", "ref_yaw", "mpc_yaws", "path_yaws"))
        if yaw_arr is not None and yaw_arr.size > 0:
            yaw_count = min(int(yaw_arr.size), N + 1)
            X[3, :yaw_count] = [wrap_to_pi(v) for v in yaw_arr[:yaw_count]]
            if yaw_count < N + 1:
                X[3, yaw_count:] = X[3, yaw_count - 1]
        elif count >= 2:
            for k in range(count - 1):
                dx = float(X[0, k + 1] - X[0, k])
                dy = float(X[1, k + 1] - X[1, k])
                if math.hypot(dx, dy) > 1.0e-9:
                    X[3, k] = math.atan2(dy, dx)
            X[3, count - 1:] = X[3, max(0, count - 2)]

        v_arr = self._first_numeric_array_field(msg, ("vs", "v_seq", "v_seqs", "vels", "vel_seq", "speeds", "speed_seq", "traj_vs", "ref_vs"))
        if v_arr is not None and v_arr.size > 0:
            v_count = min(int(v_arr.size), N + 1)
            X[2, :v_count] = v_arr[:v_count]
            if v_count < N + 1:
                X[2, v_count:] = X[2, v_count - 1]
        elif count >= 2:
            dt_safe = max(float(dt), 1.0e-9)
            for k in range(count - 1):
                ds = math.hypot(float(X[0, k + 1] - X[0, k]), float(X[1, k + 1] - X[1, k]))
                X[2, k] = ds / dt_safe
            X[2, count - 1:] = X[2, max(0, count - 2)]

        X[0, 0] = float(self.ego_x)
        X[1, 0] = float(self.ego_y)
        X[2, 0] = float(self.ego_v)
        X[3, 0] = wrap_to_pi(float(self.ego_yaw))
        return X

    def initial_guidance_callback(self, msg: MPCControlSeq) -> None:
        """Cache one newly received /initial_guidanceN seed for the next MPC recycle.

        If the message is a failed/placeholder guidance, do not cache it.
        The next step() call will then build the nominal from ORDER.
        """
        seq_id = int(msg.seq_id) if hasattr(msg, "seq_id") else -1

        # An empty /initial_guidanceN is an explicit "no external guidance"
        # signal from the guidance planner. Handle it before checking cost, because
        # empty slots may intentionally carry the invalid-guidance sentinel cost.
        # Do not cache a fabricated external guidance here: step() will generate
        # the planner-specific default ORDER seed from the current x0 and keep the
        # optimization goal at the global /setup goal.
        try:
            yawrates = np.asarray(list(msg.yawrates), dtype=float).reshape(-1)
            accs = np.asarray(list(msg.accs), dtype=float).reshape(-1)
        except Exception as exc:
            self._clear_invalid_initial_guidance(str(exc), seq_id=seq_id)
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] invalid initial guidance on %s: %s; using default ORDER seed",
                int(self.cfg.planner_id), str(self.initial_guidance_topic), str(exc),
            )
            return

        n_ctrl = int(self.cfg.N)
        m = min(n_ctrl, int(yawrates.size), int(accs.size))
        if m <= 0:
            self._clear_invalid_initial_guidance("empty_initial_guidance_use_default", seq_id=seq_id)
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] empty initial guidance on %s; generating default ORDER seed",
                int(self.cfg.planner_id), str(self.initial_guidance_topic),
            )
            return

        cost_ok, cost_reason = self._initial_guidance_cost_is_valid(msg)
        if not cost_ok:
            self._clear_invalid_initial_guidance(cost_reason, seq_id=seq_id)
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] rejected initial guidance on %s: %s; using default ORDER seed",
                int(self.cfg.planner_id), str(self.initial_guidance_topic), str(cost_reason),
            )
            return
        if (not np.all(np.isfinite(yawrates[:m]))) or (not np.all(np.isfinite(accs[:m]))):
            self._clear_invalid_initial_guidance("non-finite control sequence", seq_id=seq_id)
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] non-finite initial guidance on %s; using ORDER seed",
                int(self.cfg.planner_id), str(self.initial_guidance_topic),
            )
            return

        U = np.zeros((2, n_ctrl), dtype=float)
        U[0, :m] = np.clip(yawrates[:m], float(self.cfg.yawrate_min), float(self.cfg.yawrate_max))
        U[1, :m] = np.clip(accs[:m], float(self.cfg.a_min), float(self.cfg.a_max))
        if m < n_ctrl:
            U[:, m:] = U[:, m - 1:m]
        if bool(getattr(self.cfg, "no_acc_flag", False)):
            U[1, :] = 0.0

        # MPC uses one fixed sampling period; guidance message dt/dt_seq is not an OCP variable.
        dt = float(self.cfg.dt_init)

        X = self._extract_guidance_state_nominal(msg, U, dt)
        if X is not None and not self._array_is_finite(X):
            self._clear_invalid_initial_guidance("non-finite state nominal", seq_id=seq_id)
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] non-finite initial guidance state on %s; using ORDER seed",
                int(self.cfg.planner_id), str(self.initial_guidance_topic),
            )
            return
        stamp = rospy.Time.now()
        if hasattr(msg, "header") and getattr(msg.header, "stamp", rospy.Time(0)).to_sec() > 0.0:
            stamp = msg.header.stamp

        self.initial_guidance_recv_id += 1
        self.latest_initial_guidance = {
            "recv_id": int(self.initial_guidance_recv_id),
            "seq_id": int(seq_id),
            "stamp": stamp,
            "recv_stamp": rospy.Time.now(),
            "U": U,
            "X": X,
            "dt": float(self.cfg.dt_init),
            "count": int(m),
        }
        self.have_initial_guidance = True

    def pop_initial_guidance_nominal(self, x0: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], float]]:
        if not bool(getattr(self, "use_initial_guidance", True)):
            self.latest_seed_info = {"used": False, "reason": "initial_guidance_disabled"}
            return None
        seed = self.latest_initial_guidance if isinstance(self.latest_initial_guidance, dict) else {}
        recv_id = int(seed.get("recv_id", 0))
        if recv_id <= 0 or recv_id == int(self.last_used_initial_guidance_recv_id):
            self.latest_seed_info = {"used": False, "reason": "no_new_initial_guidance"}
            return None

        now = rospy.Time.now()
        timeout_s = float(getattr(self, "initial_guidance_timeout_s", 0.0))
        if timeout_s > 0.0:
            try:
                age = (now - seed.get("recv_stamp", seed.get("stamp", now))).to_sec()
            except Exception:
                age = float("inf")
            if age > timeout_s:
                self.last_used_initial_guidance_recv_id = recv_id
                self.latest_seed_info = {"used": False, "reason": "stale_initial_guidance", "age_s": float(age)}
                return None

        self.last_used_initial_guidance_recv_id = recv_id
        try:
            U = np.asarray(seed.get("U"), dtype=float).reshape(2, int(self.cfg.N))
            dt = float(self.cfg.dt_init)
        except Exception as exc:
            self.latest_seed_info = {"used": False, "reason": "invalid_cached_initial_guidance", "detail": str(exc)}
            return None
        if (not np.isfinite(dt)) or dt <= 1.0e-9 or (not np.all(np.isfinite(U))):
            self.latest_seed_info = {"used": False, "reason": "invalid_cached_initial_guidance", "detail": "bad U/dt"}
            return None

        X = seed.get("X", None)
        if X is None:
            try:
                X = rollout_ego(np.asarray(x0, dtype=float).reshape(4), U, dt, self.cfg)
            except Exception:
                X = None
        else:
            try:
                X = np.asarray(X, dtype=float).copy()
                if X.ndim == 2:
                    if X.shape[0] not in (4, 5) and X.shape[1] in (4, 5):
                        X = X.T
                    if X.shape[0] >= 4:
                        X[:4, 0] = np.asarray(x0, dtype=float).reshape(4)
                    else:
                        X = None
                else:
                    X = None
            except Exception:
                X = None
        if X is not None and not self._array_is_finite(X):
            self.latest_seed_info = {"used": False, "reason": "invalid_cached_initial_guidance", "detail": "bad X"}
            return None

        self.latest_seed_info = {
            "used": True,
            "source": "initial_guidance",
            "topic": str(self.initial_guidance_topic),
            "seq_id": int(seed.get("seq_id", -1)),
            "recv_id": int(recv_id),
            "dt": float(self.cfg.dt_init),
            "count": int(seed.get("count", int(self.cfg.N))),
        }
        return U, X, dt

    # ------------------------------------------------------------------
    # State / planning
    # ------------------------------------------------------------------

    @staticmethod
    def _order_sign_to_value(sign: int, lower: float, upper: float) -> float:
        sign_i = int(sign)
        if sign_i < 0:
            return float(lower)
        if sign_i > 0:
            return float(upper)
        return 0.0

    def order_acc_yawrate_signs(self, order: Optional[int] = None) -> Tuple[int, int]:
        """Map ORDER to the setter-owned acc/yawrate ordering.

        ORDER uses acc-major / yawrate-minor indexing:
          order = acc_index * len(yawrate_order) + yawrate_index + 1.
        """
        acc_order = self.acc_order
        yawrate_order = self.yawrate_order
        idx = int(self.ORDER if order is None else order) - 1
        max_idx = len(acc_order) * len(yawrate_order) - 1
        idx = max(0, min(idx, max_idx))
        acc_idx = idx // len(yawrate_order)
        yaw_idx = idx % len(yawrate_order)
        return int(acc_order[acc_idx]), int(yawrate_order[yaw_idx])

    def order_yawrate_bias(self) -> float:
        explicit = float(getattr(self, "fallback_yawrate_bias", float("nan")))
        if np.isfinite(explicit):
            return float(np.clip(explicit, self.cfg.yawrate_min, self.cfg.yawrate_max))
        _, yaw_sign = self.order_acc_yawrate_signs()
        return self._order_sign_to_value(
            yaw_sign,
            float(self.cfg.yawrate_min),
            float(self.cfg.yawrate_max),
        )

    def order_acc_bias(self) -> float:
        if bool(getattr(self.cfg, "no_acc_flag", False)):
            return 0.0
        explicit = float(getattr(self, "fallback_acc_bias", float("nan")))
        if np.isfinite(explicit):
            return float(np.clip(explicit, self.cfg.a_min, self.cfg.a_max))
        acc_sign, _ = self.order_acc_yawrate_signs()
        return self._order_sign_to_value(
            acc_sign,
            float(self.cfg.a_min),
            float(self.cfg.a_max),
        )

    def make_order_nominal_seed(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Build a constant-control nominal seed for the current ORDER."""
        N = int(self.cfg.N)
        U = np.zeros((2, N), dtype=float)
        U[0, :] = self.order_yawrate_bias()
        U[1, :] = self.order_acc_bias()
        dt = float(self.cfg.dt_init)
        try:
            X = rollout_ego(x0, U, dt, self.cfg)
        except Exception:
            X = np.repeat(np.asarray(x0, dtype=float).reshape(4, 1), N + 1, axis=1)
        return U, X, dt

    def make_order_fallback_plan(
        self,
        x0: np.ndarray,
        vertex_set: np.ndarray,
        halfspaces: np.ndarray,
        exitflag: int,
        vertex_types: Optional[np.ndarray] = None,
        dynamic_halfspace_mask: Optional[np.ndarray] = None,
    ) -> PlanResult:
        N = int(self.cfg.N)
        print("###################")
        print("# fallback #")
        print("###################")
        U, X, dt = self.make_order_nominal_seed(x0)
        plan = PlanResult(
            U=U,
            X=X,
            dt=dt,
            exitflag=int(exitflag),
            cost=float("inf"),
            vertex_set=vertex_set,
            halfspaces=halfspaces,
            slack=np.zeros(N, dtype=float),
            normal_slack=np.zeros(N, dtype=float),
            vertex_types=vertex_types,
            dynamic_halfspace_mask=dynamic_halfspace_mask,
        )
        plan.is_order_fallback = True
        plan.ocp_feasible = False
        plan.original_ocp_feasible = False
        plan.paper_feasible = False
        plan.zero_slack_feasible = False
        plan.s_r_feasible = False
        plan.selection_eligible = False
        plan.J_global = 1.0e15
        plan.global_cost = 1.0e15
        return plan

    def _setup_callback(self, msg: Setup) -> None:
        """Cache /setup and apply runtime field/sector comparison overrides."""
        raw = setup_msg_to_raw(msg)
        if isinstance(raw, dict) and raw:
            self._setup_raw = raw
            self._setup_event.set()

            mpc_cfg = raw.get("mpc", {})
            if isinstance(mpc_cfg, dict) and "cal_time_log_flag" in mpc_cfg:
                self.cal_time_log_flag = _as_bool(mpc_cfg.get("cal_time_log_flag"), True)

            # field_level_comparison.py republishes the setter-owned Setup
            # message with only normal_cfg.field_flag changed between runs.
            # self.cfg is shared with AcadosMPCPlanner, so changing this one
            # attribute updates the collision-field evaluator without rebuilding
            # the OCP or changing any other MPC parameter.
            cfg_obj = getattr(self, "cfg", None)
            normal_cfg = raw.get("normal_cfg", {})
            if cfg_obj is not None and isinstance(normal_cfg, dict) and "field_flag" in normal_cfg:
                try:
                    new_flag = int(round(float(normal_cfg["field_flag"])))
                    if new_flag not in (0, 1, 2, 3, 4):
                        raise ValueError("field_flag out of range")
                    old_flag = int(getattr(cfg_obj, "field_flag", new_flag))
                    if old_flag != new_flag:
                        cfg_obj.field_flag = new_flag
                        self._loginfo(
                            "[%s] runtime /setup field_flag update: %d -> %d",
                            self.planner_key, old_flag, new_flag,
                        )
                except Exception as exc:
                    rospy.logerr("[%s] invalid runtime field_flag: %s", self.planner_key, exc)

            # Runtime sector update.  /setup may be republished between runs while
            # this planner process stays alive.  Reconstruct the effective planner
            # config using the same precedence as startup, then refresh the sector
            # center ray used as n(phi_c) by the regulation constraint.
            if cfg_obj is not None:
                try:
                    planner_id_now = int(getattr(
                        self, "guidance_topic_id",
                        infer_planner_id_from_key(self.planner_key, int(getattr(cfg_obj, "planner_id", 0)))
                    ))
                    sector_cfg, _, _ = load_planner_config(
                        raw_config=raw, planner_id=planner_id_now
                    )
                    sector_runtime = get_planner_runtime_config(raw, self.planner_key)
                    _apply_config_overrides(sector_cfg, sector_runtime)
                    new_sector_id = normalize_sector_config(sector_cfg)

                    old_sector_id = int(getattr(self, "sector_id", getattr(cfg_obj, "sector_id", new_sector_id)))
                    old_rays = np.asarray(
                        getattr(self, "sector_rays_body", sector_rays_body_from_id(
                            float(cfg_obj.ego_L), float(cfg_obj.ego_W), old_sector_id
                        )),
                        dtype=float,
                    ).reshape(3, 2)

                    cfg_obj.sector_id = int(new_sector_id)
                    cfg_obj.designated_surface_id = int(new_sector_id)
                    cfg_obj.sector_dir = str(getattr(sector_cfg, "sector_dir", getattr(cfg_obj, "sector_dir", "outward")))
                    normalize_sector_config(cfg_obj)

                    new_rays = sector_rays_body_from_id(
                        float(cfg_obj.ego_L), float(cfg_obj.ego_W), int(new_sector_id)
                    )
                    if str(getattr(cfg_obj, "sector_dir", "outward")).strip().lower() == "inward":
                        new_rays = tuple(-np.asarray(ray, dtype=float) for ray in new_rays)
                    new_rays_arr = np.asarray(new_rays, dtype=float).reshape(3, 2)

                    self.sector_id = int(new_sector_id)
                    self.sector_rays_body = tuple(new_rays_arr[i].copy() for i in range(3))

                    # Acknowledge only after cfg.sector_id and sector_rays_body are both updated.
                    ack_pub = getattr(self, "sector_ack_pub", None)
                    if ack_pub is not None:
                        ack_pub.publish(Int32(data=int(new_sector_id)))
                        self._loginfo(
                            "[%s] runtime sector ACK: sector_id=%d -> %s",
                            self.planner_key, int(new_sector_id), self.sector_ack_topic,
                        )

                    if old_sector_id != int(new_sector_id) or not np.allclose(old_rays, new_rays_arr):
                        lc = new_rays_arr[2]
                        rospy.loginfo(
                            "[%s] runtime /setup sector update: %d -> %d, n(phi_c)_body=[%.6f, %.6f]",
                            self.planner_key, old_sector_id, int(new_sector_id), float(lc[0]), float(lc[1]),
                        )
                except Exception as exc:
                    rospy.logerr("[%s] invalid runtime sector update: %s", self.planner_key, exc)

    def _wait_for_setup_raw(self) -> Dict[str, Any]:
        while not rospy.is_shutdown():
            if self._setup_event.wait(0.05):
                raw = self._setup_raw
                if isinstance(raw, dict) and raw:
                    return raw
        raise rospy.ROSInterruptException(
            "shutdown while waiting for %s" % self.setup_topic
        )

    def build_x0(self) -> np.ndarray:
        return np.array([self.ego_x, self.ego_y, self.ego_v, self.ego_yaw], dtype=float)

    def step(self) -> None:
        # Solve each scheduled MPC tick using the newest cached pose and cmd_vel.
        if not (self.have_bounds and self.have_goal):
            return
        # Snapshot the latest cached pose and command every MPC tick.
        # Reuse the cached pose when no new callback arrived; only the first
        # valid pose is required to begin planning.
        with self.tracked_pose_lock:
            if not self.have_ego:
                return
            # Pose callback owns x/y/yaw; speed is the last /cmd_vel.linear.x.
            self.ego_v = float(self.latest_cmd_vel)
            ego_snapshot = self.build_x0().copy()
            self.planning_cycle_id += 1
            planning_round_id = int(self.planning_cycle_id)

        # Complete planner-cycle timing, matching vertex_selector.py's
        # perf_counter + throttled phase-breakdown style.
        step_t0 = time.perf_counter()
        input_t0 = step_t0
        input_ms = 0.0
        halfspace_ms = 0.0
        seed_ms = 0.0
        solve_ms = 0.0
        post_ms = 0.0
        hyperplane_pub_ms = 0.0
        control_pub_ms = 0.0
        halfspace_t0 = None
        seed_t0 = None
        solve_t0 = None

        x0 = np.asarray(ego_snapshot, dtype=float).reshape(4).copy()
        # Freeze the exact h=0 selector data at the same planning snapshot.
        # Later selector/horizon callbacks cannot change this cycle's
        # predicted/true-normal comparison context.
        h0_snapshot = self.current_vertex_snapshot
        h0_vertex_set = np.asarray(h0_snapshot[0], dtype=float).reshape(-1, 2).copy()
        h0_vertex_normals = np.asarray(h0_snapshot[1], dtype=float).reshape(-1, 2).copy()
        h0_vertex_types = np.asarray(h0_snapshot[2], dtype=np.int32).reshape(-1).copy()
        h0_group_ids = np.asarray(h0_snapshot[3], dtype=np.int32).reshape(-1).copy()
        goal = np.array([self.goal_x, self.goal_y], dtype=float)

        # vertex_selector.py has already applied FOV/radius filtering and
        # generated MATLAB-style pseudo vertices.  Use the latest received
        # selected features directly; do not re-select in the MPC node.
        vertex_set = np.asarray(self.latest_vertex_set, dtype=float).reshape(-1, 2)
        vertex_normals = np.asarray(self.latest_vertex_normals, dtype=float).reshape(-1, 2)
        vertex_types = np.asarray(self.latest_vertex_types, dtype=np.int32).reshape(-1)
        if bool(self.have_horizon_vertices):
            stage_vertex_set = np.asarray(self.latest_vertex_set_horizon, dtype=float).reshape(int(self.cfg.N), -1, 2)
            stage_vertex_normals = np.asarray(self.latest_vertex_normals_horizon, dtype=float).reshape(int(self.cfg.N), -1, 2)
            vertex_set = stage_vertex_set[0].copy()
            vertex_normals = stage_vertex_normals[0].copy()
        else:
            stage_vertex_set = np.repeat(vertex_set.reshape(1, -1, 2), int(self.cfg.N), axis=0)
            stage_vertex_normals = np.repeat(vertex_normals.reshape(1, -1, 2), int(self.cfg.N), axis=0)
        input_ms = (time.perf_counter() - input_t0) * 1000.0
        halfspace_t0 = time.perf_counter()
        # Recompute only the ego-dependent halfspace rows at the MPC tick.
        # This is not vertex selection; vertex_selector.py still owns which
        # vertices exist in vertex_set.
        halfspaces = selected_vertices_normals_to_halfspaces(
            vertex_set,
            vertex_normals,
            x0,
            int(self.cfg.vertex_capacity),
            self.cfg,
            disabled_c_value=float(self.cfg.disabled_c_value),
        )
        dynamic_halfspace_mask = vertex_types_to_dynamic_mask(vertex_types, int(self.cfg.vertex_capacity))
        self.latest_halfspaces = halfspaces
        self.latest_dynamic_halfspace_mask = dynamic_halfspace_mask
        halfspace_ms = (time.perf_counter() - halfspace_t0) * 1000.0
        seed_t0 = time.perf_counter()

        feature_obs_indices = np.asarray(
            self.latest_selected_obs_indices, dtype=np.int32
        ).reshape(-1).copy()

        try:
            nominal = self.pop_initial_guidance_nominal(x0)
            solve_kwargs: Dict[str, Any] = dict(
                x0=x0,
                goal=goal,
                vertex_set=vertex_set,
                vertex_normals=vertex_normals,
                stage_vertex_set=stage_vertex_set,
                stage_vertex_normals=stage_vertex_normals,
                feature_obs_indices=feature_obs_indices,
                halfspaces=halfspaces,
                dynamic_halfspace_mask=dynamic_halfspace_mask,
                vertex_types=vertex_types,
                order=int(self.ORDER),
                sqp_iter=int(self.SQP_iter),
                sector_rays_body=self.sector_rays_body,
            )
            fallback_U_nom, fallback_X_nom, _ = self.make_order_nominal_seed(x0)
            solve_kwargs["fallback_U"] = fallback_U_nom
            solve_kwargs["fallback_X"] = fallback_X_nom
            if nominal is not None:
                U_nom, X_nom, dt_nom = nominal
                solve_kwargs["initial_U"] = U_nom
                solve_kwargs["initial_X"] = X_nom
                solve_kwargs["initial_dt"] = float(self.cfg.dt_init)
                solve_kwargs["reference_states"] = X_nom
                solve_kwargs["original_goal"] = goal
                # MATLAB refines each candidate toward its own local guidance endpoint.
                if X_nom is not None and np.asarray(X_nom).ndim == 2 and np.asarray(X_nom).shape[1] > 0:
                    solve_kwargs["goal"] = np.asarray(X_nom, dtype=float)[0:2, -1]
            else:
                # No usable initial_guidance was received (planner 1--3), or
                # initial guidance is intentionally disabled (planner 4--6).
                # Run the MPC from the planner-specific fixed-bias nominal
                # rather than publishing an invalid candidate immediately.
                U_nom, X_nom, dt_nom = self.make_order_nominal_seed(x0)
                self.latest_seed_info = {
                    "used": False,
                    "reason": "fixed_bias_nominal",
                    "acc_bias": float(self.order_acc_bias()),
                    "yawrate_bias": float(self.order_yawrate_bias()),
                }
                solve_kwargs["initial_U"] = U_nom
                solve_kwargs["initial_X"] = X_nom
                solve_kwargs["initial_dt"] = float(self.cfg.dt_init)
                solve_kwargs["original_goal"] = goal
                # No usable initial guidance: keep solve_kwargs["goal"] equal to
                # the global /setup.config.x_end/y_end target. X_nom is only
                # the fixed-bias warm-start initialization; it is not p_r.

            seed_ms = (time.perf_counter() - seed_t0) * 1000.0
            solve_t0 = time.perf_counter()
            plan = self.planner.solve(**solve_kwargs)
            solve_ms = (time.perf_counter() - solve_t0) * 1000.0
            seed = self.latest_seed_info if isinstance(self.latest_seed_info, dict) else {}
            if seed.get("used", False):
                rospy.logdebug(
                    "[MPC_planner%d] SQP nominal from %s seq=%d recv=%d dt=%.4f",
                    int(self.cfg.planner_id), str(seed.get("topic", "")),
                    int(seed.get("seq_id", -1)), int(seed.get("recv_id", -1)),
                    float(self.cfg.dt_init),
                )
        except Exception as exc:
            if solve_t0 is not None:
                solve_ms = (time.perf_counter() - solve_t0) * 1000.0
            elif seed_t0 is not None:
                seed_ms = (time.perf_counter() - seed_t0) * 1000.0
            rospy.logerr_throttle(
                1.0,
                "[MPC_planner%d] linearized MPC solve failed: %s",
                int(self.cfg.planner_id), str(exc)
            )
            plan = self.make_order_fallback_plan(
                x0=x0,
                vertex_set=vertex_set,
                halfspaces=halfspaces,
                exitflag=-999,
                vertex_types=vertex_types,
                dynamic_halfspace_mask=dynamic_halfspace_mask,
            )
            plan.trajectory_type = 0
            plan.trajectory_stage = "single_dual_constraint_fallback"
            plan.is_order_fallback = True
            self.planner.last_result = plan
            # Never leak candidates from a preceding planning cycle when an
            # exception happens before/during the current solve.
            plan.intermediate_results = []
            plan.successful_iteration_count = 0
            self.planner.last_iteration_results = []
            self.planner.last_feasible_candidates = []

        post_t0 = time.perf_counter()

        # J_global is evaluated for every successful outer-SQP iterate.
        # Feasibility is decided here, uniformly across MPC_planner0..6,
        # only from the optimized regulation slack s_r^k:
        #     feasible <=> s_r^k <= 1e-3 for every horizon stage k.
        # The mask is retained for diagnostics/metadata only; it does not select
        # the published candidate.
        all_iteration_plans = list(getattr(self.planner, "last_iteration_results", []))

        def _s_r_feasible(candidate):
            try:
                s_r = np.asarray(
                    getattr(candidate, "normal_slack", np.zeros(0)), dtype=float
                ).reshape(-1)
                feasible = bool(
                    s_r.size > 0
                    and np.all(np.isfinite(s_r))
                    and np.all(s_r <= float(self.s_r_feasibility_tol))
                )
                s_r_max = float(np.max(s_r)) if s_r.size else float("inf")
            except Exception:
                feasible = False
                s_r_max = float("inf")

            # Keep aliases synchronized for downstream consumers built against
            # older MPCControlSeq / PlanResult conventions.
            candidate.s_r_feasible = bool(feasible)
            candidate.s_r_max = float(s_r_max)
            candidate.original_ocp_feasible = bool(feasible)
            candidate.zero_slack_feasible = bool(feasible)
            candidate.selection_eligible = bool(feasible)
            return bool(feasible)

        feasible_plans = [cand for cand in all_iteration_plans if _s_r_feasible(cand)]
        plan.feasible_candidates = list(feasible_plans)
        self.planner.last_feasible_candidates = list(feasible_plans)

        # Feasible masking and J_global evaluation are intentionally kept for
        # every successful SQP iterate as diagnostics. They DO NOT affect which
        # candidate is published. The published control/trajectory is always the
        # final successful outer-SQP iteration result.
        if all_iteration_plans:
            self.latest_plan = all_iteration_plans[-1]
            selection_source = "last_iteration"
        else:
            # No successful SQP iterate exists (e.g. solver exception/failure).
            # Keep the deterministic ORDER fallback as the only publishable plan.
            self.latest_plan = plan
            selection_source = "solver_fallback"

        # Record the last published planning cycle; do not gate subsequent solves.
        self.last_planned_round_id = int(planning_round_id)
        post_ms = (time.perf_counter() - post_t0) * 1000.0

        # No MPC-side RViz/debug hyperplane publication.
        hyperplane_pub_ms = 0.0

        # Publish exactly ONE planner-local MPCControlSeq for this planning round:
        # the final successful outer-SQP iteration result, irrespective of feasibility
        # and irrespective of J_global ranking. plotter.py reconstructs the matching
        # trajectory from this same MPCControlSeq.
        selected_sqp_iteration = int(getattr(self.latest_plan, "sqp_iteration", -1))
        selected_trajectory_type = (
            0 if selection_source == "solver_fallback"
            else max(1, selected_sqp_iteration + 1)
        )
        control_pub_t0 = time.perf_counter()
        self.publish_mpc_control_seq(
            self.latest_plan, x0, planning_round_id=planning_round_id,
            trajectory_type=selected_trajectory_type,
            normal_h0_vertex_set=h0_vertex_set,
            normal_h0_vertex_normals=h0_vertex_normals,
            normal_h0_vertex_types=h0_vertex_types,
            normal_h0_group_ids=h0_group_ids,
        )
        control_pub_ms = (time.perf_counter() - control_pub_t0) * 1000.0

        # Always expose the performance breakdown, like vertex_selector.py.
        # Throttling limits synchronous console I/O while keeping bottlenecks visible.
        total_step_ms = (time.perf_counter() - step_t0) * 1000.0
        accounted_ms = (
            input_ms + halfspace_ms + seed_ms + solve_ms + post_ms
            + hyperplane_pub_ms + control_pub_ms
        )
        other_ms = max(0.0, total_step_ms - accounted_ms)
        timing_plan = self.latest_plan if self.latest_plan is not None else plan
        solve_timing = getattr(timing_plan, "timing_ms", {}) or {}
        self._loginfo_throttle(
            0.5,
            "[MPC_planner%d][timing] total=%.3f ms | input=%.3f | halfspace=%.3f | seed=%.3f | solve=%.3f | post=%.3f | hyperplane_pub=%.3f | control_pub=%.3f | other=%.3f",
            int(self.cfg.planner_id), float(total_step_ms), float(input_ms),
            float(halfspace_ms), float(seed_ms), float(solve_ms), float(post_ms),
            float(hyperplane_pub_ms), float(control_pub_ms), float(other_ms),
        )
        self._loginfo_throttle(
            0.5,
            "[MPC_planner%d][solve_timing] total=%.3f ms | input_prep=%.3f | halfspace=%.3f | stage_prep=%.3f | nominal_prep=%.3f | construct_lin=%.3f | warm_start=%.3f | phase0=%.3f | phase1=%.3f | phase2=%.3f | main=%.3f | extract=%.3f | rollout=%.3f | candidate=%.3f | jglobal=%.3f | warm_update=%.3f | finalize=%.3f | fallback=%.3f | other=%.3f",
            int(self.cfg.planner_id),
            float(solve_timing.get("total", solve_ms)),
            float(solve_timing.get("input_prep", 0.0)),
            float(solve_timing.get("halfspace", 0.0)),
            float(solve_timing.get("stage_prep", 0.0)),
            float(solve_timing.get("nominal_prep", 0.0)),
            float(solve_timing.get("construct_lin", 0.0)),
            float(solve_timing.get("warm_start", 0.0)),
            float(solve_timing.get("phase0", 0.0)),
            float(solve_timing.get("phase1", 0.0)),
            float(solve_timing.get("phase2", 0.0)),
            float(solve_timing.get("main", 0.0)),
            float(solve_timing.get("extract", 0.0)),
            float(solve_timing.get("rollout", 0.0)),
            float(solve_timing.get("candidate_finalize", 0.0)),
            float(solve_timing.get("jglobal", 0.0)),
            float(solve_timing.get("warm_update", 0.0)),
            float(solve_timing.get("finalize", 0.0)),
            float(solve_timing.get("fallback", 0.0)),
            float(solve_timing.get("other", 0.0)),
        )
        sqp_timing_rows = getattr(timing_plan, "iteration_timing_ms", []) or []
        if sqp_timing_rows:
            sqp_summary = " ; ".join(
                "i%d total=%.3f construct_lin=%.3f phase0=%.3f phase1=%.3f phase2=%.3f main=%.3f extract=%.3f rollout=%.3f candidate=%.3f jglobal=%.3f warm_update=%.3f" % (
                    int(row.get("sqp", -1)),
                    float(row.get("total", 0.0)),
                    float(row.get("construct_lin", 0.0)),
                    float(row.get("phase0", 0.0)),
                    float(row.get("phase1", 0.0)),
                    float(row.get("phase2", 0.0)),
                    float(row.get("main", 0.0)),
                    float(row.get("extract", 0.0)),
                    float(row.get("rollout", 0.0)),
                    float(row.get("candidate_finalize", 0.0)),
                    float(row.get("jglobal", 0.0)),
                    float(row.get("warm_update", 0.0)),
                )
                for row in sqp_timing_rows
            )
            self._loginfo_throttle(
                0.5, "[MPC_planner%d][sqp_timing] %s",
                int(self.cfg.planner_id), sqp_summary,
            )

        # /mpc_pred_traj is no longer published here.  plotter.py
        # visualizes all planner predictions from /mpc_control_seq*.

        selected_dyn_vertices = int(np.count_nonzero(dynamic_halfspace_mask[: int(vertex_set.shape[0])]))

        active_plan = self.latest_plan if self.latest_plan is not None else plan
        if selection_source == "last_iteration":
            self._loginfo_throttle(
                0.5,
                "[MPC_planner%d] seq=%d published final SQP iteration; feasible=%d J_global=%.3e sqp_iter=%d feasible_count=%d total_iter=%d s_r_max=%.3e phi_min=%.3e",
                int(self.cfg.planner_id), int(planning_round_id),
                int(bool(getattr(active_plan, "s_r_feasible", False))),
                float(getattr(active_plan, "J_global", getattr(active_plan, "global_cost", float("inf")))),
                int(getattr(active_plan, "sqp_iteration", -1)),
                len(feasible_plans), len(all_iteration_plans),
                float(getattr(active_plan, "s_r_max", float("inf"))),
                float(getattr(active_plan, "phi_min", float("nan"))),
            )
        else:
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] seq=%d no successful SQP iteration; published deterministic solver fallback; exit=%d ORDER=%d SQP_iter=%d",
                int(self.cfg.planner_id), int(planning_round_id), int(plan.exitflag),
                int(self.ORDER), int(self.SQP_iter),
            )

        # self.ego_trail.append((float(x0[0]), float(x0[1])))
        # if len(self.ego_trail) > int(self.cfg.ego_trail_max):
        #     self.ego_trail = self.ego_trail[-int(self.cfg.ego_trail_max):]
        # self.publish_ego_traj()

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def publish_separating_hyperplane_data(
        self, plan: Optional[PlanResult], planning_round_id: int = -1
    ) -> None:
        if self.pub_separating_hyperplane_data is None:
            return
        """Publish the exact proposed avoidance affine blocks used by acados.

        One row has 14 fields:
          [seq_id, h, t, pbar_x, pbar_y, pbar_v, pbar_yaw,
           g_bar, dg_dpx, dg_dpy, dg_dv, dg_dyaw, s_k, Phi_bar]

        For proposed MPC block 0, g_bar = d_safe - Phi_bar and acados imposes

            g_bar + grad_g^T (x_{k+1} - xbar_k) <= s_k.

        Therefore the 2-D world-frame slice drawn by plotter.py is

            dg_dpx (x-pbar_x) + dg_dpy (y-pbar_y) + g_bar = s_k,

        with v/yaw fixed at the collision linearization point.
        """
        msg = Float64MultiArray()
        fields = 14
        rows = []

        if plan is not None and int(getattr(self.cfg, "mpc_flag", 4)) == 4:
            lin = getattr(plan, "avoidance_linearization", None)
            nominal = getattr(plan, "collision_nominal", None)
            if lin is not None and nominal is not None:
                try:
                    lin_arr = np.asarray(lin, dtype=float).reshape(-1, 7)
                    nom_arr = np.asarray(nominal, dtype=float).reshape(-1, 4)
                    slack_arr = np.asarray(
                        getattr(plan, "slack", np.zeros(0)), dtype=float
                    ).reshape(-1)
                    H = min(int(self.cfg.N), lin_arr.shape[0], nom_arr.shape[0])
                    hub = float(getattr(self.cfg, "h_upper_bound", 1.0e15))
                    d_safe = float(getattr(self.cfg, "d_safe", 0.0))
                    dt = float(getattr(plan, "dt", self.cfg.dt_init))
                    for k0 in range(H):
                        block = lin_arr[k0]
                        xbar = nom_arr[k0]
                        if (not np.all(np.isfinite(block[:5])) or
                                not np.all(np.isfinite(xbar)) or
                                float(block[0]) <= -0.5 * hub):
                            continue
                        s_k = float(slack_arr[k0]) if k0 < slack_arr.size else 0.0
                        if not np.isfinite(s_k):
                            s_k = 0.0
                        g_bar = float(block[0])
                        rows.append([
                            float(planning_round_id), float(k0 + 1),
                            float(k0 + 1) * dt,
                            float(xbar[0]), float(xbar[1]), float(xbar[2]), float(xbar[3]),
                            g_bar, float(block[1]), float(block[2]),
                            float(block[3]), float(block[4]), max(s_k, 0.0),
                            d_safe - g_bar,
                        ])
                except Exception as exc:
                    rospy.logwarn_throttle(
                        1.0,
                        "[MPC_planner%d] failed to encode separating hyperplanes: %s",
                        int(self.cfg.planner_id), str(exc),
                    )

        dim_stage = MultiArrayDimension()
        dim_stage.label = "stage"
        dim_stage.size = len(rows)
        dim_stage.stride = len(rows) * fields
        dim_field = MultiArrayDimension()
        dim_field.label = (
            "fields:seq,h,t,pbar_x,pbar_y,pbar_v,pbar_yaw,"
            "g_bar,dg_dpx,dg_dpy,dg_dv,dg_dyaw,s_k,Phi_bar"
        )
        dim_field.size = fields
        dim_field.stride = fields
        msg.layout.dim = [dim_stage, dim_field]
        msg.layout.data_offset = 0
        msg.data = [float(v) for row in rows for v in row]
        self.pub_separating_hyperplane_data.publish(msg)

    def publish_mpc_control_seq(
        self,
        plan: PlanResult,
        start_x0: Optional[np.ndarray] = None,
        planning_round_id: int = -1,
        trajectory_type: int = 0,
        normal_stage_vertex_set: Optional[np.ndarray] = None,
        normal_stage_group_ids: Optional[np.ndarray] = None,
        normal_h0_vertex_set: Optional[np.ndarray] = None,
        normal_h0_vertex_normals: Optional[np.ndarray] = None,
        normal_h0_vertex_types: Optional[np.ndarray] = None,
        normal_h0_group_ids: Optional[np.ndarray] = None,
    ) -> None:
        msg = MPCControlSeq()
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
        if hasattr(msg, "seq_id"):
            msg.seq_id = int(planning_round_id)

        if hasattr(msg, "planner_id"):
            msg.planner_id = int(self.guidance_topic_id)

        if hasattr(msg, "trajectory_type"):
            msg.trajectory_type = int(trajectory_type)

        # Keep the solver/local ranking value and the global candidate-ranking
        # objective separate.  plan.cost remains J_local + configured slack cost;
        # J_global is computed independently for candidate selection.
        local_cost = float(getattr(plan, "cost", float("inf")))
        if not np.isfinite(local_cost):
            local_cost = 1.0e100
        if hasattr(msg, "cost"):
            msg.cost = float(local_cost)

        j_global = float(getattr(plan, "J_global", getattr(plan, "global_cost", float("inf"))))
        if not np.isfinite(j_global):
            j_global = 1.0e100
        for field_name in ("global_cost", "J_global", "j_global"):
            if hasattr(msg, field_name):
                setattr(msg, field_name, float(j_global))

        # Publish the nonlinear-candidate diagnostics that ego_mover needs for
        # Algorithm-1 filtering.  All assignments are guarded so older message
        # definitions remain importable; for strict filtering the corresponding
        # fields must exist in MPCControlSeq.msg.
        if hasattr(msg, "exitflag"):
            msg.exitflag = int(plan.exitflag)
        if hasattr(msg, "exit_flag"):
            msg.exit_flag = int(plan.exitflag)

        # Feasibility metadata uses the same regulation-slack criterion as
        # candidate masking above: max_k s_r^k <= 1e-3. A deterministic
        # solver fallback is not a successful SQP candidate and is marked infeasible.
        try:
            s_r_for_feasibility = np.asarray(
                getattr(plan, "normal_slack", np.zeros(0)), dtype=float
            ).reshape(-1)
            s_r_feasible = bool(
                not bool(getattr(plan, "is_order_fallback", False))
                and s_r_for_feasibility.size > 0
                and np.all(np.isfinite(s_r_for_feasibility))
                and np.all(s_r_for_feasibility <= float(self.s_r_feasibility_tol))
            )
        except Exception:
            s_r_feasible = False
        original_ocp_feasible = bool(s_r_feasible)
        for field_name in ("original_ocp_feasible", "ocp_feasible", "selection_eligible"):
            if hasattr(msg, field_name):
                setattr(msg, field_name, bool(original_ocp_feasible))
        for field_name, value in (
            ("original_ocp_avoidance_residual_max", getattr(plan, "original_ocp_avoidance_residual_max", float("inf"))),
            ("original_ocp_slack_nonneg_residual_max", getattr(plan, "original_ocp_slack_nonneg_residual_max", float("inf"))),
            ("original_ocp_mitigation_residual_max", getattr(plan, "original_ocp_mitigation_residual_max", float("inf"))),
        ):
            if hasattr(msg, field_name):
                setattr(msg, field_name, float(value))
        if hasattr(msg, "sqp_iteration"):
            msg.sqp_iteration = int(getattr(plan, "sqp_iteration", -1))
        for field_name in ("candidate_batch_done", "batch_done"):
            if hasattr(msg, field_name):
                setattr(msg, field_name, False)

        # Feasibility/residual metadata.  Legacy zero-slack field names are
        # populated as aliases so this remains compatible
        # with an already-built MPCControlSeq.msg.
        zero_slack_avoidance_residual_max = float(
            getattr(plan, "zero_slack_avoidance_residual_max", float("inf"))
        )
        zero_slack_regulation_residual_max = float(
            getattr(plan, "zero_slack_normal_alignment_residual_max", float("inf"))
        )
        zero_slack_feasible = bool(s_r_feasible)
        phi_min = float(getattr(plan, "phi_min", float("-inf")))

        for field_name in (
            "zero_slack_avoidance_residual_max",
            "avoidance_residual_max",
            "collision_residual_max",
        ):
            if hasattr(msg, field_name):
                setattr(msg, field_name, float(zero_slack_avoidance_residual_max))
        for field_name in (
            "zero_slack_normal_alignment_residual_max",
            "normal_alignment_residual_max",
            "regulation_residual_max",
        ):
            if hasattr(msg, field_name):
                setattr(msg, field_name, float(zero_slack_regulation_residual_max))

        if hasattr(msg, "phi_min"):
            msg.phi_min = float(phi_min)
        # Legacy clearance fields carry the same Phi_min value.  ego_mover
        # explicitly prefers phi_min, but this keeps old message definitions usable.
        for field_name in ("d_min", "min_clearance", "minimum_clearance", "clearance_min"):
            if hasattr(msg, field_name):
                setattr(msg, field_name, float(phi_min))

        # Keep optimized slack magnitudes as diagnostics. normal_slack is s_r and
        # directly defines feasibility through the 1e-3 threshold above.
        normal_slack = np.asarray(getattr(plan, "normal_slack", np.zeros(0)), dtype=float).reshape(-1)
        normal_slack_max = float(np.max(normal_slack)) if normal_slack.size else 0.0
        avoidance_slack = np.asarray(getattr(plan, "slack", np.zeros(0)), dtype=float).reshape(-1)
        avoidance_slack_max = float(np.max(np.abs(avoidance_slack))) if avoidance_slack.size else 0.0
        if hasattr(msg, "normal_slack_max"):
            msg.normal_slack_max = float(normal_slack_max)
        if hasattr(msg, "avoidance_slack_max"):
            msg.avoidance_slack_max = float(avoidance_slack_max)

        # Prefer a dedicated field when available. Existing boolean aliases are
        # populated with the SAME s_r <= 1e-3 semantics so ego_mover can use
        # this code without requiring an immediate ROS message rebuild.
        for field_name in (
            "zero_slack_feasible",
            "nonlinear_zero_slack_feasible",
            "paper_feasible",
            "collision_free",
        ):
            if hasattr(msg, field_name):
                setattr(msg, field_name, bool(zero_slack_feasible))

        stage_label = (
            "single_dual_constraint_fallback"
            if bool(getattr(plan, "is_order_fallback", False))
            else "single_dual_constraint_traj"
        )
        if hasattr(msg, "trajectory_stage"):
            msg.trajectory_stage = stage_label

        dt = float(self.cfg.dt_init)
        msg.dt = dt

        # Store the ego state that was used as the MPC initial condition.
        # ego_mover uses this to reject low-cost candidates solved from an
        # outdated/different ego pose.  These assignments are guarded so older
        # MPCControlSeq.msg definitions remain loadable until the msg is rebuilt.
        try:
            x0_msg = np.asarray(start_x0 if start_x0 is not None else self.build_x0(), dtype=float).reshape(-1)
        except Exception:
            x0_msg = np.asarray(self.build_x0(), dtype=float).reshape(-1)
        if x0_msg.size >= 4:
            if hasattr(msg, "start_ego_x"):
                msg.start_ego_x = float(x0_msg[0])
            if hasattr(msg, "start_ego_y"):
                msg.start_ego_y = float(x0_msg[1])
            if hasattr(msg, "start_ego_z"):
                msg.start_ego_z = float(self.ego_z)
            if hasattr(msg, "start_ego_v"):
                msg.start_ego_v = float(x0_msg[2])
            if hasattr(msg, "start_ego_yaw"):
                msg.start_ego_yaw = float(wrap_to_pi(float(x0_msg[3])))

        # utils_linearized.py planner convention used here:
        # U[0, :] = yawrate, U[1, :] = acc.  The state arrays published in
        # MPCControlSeq are NOT copied from plan.X / solver states; they are
        # reconstructed by rolling the returned control sequence out from the
        # exact MPC start state.
        if plan.U is None or np.asarray(plan.U).size == 0:
            n = int(self.cfg.N)
            yawrates = np.zeros(n, dtype=float)
            accs = np.zeros(n, dtype=float)
        else:
            U = np.asarray(plan.U, dtype=float)
            if U.ndim == 1:
                U = U.reshape(2, -1)
            if U.shape[0] < 2:
                n = int(self.cfg.N)
                yawrates = np.zeros(n, dtype=float)
                accs = np.zeros(n, dtype=float)
            else:
                yawrates = U[0, :].astype(float)
                accs = U[1, :].astype(float)

        msg.yawrates = [float(v) for v in yawrates]
        msg.accs = [float(a) for a in accs]

        U_msg = np.vstack((yawrates, accs))
        X_msg = rollout_ego(x0_msg[:4], U_msg, dt, self.cfg)
        n_state = int(X_msg.shape[1])
        if hasattr(msg, "xs"):
            msg.xs = [float(v) for v in X_msg[0, :n_state]]
        if hasattr(msg, "ys"):
            msg.ys = [float(v) for v in X_msg[1, :n_state]]
        if hasattr(msg, "vs"):
            msg.vs = [float(v) for v in X_msg[2, :n_state]]
        if hasattr(msg, "yaws"):
            msg.yaws = [float(v) for v in X_msg[3, :n_state]]

        # Publish N_0(0) from the CURRENT planning snapshot (k=0), not k=1.
        # The exact same frozen /selected_vertex_data geometry and x0 are packed
        # into this message so repeat_scenario computes the comparison true normal
        # from identical vertices, group ids, ego pose, and planning round.
        #
        # For the proposed avoidance block g_a = d_s - Phi,
        # grad(g_a) = -grad(Phi), hence N_0(0) is the normalized position
        # gradient (dg/dpx, dg/dpy) of a genuine h_steps=[0] field evaluation.
        normal_candidate_msg = Float64MultiArray()
        normal_pub_stamp = float(rospy.Time.now().to_sec())
        normal_stage = -1
        normal_phi = float("nan")
        normal_nx = float("nan")
        normal_ny = float("nan")
        normal_nominal_state = np.full(4, np.nan, dtype=float)
        geometry_rows = np.zeros((0, 3), dtype=float)
        try:
            V0 = np.asarray(
                normal_h0_vertex_set if normal_h0_vertex_set is not None else np.zeros((0, 2)),
                dtype=float,
            ).reshape(-1, 2)
            N0 = np.asarray(
                normal_h0_vertex_normals if normal_h0_vertex_normals is not None else np.zeros((0, 2)),
                dtype=float,
            ).reshape(-1, 2)
            T0 = np.asarray(
                normal_h0_vertex_types if normal_h0_vertex_types is not None else np.zeros(0),
                dtype=np.int32,
            ).reshape(-1)
            G0 = np.asarray(
                normal_h0_group_ids if normal_h0_group_ids is not None else np.zeros(0),
                dtype=np.int32,
            ).reshape(-1)
            n0 = min(V0.shape[0], N0.shape[0], T0.size, G0.size)
            if n0 > 0 and x0_msg.size >= 4:
                V0 = np.ascontiguousarray(V0[:n0], dtype=float)
                N0 = np.ascontiguousarray(N0[:n0], dtype=float)
                T0 = np.ascontiguousarray(T0[:n0], dtype=np.int32)
                G0 = np.ascontiguousarray(G0[:n0], dtype=np.int32)
                x0_normal = np.asarray(x0_msg[:4], dtype=float).reshape(1, 4)
                lin0 = build_external_avoidance_linearization_sequence(
                    x0_normal, V0, N0, T0, G0,
                    h_steps=np.zeros(1, dtype=float),
                    dt_vals=np.full(1, float(self.cfg.dt_init), dtype=float),
                    cfg=self.cfg,
                    sector_rays_body=self.sector_rays_body,
                )
                block0 = np.asarray(lin0, dtype=float).reshape(-1, 7)[0]
                hub = float(getattr(self.cfg, "h_upper_bound", 1.0e15))
                gx = float(block0[1])
                gy = float(block0[2])
                gn = math.hypot(gx, gy)
                if (np.all(np.isfinite(block0[:5])) and
                        float(block0[0]) > -0.5 * hub and
                        np.isfinite(gn) and gn > 1.0e-12):
                    normal_stage = 0
                    d_safe = max(abs(float(getattr(self.cfg, "d_safe", 0.2))), 1.0e-12)
                    normal_phi = float(d_safe - float(block0[0]))
                    normal_nx = gx / gn
                    normal_ny = gy / gn
                    normal_nominal_state = x0_normal[0].copy()

                    good = np.all(np.isfinite(V0), axis=1) & (G0 >= 0)
                    if np.any(good):
                        geometry_rows = np.column_stack((
                            V0[good], G0[good].astype(float)
                        ))
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[MPC_planner%d] failed to evaluate k=0 predicted impact normal: %s",
                int(self.guidance_topic_id), str(exc),
            )

        # Extended payload keeps the exact k=0 evaluation context.
        # Layout:
        # [planner_id, seq_id, trajectory_type, sqp_iteration, publish_stamp_s,
        #  dt, normal_stage(=0), Phi, nx_world, ny_world, state_count,
        #  rollout states (4*state_count),
        #  geometry_count, x0_x, x0_y, x0_v, x0_yaw,
        #  h=0 geometry rows (x,y,group_id) * geometry_count].
        state_rows = np.asarray(X_msg, dtype=float).T.reshape(-1, 4)
        normal_candidate_msg.data = [
            float(self.guidance_topic_id),
            float(planning_round_id),
            float(trajectory_type),
            float(getattr(plan, "sqp_iteration", -1)),
            normal_pub_stamp,
            float(dt),
            float(normal_stage),
            float(normal_phi),
            float(normal_nx),
            float(normal_ny),
            float(state_rows.shape[0]),
        ] + state_rows.reshape(-1).astype(float).tolist() + [
            float(geometry_rows.shape[0]),
        ] + normal_nominal_state.astype(float).tolist() + geometry_rows.reshape(-1).astype(float).tolist()
        self.pub_predicted_impact_normal_candidate.publish(normal_candidate_msg)

        # Publish the optimized regulation-slack sequence s^r_k.
        # MPCControlSeq.msg defines this as float64[] normal_slacks, and
        # ego_mover uses it for the selected-s^r diagnostic/output.
        normal_slacks = np.asarray(
            getattr(plan, "normal_slack", np.zeros(0)), dtype=float
        ).reshape(-1)
        if hasattr(msg, "normal_slacks"):
            msg.normal_slacks = [float(s) for s in normal_slacks]
        self.pub_mpc_control_seq.publish(msg)

    def obstacle_visible_in_current_fov(self, obs: Dict[str, Any], ego: np.ndarray) -> bool:
        V = np.asarray(obs.get("V", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
        if V.shape[0] == 0:
            return False
        c = np.asarray(obs.get("center", np.mean(V, axis=0)), dtype=float).reshape(2)
        r = float(obs.get("radius", 0.0))
        ego_pos = ego[:2]
        if max(0.0, float(np.linalg.norm(c - ego_pos)) - r) > float(self.cfg.obs_sensing_R):
            return False
        fov_half = math.radians(float(self.cfg.obs_fov_deg))
        for vtx in V:
            rel = vtx - ego_pos
            rel_angle = wrap_to_pi(math.atan2(rel[1], rel[0]) - float(ego[3]))
            if abs(rel_angle) <= fov_half:
                return True
        return False

    def publish_percepted_obstacle_centers(self) -> None:
        if self.pub_percepted_obs is None:
            return
        # Publish perceived obstacles as obstacle_polygon so obs_types,
        # velocities, radii, and polygon vertices are preserved for plotter/RViz.
        msg = obstacle_polygon()
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id

        max_cnt = len(msg.obs_xs) if hasattr(msg, "obs_xs") else 0
        if max_cnt <= 0:
            self.pub_percepted_obs.publish(msg)
            return

        ego = self.build_x0()
        selected = []
        for obs in self.obstacles:
            if self.obstacle_visible_in_current_fov(obs, ego):
                selected.append(obs)

        msg.obs_cnt = min(len(selected), max_cnt)
        max_vertices = int(self.cfg.msg_max_vertices)
        for i in range(msg.obs_cnt):
            obs = selected[i]
            V = np.asarray(obs.get("V", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
            c = np.asarray(obs.get("center", np.mean(V, axis=0) if V.size else np.zeros(2)), dtype=float).reshape(2)
            vel = np.asarray(obs.get("vel", np.zeros(2)), dtype=float).reshape(2)
            obs_type = str(obs.get("type", "static")).strip().lower()
            obs_type_i = 1 if obs_type in ("dyn", "dynamic", "moving", "1", "true") else 0

            msg.obs_xs[i] = float(c[0])
            msg.obs_ys[i] = float(c[1])
            if hasattr(msg, "obs_zs"):
                msg.obs_zs[i] = float(self.goal_z)
            msg.obs_vxs[i] = float(vel[0]) if obs_type_i == 1 else 0.0
            msg.obs_vys[i] = float(vel[1]) if obs_type_i == 1 else 0.0
            if hasattr(msg, "obs_vzs"):
                msg.obs_vzs[i] = 0.0
            if hasattr(msg, "obs_rs"):
                msg.obs_rs[i] = float(obs.get("radius", 0.0))
            if hasattr(msg, "obs_types"):
                msg.obs_types[i] = int(obs_type_i)
            if hasattr(msg, "obs_vertex_counts"):
                nv = min(int(V.shape[0]), max_vertices)
                msg.obs_vertex_counts[i] = int(nv)
                for j in range(nv):
                    flat_idx = i * max_vertices + j
                    if flat_idx >= len(msg.obs_poly_xs) or flat_idx >= len(msg.obs_poly_ys):
                        break
                    msg.obs_poly_xs[flat_idx] = float(V[j, 0])
                    msg.obs_poly_ys[flat_idx] = float(V[j, 1])
                    if hasattr(msg, "obs_poly_zs") and flat_idx < len(msg.obs_poly_zs):
                        msg.obs_poly_zs[flat_idx] = 0.0

        self.pub_percepted_obs.publish(msg)

    def publish_mpc_pred_traj(self, plan: PlanResult) -> None:
        stamp = rospy.Time.now()
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = stamp
        mk.ns = "acados_vertex_set_mpc_pred"
        mk.id = 0
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = self.marker_width * 2.0
        _set_marker_color(mk, self.mpc_pred_color)
        if plan.X is not None and plan.X.size:
            for k in range(plan.X.shape[1]):
                p = Point()
                p.x = float(plan.X[0, k])
                p.y = float(plan.X[1, k])
                p.z = float(k * self.cfg.dt_init) * self.t_scale
                mk.points.append(p)
        # Deprecated: plotter.py publishes predicted trajectories from MPCControlSeq.
        return

    def publish_selected_vertices(self, vertex_set: np.ndarray) -> None:
        # Kept as a no-op for compatibility with old launch/config files.
        # Selected vertex visualization is performed only by plotter.py.
        return

    def publish_ego_traj(self) -> None:
        if self.pub_ego_traj is None:
            return
        stamp = rospy.Time.now()
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = stamp
        mk.ns = "acados_vertex_set_ego_traj"
        mk.id = 0
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = self.marker_width * 2.0
        _set_marker_color(mk, self.ego_traj_color)
        for x, y in self.ego_trail:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            mk.points.append(p)
        self.pub_ego_traj.publish(mk)


def main() -> None:
    rospy.init_node(infer_planner_key(), anonymous=False)
    node = MPCPlannerTrajectoryNode()
    time.sleep(1.0)
    rate = rospy.Rate(node.hz)
    while not rospy.is_shutdown():
        start_time = time.time()
        
        node.step()
        
        print("MPC_planner" + str(node.ORDER) + ": %s ms ---" %
              (((time.time() - start_time) * 1000)))
        rate.sleep()


if __name__ == "__main__":
    main()
