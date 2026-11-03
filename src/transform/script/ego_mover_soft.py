#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ego_mover.py

Owns /ego_state and ego dynamics only.
All RViz visualization publishers are owned by plotter.py.

Setup role
----------
- Subscribes to /setup; it never reads any JSON configuration directly.
- Uses the "ego" field for initial ego state, body size, update rate, frame and topics.
- Uses the "mpc" field only for dynamics limits and dt defaults through utils_linearized.MPCConfig.
- Receives the goal only from setter via /ego_goal_set and never publishes or changes the goal.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import re
import time
import threading
import copy
import tf2_ros

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion
from setup_config import setup_msg_to_raw

from drone_msgs.msg import Ego_goal_setting, Ego_state, MPCControlSeq, Setup
from std_msgs.msg import Float64MultiArray, Int32, MultiArrayDimension

from utils_linearized import (
    build_external_dual_collision_linearization_sequence,
    collision_safe_distance,
    collision_smoothing_lambda,
    load_planner_config,
    regulation_alignment_Ns,
    ros_param_override,
    step_ego_dynamics,
    wrap_to_pi,
)


def _wait_for_setup(setup_topic: str = "/setup") -> Dict[str, Any]:
    """Receive the latched Setup message published by setter.py."""
    rospy.loginfo("[%s] waiting for setup on %s", rospy.get_name(), setup_topic)
    msg = rospy.wait_for_message(setup_topic, Setup)
    raw = setup_msg_to_raw(msg)
    if not isinstance(raw, dict):
        raise ValueError("/setup did not decode to a configuration dictionary")
    return raw


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def _get_nested(block: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    sub = block.get(section, {})
    if isinstance(sub, dict) and key in sub:
        return sub[key]
    return default


def _get_first(block: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in block:
            return block[key]
    return default




def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _finite_or_large(value: Any, large: float = 1.0e100) -> float:
    try:
        out = float(value)
    except Exception:
        return float(large)
    if not np.isfinite(out):
        return float(large)
    return out


def _ellipse_axes_from_lengths(ego_L: float, ego_W: float) -> Tuple[float, float, Tuple[float, float]]:
    """Return (major_axis, minor_axis, body_axis_scale) for the ego ellipse."""
    L = max(abs(float(ego_L)), 1.0e-12)
    W = max(abs(float(ego_W)), 1.0e-12)
    minor = min(L, W)
    major = max(L, W)
    return float(major), float(minor), (float(L / minor), float(W / minor))


class EgoMoverNode:
    def __init__(self):
        self.setup_topic = str(rospy.get_param("~setup_topic", "/setup"))
        raw_cfg = _wait_for_setup(self.setup_topic)
        self.raw_setup = raw_cfg

        self.cfg, self.normal_cfg, raw_cfg = load_planner_config(raw_config=raw_cfg)
        ros_param_override(self.cfg, self.normal_cfg, rospy)

        ego_cfg = raw_cfg.get("ego", {}) if isinstance(raw_cfg.get("ego"), dict) else {}

        # ego_mover runs independently of the MPC discretization.  With the
        # received /setup message this is 20 Hz (0.05 s), while an MPC dt of 0.1 s holds
        # each selected control for two ego_mover updates.
        self.hz = float(rospy.get_param("~hz", ego_cfg.get("hz", 20.0)))
        if not np.isfinite(self.hz) or self.hz <= 0.0:
            rospy.logwarn("[ego_mover] invalid hz=%r; falling back to 20 Hz", self.hz)
            self.hz = 20.0
        # Cartographer tracked_pose is map-referenced. MPC and goals use map.
        self.frame_id = "map"
        self.world_frame_id = "map"
        self.publish_map_state = _as_bool(rospy.get_param("~publish_map_state", True), True)
        self.map_state_topic = str(rospy.get_param("~map_state_topic", "/ego_state_map"))
        self.world_tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.world_tf_listener = tf2_ros.TransformListener(self.world_tf_buffer)
        self.reset_from_goal_once = _as_bool(
            rospy.get_param("~reset_from_goal_once", ego_cfg.get("reset_from_goal_once", True)),
            True,
        )
        self.reset_on_every_goal = _as_bool(
            rospy.get_param("~reset_on_every_goal", ego_cfg.get("reset_on_every_goal", False)),
            False,
        )

        self.ego_x = float(rospy.get_param("~ego_x", _get_first(ego_cfg, ["ego_x", "ego_x0", "x"], self.cfg.ego_x)))
        self.ego_y = float(rospy.get_param("~ego_y", _get_first(ego_cfg, ["ego_y", "ego_y0", "y"], self.cfg.ego_y)))
        self.ego_z = float(rospy.get_param("~ego_z", _get_first(ego_cfg, ["ego_z", "ego_z0", "z"], self.cfg.ego_z)))
        self.ego_v = float(rospy.get_param("~ego_v", _get_first(ego_cfg, ["ego_v", "ego_v0", "v"], self.cfg.ego_v0)))
        self.ego_yaw = wrap_to_pi(float(rospy.get_param("~ego_yaw", _get_first(ego_cfg, ["ego_yaw", "ego_yaw0", "yaw"], self.cfg.ego_yaw0))))
        self.ego_L = float(ego_cfg.get("ego_L", self.cfg.ego_L))
        self.ego_W = float(ego_cfg.get("ego_W", self.cfg.ego_W))
        self.ego_R = float(rospy.get_param("~ego_R", ego_cfg.get("ego_R", self.cfg.ego_R)))
        self.ego_major_axis_m, self.ego_minor_axis_m, self.ego_axis_scale = _ellipse_axes_from_lengths(self.ego_L, self.ego_W)

        # Real robot: Cartographer /tracked_pose determines the map state.
        # Use /cmd_vel.linear.x as forward velocity (no pose differentiation).
        # Set ~use_real_robot:=false to retain the original simulation behavior.
        self.use_real_robot = _as_bool(rospy.get_param("~use_real_robot", True), True)
        self.pose_topic = str(rospy.get_param("~pose_topic", "/tracked_pose"))
        self.use_tracked_pose = _as_bool(rospy.get_param("~use_tracked_pose", True), True)
        # Cartographer's tracked frame can differ from base_link (e.g. imu_link).
        self.tracking_frame = str(rospy.get_param("~tracking_frame", "base_link")).lstrip("/")
        self.robot_frame = str(rospy.get_param("~robot_frame", "base_link")).lstrip("/")
        self.cmd_vel_topic = str(rospy.get_param("~cmd_vel_topic", "/cmd_vel"))
        self.command_timeout_s = max(0.05, float(rospy.get_param("~command_timeout_s", 1.0)))
        self.previous_velocity_pose = None
        self.velocity_filter_alpha = min(1.0, max(0.0, float(rospy.get_param("~velocity_filter_alpha", 1.0))))
        self.velocity_min_dt = max(0.001, float(rospy.get_param("~velocity_min_dt", 0.02)))
        self.velocity_max_dt = max(self.velocity_min_dt, float(rospy.get_param("~velocity_max_dt", 1.0)))
        self.velocity_jump_threshold = max(0.0, float(rospy.get_param("~velocity_jump_threshold", 0.75)))
        self.state_lock = threading.RLock()
        self.latest_cmd_vel = None
        self.have_tracked_pose = False

        # Raw-control mode: the selected MPC trajectory's control inputs
        # (yawrate, acc) are applied exactly as optimized (zero-order hold per
        # MPC interval).  No yaw-rate low-pass filtering is performed.
        rospy.loginfo("[ego_mover] raw control mode: applying selected trajectory's (yawrate, acc) directly")


        # Benchmark/repeat-scenario reset state.  A /scenario_reset Int32 message
        # restores exactly this initial state without restarting the ROS node.
        self.scenario_start_x = float(self.ego_x)
        self.scenario_start_y = float(self.ego_y)
        self.scenario_start_z = float(self.ego_z)
        self.scenario_start_v = float(self.ego_v)
        self.scenario_start_yaw = float(self.ego_yaw)
        self.scenario_managed_mode = False
        self.scenario_run_enabled = True
        self.scenario_seed = -1

        # Bounds/goal are initialized from /setup as startup defaults.
        # Runtime goal authority belongs exclusively to setter via /ego_goal_set.
        self.x_min = float(rospy.get_param("~x_min", _get_first(ego_cfg, ["x_min"], self.cfg.x_min)))
        self.x_max = float(rospy.get_param("~x_max", _get_first(ego_cfg, ["x_max"], self.cfg.x_max)))
        self.y_min = float(rospy.get_param("~y_min", _get_first(ego_cfg, ["y_min"], self.cfg.y_min)))
        self.y_max = float(rospy.get_param("~y_max", _get_first(ego_cfg, ["y_max"], self.cfg.y_max)))
        self.goal_x = float(rospy.get_param("~goal_x", _get_first(ego_cfg, ["goal_x", "x_end"], self.cfg.goal_x)))
        self.goal_y = float(rospy.get_param("~goal_y", _get_first(ego_cfg, ["goal_y", "y_end"], self.cfg.goal_y)))
        self.goal_z = float(rospy.get_param("~goal_z", _get_first(ego_cfg, ["goal_z", "z_end"], self.cfg.goal_z)))

        # Paper global candidate objective.  The planner-reported cost contains
        #
        #   sum_{k=1}^H ||x_k-x_G||_Q^2
        #   + sum_{k=1}^H [-rho_Phi*Phi_k + rho_s*s_k
        #                  + rho_c*N_k(0)^T*n(phi_c)].
        #
        # ego_mover owns the previously selected trajectory and therefore adds
        # exactly the remaining paper term
        #
        #   sum_{k=0}^{H-1} ||x_k-tilde{x}^{prev}_k||_{Q_sim}^2.
        #
        # The first planning round has no previous trajectory, so this term is 0.
        mpc_raw_cfg = raw_cfg.get("mpc", {}) if isinstance(raw_cfg.get("mpc"), dict) else {}
        self.traj_similarity_weight = max(
            0.0,
            float(rospy.get_param(
                "~traj_similarity_weight",
                ego_cfg.get("traj_similarity_weight", 1.0),
            )),
        )
        self.traj_similarity_pos_weight = max(
            0.0,
            float(rospy.get_param(
                "~traj_similarity_pos_weight",
                ego_cfg.get("traj_similarity_pos_weight", 20.0),
            )),
        )
        self.traj_similarity_yaw_weight = max(
            0.0,
            float(rospy.get_param(
                "~traj_similarity_yaw_weight",
                ego_cfg.get("traj_similarity_yaw_weight", 0.5),
            )),
        )
        self.traj_similarity_v_weight = max(
            0.0,
            float(rospy.get_param(
                "~traj_similarity_v_weight",
                ego_cfg.get("traj_similarity_v_weight", 0.5),
            )),
        )
        # Stores tilde{x}^{prev}_{0:H-1} = x^{selected,prev}_{1:H}.
        self.previous_selected_reference_states = None

        # Kept only as a diagnostic/backward-compatible mirror of the planner
        # parameter. rho_c is already included in planner_global_cost and MUST NOT
        # be added again in ego_mover.
        self.rho_c_global = max(
            0.0,
            float(rospy.get_param(
                "~rho_c_global",
                mpc_raw_cfg.get("rho_c_global", 100.0),
            )),
        )

        self.goal_tolerance = float(rospy.get_param(
            "~goal_tolerance",
            ego_cfg.get("goal_tolerance", max(float(self.ego_R), 0.40)),
        ))
        self.boundary_margin = float(rospy.get_param(
            "~boundary_margin",
            ego_cfg.get("boundary_margin", 0.0),
        ))
        # Boundary exit no longer shuts down ego_mover.  Boundary feasibility is
        # handled inside the MPC with a shared soft boundary slack.  Keep the
        # attribute only for backward-compatible parameter files, but ignore it
        # in termination_reason().
        self.stop_on_boundary_exit = False
        # The mover never creates, switches, or publishes goals.
        # Only setter may define the runtime goal through /ego_goal_set.
        self.stop_on_goal = _as_bool(
            rospy.get_param("~stop_on_goal", ego_cfg.get("stop_on_goal", False)),
            False,
        )
        self.stopped = False

        # Monotonic common planning-round ID.  It is published in Ego_state.header.seq
        # and copied unchanged by every MPC planner into MPCControlSeq.seq_id.
        # IMPORTANT: this ID is held constant until ego_mover has selected the
        # candidate set for that round; it is NOT incremented at every state tick.
        self.planning_round_id = 0

        # Candidate collection/validity rule.
        # Each MPC_planner publishes exactly ONE planner-local winner per common
        # planning round.  Therefore round completion is source-driven: snapshot
        # the number of currently connected MPCControlSeq publishers and close the
        # round as soon as every connected publisher has responded.  A timeout is
        # retained only as a failure fallback for a crashed/stalled planner.
        self.planning_period_s = 1.0 / max(float(self.hz), 1.0e-9)
        self.trajectory_receive_window_s = max(
            0.8 * float(self.planning_period_s),
            float(rospy.get_param(
                "~planner_result_timeout_s",
                ego_cfg.get("planner_result_timeout_s", max(0.25, 5.0 * float(self.planning_period_s))),
            )),
        )
        self.traj_start_position_threshold = max(
            0.0,
            float(rospy.get_param(
                "~traj_start_position_threshold",
                ego_cfg.get("traj_start_position_threshold", 0.5),
            )),
        )
        self.planning_round_start_xy: Dict[int, Tuple[float, float]] = {}
        # Exact state published at the beginning of each common planning round.
        # Used only as a fallback when an older MPCControlSeq.msg has no start_ego_* fields.
        self.planning_round_start_state: Dict[int, Tuple[float, float, float, float]] = {}


        # Collect planner trajectories by common planning-round seq_id.  Each
        # MPC_plannerN publishes one selected trajectory for the round.
        self.control_candidates_per_seq = max(
            1,
            int(rospy.get_param(
                "~control_candidates_per_seq",
                ego_cfg.get("control_candidates_per_seq", 6),
            )),
        )
        # Compatibility-only configured count.  Runtime completion uses the ROS
        # subscriber connection counts instead, so running 4 planners expects 4
        # responses even if several publishers share one topic.
        self.control_sources_per_seq = int(self.control_candidates_per_seq)
        self.pending_cmd = None  # selected candidate, retained for compatibility
        self.pending_candidates: List[Dict[str, Any]] = []
        self.pending_sources_seen = set()
        self.pending_expected_source_counts: Dict[str, int] = {}
        self.pending_source_receive_counts: Dict[str, int] = {}
        self.pending_messages_received = 0
        self.pending_candidates_rejected = 0
        self.pending_callbacks_inflight = 0
        self.pending_seq_id = None
        self.pending_first_recv_stamp = rospy.Time(0)
        self.pending_last_recv_stamp = rospy.Time(0)
        self.pending_window_timer = None
        self.pending_lock = threading.RLock()

        # Source-barrier planning-cycle closure.  Timeout is only a fallback.
        self.selection_feasibility_tol = max(
            1.0e-9,
            float(rospy.get_param(
                "~selection_feasibility_tol",
                ego_cfg.get("selection_feasibility_tol", max(float(getattr(self.cfg, "acados_tol", 1.0e-6)), 1.0e-6)),
            )),
        )

        # Two-tier selection rule over ALL trajectories received in one planning cycle:
        #   1) if at least one candidate is feasible for the original/nonsmooth OCP,
        #      minimize J_global among only those feasible candidates;
        #   2) if no feasible candidate exists, minimize J_global over ALL candidates.
        #
        # Here J_global = J_planner + J_sim. J_planner contains the planner-side
        # global objective terms; J_sim is the previous-trajectory Q_sim term
        # computed here. Feasibility is used only as a selection mask, never to
        # suppress J_global evaluation/collection.
        self.cmd_base_cost = 1.0e100
        self.cmd_terminal_cost = 1.0e100
        self.cmd_slack_cost = 0.0
        self.cmd_shared_slack = 0.0
        self.cmd_selection_cost = 1.0e100
        self.cmd_planner_global_cost = 1.0e100
        self.cmd_phi_min = float("-inf")
        self.cmd_min_clearance = float("-inf")  # legacy alias of cmd_phi_min
        self.cmd_zero_slack_feasible = False
        self.cmd_paper_feasible = False  # legacy alias of cmd_zero_slack_feasible
        self.cmd_selection_reason = "uninitialized"
        self.cmd_trajectory_type = 0

        # Keep cfg synchronized for the dynamics helper.
        self.cfg.ego_x = self.ego_x
        self.cfg.ego_y = self.ego_y
        self.cfg.ego_z = self.ego_z
        self.cfg.ego_v0 = self.ego_v
        self.cfg.ego_yaw0 = self.ego_yaw
        self.cfg.ego_L = self.ego_L
        self.cfg.ego_W = self.ego_W
        self.cfg.ego_R = self.ego_R

        self.have_goal_init = False
        self.have_cmd = False
        self.cmd_seq_id = -1
        self.cmd_index = 0
        self.cmd_dt = float(self.cfg.dt_init)
        self.cmd_dts: List[float] = []
        self.cmd_cost = 1.0e100
        self.cmd_yawrates: List[float] = []
        self.cmd_accs: List[float] = []
        self.cmd_start_stamp = rospy.Time(0)
        self.last_cmd_stamp = rospy.Time(0)
        self.last_step_stamp = rospy.Time(0)

        ego_goal_topic = str(rospy.get_param(
            "~ego_goal_topic",
            _get_nested(ego_cfg, "input_topics", "ego_goal_set", "/ego_goal_set"),
        ))
        control_topic_default = _get_nested(ego_cfg, "input_topics", "mpc_control_seq", "/mpc_control_seq")
        control_topics_value = rospy.get_param(
            "~mpc_control_seq_topics",
            _get_nested(ego_cfg, "input_topics", "mpc_control_seq_topics", None),
        )
        control_topics = _as_string_list(control_topics_value)
        control_topic_scalar = str(rospy.get_param("~mpc_control_seq_topic", control_topic_default))
        if control_topic_scalar not in control_topics:
            control_topics.append(control_topic_scalar)

        # Always include MPC_planner0 plus the nine ORDER-grid planners.
        # planner0 is the previous-selected-trajectory nominal source; planners
        # 1..9 are the acc_order x yawrate_order guidance/order sources.
        for i in range(0, 10):
            topic = "/mpc_control_seq%d" % i
            if topic not in control_topics:
                control_topics.append(topic)

        # If /setup has per-planner control publishers, subscribe to
        # all of them so the lowest-cost candidate among MPC_planner1/2/3/4 can
        # be selected even when their topics are not remapped to one topic.
        planners_cfg = raw_cfg.get("planners", {}) if isinstance(raw_cfg.get("planners"), dict) else {}
        for block in planners_cfg.values():
            if not isinstance(block, dict):
                continue
            pubs = block.get("publishers", {}) if isinstance(block.get("publishers"), dict) else {}
            topic = pubs.get("mpc_control_seq", None)
            for item in _as_string_list(topic):
                if item not in control_topics:
                    control_topics.append(item)

        self.control_subscriber_queue_size = max(10, int(rospy.get_param(
            "~control_subscriber_queue_size", ego_cfg.get("control_subscriber_queue_size", 100)
        )))

        # Track which MPC candidates were generated from a real initial guidance.
        # visibility_prm may publish placeholder MPCControlSeq messages on unused
        # /initial_guidanceN topics; as in the MPC planners, a very large cost is
        # treated as an invalid/missing guidance.
        self.initial_guidance_invalid_cost_threshold = float(rospy.get_param(
            "~initial_guidance_invalid_cost_threshold",
            ego_cfg.get("initial_guidance_invalid_cost_threshold", 1.0e99),
        ))
        self.initial_guidance_timeout_s = max(
            0.0,
            float(rospy.get_param(
                "~initial_guidance_timeout_s",
                ego_cfg.get("initial_guidance_timeout_s", max(0.25, 4.0 / max(float(self.hz), 1.0))),
            )),
        )
        self.initial_guidance_status_by_slot: Dict[int, Dict[str, Any]] = {}
        self.control_topic_to_guidance_slot: Dict[str, int] = {}
        self.control_topic_initial_guidance_enabled: Dict[str, bool] = {}
        self.initial_guidance_topic_to_slot: Dict[str, int] = {}

        # MPC_planner0 deliberately has no initial-guidance slot.
        self.control_topic_initial_guidance_enabled["/mpc_control_seq0"] = False

        # Standard 1..9 planner/guidance pairing.  Per-planner /setup settings
        # below override this default when initial guidance is disabled.
        for i in range(1, 10):
            control_topic = "/mpc_control_seq%d" % i
            self.control_topic_to_guidance_slot[control_topic] = i
            self.control_topic_initial_guidance_enabled[control_topic] = True
            self.initial_guidance_topic_to_slot["/initial_guidance%d" % i] = i

        # Also honor any per-planner topic remapping carried by /setup.
        for planner_name, block in planners_cfg.items():
            if not isinstance(block, dict):
                continue
            text = str(planner_name)
            explicit_id = block.get("planner_id", None)
            try:
                planner_id = int(explicit_id) if explicit_id is not None else None
            except Exception:
                planner_id = None
            if planner_id is None:
                match = re.search(r"MPC_planner(\d+)$", text, re.IGNORECASE)
                planner_id = int(match.group(1)) if match else -1
            # planner_id=0 is valid only for the explicitly registered MPC_planner0.
            if planner_id < 0 or (planner_id == 0 and text.lower() != "mpc_planner0"):
                continue

            pubs = block.get("publishers", {}) if isinstance(block.get("publishers"), dict) else {}
            inputs = block.get("input_topics", {}) if isinstance(block.get("input_topics"), dict) else {}
            guidance_enabled = _as_bool(block.get("use_initial_guidance", True), True)
            configured_control_topics = _as_string_list(pubs.get("mpc_control_seq", None))
            if not configured_control_topics:
                configured_control_topics = ["/mpc_control_seq%d" % int(planner_id)]
            for control_topic in configured_control_topics:
                control_topic = str(control_topic)
                self.control_topic_to_guidance_slot[control_topic] = int(planner_id)
                self.control_topic_initial_guidance_enabled[control_topic] = bool(guidance_enabled)

            # Only guidance-enabled planners own /initial_guidanceN slots.
            # MPC_planner0 uses /selected_traj as its nominal source instead.
            if guidance_enabled and planner_id > 0:
                guidance_value = inputs.get("initial_guidance", None)
                if guidance_value is None:
                    guidance_value = block.get("initial_guidance_topic", None)
                if guidance_value is None:
                    guidance_value = pubs.get("initial_guidance", None)
                guidance_topics = _as_string_list(guidance_value)
                if not guidance_topics:
                    guidance_topics = ["/initial_guidance%d" % int(planner_id)]
                for guidance_topic in guidance_topics:
                    self.initial_guidance_topic_to_slot[str(guidance_topic)] = int(planner_id)

        ros_cfg = raw_cfg.get("ros", {}) if isinstance(raw_cfg.get("ros"), dict) else {}
        selected_vertex_horizon_default = _get_nested(
            ros_cfg, "input_topics", "selected_vertex_horizon_data", "/selected_vertex_horizon_data"
        )
        self.selected_vertex_horizon_topic = str(rospy.get_param(
            "~selected_vertex_horizon_data_topic", selected_vertex_horizon_default
        ))

        ego_state_topic = str(rospy.get_param(
            "~ego_state_topic",
            _get_nested(ego_cfg, "publishers", "ego_state", "/ego_state"),
        ))
        rospy.Subscriber(ego_goal_topic, Ego_goal_setting, self.goal_callback, queue_size=1)
        rospy.Subscriber(
            self.selected_vertex_horizon_topic, Float64MultiArray,
            self.selected_vertex_horizon_callback, queue_size=1,
        )
        self.scenario_reset_topic = str(rospy.get_param("~scenario_reset_topic", "/scenario_reset"))
        self.scenario_start_topic = str(rospy.get_param("~scenario_start_topic", "/scenario_start"))
        self.scenario_ego_ready_topic = str(rospy.get_param("~scenario_ego_ready_topic", "/scenario_ego_ready"))
        rospy.Subscriber(self.scenario_reset_topic, Int32, self.scenario_reset_callback, queue_size=1)
        rospy.Subscriber(self.scenario_start_topic, Int32, self.scenario_start_callback, queue_size=1)
        self.control_subscribers = []
        self.control_subscribers_by_topic: Dict[str, Any] = {}
        for topic in control_topics:
            sub = rospy.Subscriber(
                topic, MPCControlSeq, self.control_callback,
                callback_args=topic, queue_size=self.control_subscriber_queue_size,
            )
            self.control_subscribers.append(sub)
            self.control_subscribers_by_topic[str(topic)] = sub

        # Subscribe to initial-guidance topics for diagnostics/compatibility only.
        # Candidate selection itself is based only on original-OCP feasibility and
        # J_global; ego_mover does not modify or republish the guidance.
        self.initial_guidance_subscribers = []
        for guidance_topic, slot in sorted(self.initial_guidance_topic_to_slot.items()):
            self.initial_guidance_subscribers.append(
                rospy.Subscriber(
                    guidance_topic, MPCControlSeq, self.initial_guidance_callback,
                    callback_args=(int(slot), str(guidance_topic)), queue_size=1,
                )
            )

        self.pub_ego = rospy.Publisher(ego_state_topic, Ego_state, queue_size=1)
        self.pub_ego_map = (rospy.Publisher(self.map_state_topic, Ego_state, queue_size=1)
                            if self.publish_map_state else None)
        rospy.loginfo("[ego_mover] /ego_state: %s MPC pose from %s; %s: map mirror",
                      self.frame_id, self.pose_topic, self.map_state_topic)
        self.pub_cmd_vel = None
        if self.use_real_robot:
            self.pub_cmd_vel = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
            rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
            if self.use_tracked_pose:
                rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=1)
            rospy.on_shutdown(self.publish_stop_command)
            rospy.loginfo("[ego_mover] REAL mode: pose=%s cmd=%s (velocity from /cmd_vel.linear.x)",
                          self.pose_topic, self.cmd_vel_topic)

        self.selected_sr_topic = str(rospy.get_param("~selected_sr_topic", "/selected_sr_seq"))
        self.pub_selected_sr = rospy.Publisher(
            self.selected_sr_topic, Float64MultiArray, queue_size=1, latch=True
        )
        self.selected_traj_topic = str(rospy.get_param("~selected_traj_topic", "/selected_traj"))
        self.pub_selected_traj = rospy.Publisher(
            self.selected_traj_topic, Float64MultiArray, queue_size=1, latch=True
        )
        self.pub_scenario_ego_ready = rospy.Publisher(self.scenario_ego_ready_topic, Int32, queue_size=1, latch=True)
        self.setup_subscriber = rospy.Subscriber(
            self.setup_topic, Setup, self.setup_callback, queue_size=1
        )


        # rospy.loginfo(
        #     "[ego_mover] /setup ego initialized at x=%.3f y=%.3f z=%.3f v=%.3f yaw=%.3f, update_hz=%.1f, goal=(%.3f, %.3f), bounds=[%.3f %.3f]x[%.3f %.3f]",
        #     self.ego_x, self.ego_y, self.ego_z, self.ego_v, self.ego_yaw, self.hz,
        #     self.goal_x, self.goal_y, self.x_min, self.x_max, self.y_min, self.y_max,
        # )
        # rospy.loginfo(
        #     "[ego_mover] candidate receive window=0.8*(1/Hz)=%.4f s at Hz=%.3f; start-distance valid threshold=%.3f m",
        #     float(self.trajectory_receive_window_s), float(self.hz),
        #     float(self.traj_start_position_threshold),
        # )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped) -> None:
        """Use Cartographer tracked_pose as the authoritative map pose."""
        source_frame = str(msg.header.frame_id).lstrip("/")
        if source_frame != self.world_frame_id:
            rospy.logwarn_throttle(2.0, "[ego_mover] tracked_pose frame=%s; expected %s",
                                   source_frame, self.world_frame_id)
            return
        p = msg.pose.position
        q = msg.pose.orientation
        tracked_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        x, y, z, yaw = float(p.x), float(p.y), float(p.z), float(tracked_yaw)
        if not all(math.isfinite(v) for v in (x, y, z, yaw)):
            return
        if self.tracking_frame != self.robot_frame:
            try:
                # T_map_robot = T_map_tracking * T_tracking_robot
                t = self.world_tf_buffer.lookup_transform(
                    self.tracking_frame, self.robot_frame,
                    rospy.Time(0), rospy.Duration(0.05))
                tq = t.transform.rotation
                dyaw = euler_from_quaternion([tq.x, tq.y, tq.z, tq.w])[2]
                dx, dy, dz = (float(t.transform.translation.x),
                              float(t.transform.translation.y),
                              float(t.transform.translation.z))
                c, si = math.cos(yaw), math.sin(yaw)
                x, y, z, yaw = (x + c*dx - si*dy, y + si*dx + c*dy,
                                z + dz, wrap_to_pi(yaw + dyaw))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                rospy.logwarn_throttle(2.0, "[ego_mover] tracking -> robot TF unavailable: %s", str(exc))
                return
        with self.state_lock:
            self.ego_x, self.ego_y, self.ego_z = x, y, z
            self.ego_yaw = wrap_to_pi(yaw)
            self.have_tracked_pose = True
            if self.latest_cmd_vel is not None:
                self.ego_v = self.latest_cmd_vel

    def cmd_vel_callback(self, msg: Twist) -> None:
        speed = float(msg.linear.x)
        if not math.isfinite(speed):
            return
        with self.state_lock:
            self.latest_cmd_vel = speed
            self.ego_v = speed

    def publish_stop_command(self) -> None:
        # Emergency/hold stop is immediate.
        with self.state_lock:
            self.commanded_v = 0.0
            self.latest_cmd_vel = 0.0
            self.ego_v = 0.0
        if self.pub_cmd_vel is not None:
            self.pub_cmd_vel.publish(Twist())

    def publish_robot_command(self, now: rospy.Time, interval_s: float,
                              yawrate: float = None, acc: float = None) -> None:
        if self.pub_cmd_vel is None:
            return
        if (self.stopped or (self.scenario_managed_mode and not self.scenario_run_enabled)
                or not self.have_cmd
                or (now - self.last_cmd_stamp).to_sec() > self.command_timeout_s):
            self.publish_stop_command()
            return
        if yawrate is None or acc is None:
            yawrate, acc, _, _ = self.select_command(now)
        if not math.isfinite(float(yawrate)):
            yawrate = 0.0
        if not math.isfinite(float(acc)):
            acc = 0.0
        with self.state_lock:
            if not hasattr(self, "commanded_v"):
                self.commanded_v = float(self.ego_v)
        v_min = float(getattr(self.cfg, "v_min", 0.0))
        v_max = float(getattr(self.cfg, "v_max", getattr(self.cfg, "V_max", 0.6)))
        yawrate_min = float(getattr(self.cfg, "yawrate_min", -1.0))
        yawrate_max = float(getattr(self.cfg, "yawrate_max", 1.0))
        interval_s = min(max(float(interval_s), 0.0), 0.1)
        cmd = Twist()
        with self.state_lock:
            self.commanded_v = max(v_min, min(v_max, self.commanded_v + float(acc) * interval_s))
            cmd.linear.x = float(self.commanded_v)
            self.latest_cmd_vel = float(cmd.linear.x)
            self.ego_v = float(cmd.linear.x)
        cmd.angular.z = 1* max(yawrate_min, min(yawrate_max, float(yawrate)))
        self.pub_cmd_vel.publish(cmd)

    def setup_callback(self, msg: Setup) -> None:
        """Keep the latest setter-owned /setup payload."""
        try:
            self.raw_setup = setup_msg_to_raw(msg)
        except Exception as exc:
            rospy.logerr_throttle(1.0, "[%s] failed to decode /setup: %s", rospy.get_name(), exc)

    def clear_selected_traj(self) -> None:
        """Clear the latched previous selected trajectory for MPC_planner0."""
        pub = getattr(self, "pub_selected_traj", None)
        if pub is None:
            return
        msg = Float64MultiArray()
        msg.data = []
        pub.publish(msg)

    def scenario_reset_callback(self, msg: Int32) -> None:
        """Reset ego state and planner-command history for one benchmark seed."""
        if self.use_real_robot:
            rospy.logwarn("[ego_mover] ignoring simulated state reset on real robot")
            return
        seed = int(msg.data)
        self.scenario_seed = seed
        self.scenario_managed_mode = True
        self.scenario_run_enabled = False

        self.ego_x = float(self.scenario_start_x)
        self.ego_y = float(self.scenario_start_y)
        self.ego_z = float(self.scenario_start_z)
        self.ego_v = float(self.scenario_start_v)
        self.ego_yaw = wrap_to_pi(float(self.scenario_start_yaw))

        # Do not execute a command selected in the preceding scenario.
        self.have_cmd = False
        self.cmd_seq_id = -1
        self.cmd_index = 0
        self.cmd_yawrates = []
        self.cmd_accs = []
        self.cmd_dts = []
        self.cmd_start_stamp = rospy.Time(0)
        self.last_cmd_stamp = rospy.Time(0)
        self.last_step_stamp = rospy.Time.now()
        self.pending_cmd = None
        self.pending_candidates = []
        self.pending_sources_seen.clear()
        self.pending_expected_source_counts.clear()
        self.pending_source_receive_counts.clear()
        self.pending_messages_received = 0
        self.pending_candidates_rejected = 0
        self.pending_callbacks_inflight = 0
        self._cancel_pending_window_timer()
        self.pending_seq_id = None
        self.pending_first_recv_stamp = rospy.Time(0)
        self.pending_last_recv_stamp = rospy.Time(0)
        self.initial_guidance_status_by_slot.clear()
        self.previous_selected_reference_states = None
        self.clear_selected_traj()
        self.planning_round_start_xy.clear()
        self.planning_round_start_state.clear()
        self.stopped = False

        # Scenario reset does not modify the goal; setter remains authoritative.
        self.publish_ego_state()
        self.pub_scenario_ego_ready.publish(Int32(data=seed))
        rospy.loginfo(
            "[ego_mover] scenario reset seed=%d -> start=(%.3f, %.3f), goal=(%.3f, %.3f)",
            seed, self.ego_x, self.ego_y, self.goal_x, self.goal_y,
        )

    def scenario_start_callback(self, msg: Int32) -> None:
        seed = int(msg.data)
        if not self.scenario_managed_mode or seed != int(self.scenario_seed):
            return

        # Drop any candidate produced while the ego was held and obstacles were
        # still resetting.  Only post-start planner outputs may move the ego.
        self.have_cmd = False
        self.cmd_seq_id = -1
        self.cmd_index = 0
        self.cmd_yawrates = []
        self.cmd_accs = []
        self.cmd_dts = []
        self.cmd_start_stamp = rospy.Time(0)
        self.last_cmd_stamp = rospy.Time(0)
        self.pending_cmd = None
        self.pending_candidates = []
        self.pending_sources_seen.clear()
        self._cancel_pending_window_timer()
        self.pending_seq_id = None
        self.pending_first_recv_stamp = rospy.Time(0)
        self.pending_last_recv_stamp = rospy.Time(0)
        self.initial_guidance_status_by_slot.clear()
        self.previous_selected_reference_states = None
        self.clear_selected_traj()
        self.planning_round_start_xy.clear()
        self.planning_round_start_state.clear()
        self.last_step_stamp = rospy.Time.now()
        self.scenario_run_enabled = True
        # Candidates produced during the reset/hold phase were just discarded.
        # Advance once so all planners are forced to consume a fresh common round.
        self.publish_ego_state()
        rospy.loginfo(
            "[ego_mover] scenario start seed=%d planning_round=%d",
            seed, int(self.planning_round_id),
        )

    def selected_vertex_horizon_callback(self, msg: Float64MultiArray) -> None:
        """Cache the selector-owned H x K predicted/inflated feature tensor.

        Payload fields are [x, y, nx, ny, type, group_id], matching the MPC
        planner input.  This lets trajectory selection recompute N_k(0) at every
        candidate stage without changing any MPC planner or ROS message type.
        """
        dims = list(getattr(getattr(msg, "layout", None), "dim", []))
        if len(dims) < 3:
            rospy.logwarn_throttle(1.0, "[ego_mover] malformed horizon vertex layout")
            return

        H_msg = max(0, int(dims[0].size))
        K_msg = max(0, int(dims[1].size))
        F_msg = max(0, int(dims[2].size))
        if H_msg <= 0 or K_msg <= 0 or F_msg < 6:
            return

        raw = np.asarray(msg.data, dtype=float).reshape(-1)
        needed = H_msg * K_msg * F_msg
        if raw.size < needed:
            rospy.logwarn_throttle(
                1.0,
                "[ego_mover] truncated horizon vertex payload: got=%d need=%d",
                int(raw.size), int(needed),
            )
            return

        tensor = raw[:needed].reshape(H_msg, K_msg, F_msg)
        K = min(K_msg, int(getattr(self.cfg, "max_vertex_num", K_msg)))
        used = tensor[:, :K, :]

        self.latest_vertex_set_horizon = np.ascontiguousarray(used[:, :, 0:2], dtype=float)
        self.latest_vertex_normals_horizon = np.ascontiguousarray(used[:, :, 2:4], dtype=float)
        self.latest_vertex_types = np.rint(used[0, :, 4]).astype(np.int32)
        self.latest_selected_obs_indices = np.rint(used[0, :, 5]).astype(np.int32)

    def goal_callback(self, msg: Ego_goal_setting) -> None:
        """Update only the runtime goal; /setup owns bounds and ego size."""
        goal_changed = False
        if hasattr(msg, "x_end"):
            new_goal_x = float(msg.x_end)
            goal_changed = goal_changed or (abs(new_goal_x - float(self.goal_x)) > 1.0e-9)
            self.goal_x = new_goal_x
        if hasattr(msg, "y_end"):
            new_goal_y = float(msg.y_end)
            goal_changed = goal_changed or (abs(new_goal_y - float(self.goal_y)) > 1.0e-9)
            self.goal_y = new_goal_y
        if hasattr(msg, "z_end"):
            self.goal_z = float(msg.z_end)

        # Deliberately ignore geometry/bounds carried by legacy goal messages.
        # Those fields are initialized from /setup, not /ego_goal_set.

        if goal_changed:
            # A setter-side goal change invalidates the previous trajectory reference.
            self.previous_selected_reference_states = None
            self.clear_selected_traj()

        should_reset = self.reset_on_every_goal or (self.reset_from_goal_once and not self.have_goal_init)
        if not should_reset:
            return
        self.have_goal_init = True

    def _extract_time_intervals(self, msg: MPCControlSeq, n: int, default_dt: float) -> List[float]:
        intervals: List[float] = []

        # Support both the current scalar dt message and possible future
        # per-stage interval fields without requiring another ego_mover change.
        for field_name in ("time_intervals", "time_interval", "dts", "dt_seq", "dt_list"):
            if not hasattr(msg, field_name):
                continue
            value = getattr(msg, field_name)
            if isinstance(value, (list, tuple)):
                intervals = [float(v) for v in value]
            else:
                try:
                    intervals = [float(value)]
                except Exception:
                    intervals = []
            if len(intervals) > 0:
                break

        if len(intervals) == 0:
            intervals = [float(default_dt)] * int(n)
        elif len(intervals) == 1 and n > 1:
            intervals = [float(intervals[0])] * int(n)
        elif len(intervals) < n:
            intervals = list(intervals) + [float(intervals[-1])] * (int(n) - len(intervals))

        cleaned: List[float] = []
        fallback = float(default_dt) if np.isfinite(default_dt) and default_dt > 1.0e-9 else float(self.cfg.dt_init)
        for value in intervals[:n]:
            dt_i = float(value)
            if not np.isfinite(dt_i) or dt_i <= 1.0e-9:
                dt_i = fallback
            cleaned.append(dt_i)
        return cleaned

    def _rollout_control_sequence_relative(self, yawrates, accs, dts, start_state=None):
        n = min(len(yawrates), len(accs), len(dts))
        if n <= 0:
            return np.zeros((0, 4), dtype=float)

        if start_state is None:
            x0 = float(self.ego_x)
            y0 = float(self.ego_y)
            v0 = float(self.ego_v)
            yaw0 = float(self.ego_yaw)
        else:
            start = np.asarray(start_state, dtype=float).reshape(-1)
            if start.size < 4 or not np.all(np.isfinite(start[:4])):
                raise ValueError("start_state must contain finite [x, y, v, yaw]")
            x0, y0, v0, yaw0 = [float(v) for v in start[:4]]
            yaw0 = wrap_to_pi(yaw0)
        state = np.array([x0, y0, v0, yaw0], dtype=float)
        rows = [[0.0, 0.0, 0.0, 0.0]]

        for k in range(n):
            dt_i = float(dts[k])
            if not np.isfinite(dt_i) or dt_i <= 1.0e-9:
                dt_i = float(self.cfg.dt_init)
            yawrate = float(yawrates[k]) if np.isfinite(float(yawrates[k])) else 0.0
            acc = float(accs[k]) if np.isfinite(float(accs[k])) else 0.0

            def f(sv):
                return np.array([
                    float(sv[2]) * math.cos(float(sv[3])),
                    float(sv[2]) * math.sin(float(sv[3])),
                    acc,
                    yawrate,
                ], dtype=float)

            k1 = f(state)
            k2 = f(state + 0.5 * dt_i * k1)
            k3 = f(state + 0.5 * dt_i * k2)
            k4 = f(state + dt_i * k3)
            state = state + (dt_i / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state[3] = wrap_to_pi(float(state[3]))
            rows.append([
                float(state[0] - x0),
                float(state[1] - y0),
                float(state[2] - v0),
                float(wrap_to_pi(float(state[3]) - yaw0)),
            ])

        return np.asarray(rows, dtype=float)

    def _rollout_control_sequence_absolute(self, yawrates, accs, dts, start_state=None):
        if start_state is None:
            start = np.array([self.ego_x, self.ego_y, self.ego_v, self.ego_yaw], dtype=float)
        else:
            start = np.asarray(start_state, dtype=float).reshape(-1)
            if start.size < 4 or not np.all(np.isfinite(start[:4])):
                raise ValueError("start_state must contain finite [x, y, v, yaw]")
            start = start[:4].astype(float, copy=True)
        start[3] = wrap_to_pi(float(start[3]))

        rel_states = self._rollout_control_sequence_relative(
            yawrates, accs, dts, start_state=start
        )
        if rel_states.shape[0] == 0:
            return np.zeros((0, 4), dtype=float)

        abs_states = rel_states.copy()
        abs_states[:, 0] += float(start[0])
        abs_states[:, 1] += float(start[1])
        abs_states[:, 2] += float(start[2])
        abs_states[:, 3] = np.asarray([
            wrap_to_pi(float(start[3]) + float(dyaw))
            for dyaw in rel_states[:, 3]
        ], dtype=float)
        return abs_states

    def candidate_similarity_cost(
        self, yawrates, accs, dts, start_state=None, candidate_states=None
    ):
        """Return the paper Q_sim cost against the shifted previous trajectory.

        The current candidate contributes x_0,...,x_{H-1}.  After a candidate is
        selected, x_1,...,x_H from that trajectory are stored as
        tilde{x}^{prev}_0,...,tilde{x}^{prev}_{H-1} for the next planning round.
        Prefer the exact optimized MPC X when available so the ranking trajectory
        is identical to the one rendered in /mpc_pred_trajs and /selected_traj.
        """
        states = np.zeros((0, 4), dtype=float)
        if candidate_states is not None:
            try:
                exact = np.asarray(candidate_states, dtype=float).reshape(-1, 4)
                if exact.shape[0] > 0 and np.all(np.isfinite(exact)):
                    states = exact
            except Exception:
                states = np.zeros((0, 4), dtype=float)

        if states.shape[0] == 0:
            states = self._rollout_control_sequence_absolute(
                yawrates, accs, dts, start_state=start_state
            )
        prev = self.previous_selected_reference_states
        if (states.shape[0] <= 1 or prev is None
                or float(self.traj_similarity_weight) <= 0.0):
            H0 = max(min(len(yawrates), len(accs), len(dts), max(states.shape[0] - 1, 0)), 0)
            return 0.0, np.zeros(H0, dtype=float), "no-previous-trajectory"

        prev = np.asarray(prev, dtype=float).reshape(-1, 4)
        H = min(
            len(yawrates), len(accs), len(dts),
            states.shape[0] - 1, prev.shape[0],
        )
        if H <= 0:
            return 0.0, np.zeros(0, dtype=float), "empty-similarity-horizon"

        # Paper index k=0,...,H-1: do not use the candidate terminal x_H here.
        cur = np.asarray(states[:H, :], dtype=float)
        ref = np.asarray(prev[:H, :], dtype=float)
        dx = cur[:, 0] - ref[:, 0]
        dy = cur[:, 1] - ref[:, 1]
        dv = cur[:, 2] - ref[:, 2]
        dyaw_raw = cur[:, 3] - ref[:, 3]
        dyaw = np.arctan2(np.sin(dyaw_raw), np.cos(dyaw_raw))

        per_k = float(self.traj_similarity_weight) * (
            float(self.traj_similarity_pos_weight) * (dx * dx + dy * dy)
            + float(self.traj_similarity_v_weight) * (dv * dv)
            + float(self.traj_similarity_yaw_weight) * (dyaw * dyaw)
        )
        if not np.all(np.isfinite(per_k)):
            return 1.0e100, np.full(H, 1.0e100, dtype=float), "nonfinite-similarity"
        return float(np.sum(per_k)), per_k, "shifted-previous-selected-trajectory"

    def _update_previous_selected_reference(self, candidate) -> None:
        """Store x_1,...,x_H of the exact selected MPC candidate for next-round Q_sim."""
        yawrates = list(candidate.get("yawrates", []))
        accs = list(candidate.get("accs", []))
        dts = list(candidate.get("dts", []))

        # Prefer the exact optimized state trajectory carried by MPCControlSeq.
        # This is the same X trajectory visualized in /mpc_pred_trajs.
        optimized_states = candidate.get("optimized_states")
        if optimized_states is not None:
            try:
                states = np.asarray(optimized_states, dtype=float).reshape(-1, 4)
                if states.shape[0] == 0 or not np.all(np.isfinite(states)):
                    states = np.zeros((0, 4), dtype=float)
            except Exception:
                states = np.zeros((0, 4), dtype=float)
        else:
            states = np.zeros((0, 4), dtype=float)

        # Legacy fallback for old MPCControlSeq messages without xs/ys/vs/yaws.
        if states.shape[0] == 0:
            states = self._rollout_control_sequence_absolute(
                yawrates, accs, dts, start_state=candidate.get("start_state")
            )
        H = min(len(yawrates), len(accs), len(dts), max(states.shape[0] - 1, 0))
        if H <= 0:
            self.previous_selected_reference_states = None
            return
        self.previous_selected_reference_states = np.asarray(
            states[1:H + 1, :], dtype=float
        ).copy()

    def candidate_global_cost_components(self, yawrates, accs, dts, start_state=None):
        """Return legacy diagnostics only; never alter paper-objective ranking.

        Candidate selection uses the planner-reported paper objective plus the
        Q_sim previous-trajectory term.  Boundary slack is not part of that
        objective, so this helper reports its geometric value for diagnostics
        but assigns it zero selection cost.
        """
        states = self._rollout_control_sequence_absolute(
            yawrates, accs, dts, start_state=start_state
        )
        if states.shape[0] == 0:
            if start_state is not None:
                start = np.asarray(start_state, dtype=float).reshape(-1)
                sx = float(start[0]) if start.size > 0 else float(self.ego_x)
                sy = float(start[1]) if start.size > 1 else float(self.ego_y)
            else:
                sx, sy = float(self.ego_x), float(self.ego_y)
            return 1.0e100, 1.0e100, 1.0e100, 1.0e100, sx, sy

        terminal_x = float(states[-1, 0])
        terminal_y = float(states[-1, 1])
        dx = terminal_x - float(self.goal_x)
        dy = terminal_y - float(self.goal_y)
        terminal_cost = float(dx * dx + dy * dy)

        # The MPC comments describe a shared soft boundary slack.  Recompute the
        # minimum shared slack required by this candidate rollout instead of
        # trusting msg.cost from a planner.
        x = states[:, 0]
        y = states[:, 1]
        violations = np.maximum.reduce([
            np.maximum(float(self.x_min) - x, 0.0),
            np.maximum(x - float(self.x_max), 0.0),
            np.maximum(float(self.y_min) - y, 0.0),
            np.maximum(y - float(self.y_max), 0.0),
        ])
        shared_slack = float(np.max(violations)) if violations.size > 0 else 0.0
        # Boundary slack is diagnostic only and is intentionally excluded
        # from the paper candidate-selection objective.
        slack_cost = 0.0
        base_cost = terminal_cost
        return base_cost, terminal_cost, slack_cost, shared_slack, terminal_x, terminal_y

    @staticmethod
    def _msg_first_float(msg: MPCControlSeq, names, default=None):
        for name in names:
            if not hasattr(msg, name):
                continue
            try:
                value = float(getattr(msg, name))
            except Exception:
                continue
            if np.isfinite(value):
                return value
        return default

    @staticmethod
    def _msg_first_bool(msg: MPCControlSeq, names):
        for name in names:
            if hasattr(msg, name):
                return _as_bool(getattr(msg, name), False)
        return None

    def _candidate_start_state_from_msg(self, msg: MPCControlSeq, seq_id: int):
        """Return the exact MPC initial state [x,y,v,yaw] for this candidate.

        New MPC planners publish start_ego_* in every MPCControlSeq.  For an older
        message definition, fall back to the state first published by ego_mover for
        the same planning-round seq_id.  Never silently substitute the *current*
        ego pose, because that changes the trajectory that was feasibility-checked.
        """
        names = ("start_ego_x", "start_ego_y", "start_ego_v", "start_ego_yaw")
        vals = []
        complete = True
        for name in names:
            if not hasattr(msg, name):
                complete = False
                break
            try:
                value = float(getattr(msg, name))
            except Exception:
                complete = False
                break
            if not np.isfinite(value):
                complete = False
                break
            vals.append(value)
        if complete and len(vals) == 4:
            vals[3] = wrap_to_pi(vals[3])
            return np.asarray(vals, dtype=float), "msg-start_ego"

        fallback = (float(self.ego_x), float(self.ego_y), float(self.ego_v), float(self.ego_yaw))
        if fallback is not None:
            start = np.asarray(fallback, dtype=float).reshape(-1)
            if start.size >= 4 and np.all(np.isfinite(start[:4])):
                start = start[:4].astype(float, copy=True)
                start[3] = wrap_to_pi(float(start[3]))
                rospy.logwarn_throttle(
                    2.0,
                    "[ego_mover] MPCControlSeq has no complete start_ego_*; "
                    "using stored planning-round start state for seq=%d",
                    int(seq_id),
                )
                return start, "stored-round-start"

        rospy.logerr_throttle(
            1.0,
            "[ego_mover] cannot recover MPC start state for seq=%d; candidate rejected",
            int(seq_id),
        )
        return None, "missing-start-state"

    def candidate_zero_slack_feasibility(self, msg: MPCControlSeq):
        """Return paper-selection feasibility with only regulation slack s^r = 0.

        The avoidance slack s_k (= s^a_k) remains active.  New planners publish
        ``zero_slack_feasible`` with this meaning.  Older aliases are accepted
        only as compatibility fallbacks.
        """
        explicit = self._msg_first_bool(
            msg,
            (
                "zero_slack_feasible",
                "nonlinear_zero_slack_feasible",
                "ocp_feasible",
                "paper_feasible",
                "selection_eligible",
            ),
        )
        if explicit is not None:
            return bool(explicit), "planner-zero-slack-flag"

        tol = float(self.selection_feasibility_tol)
        g_a = self._msg_first_float(
            msg,
            (
                "zero_slack_avoidance_residual_max",
                "avoidance_residual_max",
                "collision_residual_max",
            ),
            None,
        )
        g_r = self._msg_first_float(
            msg,
            (
                "zero_slack_normal_alignment_residual_max",
                "normal_alignment_residual_max",
                "regulation_residual_max",
            ),
            None,
        )
        if g_a is not None:
            feasible = bool(float(g_a) <= tol)
            impact_enabled = _as_bool(
                getattr(self.cfg, "enable_impact_normal_constraint", True), True
            )
            if impact_enabled:
                if g_r is None:
                    return False, "missing-zero-slack-regulation-residual"
                feasible = bool(feasible and float(g_r) <= tol)
            return feasible, "planner-zero-slack-residuals"

        rospy.logwarn_throttle(
            2.0,
            "[ego_mover] MPCControlSeq has no zero-slack nonlinear feasibility metadata; candidate is treated as infeasible",
        )
        return False, "missing-zero-slack-metadata"

    def candidate_phi_min(self, msg: MPCControlSeq) -> float:
        """Return min_k(Phi_k) for diagnostics/backward-compatible logging."""
        value = self._msg_first_float(
            msg,
            (
                "phi_min",
                "min_phi",
                "minimum_phi",
                # Compatibility aliases populated with exactly the same Phi_min
                # value by the modified MPC planners.
                "min_clearance",
                "d_min",
                "minimum_clearance",
                "clearance_min",
            ),
            None,
        )
        return float(value) if value is not None else float("-inf")

    def candidate_impact_alignment_sum(
        self, yawrates, accs, dts
    ) -> Tuple[float, np.ndarray, str]:
        """Return sum_k N_k(0)^T n(phi_c) for one candidate trajectory.

        Diagnostic helper only.  The planner now evaluates and includes
        rho_c * sum_k N_k(0)^T n(phi_c) directly in planner_global_cost, so this
        helper is NOT used by candidate ranking and rho_c is never double-counted.

        A stage with no valid obstacle field has no defined N_k(0); its contribution
        is set to zero rather than fabricating an impact direction.
        """
        if (self.latest_vertex_set_horizon is None
                or self.latest_vertex_normals_horizon is None
                or self.latest_vertex_types is None
                or self.latest_selected_obs_indices is None):
            rospy.logwarn_throttle(
                1.0,
                "[ego_mover] horizon feature tensor unavailable; all-k alignment cost set to zero",
            )
            return 0.0, np.zeros(0, dtype=float), "horizon-features-unavailable"

        states = self._rollout_control_sequence_absolute(yawrates, accs, dts)
        if states.shape[0] <= 1:
            return 0.0, np.zeros(0, dtype=float), "empty-rollout"

        Vh = np.asarray(self.latest_vertex_set_horizon, dtype=float)
        Nh = np.asarray(self.latest_vertex_normals_horizon, dtype=float)
        types = np.asarray(self.latest_vertex_types, dtype=np.int32).reshape(-1)
        groups = np.asarray(self.latest_selected_obs_indices, dtype=np.int32).reshape(-1)

        H = min(
            len(yawrates), len(accs), len(dts), states.shape[0] - 1,
            Vh.shape[0], Nh.shape[0], int(getattr(self.cfg, "N", states.shape[0] - 1)),
        )
        if H <= 0:
            return 0.0, np.zeros(0, dtype=float), "empty-horizon"

        K = min(Vh.shape[1], Nh.shape[1], types.size, groups.size)
        if K <= 0:
            return 0.0, np.zeros(H, dtype=float), "empty-feature-set"

        X_stage = np.ascontiguousarray(states[1:H + 1, :], dtype=float)
        vertex_set = np.ascontiguousarray(Vh[:H, :K, :], dtype=float)
        vertex_normals = np.ascontiguousarray(Nh[:H, :K, :], dtype=float)
        vertex_types = np.ascontiguousarray(types[:K], dtype=np.int32)
        feature_obs_indices = np.ascontiguousarray(groups[:K], dtype=np.int32)
        h_steps = np.arange(1, H + 1, dtype=float)
        dt_vals = np.asarray(dts[:H], dtype=float)

        try:
            dual = build_external_dual_collision_linearization_sequence(
                X_stage, vertex_set, vertex_normals, vertex_types,
                feature_obs_indices, h_steps, dt_vals, self.cfg,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[ego_mover] all-k impact alignment evaluation failed: %s",
                str(exc),
            )
            return 0.0, np.zeros(H, dtype=float), "all-k-evaluation-failed"

        dual = np.asarray(dual, dtype=float).reshape(H, 14)
        ds = float(collision_safe_distance(self.cfg))
        lam = float(collision_smoothing_lambda(self.cfg))
        Ns = float(regulation_alignment_Ns(self.cfg))
        hub = float(getattr(self.cfg, "h_upper_bound", 1.0e15))

        # The dual kernel stores the active-branch residual
        #   g_k = lam*log(2-exp(Phi_k/lam)) - (N_s-alignment_k).
        # With delta=Phi/lam-log(2)<0, evaluate the log term through expm1
        # and recover alignment without re-evaluating the field.  Rows with
        # Phi/lam >= log(2) are automatically satisfied and stored disabled.
        phi = ds - dual[:, 0]
        log2 = math.log(2.0)
        u = phi / lam
        active = (u < log2) & (dual[:, 7] > (-0.5 * hub))
        rhs = np.zeros_like(phi)
        if np.any(active):
            delta = u[active] - log2
            qgap = -np.expm1(delta)
            good = qgap > 0.0
            rhs_active = np.full(delta.shape, np.nan, dtype=float)
            rhs_active[good] = lam * (log2 + np.log(qgap[good]))
            rhs[active] = rhs_active
        alignment = Ns - rhs + dual[:, 7]

        valid = (
            active
            & np.isfinite(alignment)
            & np.isfinite(phi)
            & (dual[:, 0] > (-0.5 * hub))
        )
        alignment = np.where(valid, np.clip(alignment, -1.0, 1.0), 0.0)
        alignment_sum = float(np.sum(alignment))
        return alignment_sum, alignment, "all-k-cpp-field"

    def _cancel_pending_window_timer(self) -> None:
        timer = getattr(self, "pending_window_timer", None)
        self.pending_window_timer = None
        if timer is None:
            return
        try:
            timer.shutdown()
        except Exception:
            pass

    def _connected_control_source_counts(self) -> Dict[str, int]:
        """Return {topic: connected publisher count} for MPCControlSeq inputs."""
        counts: Dict[str, int] = {}
        for topic, sub in getattr(self, "control_subscribers_by_topic", {}).items():
            try:
                n = int(sub.get_num_connections())
            except Exception:
                n = 0
            if n > 0:
                counts[str(topic)] = n
        return counts

    def _refresh_pending_expected_sources(self, current_source: str = "") -> None:
        """Grow the expected-source snapshot; never shrink it within a round."""
        connected = self._connected_control_source_counts()
        for topic, count in connected.items():
            self.pending_expected_source_counts[topic] = max(
                int(self.pending_expected_source_counts.get(topic, 0)), int(count)
            )
        current_source = str(current_source).strip()
        if current_source:
            # A callback proves at least one publisher exists even if ROS master
            # connection accounting has not settled yet.
            self.pending_expected_source_counts[current_source] = max(
                1, int(self.pending_expected_source_counts.get(current_source, 0))
            )

    def _expected_response_totals(self) -> Tuple[int, int, List[str]]:
        expected_total = int(sum(max(0, int(v)) for v in self.pending_expected_source_counts.values()))
        received_total = 0
        missing = []
        for topic, expected in sorted(self.pending_expected_source_counts.items()):
            expected = max(0, int(expected))
            received = max(0, int(self.pending_source_receive_counts.get(topic, 0)))
            received_total += min(received, expected)
            if received < expected:
                missing.append("%s:%d/%d" % (topic, received, expected))
        return expected_total, received_total, missing

    def _start_pending_window_timer(self, seq_id: int) -> None:
        """Start one-shot failure timeout; normal closure is publisher-barrier driven."""
        self._cancel_pending_window_timer()
        duration = max(float(self.trajectory_receive_window_s), 1.0e-6)
        expected_seq = int(seq_id)
        self.pending_window_timer = rospy.Timer(
            rospy.Duration(duration),
            lambda event, sid=expected_seq: self._candidate_window_timer_callback(event, sid),
            oneshot=True,
        )

    def _candidate_window_timer_callback(self, _event, expected_seq: int) -> None:
        with self.pending_lock:
            if self.pending_seq_id is None or int(self.pending_seq_id) != int(expected_seq):
                return
            self.pending_window_timer = None
            # Never close while a callback for this round is still computing its
            # candidate metadata.  Re-arm a short watchdog instead.
            if int(self.pending_callbacks_inflight) > 0:
                self.pending_window_timer = rospy.Timer(
                    rospy.Duration(0.005),
                    lambda event, sid=int(expected_seq): self._candidate_window_timer_callback(event, sid),
                    oneshot=True,
                )
                return
        self._maybe_activate_completed_round(force=True, force_reason="planner-result-timeout")

    def _reset_pending_round_state(self) -> None:
        self.pending_cmd = None
        self.pending_candidates = []
        self.pending_sources_seen.clear()
        self.pending_expected_source_counts.clear()
        self.pending_source_receive_counts.clear()
        self.pending_messages_received = 0
        self.pending_candidates_rejected = 0
        self.pending_callbacks_inflight = 0
        self._cancel_pending_window_timer()
        self.pending_seq_id = None
        self.pending_first_recv_stamp = rospy.Time(0)
        self.pending_last_recv_stamp = rospy.Time(0)

    def _finish_pending_round_without_candidate(self) -> None:
        """Close a completed seq_id that produced no usable trajectory candidate."""
        if self.pending_seq_id is None:
            return
        finished_seq = int(self.pending_seq_id)
        expected_total, received_total, missing = self._expected_response_totals()
        print("candidate trajectory_count: 0")
        print(
            "planner response_count: %d/%d, rejected=%d, missing=%s"
            % (received_total, expected_total, int(self.pending_candidates_rejected),
               ",".join(missing) if missing else "none")
        )
        self._reset_pending_round_state()
        # Keep the previously active command, but open a fresh common planning
        # round so planners cannot deadlock when no valid trajectory arrives.
        # No global planning-round sequence in asynchronous mode.
        rospy.logwarn_throttle(
            1.0,
            "[ego_mover] seq=%d completed with no usable candidates; keeping previous command and advancing planning round",
            finished_seq,
        )

    def _maybe_activate_completed_round(self, force: bool = False, force_reason: str = "newer-seq") -> None:
        """Close when every connected MPC publisher responded, or on timeout."""
        with self.pending_lock:
            if self.pending_seq_id is None:
                return
            if self.pending_first_recv_stamp.to_sec() <= 0.0:
                return
            if int(self.pending_callbacks_inflight) > 0:
                return

            self._refresh_pending_expected_sources()
            expected_total, received_total, missing = self._expected_response_totals()
            all_expected_received = bool(expected_total > 0 and not missing)

            now = rospy.Time.now()
            elapsed = max(0.0, (now - self.pending_first_recv_stamp).to_sec())
            timed_out = elapsed >= float(self.trajectory_receive_window_s)
            should_close = bool(force) or all_expected_received or timed_out
            if not should_close:
                return

            if all_expected_received and not force:
                close_reason = "all-connected-planners-responded"
            elif force:
                close_reason = str(force_reason)
            else:
                close_reason = "planner-result-timeout"

            rospy.loginfo(
                "[ego_mover] closing seq=%s reason=%s elapsed=%.4f responses=%d/%d "
                "usable=%d rejected=%d missing=%s",
                str(self.pending_seq_id), close_reason, float(elapsed),
                int(received_total), int(expected_total), len(self.pending_candidates),
                int(self.pending_candidates_rejected), ",".join(missing) if missing else "none",
            )

            if self.pending_candidates:
                self.activate_pending_command()
            else:
                self._finish_pending_round_without_candidate()

    @staticmethod
    def _extract_regulation_slack_sequence(msg: MPCControlSeq, n: int) -> List[float]:
        """Extract optimized s^r_k sequence from an MPCControlSeq message.

        Preferred message field is ``normal_slacks``.  Aliases are accepted so the
        ego selector remains compatible with local message naming conventions.
        """
        for field_name in (
            "normal_slacks", "normal_slack", "regulation_slacks",
            "regulation_slack", "s_r", "sr_seq",
        ):
            if not hasattr(msg, field_name):
                continue
            raw = getattr(msg, field_name)
            try:
                vals = [float(v) for v in raw]
            except Exception:
                continue
            if len(vals) == 0:
                continue
            vals = vals[:max(int(n), 0)]
            if len(vals) < int(n):
                vals.extend([0.0] * (int(n) - len(vals)))
            return vals
        return []

    def initial_guidance_callback(self, msg: MPCControlSeq, callback_args=None) -> None:
        """Cache whether /initial_guidanceN currently carries a usable guidance."""
        slot = -1
        source_topic = ""
        if isinstance(callback_args, tuple) and len(callback_args) >= 2:
            try:
                slot = int(callback_args[0])
            except Exception:
                slot = -1
            source_topic = str(callback_args[1])
        if slot <= 0:
            return

        cost = self._msg_first_float(msg, ("cost", "global_cost", "J_global", "j_global"), None)
        cost_valid = bool(
            cost is None
            or (np.isfinite(float(cost))
                and float(cost) < float(self.initial_guidance_invalid_cost_threshold))
        )

        try:
            yawrates = np.asarray(list(msg.yawrates), dtype=float).reshape(-1)
            accs = np.asarray(list(msg.accs), dtype=float).reshape(-1)
            n = min(int(yawrates.size), int(accs.size))
            controls_valid = bool(
                n > 0
                and np.all(np.isfinite(yawrates[:n]))
                and np.all(np.isfinite(accs[:n]))
            )
        except Exception:
            controls_valid = False

        valid = bool(cost_valid and controls_valid)
        self.initial_guidance_status_by_slot[int(slot)] = {
            "valid": valid,
            "cost": float(cost) if cost is not None and np.isfinite(float(cost)) else 1.0e100,
            "seq_id": int(msg.seq_id) if hasattr(msg, "seq_id") else -1,
            "recv_stamp": rospy.Time.now(),
            "source_topic": source_topic,
        }

    def candidate_initial_guidance_status(
        self, msg: MPCControlSeq, source_topic: str
    ) -> Tuple[bool, str]:
        """Return whether this planner candidate had a valid initial-guidance source."""
        explicit = self._msg_first_bool(
            msg,
            (
                "used_initial_guidance",
                "initial_guidance_used",
                "has_initial_guidance",
                "guidance_available",
            ),
        )
        if explicit is not None:
            return bool(explicit), "planner-guidance-flag"

        source_topic = str(source_topic)
        if not bool(self.control_topic_initial_guidance_enabled.get(source_topic, True)):
            return False, "initial-guidance-disabled"

        slot = self.control_topic_to_guidance_slot.get(source_topic)
        if slot is None:
            return False, "unmapped-control-topic"
        status = self.initial_guidance_status_by_slot.get(int(slot))
        if not isinstance(status, dict) or not bool(status.get("valid", False)):
            return False, "no-valid-initial-guidance"

        timeout_s = float(self.initial_guidance_timeout_s)
        if timeout_s > 0.0:
            recv_stamp = status.get("recv_stamp", rospy.Time(0))
            try:
                age_s = (rospy.Time.now() - recv_stamp).to_sec()
            except Exception:
                age_s = float("inf")
            if (not np.isfinite(age_s)) or age_s > timeout_s:
                return False, "stale-initial-guidance"

        return True, "cached-valid-initial-guidance"

    @staticmethod
    def _candidate_selection_cost(candidate: Dict[str, Any]) -> float:
        value = candidate.get("selection_cost", 1.0e100)
        try:
            value = float(value)
        except Exception:
            return 1.0e100
        return value if np.isfinite(value) else 1.0e100

    def control_callback(self, msg: MPCControlSeq, source_topic: str = "") -> None:
        """Collect latest result from each planner; no common planning ID/barrier."""
        seq_id = int(getattr(msg, "seq_id", -1))
        trajectory_type = int(getattr(msg, "trajectory_type", 0))
        source_key = str(source_topic).strip() or "__anonymous_control_source__"
        candidate_accepted = False
        try:
            yawrates = [float(v) for v in msg.yawrates]
            accs = [float(a) for a in msg.accs]
            n = min(len(yawrates), len(accs))
            if n <= 0:
                rospy.logwarn_throttle(1.0, "[ego_mover] received empty /mpc_control_seq candidate")
                return

            dt = float(getattr(msg, "dt", self.cfg.dt_init))
            if not np.isfinite(dt) or dt <= 1.0e-9:
                dt = float(self.cfg.dt_init)
            dts = self._extract_time_intervals(msg, n, dt)
            regulation_slacks = self._extract_regulation_slack_sequence(msg, n)

            # Preserve the exact optimized state trajectory published by the MPC.
            # plotter.py prefers these same xs/ys/vs/yaws fields for /mpc_pred_trajs,
            # so keeping them here guarantees that /selected_traj can be an exact
            # member of the visualized MPC candidate set instead of a fresh RK4
            # reconstruction from controls.
            optimized_states = None
            required_state_fields = ("xs", "ys", "vs", "yaws")
            if all(hasattr(msg, name) for name in required_state_fields):
                try:
                    xs = np.asarray(list(msg.xs), dtype=float).reshape(-1)
                    ys = np.asarray(list(msg.ys), dtype=float).reshape(-1)
                    vs = np.asarray(list(msg.vs), dtype=float).reshape(-1)
                    yaws = np.asarray(list(msg.yaws), dtype=float).reshape(-1)
                    n_state = min(xs.size, ys.size, vs.size, yaws.size)
                    if n_state > 0:
                        exact = np.column_stack((
                            xs[:n_state], ys[:n_state], vs[:n_state], yaws[:n_state]
                        ))
                        if np.all(np.isfinite(exact)):
                            exact[:, 3] = np.asarray(
                                [wrap_to_pi(float(v)) for v in exact[:, 3]], dtype=float
                            )
                            optimized_states = exact
                except Exception:
                    optimized_states = None

            candidate_start_state, start_state_source = self._candidate_start_state_from_msg(msg, seq_id)
            if candidate_start_state is None:
                return

            # Local rollout is retained only for diagnostics / legacy cost fallback.
            # It must use the same x0 as the MPC feasibility check.
            (
                legacy_base_cost,
                terminal_cost,
                legacy_boundary_cost,
                shared_boundary_slack,
                terminal_x,
                terminal_y,
            ) = self.candidate_global_cost_components(
                yawrates[:n], accs[:n], dts, start_state=candidate_start_state
            )

            planner_global_cost = self._msg_first_float(
                msg, ("global_cost", "J_global", "j_global", "cost"), None
            )
            if planner_global_cost is None:
                planner_global_cost = float(legacy_base_cost)
                cost_source = "legacy-local-fallback"
                rospy.logwarn_throttle(
                    2.0,
                    "[ego_mover] MPCControlSeq has no finite J_global; using legacy local cost",
                )
            else:
                planner_global_cost = float(planner_global_cost)
                cost_source = "planner-paper-global-without-Qsim"

            # Paper feasibility: "zero slack" means only the auxiliary regulation
            # slack s^r is zero; the avoidance slack s_k remains allowed.
            zero_slack_feasible, feasibility_source = self.candidate_zero_slack_feasibility(msg)
            phi_min = self.candidate_phi_min(msg)
            similarity_cost, similarity_per_k, similarity_source = self.candidate_similarity_cost(
                yawrates[:n], accs[:n], dts, start_state=candidate_start_state,
                candidate_states=optimized_states,
            )
            # rho_c is already part of planner_global_cost.  Add ONLY Q_sim here.
            selection_cost = float(planner_global_cost) + float(similarity_cost)
            # Feasibility comes from per-candidate planner metadata. Missing metadata
            # is treated conservatively as infeasible, while J_global is still retained
            # for the all-infeasible fallback selection below.
            original_ocp_feasible = self._msg_first_bool(
                msg, ("original_ocp_feasible", "ocp_feasible", "selection_eligible")
            )
            if original_ocp_feasible is None:
                original_ocp_feasible = False
                original_ocp_feasibility_source = "missing-original-ocp-metadata"
            else:
                original_ocp_feasible = bool(original_ocp_feasible)
                original_ocp_feasibility_source = "planner-original-ocp-flag"
            has_initial_guidance, initial_guidance_source = self.candidate_initial_guidance_status(
                msg, source_key
            )

            candidate = {
                "seq_id": seq_id,
                "source_topic": source_key,
                "trajectory_type": trajectory_type,
                "cost": planner_global_cost,
                "base_cost": planner_global_cost,
                "planner_global_cost": planner_global_cost,
                "cost_source": cost_source,
                "terminal_cost": terminal_cost,
                "slack_cost": legacy_boundary_cost,
                "shared_slack": shared_boundary_slack,
                "terminal_x": terminal_x,
                "terminal_y": terminal_y,
                "selection_cost": selection_cost,
                "similarity_cost": float(similarity_cost),
                "similarity_per_k": similarity_per_k.tolist(),
                "similarity_source": similarity_source,
                "original_ocp_feasible": bool(original_ocp_feasible),
                "original_ocp_feasibility_source": original_ocp_feasibility_source,
                "start_state": candidate_start_state.tolist(),
                "start_state_source": start_state_source,
                "zero_slack_feasible": bool(zero_slack_feasible),
                "paper_feasible": bool(zero_slack_feasible),
                "feasibility_source": feasibility_source,
                "has_initial_guidance": bool(has_initial_guidance),
                "initial_guidance_source": initial_guidance_source,
                "phi_min": float(phi_min),
                "min_clearance": float(phi_min),
                "dt": dt,
                "dts": dts,
                "yawrates": yawrates[:n],
                "accs": accs[:n],
                "optimized_states": (
                    optimized_states.tolist() if optimized_states is not None else None
                ),
                "regulation_slacks": regulation_slacks,
                "stamp": rospy.Time.now(),
            }

            # Each MPC_plannerN publishes one trajectory.  Keep all planner outputs
            # that arrive inside the 0.8-period receive window.
            with self.pending_lock:
                # The round may have been force-closed only if seq changed; otherwise
                # timer closure is blocked while this callback is in flight.
                self.pending_candidates = [c for c in self.pending_candidates
                                           if c.get("source_topic") != source_key]
                candidate["arrival_wall_time"] = time.monotonic()
                self.pending_candidates.append(candidate)
                candidate_accepted = True

            rospy.logdebug(
                "[ego_mover] queued seq=%d topic=%s type=%d feasible=%s guidance=%s "
                "J_planner=%.6e J_sim=%.6e J_global=%.6e total_candidates=%d sources=%d",
                seq_id, source_key, trajectory_type, bool(zero_slack_feasible),
                bool(has_initial_guidance), float(planner_global_cost),
                float(similarity_cost), float(selection_cost),
                len(self.pending_candidates), len(self.pending_sources_seen),
            )
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "[ego_mover] invalid MPC candidate from %s: %s", source_key, exc)

    def activate_pending_command(self) -> None:
        if len(self.pending_candidates) == 0:
            self._finish_pending_round_without_candidate()
            return

        # Two-tier selection over every trajectory received in this cycle:
        #   1) if feasible trajectories exist, choose the feasible one with minimum J_global;
        #   2) otherwise, choose the minimum-J_global trajectory over ALL candidates.
        # J_global has already been computed/stored for every candidate regardless
        # of feasibility.
        feasible_candidates = [
            c for c in self.pending_candidates
            if bool(c.get("original_ocp_feasible", False))
        ]
        if feasible_candidates:
            candidate = min(feasible_candidates, key=self._candidate_selection_cost)
            selection_reason = "original-ocp-feasible-min-J_global"
        else:
            candidate = min(self.pending_candidates, key=self._candidate_selection_cost)
            selection_reason = "no-feasible-fallback-min-J_global"
            rospy.logwarn(
                "[ego_mover] no original-OCP-feasible candidate for seq=%s; "
                "falling back to minimum J_global over all %d received candidates",
                str(self.pending_seq_id), len(self.pending_candidates),
            )

        self.pending_cmd = candidate
        self.cmd_yawrates = list(candidate["yawrates"])
        self.cmd_accs = list(candidate["accs"])
        self.cmd_dt = float(candidate["dt"])
        self.cmd_dts = list(candidate.get("dts", [self.cmd_dt] * len(self.cmd_yawrates)))
        self.cmd_base_cost = float(candidate.get("base_cost", candidate.get("cost", 1.0e100)))
        self.cmd_planner_global_cost = float(candidate.get("planner_global_cost", self.cmd_base_cost))
        self.cmd_terminal_cost = float(candidate.get("terminal_cost", 0.0))
        self.cmd_slack_cost = float(candidate.get("slack_cost", 0.0))
        self.cmd_shared_slack = float(candidate.get("shared_slack", 0.0))
        self.cmd_selection_cost = float(candidate.get("selection_cost", self.cmd_planner_global_cost))
        self.cmd_phi_min = float(candidate.get("phi_min", float("-inf")))
        self.cmd_min_clearance = float(self.cmd_phi_min)
        self.cmd_zero_slack_feasible = bool(candidate.get("zero_slack_feasible", False))
        self.cmd_paper_feasible = bool(candidate.get("paper_feasible", self.cmd_zero_slack_feasible))
        self.cmd_selection_reason = selection_reason
        self.cmd_trajectory_type = int(candidate.get("trajectory_type", 0))
        self.cmd_cost = float(self.cmd_selection_cost)
        self.cmd_seq_id = int(candidate["seq_id"])
        self.cmd_index = 0
        self.have_cmd = True
        now = rospy.Time.now()
        self.cmd_start_stamp = now
        self.last_cmd_stamp = now

        # Publish exactly the MPC candidate trajectory that was selected.
        # /mpc_pred_trajs uses msg.xs/ys/vs/yaws directly when available, so use
        # the same optimized X here.  Re-integrating yawrates/accs can produce a
        # slightly different curve and makes /selected_traj appear not to belong
        # to /mpc_pred_trajs even though it came from the same MPCControlSeq.
        optimized_states = candidate.get("optimized_states")
        if optimized_states is not None:
            try:
                selected_states = np.asarray(optimized_states, dtype=float).reshape(-1, 4)
                if selected_states.shape[0] == 0 or not np.all(np.isfinite(selected_states)):
                    selected_states = np.zeros((0, 4), dtype=float)
            except Exception:
                selected_states = np.zeros((0, 4), dtype=float)
        else:
            selected_states = np.zeros((0, 4), dtype=float)

        # Backward-compatible fallback for legacy MPCControlSeq without X fields.
        if selected_states.shape[0] == 0:
            selected_start_state = np.asarray(candidate["start_state"], dtype=float)
            selected_states = self._rollout_control_sequence_absolute(
                self.cmd_yawrates, self.cmd_accs, self.cmd_dts,
                start_state=selected_start_state,
            )
        traj_msg = Float64MultiArray()
        if selected_states.size > 0:
            traj_msg.layout.dim = []
            dim0 = MultiArrayDimension()
            dim0.label = "state"
            dim0.size = int(selected_states.shape[0])
            dim0.stride = int(selected_states.shape[0] * selected_states.shape[1])
            dim1 = MultiArrayDimension()
            dim1.label = "field[x,y,v,yaw]"
            dim1.size = int(selected_states.shape[1])
            dim1.stride = int(selected_states.shape[1])
            traj_msg.layout.dim = [dim0, dim1]
            traj_msg.layout.data_offset = 0
            traj_msg.data = selected_states.reshape(-1).astype(float).tolist()
        else:
            traj_msg.data = []
        self.pub_selected_traj.publish(traj_msg)

        # Publish ONLY the selected trajectory's optimized regulation-slack sequence.
        # For mpc_flag=4 this is s^r_0,...,s^r_{H-1} = plan.normal_slack.
        sr_seq = [float(v) for v in candidate.get("regulation_slacks", [])]
        sr_msg = Float64MultiArray()
        sr_msg.data = sr_seq
        self.pub_selected_sr.publish(sr_msg)
        if sr_seq:
            rospy.loginfo(
                "[ego_mover] selected s^r seq=%s",
                "[" + ", ".join("%.6g" % v for v in sr_seq) + "]",
            )
        else:
            rospy.logwarn_throttle(
                2.0,
                "[ego_mover] selected candidate has no s^r sequence; "
                "add float64[] normal_slacks to MPCControlSeq.msg and publish plan.normal_slack",
            )

        # This selected trajectory becomes tilde{x}^{prev} (one-stage shifted)
        # for the next planning round.
        self._update_previous_selected_reference(candidate)

        candidate_count = len(self.pending_candidates)

        # Report the trajectory/planner ID encoded in its source topic rather than
        # the asynchronous arrival order in pending_candidates.  For example,
        # /mpc_control_seq4 is always reported as trajectory 4.
        source_topic = str(candidate.get("source_topic", ""))
        selected_traj_id = -1
        marker = "mpc_control_seq"
        if marker in source_topic:
            suffix = source_topic.rsplit(marker, 1)[1]
            if suffix.isdigit():
                selected_traj_id = int(suffix)

        print("candidate trajectory_count:", candidate_count)
        print(
            "selected traj: %d / total: %d | seq=%d topic=%s type=%d J_global=%.6e exact_X=%s"
            % (
                int(selected_traj_id), int(candidate_count),
                int(self.cmd_seq_id), source_topic,
                int(self.cmd_trajectory_type), float(self.cmd_selection_cost),
                "yes" if candidate.get("optimized_states") is not None else "no",
            )
        )
        print("async candidate selection: received=%d (no publisher barrier)" % candidate_count)
        received_source_count = len(self.pending_sources_seen)
        self._reset_pending_round_state()


        # rospy.loginfo(
        #     "[ego_mover] activated seq=%d type=%d reason=%s J_planner=%.6e J_sim=%.6e J_global=%.6e from %d valid candidates (%d sources received) dt=%.4f len=%d",
        #     int(self.cmd_seq_id), int(self.cmd_trajectory_type), str(self.cmd_selection_reason),
        #     float(self.cmd_planner_global_cost), float(candidate.get("similarity_cost", 0.0)),
        #     float(self.cmd_selection_cost), int(candidate_count), int(received_source_count),
        #     float(self.cmd_dt), len(self.cmd_yawrates),
        # )

    # ------------------------------------------------------------------
    # Dynamics update
    # ------------------------------------------------------------------

    def build_state(self) -> np.ndarray:
        return np.array([self.ego_x, self.ego_y, self.ego_v, self.ego_yaw], dtype=float)

    def command_index_from_elapsed(self, elapsed_s: float) -> int:
        n = min(len(self.cmd_yawrates), len(self.cmd_accs), len(self.cmd_dts))
        if n <= 0:
            return -1
        if elapsed_s <= 0.0:
            return 0

        cumulative = 0.0
        for idx in range(n):
            cumulative += max(float(self.cmd_dts[idx]), 1.0e-9)
            if elapsed_s < cumulative:
                return idx
        return n - 1

    def select_command(self, now: rospy.Time = None) -> Tuple[float, float, float, int]:
        if not self.have_cmd or len(self.cmd_yawrates) == 0 or len(self.cmd_accs) == 0:
            return 0.0, 0.0, float(self.cfg.dt_init), -1

        if now is None:
            now = rospy.Time.now()

        elapsed_s = (now - self.cmd_start_stamp).to_sec() if self.cmd_start_stamp.to_sec() > 0.0 else 0.0
        idx = self.command_index_from_elapsed(elapsed_s)
        if idx < 0:
            return 0.0, 0.0, float(self.cfg.dt_init), -1

        yawrate = float(self.cmd_yawrates[idx])
        acc = float(self.cmd_accs[idx])
        dt = float(self.cmd_dts[idx]) if idx < len(self.cmd_dts) else float(self.cmd_dt)
        self.cmd_index = int(idx)

        return yawrate, acc, dt, idx

    def termination_reason(self) -> Tuple[bool, str]:
        # Do not stop when the ego leaves the configured map boundary.  The
        # planner now has a shared soft boundary slack, so temporary boundary
        # violation should be resolved by MPC cost/constraints instead of
        # killing the ROS node.
        if self.stop_on_goal:
            dist_goal = math.hypot(self.ego_x - self.goal_x, self.ego_y - self.goal_y)
            if dist_goal <= float(self.goal_tolerance):
                return True, (
                    "ego reached goal: dist=%.3f <= %.3f, goal=(%.3f, %.3f)"
                    % (dist_goal, self.goal_tolerance, self.goal_x, self.goal_y)
                )

        return False, ""

    def shutdown_if_terminated(self) -> bool:
        if self.stopped:
            return True
        should_stop, reason = self.termination_reason()
        if not should_stop:
            return False
        self.stopped = True
        self.publish_stop_command()
        self.publish_ego_state()
        rospy.loginfo("[ego_mover] %s. Shutting down ego_mover.", reason)
        rospy.signal_shutdown(reason)
        return True

    def step(self) -> None:
        if self.shutdown_if_terminated():
            return

        # During benchmark reset, hold the ego exactly at its configured start
        # until repeat_scenario.py confirms that both ego and obstacles are ready.
        if self.scenario_managed_mode and not self.scenario_run_enabled:
            self.last_step_stamp = rospy.Time.now()
            self.publish_stop_command()
            self.publish_ego_state()
            return

        # Close the current planning round when all connected planners responded;
        # the timeout is only a stalled-planner fallback.
        with self.pending_lock:
            # Only consider results received since the previous selection tick.
            if self.pending_candidates:
                self.pending_candidates = [c for c in self.pending_candidates
                                           if time.monotonic() - c.get("arrival_wall_time", 0.0) <= 0.25]
                if self.pending_candidates:
                    self.pending_seq_id = None  # no round gating
                    self.activate_pending_command()

        now = rospy.Time.now()
        if self.last_step_stamp.to_sec() > 0.0:
            sim_dt = (now - self.last_step_stamp).to_sec()
        else:
            sim_dt = 1.0 / max(float(self.hz), 1.0)
        self.last_step_stamp = now
        if not np.isfinite(sim_dt) or sim_dt <= 0.0:
            sim_dt = 1.0 / max(float(self.hz), 1.0)
        sim_dt = min(sim_dt, 0.1)
        
        # Apply the selected trajectory's control input exactly as optimized.
        yawrate, acc, ctrl_dt, used_idx = self.select_command(now)
        if not self.have_cmd:
            yawrate, acc = 0.0, 0.0

        if self.use_real_robot:
            self.publish_robot_command(now, sim_dt, yawrate=yawrate, acc=acc)
        elif self.have_cmd:
            x = self.build_state()
            u = np.array([yawrate, acc], dtype=float)
            print("U: ", u)
            x_next = step_ego_dynamics(x, u, sim_dt, self.cfg)
            self.ego_x = float(x_next[0])
            self.ego_y = float(x_next[1])
            self.ego_v = float(x_next[2])
            self.ego_yaw = wrap_to_pi(float(x_next[3]))
            
            # print("ego_v: ", self.ego_v)
            # print("ego_x: ", self.ego_x)
            # print("ego_v: ", self.ego_v)
            
            # self.ego_x = -4 #float(x_next[0])
            # self.ego_y = -4 #float(x_next[1])
            # self.ego_v = 1.5 #float(x_next[2])
            # self.ego_yaw = 3.14/4 #wrap_to_pi(float(x_next[3]))

            rospy.logdebug(
                "[ego_mover] seq=%d cost=%.6e idx=%d yawrate=%.4f acc=%.4f ctrl_dt=%.4f sim_dt=%.4f -> x=%.3f y=%.3f v=%.3f yaw=%.3f",
                int(self.cmd_seq_id), float(self.cmd_cost), int(used_idx), yawrate, acc, ctrl_dt, sim_dt,
                self.ego_x, self.ego_y, self.ego_v, self.ego_yaw,
            )

        if not self.use_real_robot or self.have_tracked_pose:
            self.publish_ego_state()
        self.shutdown_if_terminated()

    # ------------------------------------------------------------------
    # State publisher only.  RViz Marker publication belongs to plotter.py.
    # ------------------------------------------------------------------

    def publish_ego_state(self) -> None:
        msg = Ego_state()

        # Store the FIRST state published for this cycle only.  Repeated ego-state
        # ticks with the same seq_id must not move the trajectory start point.
        round_id = 0  # no ego_mover planning-round protocol
        if round_id not in self.planning_round_start_xy:
            self.planning_round_start_xy[round_id] = (
                float(self.ego_x), float(self.ego_y)
            )
            self.planning_round_start_state[round_id] = (
                float(self.ego_x), float(self.ego_y),
                float(self.ego_v), float(self.ego_yaw),
            )
            # Bound diagnostic history; only recent rounds can still arrive.
            if len(self.planning_round_start_xy) > 32:
                for old_id in sorted(self.planning_round_start_xy)[:-16]:
                    self.planning_round_start_xy.pop(old_id, None)
                    self.planning_round_start_state.pop(old_id, None)

        if hasattr(msg, "header"):
            msg.header.seq = round_id
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id

        msg.ego_x = float(self.ego_x)
        msg.ego_y = float(self.ego_y)
        if hasattr(msg, "ego_z"):
            msg.ego_z = float(self.ego_z)

        if hasattr(msg, "ego_v"):
            msg.ego_v = float(self.ego_v)
        if hasattr(msg, "ego_yaw"):
            msg.ego_yaw = float(self.ego_yaw)

        if hasattr(msg, "ego_vx"):
            msg.ego_vx = float(self.ego_v * math.cos(self.ego_yaw))
        if hasattr(msg, "ego_vy"):
            msg.ego_vy = float(self.ego_v * math.sin(self.ego_yaw))
        if hasattr(msg, "ego_vz"):
            msg.ego_vz = 0.0

        if hasattr(msg, "ego_L"):
            msg.ego_L = float(self.ego_L)
        if hasattr(msg, "ego_W"):
            msg.ego_W = float(self.ego_W)
        if hasattr(msg, "ego_R"):
            msg.ego_R = float(self.ego_R)

        self.pub_ego.publish(msg)
        # Publish an identical map-framed state for legacy visualization consumers.
        self.publish_ego_state_in_map(msg)
        # Do not increment here.  The same round ID may be published at several
        # ego-state ticks while the six planners finish asynchronously.  The ID
        # advances only in activate_pending_command() after the round is selected.


    def publish_ego_state_in_map(self, local_msg: Ego_state) -> None:
        """Compatibility publisher: /ego_state_map mirrors map-framed /ego_state."""
        if self.pub_ego_map is None:
            return
        world = copy.deepcopy(local_msg)
        if hasattr(world, "header"):
            world.header.frame_id = self.world_frame_id
        self.pub_ego_map.publish(world)


def main() -> None:
    rospy.init_node("ego_mover", anonymous=False)
    node = EgoMoverNode()
    time.sleep(0.2)
    # rate = rospy.Rate(node.hz)
    rate = rospy.Rate(node.hz)
    while not rospy.is_shutdown():
        node.step()
        rate.sleep()


if __name__ == "__main__":
    main()
