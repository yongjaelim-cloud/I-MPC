#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RViz plotter for the 2-D crowded-drone scenario.

Important convention:
    The simulation/planning state is 2-D only: (x, y).
    In RViz Marker points, Point.z is NOT a spatial z coordinate.
    Point.z is used only as a visualization time axis:
        z_vis = TIME_SCALE * t.
"""

import random
import os
import threading
import copy

import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_matrix, quaternion_multiply
from setup_config import setup_msg_to_raw

from drone_msgs.msg import (
    Ego_state,
    Determined_traj,
    Sampled_traj,
    obstacle_polygon,
    obstacle_drone,
    MPCControlSeq,
    Guidance,
    Setup,
    Ego_goal_setting,
)
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import math

try:
    import utils_linearized as _utils_linearized
    _UTILS_LINEARIZED_IMPORT_ERROR = None
except Exception as _utils_linearized_exc:
    _utils_linearized = None
    _UTILS_LINEARIZED_IMPORT_ERROR = repr(_utils_linearized_exc)


TIME_SCALE = 1.0
PRED_HORIZON = 20
PRED_DT = 0.1
PRED_VIS_STEP = 10

MAX_OBS = 200
MAX_VERTICES = 10


def _load_setup_config():
    """Receive the latched /setup message published by setter.py.

    plotter.py never opens mpc_planner_common.json. The setter is the sole
    JSON reader; this node reconstructs the nested configuration only from
    drone_msgs/Setup received on /setup.
    """
    msg = rospy.wait_for_message("/setup", Setup)
    raw = setup_msg_to_raw(msg)
    cfg = raw.get("config", {}) if isinstance(raw.get("config", {}), dict) else {}
    normal_cfg = raw.get("normal_cfg", {}) if isinstance(raw.get("normal_cfg", {}), dict) else {}
    plotter_cfg = raw.get("plotter", {}) if isinstance(raw.get("plotter", {}), dict) else {}
    return raw, cfg, normal_cfg, plotter_cfg

def _cfg_get(cfg, primary, fallback, default):
    if primary in cfg:
        return cfg[primary]
    if fallback is not None and fallback in cfg:
        return cfg[fallback]
    return default


def _get_nested(block, section, key, default):
    sub = block.get(section, {}) if isinstance(block, dict) else {}
    if isinstance(sub, dict) and key in sub:
        return sub[key]
    return default


def _as_string_list(value):
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


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def _rgba(value, default):
    try:
        arr = list(value)
        if len(arr) == 3:
            arr.append(default[3])
        if len(arr) >= 4:
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    except Exception:
        pass
    return tuple(float(v) for v in default)


def _normalize_sector_id(value):
    try:
        sid = int(value)
    except Exception:
        sid = 1
    if sid not in (1, 2, 3, 4):
        sid = 1
    return sid


def _wrap_yaw_np(yaw):
    return math.atan2(math.sin(float(yaw)), math.cos(float(yaw)))


def _ellipse_axes_from_lengths(ego_L, ego_W):
    """Return overestimated ego-ellipse full axes and local-frame scales.

    ego_L/ego_W are full rectangle lengths.  The conservative ellipse that
    contains the rectangle and passes through its four corners has full axes
    sqrt(2)*ego_L and sqrt(2)*ego_W.  The scale ratio is unchanged by the
    common sqrt(2), but the minor/major radii used for safe-boundary inflation
    must use the overestimated axes.
    """
    L = max(abs(float(ego_L)), 1.0e-12)
    W = max(abs(float(ego_W)), 1.0e-12)
    ellipse_L = math.sqrt(2.0) * L
    ellipse_W = math.sqrt(2.0) * W
    minor = min(ellipse_L, ellipse_W)
    major = max(ellipse_L, ellipse_W)
    return float(major), float(minor), (float(ellipse_L / minor), float(ellipse_W / minor))


class _RVizWorldProjector:
    """Project a visualization message to odom without touching MPC data.

    Marker.points are coordinates in Marker.pose's local coordinate system.
    Transforming the pose (rather than each point) avoids copying thousands
    of cached points. The latest TF avoids rosbag vs wall-clock stamp mismatch.
    """
    def __init__(self, tf_buffer, target_frame):
        self.tf_buffer = tf_buffer
        self.target_frame = str(target_frame).lstrip("/")

    def _tf(self, source, cache):
        source = str(source or "base_link").lstrip("/")
        if source == self.target_frame:
            return None
        if source not in cache:
            cache[source] = self.tf_buffer.lookup_transform(
                self.target_frame, source, rospy.Time(0), rospy.Duration(0.0))
        return cache[source]

    @staticmethod
    def _pose(pose, transform):
        if transform is None:
            return pose
        t = transform.transform.translation
        q = transform.transform.rotation
        R = quaternion_matrix((q.x, q.y, q.z, q.w))[:3, :3]
        p = pose.position
        x = R[0, 0]*p.x + R[0, 1]*p.y + R[0, 2]*p.z + t.x
        y = R[1, 0]*p.x + R[1, 1]*p.y + R[1, 2]*p.z + t.y
        z = R[2, 0]*p.x + R[2, 1]*p.y + R[2, 2]*p.z + t.z
        oq = pose.orientation
        rot = (oq.x, oq.y, oq.z, oq.w)
        if sum(float(v)*float(v) for v in rot) <= 1.0e-12:
            rot = (0.0, 0.0, 0.0, 1.0)
        new_q = quaternion_multiply((q.x, q.y, q.z, q.w), rot)
        out = copy.copy(pose)
        out.position = copy.copy(p)
        out.orientation = copy.copy(oq)
        out.position.x, out.position.y, out.position.z = float(x), float(y), float(z)
        out.orientation.x, out.orientation.y = float(new_q[0]), float(new_q[1])
        out.orientation.z, out.orientation.w = float(new_q[2]), float(new_q[3])
        return out

    def _marker(self, marker, cache):
        source = str(marker.header.frame_id or "base_link").lstrip("/")
        out = copy.copy(marker)
        out.header = copy.copy(marker.header)
        out.header.frame_id = self.target_frame
        out.header.stamp = rospy.Time(0)
        if marker.action not in (Marker.DELETE, Marker.DELETEALL):
            out.pose = self._pose(marker.pose, self._tf(source, cache))
        return out

    def convert(self, msg):
        if self.target_frame == "base_link":
            return msg
        cache = {}
        try:
            if isinstance(msg, Marker):
                return self._marker(msg, cache)
            if isinstance(msg, MarkerArray):
                out = MarkerArray()
                out.markers = [self._marker(m, cache) for m in msg.markers]
                return out
            if isinstance(msg, Path):
                out = copy.copy(msg)
                out.header = copy.copy(msg.header)
                out.header.frame_id = self.target_frame
                out.header.stamp = rospy.Time(0)
                poses = []
                for ps in msg.poses:
                    src = str(ps.header.frame_id or msg.header.frame_id or "base_link").lstrip("/")
                    converted = copy.copy(ps)
                    converted.header = copy.copy(ps.header)
                    converted.header.frame_id = self.target_frame
                    converted.header.stamp = rospy.Time(0)
                    converted.pose = self._pose(ps.pose, self._tf(src, cache))
                    poses.append(converted)
                out.poses = poses
                return out
            return msg
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "[plotter] RViz frame conversion to %s unavailable: %s",
                                  self.target_frame, str(exc))
            return None


class _ProjectedPublisher:
    """Proxy for ROS visualization publishers; expose normal Publisher APIs."""
    def __init__(self, publisher, projector):
        self._publisher = publisher
        self._projector = projector

    def publish(self, msg):
        converted = self._projector.convert(msg)
        if converted is not None:
            self._publisher.publish(converted)

    def __getattr__(self, name):
        return getattr(self._publisher, name)


class Frenet2CartesianTransformer:
    def __init__(self):
        self.ego_cart_x = 0.0
        self.ego_cart_y = 0.0
        self.ego_v = 0.0
        self.ego_yaw = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.x_min = 0.0
        self.x_max = 20.0
        self.y_min = 0.0
        self.y_max = 20.0

        self.moving_obstacle_R = 0.2
        self.ego_R = 0.2
        self.ego_FOV = 1.0
        self.obs_fov_deg = 80.0

        self.raw_cfg, self.config_cfg, self.normal_cfg, self.plotter_cfg = _load_setup_config()
        mpc_cfg = self.raw_cfg.get("mpc", {}) if isinstance(self.raw_cfg.get("mpc", {}), dict) else {}
        self.selector_cfg = self.raw_cfg.get("vertex_selector", {}) if isinstance(self.raw_cfg.get("vertex_selector", {}), dict) else {}
        ego_cfg = self.raw_cfg.get("ego", {}) if isinstance(self.raw_cfg.get("ego", {}), dict) else {}
        self.ego_cart_x = float(rospy.get_param("~ego_x", ego_cfg.get("ego_x", self.ego_cart_x)))
        self.ego_cart_y = float(rospy.get_param("~ego_y", ego_cfg.get("ego_y", self.ego_cart_y)))
        self.ego_v = float(rospy.get_param("~ego_v", ego_cfg.get("ego_v", self.ego_v)))
        self.ego_yaw = float(rospy.get_param("~ego_yaw", ego_cfg.get("ego_yaw", self.ego_yaw)))

        self.x_min = float(rospy.get_param("~x_min", self.config_cfg.get("x_min", self.x_min)))
        self.x_max = float(rospy.get_param("~x_max", self.config_cfg.get("x_max", self.x_max)))
        self.y_min = float(rospy.get_param("~y_min", self.config_cfg.get("y_min", self.y_min)))
        self.y_max = float(rospy.get_param("~y_max", self.config_cfg.get("y_max", self.y_max)))
        # Goal is configuration-owned and comes from setter -> latched /setup.
        self.goal_x = float(_cfg_get(self.config_cfg, "x_end", "goal_x", self.goal_x))
        self.goal_y = float(_cfg_get(self.config_cfg, "y_end", "goal_y", self.goal_y))
        self.moving_obstacle_R = float(rospy.get_param("~moving_obstacle_R", mpc_cfg.get("moving_obstacle_R", self.moving_obstacle_R)))
        self.ego_R = float(rospy.get_param("~ego_R", mpc_cfg.get("ego_R", self.ego_R)))
        self.ego_L = float(ego_cfg.get("ego_L", 0.0))
        self.ego_W = float(ego_cfg.get("ego_W", 0.0))
        self.ego_major_axis_m, self.ego_minor_axis_m, self.ego_axis_scale = _ellipse_axes_from_lengths(self.ego_L, self.ego_W)
        self.use_ego_scaled_local_frame = _as_bool(rospy.get_param(
            "~use_ego_scaled_local_frame",
            self.normal_cfg.get("use_ego_scaled_local_frame", True),
        ), True)
        self.ego_FOV = float(rospy.get_param("~FOV", _cfg_get(self.config_cfg, "FOV", "obs_sensing_R", self.ego_FOV)))
        self.obs_fov_deg = float(rospy.get_param("~obs_fov_deg", _cfg_get(self.config_cfg, "obs_fov_deg", "obs_fov_angle_deg", self.obs_fov_deg)))
        self.sector_id = _normalize_sector_id(rospy.get_param(
            "~sector_id",
            self.config_cfg.get("sector_id", 1),
        ))
        self.show_ego_sector = True
        # The vulnerable/designated sector is meaningful only while the impact-
        # normal regulation constraint is enabled in the common MPC config.
        self.enable_impact_normal_regulation = True
        self.ego_sector_topic = str(rospy.get_param(
            "~ego_sector_topic",
            self.plotter_cfg.get("ego_sector_topic", "/ego_sector"),
        ))
        self.ego_sector_radius_m = max(0.0, float(rospy.get_param(
            "~ego_sector_radius_m",
            self.plotter_cfg.get("ego_sector_radius_m", 0.55),
        )))
        self.ego_sector_arc_steps = max(4, int(rospy.get_param(
            "~ego_sector_arc_steps",
            self.plotter_cfg.get("ego_sector_arc_steps", 24),
        )))
        self.ego_sector_fill_color = _rgba(
            self.plotter_cfg.get("ego_sector_fill_color", None),
            (1.0, 0.5, 0.0, 0.34),
        )
        self.ego_sector_outline_color = _rgba(
            self.plotter_cfg.get("ego_sector_outline_color", None),
            (1.0, 0.5, 0.0, 0.95),
        )

        self.initial_safe_margin_m = max(0.0, float(rospy.get_param(
            "~initial_safe_margin_m",
            self.normal_cfg.get("initial_safe_margin_m", 0.0),
        )))
        self.configured_boundary_buffer_m = rospy.get_param(
            "~boundary_buffer_m",
            self.normal_cfg.get("boundary_buffer_m", None),
        )
        self.boundary_buffer_m = self.normalized_boundary_buffer_m(self.configured_boundary_buffer_m)
        self.halfspace_buffer_m = max(0.0, float(rospy.get_param(
            "~halfspace_buffer_m",
            mpc_cfg.get("halfspace_buffer_m", 0.0),
        )))

        self.pred_horizon = int(rospy.get_param("~pred_horizon", self.plotter_cfg.get("pred_horizon", mpc_cfg.get("N", PRED_HORIZON))))
        self.pred_dt = float(rospy.get_param("~pred_dt", self.plotter_cfg.get("pred_dt", mpc_cfg.get("dt_init", PRED_DT))))
        self.pred_vis_step = max(1, int(rospy.get_param("~pred_vis_step", self.plotter_cfg.get("pred_vis_step", 1))))
        # Safe-margin obstacle prediction is intentionally sparse.  For N=20,
        # the default interval 5 draws k = 0, 5, 10, 15, 20 instead of every
        # horizon step.
        self.safe_margin_pred_step_interval = max(1, int(rospy.get_param(
            "~safe_margin_pred_step_interval",
            self.plotter_cfg.get("safe_margin_pred_step_interval", 5),
        )))
        self.safe_margin_line_width = float(rospy.get_param("~safe_margin_line_width", self.plotter_cfg.get("safe_margin_line_width", 0.025)))
        self.safe_margin_text_enabled = _as_bool(rospy.get_param("~safe_margin_text_enabled", self.plotter_cfg.get("safe_margin_text_enabled", True)), True)
        # /obs_polygons_horizon can dominate RViz cost in dense maps.
        # Keep it local to the current ego FOV by default, matching the MPC
        # selected-obstacle view, and cap the text marker unless explicitly enabled.
        self.safe_margin_horizon_fov_only = _as_bool(rospy.get_param(
            "~safe_margin_horizon_fov_only",
            self.plotter_cfg.get("safe_margin_horizon_fov_only", True),
        ), True)
        self.safe_margin_draw_current_boundary = _as_bool(rospy.get_param(
            "~safe_margin_draw_current_boundary",
            self.plotter_cfg.get("safe_margin_draw_current_boundary", True),
        ), True)
        # Heavy selector-horizon reconstruction (grouping + convex hull + Marker
        # serialization) does not need to run at the common visualization rate.
        self.safe_margin_horizon_viz_hz = max(0.0, float(rospy.get_param(
            "~safe_margin_horizon_viz_hz",
            self.plotter_cfg.get("safe_margin_horizon_viz_hz", 20.0),
        )))
        # Static and graph-like visualizations are intentionally much slower than
        # the ego marker.  This prevents unchanged MarkerArrays from being rebuilt
        # and serialized on every common loop tick.
        self.static_viz_hz = max(0.0, float(rospy.get_param(
            "~static_viz_hz", self.plotter_cfg.get("static_viz_hz", 20.0)
        )))
        self.path_viz_hz = max(0.0, float(rospy.get_param(
            "~path_viz_hz", self.plotter_cfg.get("path_viz_hz", 20.0)
        )))
        self.visibility_viz_hz = max(0.0, float(rospy.get_param(
            "~visibility_viz_hz", self.plotter_cfg.get("visibility_viz_hz", 20.0)
        )))
        self.best_traj_dt = self.pred_dt
        self.best_traj_cost = 1.0e100
        self.best_cmd_stamp = rospy.Time(0)
        self.best_traj_states = []

        # Subscriber callbacks only update these latest-data caches.  A single
        # fixed-rate loop publishes every available RViz visualization together.
        self.latest_obstacle_msg = None
        self.latest_dynamic_obstacles_msg = None
        # Latest selector-produced perceived obstacle polygons.  These are kept
        # separately from the ground-truth /obstacle stream because SDF
        # visualization must reflect exactly the geometry available to planning.
        self.latest_percepted_obstacle_msg = None
        # TF is the sole authority for ego pose in odom (as in vertex_selector).
        self.ego_tf_frame = str(rospy.get_param("~ego_tf_frame", "base_link")).lstrip("/")
        self.ego_tf_target = "base_link"  # All MPC geometry is robot-local; do not mix odom here.
        self.rviz_target_frame = str(rospy.get_param("~rviz_target_frame", "odom")).lstrip("/")
        self.ego_tf_timeout_s = max(0.0, float(rospy.get_param("~ego_tf_timeout_s", 0.05)))
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.have_ego_tf = False
        self.latest_ego_msg = None  # kept for compatibility; /ego_state is not subscribed
        self.latest_goal_msg = None
        self.runtime_goal_received = False
        self.latest_path_msg = None
        self.latest_sampled_traj_msg = None
        # Final trajectory selected by ego_mover.  This is visualized separately
        # from the per-planner /mpc_pred_trajs candidates so the executed choice
        # is immediately identifiable in RViz.
        self.latest_selected_traj_msg = None
        self._cached_selected_traj_marker = None
        self._selected_traj_dirty = False
        self.latest_selected_vertex_data_msg = None
        self.latest_selected_vertex_horizon_data_msg = None
        # Exact proposed-MPC affine avoidance model from MPC_planner0.
        self.latest_separating_hyperplane_msg = None
        self.latest_visibility_prm_graph_msg = None
        self.latest_visibility_guidance_msg = None
        self.latest_visibility_backward_prm_graph_msg = None
        self.latest_visibility_backward_guidance_msg = None

        # 20 Hz output is kept, but expensive marker geometry is rebuilt only
        # when its source data changes.  Between updates the already-built ROS
        # message object is republished, avoiding repeated convex hulls and
        # thousands of Point allocations at every visualization tick.
        self._obs_polygon_dirty = True
        self._selector_horizon_dirty = True
        self._sampled_traj_dirty = True
        self._selected_vertex_dirty = True
        self._separating_hyperplane_dirty = True
        self._cached_obs_polygon_marker_array = None
        self._cached_selector_horizon_marker_array = None
        self._cached_separating_hyperplane_marker_array = None
        self._cached_sampled_traj_marker_array = None
        self._cached_selected_vertex_point_marker = None
        self._cached_selected_vertex_normal_array = None

        # Fast visualization mode: heavy RViz products are event-driven instead
        # of reserializing unchanged MarkerArrays at the common loop rate.
        self.fast_viz_mode = _as_bool(rospy.get_param(
            "~fast_viz_mode",
            self.plotter_cfg.get("fast_viz_mode", True),
        ), True)
        self._path_dirty = True
        self._visibility_prm_dirty = True
        self._visibility_guidance_dirty = True
        self._visibility_backward_prm_dirty = True
        self._visibility_backward_guidance_dirty = True

        # Geometry conversion caches. SDF and impact-normal visualization often
        # consume the exact same ROS message; convert it to NumPy only once.
        self._selected_vertex_field_arrays_cache_msg = None
        self._selected_vertex_field_arrays_cache = None
        self._perceived_field_arrays_cache_msg = None
        self._perceived_field_arrays_cache = None

        self.sdf_cfg = None
        self.sdf_last_error = None
        # field_flag=2 uses the canonical world-frame BP model fitted/cached by
        # utils_linearized -> collision_lin_cpp. Reuse the same model for the SDF
        # grid and impact-normal finite differences when geometry is unchanged.
        self._last_sdf_bp_model = None
        self._last_sdf_bp_reference = None
        self._last_sdf_bp_points_obj = None
        self.publish_sdf = _as_bool(rospy.get_param(
            "~publish_sdf",
            self.plotter_cfg.get("publish_sdf", True),
        ), True)
        self.sdf_topic = str(rospy.get_param(
            "~sdf_topic",
            self.plotter_cfg.get("sdf_topic", "/sdf"),
        ))
        self.sdf_grid_x_topic = str(rospy.get_param(
            "~sdf_grid_x_topic",
            self.plotter_cfg.get("sdf_grid_x_topic", "/sdf_grid_x"),
        ))
        self.sdf_grid_y_topic = str(rospy.get_param(
            "~sdf_grid_y_topic",
            self.plotter_cfg.get("sdf_grid_y_topic", "/sdf_grid_y"),
        ))
        self.sdf_grid_min = float(rospy.get_param(
            "~sdf_grid_min",
            self.plotter_cfg.get("sdf_grid_min", -5.0),
        ))
        self.sdf_grid_max = float(rospy.get_param(
            "~sdf_grid_max",
            self.plotter_cfg.get("sdf_grid_max", 5.0),
        ))
        self.sdf_grid_resolution = max(1.0e-6, float(rospy.get_param(
            "~sdf_grid_resolution",
            self.plotter_cfg.get("sdf_grid_resolution", 0.1),
        )))
        self.sdf_grid_pub_hz = max(0.0, float(rospy.get_param(
            "~sdf_grid_pub_hz",
            self.plotter_cfg.get("sdf_grid_pub_hz", 1.0),
        )))
        self.sdf_grid_batch_size = max(1, int(rospy.get_param(
            "~sdf_grid_batch_size",
            self.plotter_cfg.get("sdf_grid_batch_size", 2048),
        )))
        if self.fast_viz_mode:
            # Dense SDF visualization is diagnostic; 1 Hz is enough and prevents
            # it from competing with the planner/ego callbacks.
            self.sdf_grid_pub_hz = min(float(self.sdf_grid_pub_hz), 1.0)
        # A failed vectorized call used to trigger O(nx*ny) Python/C++ singleton
        # calls, which can freeze this node for seconds.  The fast evaluator is
        # expected to be vectorized; fail fast by default instead of entering that
        # pathological fallback.  Legacy fallback can still be enabled explicitly.
        self.sdf_allow_singleton_fallback = _as_bool(rospy.get_param(
            "~sdf_allow_singleton_fallback",
            self.plotter_cfg.get("sdf_allow_singleton_fallback", False),
        ), False)
        # Optional RViz heat-map visualization of the same SDF grid.  No extra
        # SDF evaluation is performed: publish_current_sdf() reuses sdf_grid.
        self.publish_sdf_rviz = _as_bool(rospy.get_param(
            "~publish_sdf_rviz",
            self.plotter_cfg.get("publish_sdf_rviz", True),
        ), True)
        self.sdf_rviz_topic = str(rospy.get_param(
            "~sdf_rviz_topic",
            self.plotter_cfg.get("sdf_rviz_topic", "/sdf_rviz"),
        ))
        self.sdf_rviz_z = float(rospy.get_param(
            "~sdf_rviz_z",
            self.plotter_cfg.get("sdf_rviz_z", -0.02),
        ))
        self.sdf_rviz_alpha = min(1.0, max(0.0, float(rospy.get_param(
            "~sdf_rviz_alpha",
            self.plotter_cfg.get("sdf_rviz_alpha", 0.82),
        ))))
        # Absolute value used for color saturation.  Set <= 0 for automatic
        # per-frame scaling based on max(abs(Phi)).
        self.sdf_rviz_clip_abs = float(rospy.get_param(
            "~sdf_rviz_clip_abs",
            self.plotter_cfg.get("sdf_rviz_clip_abs", 1.0),
        ))
        self.sdf_rviz_stride = max(1, int(rospy.get_param(
            "~sdf_rviz_stride",
            self.plotter_cfg.get("sdf_rviz_stride", 4),
        )))
        self.sdf_zero_contour_width = max(1.0e-4, float(rospy.get_param(
            "~sdf_zero_contour_width",
            self.plotter_cfg.get("sdf_zero_contour_width", 0.025),
        )))
        self.sdf_grid_last_stamp = rospy.Time(0)
        if self.sdf_grid_max < self.sdf_grid_min:
            self.sdf_grid_min, self.sdf_grid_max = self.sdf_grid_max, self.sdf_grid_min
        # SDF grid axes are ego-relative offsets, not fixed world coordinates.
        # For example, [-5, 5] means a 10 m x 10 m window centered on the
        # current ego position.  World query coordinates are constructed as
        # ego_position + these offsets in publish_current_sdf().
        self.sdf_grid_x = np.arange(
            self.sdf_grid_min,
            self.sdf_grid_max + 0.5 * self.sdf_grid_resolution,
            self.sdf_grid_resolution,
            dtype=np.float64,
        )
        self.sdf_grid_y = np.arange(
            self.sdf_grid_min,
            self.sdf_grid_max + 0.5 * self.sdf_grid_resolution,
            self.sdf_grid_resolution,
            dtype=np.float64,
        )
        # The SDF grid shape never changes at runtime.  Precompute flattened local
        # offsets once instead of allocating meshgrid arrays on every SDF refresh.
        _gx_local, _gy_local = np.meshgrid(
            self.sdf_grid_x, self.sdf_grid_y, indexing="xy"
        )
        self.sdf_grid_x_flat = np.ascontiguousarray(_gx_local.reshape(-1))
        self.sdf_grid_y_flat = np.ascontiguousarray(_gy_local.reshape(-1))
        # Reuse one dense query-state buffer.  The grid shape is fixed, so there
        # is no reason to allocate a new (nx*ny)x4 array on every SDF refresh.
        self.sdf_grid_state_buffer = np.empty(
            (int(self.sdf_grid_x_flat.size), 4), dtype=np.float64
        )
        del _gx_local, _gy_local

        # RViz grid coordinates are fixed ego-relative offsets. Prebuild Point
        # objects and contour-cell metadata once instead of allocating them for
        # every SDF refresh. Only colors and zero-crossing points are dynamic.
        _stride = max(1, int(self.sdf_rviz_stride))
        _z = float(self.sdf_rviz_z)
        self._sdf_rviz_heat_points = []
        self._sdf_rviz_heat_indices = []
        for _iy in range(0, int(self.sdf_grid_y.size), _stride):
            _y = float(self.sdf_grid_y[_iy])
            for _ix in range(0, int(self.sdf_grid_x.size), _stride):
                _p = Point()
                _p.x = float(self.sdf_grid_x[_ix])
                _p.y = _y
                _p.z = _z
                self._sdf_rviz_heat_points.append(_p)
                self._sdf_rviz_heat_indices.append((_iy, _ix))
        self._sdf_rviz_contour_cells = []
        _ny = int(self.sdf_grid_y.size)
        _nx = int(self.sdf_grid_x.size)
        for _iy in range(0, max(0, _ny - _stride), _stride):
            _iy1 = min(_iy + _stride, _ny - 1)
            _y0 = float(self.sdf_grid_y[_iy])
            _y1 = float(self.sdf_grid_y[_iy1])
            for _ix in range(0, max(0, _nx - _stride), _stride):
                _ix1 = min(_ix + _stride, _nx - 1)
                self._sdf_rviz_contour_cells.append((
                    _iy, _iy1, _ix, _ix1,
                    float(self.sdf_grid_x[_ix]), float(self.sdf_grid_x[_ix1]),
                    _y0, _y1,
                ))

        if self.publish_sdf and _utils_linearized is not None:
            try:
                self.sdf_cfg, _, _ = _utils_linearized.load_planner_config(raw_config=self.raw_cfg)
            except Exception as exc:
                self.sdf_last_error = repr(exc)
                rospy.logerr("[plotter] failed to load SDF configuration: %s", self.sdf_last_error)
        elif self.publish_sdf:
            self.sdf_last_error = _UTILS_LINEARIZED_IMPORT_ERROR
            rospy.logerr(
                "[plotter] utils_linearized import failed; %s is disabled: %s",
                self.sdf_topic,
                self.sdf_last_error,
            )
        self.visualization_pub_hz = float(rospy.get_param(
            "~visualization_pub_hz",
            self.plotter_cfg.get(
                "visualization_pub_hz",
                self.plotter_cfg.get("all_visualization_pub_hz", 20.0),
            ),
        ))
        self.best_cmd_timeout_s = float(rospy.get_param("~best_cmd_timeout_s", self.plotter_cfg.get("best_cmd_timeout_s", 0.25)))

        # RViz cannot usefully render obstacle polygon MarkerArrays at the full
        # simulation rate.  Keep subscriber queues at 1 and throttle only the
        # visualization so old messages are dropped instead of displayed late.
        self.obs_polygon_viz_hz = float(rospy.get_param(
            "~obs_polygon_viz_hz",
            self.plotter_cfg.get("obs_polygon_viz_hz", 20.0),
        ))
        self.percepted_obs_polygon_viz_hz = float(rospy.get_param(
            "~percepted_obs_polygon_viz_hz",
            self.plotter_cfg.get("percepted_obs_polygon_viz_hz", 20.0),
        ))
        self.last_obs_polygon_viz_stamp = rospy.Time(0)
        self.last_percepted_obs_polygon_viz_stamp = rospy.Time(0)
        self.last_obs_polygon_horizon_viz_stamp = rospy.Time(0)
        self.last_static_viz_stamp = rospy.Time(0)
        self.last_path_viz_stamp = rospy.Time(0)
        self.last_visibility_prm_viz_stamp = rospy.Time(0)
        self.last_visibility_guidance_viz_stamp = rospy.Time(0)
        self.last_visibility_backward_prm_viz_stamp = rospy.Time(0)
        self.last_visibility_backward_guidance_viz_stamp = rospy.Time(0)

        # Heavy RViz messages are throttled independently from state updates.
        # State is still updated on every callback, but visualization is only
        # republished at these rates so RViz does not build a long render queue.
        # Keep the actual ego body responsive at the ego_mover rate, while
        # keeping heavier auxiliary ego visuals (FOV sector/sector rays/text)
        # throttled.  The previous single ego_viz_hz throttled the blue ego
        # rectangle to 20 Hz even though /ego_state is published at 100 Hz.
        self.ego_vehicle_viz_hz = float(rospy.get_param(
            "~ego_vehicle_viz_hz",
            self.plotter_cfg.get("ego_vehicle_viz_hz", 100.0),
        ))
        self.ego_aux_viz_hz = float(rospy.get_param(
            "~ego_aux_viz_hz",
            self.plotter_cfg.get("ego_aux_viz_hz", self.plotter_cfg.get("ego_viz_hz", 20.0)),
        ))
        # Backward-compatible alias: old configs that read ego_viz_hz still
        # refer to the heavier auxiliary ego visualization rate.
        self.ego_viz_hz = self.ego_aux_viz_hz
        self.sampled_traj_viz_hz = float(rospy.get_param(
            "~sampled_traj_viz_hz",
            self.plotter_cfg.get("sampled_traj_viz_hz", 20.0),
        ))
        self.last_ego_vehicle_viz_stamp = rospy.Time(0)
        self.last_ego_aux_viz_stamp = rospy.Time(0)
        # Keep the old attribute for compatibility with external debugging code.
        self.last_ego_viz_stamp = self.last_ego_aux_viz_stamp
        self.last_sampled_traj_viz_stamp = rospy.Time(0)
        self.fov_arc_steps_max = max(8, int(rospy.get_param(
            "~fov_arc_steps_max",
            self.plotter_cfg.get("fov_arc_steps_max", 48),
        )))
        self.max_sampled_traj_markers = int(rospy.get_param(
            "~max_sampled_traj_markers",
            self.plotter_cfg.get("max_sampled_traj_markers", 200),
        ))
        self.show_obstacle_positions = _as_bool(rospy.get_param(
            "~show_obstacle_positions",
            self.plotter_cfg.get("show_obstacle_positions", False),
        ), False)

        # Selected-vertex RViz ownership is centralized in plotter.py.
        # vertex_selector.py publishes only /selected_vertex_data; this node
        # converts that exact planner input into RViz point/normal markers.
        self.publish_selected_vertex_markers = _as_bool(rospy.get_param(
            "~publish_selected_vertex_markers",
            self.plotter_cfg.get(
                "publish_selected_vertex_markers",
                self.selector_cfg.get("publish_markers", True),
            ),
        ), True)
        self.selected_vertex_viz_hz = float(rospy.get_param(
            "~selected_vertex_viz_hz",
            self.plotter_cfg.get("selected_vertex_viz_hz", 20.0),
        ))
        self.selected_vertex_marker_width = float(rospy.get_param(
            "~selected_vertex_marker_width",
            self.plotter_cfg.get(
                "selected_vertex_marker_width",
                self.selector_cfg.get("selected_vertex_marker_width", 0.16),
            ),
        ))
        self.selected_vertex_normal_arrow_length = float(rospy.get_param(
            "~selected_vertex_normal_arrow_length",
            self.plotter_cfg.get(
                "selected_vertex_normal_arrow_length",
                self.selector_cfg.get("selected_vertex_normal_arrow_length", 0.35),
            ),
        ))
        self.selected_vertex_normal_shaft_diameter = float(rospy.get_param(
            "~selected_vertex_normal_shaft_diameter",
            self.plotter_cfg.get(
                "selected_vertex_normal_shaft_diameter",
                self.selector_cfg.get("selected_vertex_normal_shaft_diameter", 0.035),
            ),
        ))
        self.selected_vertex_normal_head_diameter = float(rospy.get_param(
            "~selected_vertex_normal_head_diameter",
            self.plotter_cfg.get(
                "selected_vertex_normal_head_diameter",
                self.selector_cfg.get("selected_vertex_normal_head_diameter", 0.10),
            ),
        ))
        self.selected_vertex_normal_head_length = float(rospy.get_param(
            "~selected_vertex_normal_head_length",
            self.plotter_cfg.get(
                "selected_vertex_normal_head_length",
                self.selector_cfg.get("selected_vertex_normal_head_length", 0.10),
            ),
        ))
        self.selected_vertex_color = _rgba(
            self.plotter_cfg.get(
                "selected_vertex_color",
                self.selector_cfg.get("selected_vertex_color", None),
            ),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.selected_vertex_normal_color = _rgba(
            self.plotter_cfg.get(
                "selected_vertex_normal_color",
                self.selector_cfg.get("selected_vertex_normal_color", None),
            ),
            (1.0, 0.0, 1.0, 1.0),
        )
        self.selected_vertex_normal_marker_mode = str(rospy.get_param(
            "~selected_vertex_normal_marker_mode",
            self.plotter_cfg.get(
                "selected_vertex_normal_marker_mode",
                self.selector_cfg.get("normal_marker_mode", "line_list"),
            ),
        )).strip().lower()
        self.last_selected_vertex_viz_stamp = rospy.Time(0)

        self.prev_obs_polygon_marker_cnt = 0
        self.prev_obs_polygon_horizon_marker_cnt = 0
        self.prev_obs_polygon_planning_horizon_marker_cnt = 0
        self.prev_percep_obs_polygon_marker_cnt = 0
        self.prev_percep_obs_polygon_horizon_marker_cnt = 0
        self.prev_sampled_marker_cnt = 0
        self.prev_selected_vertex_normal_arrow_cnt = 0
        self.fov_marker_initialized = False
        self.ego_sector_marker_initialized = False
        self.visibility_guide_marker_initialized = False
        self.prev_visibility_guide_count = 0
        self.visibility_backward_guide_marker_initialized = False
        self.prev_visibility_backward_guide_count = 0

        self.RRRs = list(range(0, 255))
        self.GGGs = list(range(0, 255))
        self.BBBs = list(range(0, 255))
        random.shuffle(self.RRRs)
        random.shuffle(self.GGGs)
        random.shuffle(self.BBBs)

        ros_cfg = self.raw_cfg.get("ros", {}) if isinstance(self.raw_cfg.get("ros", {}), dict) else {}
        input_topics = ros_cfg.get("input_topics", {}) if isinstance(ros_cfg.get("input_topics", {}), dict) else {}

        rospy.Subscriber("/determined_traj", Determined_traj, self.path_callback, queue_size=1)
        # Runtime/configuration goal authority belongs exclusively to setter -> /setup.
        # No /ego_state subscription: ego pose is read from odom <- base_link TF.
        self.runtime_goal_sub = rospy.Subscriber(
            "/ego_goal_set", Ego_goal_setting, self.goal_callback, queue_size=1
        )

        # obstacle_polygon.msg is used for 2-D polygons.
        rospy.Subscriber(str(input_topics.get("obstacle", "/obstacle")), obstacle_polygon, self.obs_callback, queue_size=1)

        selector_pubs = self.selector_cfg.get("publishers", {}) if isinstance(self.selector_cfg.get("publishers", {}), dict) else {}
        self.percepted_obstacle_topic = str(rospy.get_param(
            "~percepted_obstacle_topic",
            self.plotter_cfg.get(
                "percepted_obstacle_topic",
                selector_pubs.get("percepted_obstacle", "/percepted_obstacle"),
            ),
        ))
        self.selected_vertex_data_topic = str(rospy.get_param(
            "~selected_vertex_data_topic",
            self.plotter_cfg.get(
                "selected_vertex_data_topic",
                selector_pubs.get("selected_vertex_data", "/selected_vertex_data"),
            ),
        ))
        self.selected_vertex_horizon_data_topic = str(rospy.get_param(
            "~selected_vertex_horizon_data_topic",
            self.plotter_cfg.get(
                "selected_vertex_horizon_data_topic",
                selector_pubs.get("selected_vertex_horizon_data", "/selected_vertex_horizon_data"),
            ),
        ))
        self.selected_vertex_normal_orientation = str(
            self.selector_cfg.get(
                "normal_orientation",
                self.normal_cfg.get("normal_orientation", "inward"),
            )
        ).strip().lower()
        self.selected_vertex_set_topic = str(rospy.get_param(
            "~selected_vertex_set_topic",
            self.plotter_cfg.get(
                "selected_vertex_set_topic",
                selector_pubs.get("selected_vertex_set", "/selected_vertex_set"),
            ),
        ))
        self.selected_vertex_normal_topic = str(rospy.get_param(
            "~selected_vertex_normal_topic",
            self.plotter_cfg.get(
                "selected_vertex_normal_topic",
                selector_pubs.get("selected_vertex_normal", "/selected_vertex_normal"),
            ),
        ))
        # Enable this if your sampler publishes this topic.
        rospy.Subscriber("/sampled_traj", Sampled_traj, self.sampled_traj_callback, queue_size=1)

        # Final selected trajectory from ego_mover.  Payload is Float64MultiArray
        # with rows [x, y, v, yaw]; only x/y are required for RViz drawing.
        self.selected_traj_topic = str(rospy.get_param(
            "~selected_traj_topic",
            self.plotter_cfg.get("selected_traj_topic", "/selected_traj"),
        ))
        self.selected_traj_marker_topic = str(rospy.get_param(
            "~selected_traj_marker_topic",
            self.plotter_cfg.get("selected_traj_marker_topic", "/selected_traj_marker"),
        ))
        self.selected_traj_viz_hz = float(rospy.get_param(
            "~selected_traj_viz_hz",
            self.plotter_cfg.get("selected_traj_viz_hz", 20.0),
        ))
        self.selected_traj_marker_width = float(rospy.get_param(
            "~selected_traj_marker_width",
            self.plotter_cfg.get("selected_traj_marker_width", 0.16),
        ))
        self.selected_traj_marker_z = float(rospy.get_param(
            "~selected_traj_marker_z",
            self.plotter_cfg.get("selected_traj_marker_z", 0.08),
        ))
        self.selected_traj_color = _rgba(
            self.plotter_cfg.get("selected_traj_color", None),
            (1.0, 1.0, 1.0, 1.0),
        )
        self.last_selected_traj_viz_stamp = rospy.Time(0)
        self.selected_traj_sub = rospy.Subscriber(
            self.selected_traj_topic, Float64MultiArray,
            self.selected_traj_callback, queue_size=1, tcp_nodelay=True,
        )

        control_topics = _as_string_list(self.plotter_cfg.get("mpc_control_seq_topics", None))
        planners_cfg = self.raw_cfg.get("planners", {}) if isinstance(self.raw_cfg.get("planners", {}), dict) else {}
        for block in planners_cfg.values():
            if not isinstance(block, dict):
                continue
            pubs = block.get("publishers", {}) if isinstance(block.get("publishers", {}), dict) else {}
            for topic in _as_string_list(pubs.get("mpc_control_seq", None)):
                if topic not in control_topics:
                    control_topics.append(topic)
        if not control_topics:
            # Planner default is the shared /mpc_control_seq topic.  Also listen
            # to the legacy indexed topics so either launch/config convention works.
            control_topics = ["/mpc_control_seq"] + ["/mpc_control_seq%d" % i for i in range(1, 10)]

        # The planners no longer publish /mpc_pred_traj*.  Plotter owns all
        # predicted-trajectory RViz output by rolling out the received
        # MPCControlSeq messages and publishing one MarkerArray.
        dedup_control_topics = []
        for topic in control_topics:
            topic = str(topic).strip()
            if topic and topic not in dedup_control_topics:
                dedup_control_topics.append(topic)

        # Keep every deduplicated configured topic.  Planner identity is recovered
        # from MPCControlSeq.seq_id, so a shared topic and indexed topics can coexist.
        self.mpc_control_seq_topics = dedup_control_topics
        # Keep Stage-1 (avoidance) and Stage-2 (mitigation) predictions
        # separately for each planner.  /mpc_pred_trajs contains exactly one
        # selected trajectory per planner: use the newest mitigation when it
        # belongs to the current planner update, otherwise fall back to avoidance.
        self.mpc_pred_candidates = {}
        # Backward-compatible cache of the currently selected marker per planner.
        self.mpc_pred_markers = {}
        self.mpc_pred_marker_width = float(rospy.get_param(
            "~mpc_pred_marker_width",
            self.plotter_cfg.get("mpc_pred_marker_width", 0.10),
        ))
        default_pred_colors = [
            (0.0, 1.0, 0.0, 1.0),   # planner1
            (1.0, 0.5, 0.0, 1.0),   # planner2
            (1.0, 0.0, 1.0, 1.0),   # planner3
            (0.0, 1.0, 1.0, 1.0),   # planner4
            (0.5, 0.0, 1.0, 1.0),   # planner5
            (1.0, 1.0, 0.0, 1.0),   # planner6
            (0.0, 0.55, 1.0, 1.0),  # planner7
            (1.0, 0.0, 0.0, 1.0),   # planner8
            (0.25, 1.0, 0.25, 1.0), # planner9
        ]
        cfg_pred_colors = self.plotter_cfg.get("mpc_pred_traj_colors", None)
        self.mpc_pred_colors = []
        for i, default_color in enumerate(default_pred_colors):
            cfg_color = None
            if isinstance(cfg_pred_colors, (list, tuple)) and i < len(cfg_pred_colors):
                cfg_color = cfg_pred_colors[i]
            self.mpc_pred_colors.append(_rgba(cfg_color, default_color))

        # Visibility-PRM assigns colors to the raw guidance candidates before
        # publishing /initial_guidance*.  Keep a separate palette for those
        # topics instead of reusing mpc_pred_traj_colors, so the dotted initial
        # guidance and the solid initial_traj keep the same candidate color.
        default_initial_guidance_colors = [
            (1.0, 0.2, 0.2, 1.0),   # candidate 1: red
            (0.0, 1.0, 0.0, 1.0),   # candidate 2: green
            (0.2, 0.6, 1.0, 1.0),   # candidate 3: blue
            (1.0, 1.0, 0.2, 1.0),   # candidate 4: yellow
            (1.0, 0.2, 1.0, 1.0),   # candidate 5: magenta
            (0.2, 1.0, 1.0, 1.0),   # candidate 6: cyan
            (1.0, 0.5, 0.0, 1.0),   # candidate 7: orange
            (0.6, 0.2, 1.0, 1.0),   # candidate 8: purple
            (0.6, 0.4, 0.2, 1.0),   # candidate 9: brown
            (1.0, 0.6, 0.8, 1.0),   # candidate 10: pink
            (0.5, 1.0, 0.0, 1.0),   # candidate 11: lime
            (0.5, 0.5, 0.5, 1.0),   # candidate 12: gray
        ]
        cfg_guidance_colors = self.plotter_cfg.get(
            "initial_guidance_colors",
            self.plotter_cfg.get("initial_guidance_traj_colors", None),
        )
        self.initial_guidance_colors = []
        palette_len = len(default_initial_guidance_colors)
        if isinstance(cfg_guidance_colors, (list, tuple)) and len(cfg_guidance_colors) > 0:
            palette_len = max(palette_len, len(cfg_guidance_colors))
        for i in range(palette_len):
            default_color = default_initial_guidance_colors[i % len(default_initial_guidance_colors)]
            cfg_color = None
            if isinstance(cfg_guidance_colors, (list, tuple)) and i < len(cfg_guidance_colors):
                cfg_color = cfg_guidance_colors[i]
            self.initial_guidance_colors.append(_rgba(cfg_color, default_color))
        self.initial_guidance_color_cycle = max(1, int(rospy.get_param(
            "~initial_guidance_color_cycle",
            self.plotter_cfg.get("initial_guidance_color_cycle", 3),
        )))
        self.use_initial_guidance_colors_for_mpc_pred = _as_bool(rospy.get_param(
            "~use_initial_guidance_colors_for_mpc_pred",
            self.plotter_cfg.get("use_initial_guidance_colors_for_mpc_pred", True),
        ), True)

        self.control_seq_subs = []
        for idx, topic in enumerate(self.mpc_control_seq_topics):
            self.control_seq_subs.append(
                rospy.Subscriber(topic, MPCControlSeq, self.mpc_control_callback, callback_args=idx + 1, queue_size=1, tcp_nodelay=True, buff_size=4 * 1024 * 1024)
            )
        self.mpc_pred_traj_subs = []

        # Visibility-PRM publishes initial guidance as MPCControlSeq on
        # /initial_guidance1..N.  MPC planners may publish the optimized/warm-
        # start-updated trajectory as visualization_msgs/Marker on
        # /initial_traj1..N.  Plot both with the same per-planner color:
        #   - initial guidance: dotted/dashed LINE_LIST
        #   - updated trajectory: solid LINE_STRIP
        self.initial_guidance_topics = self.collect_indexed_topics(
            plural_key="initial_guidance_topics",
            singular_key="initial_guidance_topic",
            publisher_key="initial_guidance",
            fallback_prefix="/initial_guidance",
            default_count=max(9, len(self.mpc_control_seq_topics)),
        )
        self.initial_traj_topics = self.collect_indexed_topics(
            plural_key="initial_traj_topics",
            singular_key="initial_traj_topic",
            publisher_key="initial_traj",
            fallback_prefix="/initial_traj",
            default_count=max(9, len(self.mpc_control_seq_topics)),
        )
        self.initial_guidance_markers = {}
        self.initial_traj_markers = {}
        # Cache the actual MPC-grid waypoints carried by /initial_guidance*.
        # These are arc-length re-sampled points (xs, ys, ts), distinct from
        # the dense B-spline samples carried by /visibility_prm_*/guidances.
        self.visibility_initial_guidance_waypoints = {}

        # High-rate, low-overhead RViz publishing. Subscriber callbacks only
        # replace the latest marker for their planner and set a dirty flag. A
        # single timer publishes each complete MarkerArray at most once per
        # flush tick, avoiding up to N repeated serializations when N planner
        # messages arrive together. The default 100 Hz keeps added latency
        # below 10 ms while substantially reducing callback blocking.
        self.fast_marker_pub_hz = float(rospy.get_param(
            "~fast_marker_pub_hz",
            self.plotter_cfg.get("fast_marker_pub_hz", 100.0),
        ))
        self.fast_marker_lock = threading.Lock()
        self.mpc_pred_markers_dirty = False
        self.initial_guidance_markers_dirty = False
        self.initial_traj_markers_dirty = False
        self.initial_guidance_marker_width = float(rospy.get_param(
            "~initial_guidance_marker_width",
            self.plotter_cfg.get("initial_guidance_marker_width", max(0.5 * float(self.mpc_pred_marker_width), 0.03)),
        ))
        self.initial_traj_marker_width = float(rospy.get_param(
            "~initial_traj_marker_width",
            self.plotter_cfg.get("initial_traj_marker_width", float(self.mpc_pred_marker_width)),
        ))
        self.initial_guidance_lifetime_s = float(rospy.get_param(
            "~initial_guidance_lifetime_s",
            self.plotter_cfg.get("initial_guidance_lifetime_s", max(float(self.best_cmd_timeout_s), 0.25)),
        ))
        self.initial_traj_lifetime_s = float(rospy.get_param(
            "~initial_traj_lifetime_s",
            self.plotter_cfg.get("initial_traj_lifetime_s", max(float(self.best_cmd_timeout_s), 0.25)),
        ))
        self.initial_guidance_subs = []
        for idx, topic in enumerate(self.initial_guidance_topics):
            if topic:
                self.initial_guidance_subs.append(
                    rospy.Subscriber(str(topic), MPCControlSeq, self.initial_guidance_callback, callback_args=idx + 1, queue_size=1, tcp_nodelay=True, buff_size=4 * 1024 * 1024)
                )
        self.initial_traj_subs = []
        for idx, topic in enumerate(self.initial_traj_topics):
            if topic:
                self.initial_traj_subs.append(
                    rospy.Subscriber(str(topic), Marker, self.initial_traj_callback, callback_args=idx + 1, queue_size=1, tcp_nodelay=True, buff_size=4 * 1024 * 1024)
                )

        self.trajectory_pub = rospy.Publisher(
            "/mpc_path_cloud", Marker, queue_size=1
        )
        self.ego_pub = rospy.Publisher("/ego_position", Marker, queue_size=1)
        self.ego_FOV_pub = rospy.Publisher("/ego_FOV", Marker, queue_size=1)
        self.ego_sector_pub = rospy.Publisher(
            self.ego_sector_topic, MarkerArray, queue_size=1
        )
        self.goal_pub = rospy.Publisher("/goal_position", Marker, queue_size=1)
        self.boundary_pub = rospy.Publisher("/boundary", Marker, queue_size=1)
        # /sdf contains a flattened [ny, nx] SDF grid.
        # /sdf_grid_x and /sdf_grid_y contain the corresponding fixed axes.
        self.sdf_pub = rospy.Publisher(self.sdf_topic, Float64MultiArray, queue_size=1)
        self.sdf_grid_x_pub = rospy.Publisher(
            self.sdf_grid_x_topic, Float64MultiArray, queue_size=1, latch=True
        )
        self.sdf_grid_y_pub = rospy.Publisher(
            self.sdf_grid_y_topic, Float64MultiArray, queue_size=1, latch=True
        )
        self.sdf_rviz_pub = rospy.Publisher(
            self.sdf_rviz_topic, MarkerArray, queue_size=1
        )
        self.publish_sdf_grid_axes()

        # Current-step (h=0) obstacle-induced impact-normal comparison.
        # GT uses the exact minimum distance on the same current geometry used by
        # the selected field, while the estimate is -grad_p(Phi)/||grad_p(Phi)||.
        self.publish_impact_normal_comparison = _as_bool(rospy.get_param(
            "~publish_impact_normal_comparison",
            self.plotter_cfg.get("publish_impact_normal_comparison", True),
        ), True)
        self.impact_normal_topic = str(rospy.get_param(
            "~impact_normal_topic",
            self.plotter_cfg.get("impact_normal_topic", "/impact_normal_comparison"),
        ))
        self.impact_normal_viz_hz = max(0.0, float(rospy.get_param(
            "~impact_normal_viz_hz",
            self.plotter_cfg.get("impact_normal_viz_hz", 5.0),
        )))
        if self.fast_viz_mode:
            # Heavy diagnostic displays do not need to track the 20/100 Hz state
            # streams. These caps cut Python marker construction/serialization
            # substantially while keeping the ego body visualization responsive.
            self.safe_margin_horizon_viz_hz = min(float(self.safe_margin_horizon_viz_hz), 5.0)
            self.static_viz_hz = min(float(self.static_viz_hz), 1.0)
            self.path_viz_hz = min(float(self.path_viz_hz), 10.0)
            self.visibility_viz_hz = min(float(self.visibility_viz_hz), 5.0)
            self.obs_polygon_viz_hz = min(float(self.obs_polygon_viz_hz), 10.0)
            self.percepted_obs_polygon_viz_hz = min(float(self.percepted_obs_polygon_viz_hz), 10.0)
            self.sampled_traj_viz_hz = min(float(self.sampled_traj_viz_hz), 10.0)
            self.selected_vertex_viz_hz = min(float(self.selected_vertex_viz_hz), 10.0)
            self.impact_normal_viz_hz = min(float(self.impact_normal_viz_hz), 5.0)
        self.impact_normal_arrow_length = max(1.0e-3, float(rospy.get_param(
            "~impact_normal_arrow_length",
            self.plotter_cfg.get("impact_normal_arrow_length", 0.75),
        )))
        self.impact_normal_fd_eps_m = max(1.0e-5, float(rospy.get_param(
            "~impact_normal_fd_eps_m",
            self.plotter_cfg.get(
                "impact_normal_fd_eps_m",
                getattr(self.sdf_cfg, "fd_position_eps_m", 1.0e-4) if self.sdf_cfg is not None else 1.0e-4,
            ),
        )))
        self.impact_normal_pub = rospy.Publisher(
            self.impact_normal_topic, MarkerArray, queue_size=1
        )
        self.last_impact_normal_viz_stamp = rospy.Time(0)

        self.obs_pos_pub = rospy.Publisher("/obs_positions", Marker, queue_size=1)
        self.percep_obs_pos_pub = rospy.Publisher(
            "/percepted_obs_positions", Marker, queue_size=1
        )

        # Split obstacle visualization into two MarkerArray topics:
        #   /obs_polygons: current-time filled ground-truth polygons only.
        #   /obs_polygons_horizon: planning-horizon polygons reconstructed ONLY
        #     from vertex_selector outputs, with plotter-side probability inflation
        #     applied from the /setup sigma_vx/sigma_vy/pc values.
        self.obs_polygon_pub = rospy.Publisher(
            "/obs_polygons", MarkerArray, queue_size=1
        )
        self.obs_polygon_horizon_pub = rospy.Publisher(
            "/obs_polygons_horizon", MarkerArray, queue_size=1
        )
        self.dynamic_obstacles_topic = str(rospy.get_param(
            "~dynamic_obstacles_topic", "/dynamic_obstacles"))
        self.dynamic_obstacles_sub = rospy.Subscriber(
            self.dynamic_obstacles_topic, obstacle_drone,
            self.dynamic_obstacles_callback, queue_size=1, tcp_nodelay=True
        )
        # /obs_polygons_horizon is the single authoritative planning-horizon
        # obstacle visualization topic.
        self.obs_polygon_planning_horizon_pub = None
        self.obs_planning_horizon_pub = rospy.Publisher(
            "/obs_planning_horizon", MarkerArray, queue_size=1
        )
        self.percep_obs_polygon_pub = rospy.Publisher(
            "/percepted_obs_polygons", MarkerArray, queue_size=1
        )
        self.percep_obs_polygon_horizon_pub = rospy.Publisher(
            "/percepted_obs_polygons_horizon", MarkerArray, queue_size=1
        )
        self.selected_vertex_set_pub = rospy.Publisher(
            self.selected_vertex_set_topic, Marker, queue_size=1
        )
        self.selected_vertex_normal_pub = rospy.Publisher(
            self.selected_vertex_normal_topic, MarkerArray, queue_size=1
        )

        # MPC_planner0 publishes the exact affine avoidance model; plotter owns
        # the RViz MarkerArray on /mpc_separating_hyperplanes.
        self.separating_hyperplane_data_topic = "/mpc_separating_hyperplane_data0"
        self.separating_hyperplane_topic = "/mpc_separating_hyperplanes"
        self.separating_hyperplane_pub = rospy.Publisher(
            self.separating_hyperplane_topic, MarkerArray, queue_size=1
        )
        self.separating_hyperplane_sub = rospy.Subscriber(
            self.separating_hyperplane_data_topic, Float64MultiArray,
            self.separating_hyperplane_callback, queue_size=1, tcp_nodelay=True
        )
        self.separating_hyperplane_line_width = max(
            float(self.safe_margin_line_width) * 1.8, 0.035
        )
        self.separating_hyperplane_color = (1.0, 0.15, 0.85, 0.95)
        # k=0 smooth-Phi feasible half-plane: magenta fill.
        self.separating_hyperplane_fill_color = (1.0, 0.15, 0.85, 0.20)
        # k=0 exact minimum-distance tangent: cyan line + cyan feasible fill.
        self.exact_min_distance_hyperplane_color = (0.0, 0.95, 0.95, 1.0)
        self.exact_min_distance_hyperplane_fill_color = (0.0, 0.95, 0.95, 0.20)
        self.exact_min_distance_hyperplane_line_width = max(
            float(self.separating_hyperplane_line_width) * 1.15, 0.04
        )
        self.percepted_obstacle_sub = rospy.Subscriber(
            self.percepted_obstacle_topic,
            obstacle_polygon,
            self.percepted_obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.selected_vertex_data_sub = rospy.Subscriber(
            self.selected_vertex_data_topic,
            obstacle_polygon,
            self.selected_vertex_data_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.selected_vertex_horizon_data_sub = rospy.Subscriber(
            self.selected_vertex_horizon_data_topic,
            Float64MultiArray,
            self.selected_vertex_horizon_data_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.sampled_traj_pub = rospy.Publisher(
            "/sampled_traj_markers", MarkerArray, queue_size=1
        )
        # Highlight the final ego_mover-selected trajectory in RViz.
        self.selected_traj_marker_pub = rospy.Publisher(
            self.selected_traj_marker_topic, Marker, queue_size=1, latch=True
        )
        self.mpc_pred_traj_pubs = rospy.Publisher(
            "/mpc_pred_trajs", MarkerArray, queue_size=1, tcp_nodelay=True
        )
        self.initial_guidance_traj_pub = rospy.Publisher(
            str(self.plotter_cfg.get("initial_guidance_output_topic", "/initial_guidance_trajs")),
            MarkerArray,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.initial_traj_pub = rospy.Publisher(
            str(self.plotter_cfg.get("initial_traj_output_topic", "/initial_trajs")),
            MarkerArray,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Backward-compatible alias for older helper code.
        self.mpc_pred_trajs_pub = self.mpc_pred_traj_pubs

        # Fast marker arrays are also flushed by the common fixed-rate
        # publication loop.  Do not run a separate timer, otherwise these
        # topics would be published at a different phase/rate.
        self.fast_marker_pub_timer = None
        # Visibility-PRM RViz ownership.  The planner publishes only Guidance data;
        # this node reconstructs the old private MarkerArray/Path outputs so RViz
        # configurations can keep using the same absolute topic names.
        self.visibility_z_scale = float(rospy.get_param(
            "~visibility_z_scale", self.plotter_cfg.get("visibility_z_scale", TIME_SCALE)
        ))
        # Diameter [m] of MPC-grid waypoint spheres drawn together with each
        # visibility-guidance B-spline line.
        self.visibility_guide_waypoint_scale = max(1.0e-4, float(rospy.get_param(
            "~visibility_guide_waypoint_scale",
            self.plotter_cfg.get("visibility_guide_waypoint_scale", 0.12),
        )))
        self.visibility_guidance_topic = str(rospy.get_param(
            "~visibility_guidance_topic",
            self.plotter_cfg.get("visibility_guidance_topic", "/visibility_prm_forward/guidances"),
        ))
        self.visibility_prm_graph_topic = str(rospy.get_param(
            "~visibility_prm_graph_topic",
            self.plotter_cfg.get("visibility_prm_graph_topic", "/visibility_prm_forward/prm_graph_data"),
        ))
        self.visibility_prm_marker_topic = str(rospy.get_param(
            "~visibility_prm_marker_topic",
            self.plotter_cfg.get("visibility_prm_marker_topic", "/visibility_prm_forward/prm_markers"),
        ))
        self.visibility_guide_marker_topic = str(rospy.get_param(
            "~visibility_guide_marker_topic",
            self.plotter_cfg.get("visibility_guide_marker_topic", "/visibility_prm_forward/guide_markers"),
        ))
        self.visibility_guide_path_prefix = str(rospy.get_param(
            "~visibility_guide_path_prefix",
            self.plotter_cfg.get("visibility_guide_path_prefix", "/visibility_prm_forward/guide_path_"),
        ))
        self.visibility_guide_path_count = max(1, int(rospy.get_param(
            "~visibility_guide_path_count",
            self.plotter_cfg.get("visibility_guide_path_count", 6),
        )))
        self.visibility_prm_pub = rospy.Publisher(
            self.visibility_prm_marker_topic, MarkerArray, queue_size=1, tcp_nodelay=True
        )
        self.visibility_guide_marker_pub = rospy.Publisher(
            self.visibility_guide_marker_topic, MarkerArray, queue_size=1, tcp_nodelay=True
        )
        self.visibility_guide_path_pubs = [
            rospy.Publisher("%s%d" % (self.visibility_guide_path_prefix, i), Path, queue_size=1, tcp_nodelay=True)
            for i in range(self.visibility_guide_path_count)
        ]
        self.visibility_prm_sub = rospy.Subscriber(
            self.visibility_prm_graph_topic,
            Guidance,
            self.visibility_prm_graph_callback,
            queue_size=1,
            tcp_nodelay=True,
            buff_size=32 * 1024 * 1024,
        )
        self.visibility_guidance_sub = rospy.Subscriber(
            self.visibility_guidance_topic,
            Guidance,
            self.visibility_guidance_callback,
            queue_size=1,
            tcp_nodelay=True,
            buff_size=16 * 1024 * 1024,
        )

        # Backward Visibility-PRM uses its own private ROS namespace.  Mirror the
        # forward visualization pipeline so both branches are visible in RViz.
        self.visibility_backward_guidance_topic = str(rospy.get_param(
            "~visibility_backward_guidance_topic",
            self.plotter_cfg.get("visibility_backward_guidance_topic", "/visibility_prm_backward/guidances"),
        ))
        self.visibility_backward_prm_graph_topic = str(rospy.get_param(
            "~visibility_backward_prm_graph_topic",
            self.plotter_cfg.get("visibility_backward_prm_graph_topic", "/visibility_prm_backward/prm_graph_data"),
        ))
        self.visibility_backward_prm_marker_topic = str(rospy.get_param(
            "~visibility_backward_prm_marker_topic",
            self.plotter_cfg.get("visibility_backward_prm_marker_topic", "/visibility_prm_backward/prm_markers"),
        ))
        self.visibility_backward_guide_marker_topic = str(rospy.get_param(
            "~visibility_backward_guide_marker_topic",
            self.plotter_cfg.get("visibility_backward_guide_marker_topic", "/visibility_prm_backward/guide_markers"),
        ))
        self.visibility_backward_guide_path_prefix = str(rospy.get_param(
            "~visibility_backward_guide_path_prefix",
            self.plotter_cfg.get("visibility_backward_guide_path_prefix", "/visibility_prm_backward/guide_path_"),
        ))
        self.visibility_backward_guide_path_count = max(1, int(rospy.get_param(
            "~visibility_backward_guide_path_count",
            self.plotter_cfg.get("visibility_backward_guide_path_count", 3),
        )))
        self.visibility_backward_prm_pub = rospy.Publisher(
            self.visibility_backward_prm_marker_topic, MarkerArray, queue_size=1, tcp_nodelay=True
        )
        self.visibility_backward_guide_marker_pub = rospy.Publisher(
            self.visibility_backward_guide_marker_topic, MarkerArray, queue_size=1, tcp_nodelay=True
        )
        self.visibility_backward_guide_path_pubs = [
            rospy.Publisher("%s%d" % (self.visibility_backward_guide_path_prefix, i), Path, queue_size=1, tcp_nodelay=True)
            for i in range(self.visibility_backward_guide_path_count)
        ]
        self.visibility_backward_prm_sub = rospy.Subscriber(
            self.visibility_backward_prm_graph_topic,
            Guidance,
            self.visibility_backward_prm_graph_callback,
            queue_size=1,
            tcp_nodelay=True,
            buff_size=32 * 1024 * 1024,
        )
        self.visibility_backward_guidance_sub = rospy.Subscriber(
            self.visibility_backward_guidance_topic,
            Guidance,
            self.visibility_backward_guidance_callback,
            queue_size=1,
            tcp_nodelay=True,
            buff_size=16 * 1024 * 1024,
        )
        # Keep visualization configuration synchronized with setter.py. /setup
        # is latched, so this callback receives the current configuration and
        # every future setter republish without reading JSON in this node.
        self.setup_sub = rospy.Subscriber(
            "/setup", Setup, self.setup_callback, queue_size=1, tcp_nodelay=True
        )

        rospy.loginfo(
            "[plotter] impact sector RViz: enabled=%s show=%s topic=%s sector_id=%d color=orange",
            str(bool(self.enable_impact_normal_regulation)),
            str(bool(self.show_ego_sector)),
            self.ego_sector_topic,
            int(self.sector_id),
        )
        rospy.loginfo(
            "[plotter] selected-vertex RViz: input=%s points=%s normals=%s mode=%s viz_hz=%.1f",
            self.selected_vertex_data_topic,
            self.selected_vertex_set_topic,
            self.selected_vertex_normal_topic,
            self.selected_vertex_normal_marker_mode,
            float(self.selected_vertex_viz_hz),
        )
        if self.publish_sdf and self.sdf_cfg is not None:
            rospy.loginfo(
                "[plotter] SDF grid publisher: topic=%s x=%s y=%s "
                "range=[%.3f, %.3f] resolution=%.3f size=%dx%d hz=%.2f "
                "field_flag=%d field=%s geometry_source=%s rviz=%s",
                self.sdf_topic,
                self.sdf_grid_x_topic,
                self.sdf_grid_y_topic,
                float(self.sdf_grid_min),
                float(self.sdf_grid_max),
                float(self.sdf_grid_resolution),
                int(self.sdf_grid_x.size),
                int(self.sdf_grid_y.size),
                float(self.sdf_grid_pub_hz),
                int(_utils_linearized.collision_field_flag(self.sdf_cfg)),
                _utils_linearized.collision_field_name(self.sdf_cfg),
                self.percepted_obstacle_topic,
                self.sdf_rviz_topic if bool(self.publish_sdf_rviz) else "disabled",
            )


        # The transform wraps Marker/MarkerArray/Path outputs, not MPC data.
        self.install_world_visualization_publishers()

    def install_world_visualization_publishers(self):
        """Wrap only RViz publishers; never transform planner state/data topics."""
        projector = _RVizWorldProjector(self.tf_buffer, self.rviz_target_frame)
        wrapped = 0
        for name, value in list(vars(self).items()):
            if getattr(value, "data_class", None) in (Marker, MarkerArray, Path):
                setattr(self, name, _ProjectedPublisher(value, projector))
                wrapped += 1
            elif isinstance(value, list) and value and all(
                    getattr(pub, "data_class", None) in (Marker, MarkerArray, Path)
                    for pub in value):
                setattr(self, name, [_ProjectedPublisher(pub, projector) for pub in value])
                wrapped += len(value)
        rospy.loginfo("[plotter] %d RViz publishers output in %s (planning remains base_link)",
                      wrapped, self.rviz_target_frame)


    def setup_callback(self, msg):
        """Apply setter-owned /setup configuration to visualization state."""
        raw = setup_msg_to_raw(msg)
        if not isinstance(raw, dict):
            return

        cfg = raw.get("config", {}) if isinstance(raw.get("config", {}), dict) else {}
        normal_cfg = raw.get("normal_cfg", {}) if isinstance(raw.get("normal_cfg", {}), dict) else {}
        plotter_cfg = raw.get("plotter", {}) if isinstance(raw.get("plotter", {}), dict) else {}
        mpc_cfg = raw.get("mpc", {}) if isinstance(raw.get("mpc", {}), dict) else {}
        selector_cfg = raw.get("vertex_selector", {}) if isinstance(raw.get("vertex_selector", {}), dict) else {}
        ego_cfg = raw.get("ego", {}) if isinstance(raw.get("ego", {}), dict) else {}

        old_L = float(getattr(self, "ego_L", 0.0))
        old_W = float(getattr(self, "ego_W", 0.0))

        # From this point on, every configuration lookup uses the latest Setup.msg.
        self.raw_cfg = raw
        self.config_cfg = cfg
        self.normal_cfg = normal_cfg
        self.plotter_cfg = plotter_cfg
        self.selector_cfg = selector_cfg

        self.x_min = float(cfg.get("x_min", getattr(self, "x_min", 0.0)))
        self.x_max = float(cfg.get("x_max", getattr(self, "x_max", 20.0)))
        self.y_min = float(cfg.get("y_min", getattr(self, "y_min", 0.0)))
        self.y_max = float(cfg.get("y_max", getattr(self, "y_max", 20.0)))
        if not self.runtime_goal_received:
            self.goal_x = float(_cfg_get(cfg, "x_end", "goal_x", getattr(self, "goal_x", 0.0)))
            self.goal_y = float(_cfg_get(cfg, "y_end", "goal_y", getattr(self, "goal_y", 0.0)))
        self.moving_obstacle_R = float(
            mpc_cfg.get("moving_obstacle_R", getattr(self, "moving_obstacle_R", 0.2))
        )
        self.ego_R = float(mpc_cfg.get("ego_R", getattr(self, "ego_R", 0.2)))

        self.ego_L = float(ego_cfg.get("ego_L", getattr(self, "ego_L", 0.0)))
        self.ego_W = float(ego_cfg.get("ego_W", getattr(self, "ego_W", 0.0)))
        self.ego_major_axis_m, self.ego_minor_axis_m, self.ego_axis_scale = (
            _ellipse_axes_from_lengths(self.ego_L, self.ego_W)
        )

        self.use_ego_scaled_local_frame = _as_bool(
            normal_cfg.get(
                "use_ego_scaled_local_frame",
                getattr(self, "use_ego_scaled_local_frame", True),
            ),
            True,
        )
        self.ego_FOV = float(
            _cfg_get(cfg, "FOV", "obs_sensing_R", getattr(self, "ego_FOV", 1.0))
        )
        self.obs_fov_deg = float(
            _cfg_get(
                cfg, "obs_fov_deg", "obs_fov_angle_deg",
                getattr(self, "obs_fov_deg", 80.0),
            )
        )
        self.sector_id = _normalize_sector_id(
            cfg.get("sector_id", getattr(self, "sector_id", 1))
        )
        self.show_ego_sector = _as_bool(
            plotter_cfg.get("show_ego_sector", getattr(self, "show_ego_sector", True)),
            True,
        )
        self.enable_impact_normal_regulation = _as_bool(
            mpc_cfg.get(
                "enable_impact_normal_constraint",
                getattr(self, "enable_impact_normal_regulation", True),
            ),
            True,
        )
        self.ego_sector_radius_m = max(
            0.0,
            float(plotter_cfg.get(
                "ego_sector_radius_m", getattr(self, "ego_sector_radius_m", 0.55)
            )),
        )
        self.ego_sector_arc_steps = max(
            4,
            int(plotter_cfg.get(
                "ego_sector_arc_steps", getattr(self, "ego_sector_arc_steps", 24)
            )),
        )
        self.ego_sector_fill_color = _rgba(
            plotter_cfg.get("ego_sector_fill_color", None),
            getattr(self, "ego_sector_fill_color", (1.0, 0.5, 0.0, 0.34)),
        )
        self.ego_sector_outline_color = _rgba(
            plotter_cfg.get("ego_sector_outline_color", None),
            getattr(self, "ego_sector_outline_color", (1.0, 0.5, 0.0, 0.95)),
        )

        # SDF visualization must use the same received setup, including ego size.
        if bool(getattr(self, "publish_sdf", False)) and _utils_linearized is not None:
            try:
                self.sdf_cfg, _, _ = _utils_linearized.load_planner_config(
                    raw_config=self.raw_cfg
                )
                self.sdf_last_error = None
                self._last_sdf_bp_model = None
                self._last_sdf_bp_reference = None
                self._last_sdf_bp_points_obj = None
            except Exception as exc:
                self.sdf_last_error = repr(exc)
                rospy.logerr_throttle(
                    2.0, "[plotter] failed to apply /setup to SDF: %s", self.sdf_last_error
                )

        if abs(self.ego_L - old_L) > 1.0e-12 or abs(self.ego_W - old_W) > 1.0e-12:
            rospy.loginfo(
                "[plotter] /setup updated ego geometry: L %.3f -> %.3f, W %.3f -> %.3f",
                old_L, self.ego_L, old_W, self.ego_W,
            )
        else:
            rospy.loginfo_throttle(
                5.0,
                "[plotter] /setup applied: goal=(%.3f, %.3f) ego_L=%.3f ego_W=%.3f FOV=%.3f sector_id=%d",
                self.goal_x, self.goal_y, self.ego_L, self.ego_W, self.ego_FOV, int(self.sector_id),
            )

    def collect_indexed_topics(self, plural_key, singular_key, publisher_key, fallback_prefix, default_count=9):
        """Collect /name1..N style topics while preserving planner-index colors."""
        indexed = {}

        def add_topic(idx, topic):
            try:
                idx_i = int(idx)
            except Exception:
                return
            text = str(topic).strip()
            if idx_i > 0 and text:
                indexed[idx_i] = text

        def infer_index(name, block, fallback_idx):
            if isinstance(block, dict):
                for key in ("planner_id", "ORDER"):
                    if key in block:
                        try:
                            value = int(block.get(key))
                            if value > 0:
                                return value
                        except Exception:
                            pass
            digits = ""
            for ch in str(name):
                if ch.isdigit():
                    digits += ch
            if digits:
                try:
                    value = int(digits)
                    if value > 0:
                        return value
                except Exception:
                    pass
            return int(fallback_idx)

        for i, topic in enumerate(_as_string_list(self.plotter_cfg.get(plural_key, None)), start=1):
            add_topic(i, topic)
        for topic in _as_string_list(self.plotter_cfg.get(singular_key, None)):
            add_topic(1, topic)

        planners_cfg = self.raw_cfg.get("planners", {}) if isinstance(self.raw_cfg.get("planners", {}), dict) else {}
        fallback_idx = 1
        for name, block in planners_cfg.items():
            if not isinstance(block, dict):
                continue
            pubs = block.get("publishers", {}) if isinstance(block.get("publishers", {}), dict) else {}
            if publisher_key not in pubs:
                continue
            idx = infer_index(name, block, fallback_idx)
            for topic in _as_string_list(pubs.get(publisher_key, None)):
                add_topic(idx, topic)
            fallback_idx += 1

        count = max(int(default_count), max(indexed.keys()) if indexed else 0)
        topics = []
        for i in range(1, count + 1):
            topics.append(indexed.get(i, "%s%d" % (str(fallback_prefix), i)))
        return topics

    def yaw_to_quaternion(self, yaw):
        """
        Convert 2D yaw angle [rad] to geometry_msgs/Quaternion-compatible values.
        roll = 0, pitch = 0, yaw = yaw
        """
        half_yaw = 0.5 * yaw

        qx = 0.0
        qy = 0.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        return qx, qy, qz, qw

    def ego_sector_body_points(self):
        """Return MATLAB-compatible body-frame sector corner points and l_c.

        Surface id mapping:
            1: left, 2: rear, 3: right, 4: front.
        """
        L = float(self.ego_L)
        W = float(self.ego_W)
        fl = np.array([ L / 2.0 - 0.02,  W / 2.0], dtype=float)
        fr = np.array([ L / 2.0 - 0.02, -W / 2.0], dtype=float)
        rr = np.array([-L / 2.0 + 0.02, -W / 2.0], dtype=float)
        rl = np.array([-L / 2.0 + 0.02,  W / 2.0], dtype=float)
        corners = {1: (fl, rl), 2: (rl, rr), 3: (rr, fr), 4: (fr, fl)}
        a_body, b_body = corners.get(_normalize_sector_id(self.sector_id), corners[1])
        la = a_body / max(float(np.linalg.norm(a_body)), 1.0e-12)
        lb = b_body / max(float(np.linalg.norm(b_body)), 1.0e-12)
        lc = la + lb
        lc = lc / max(float(np.linalg.norm(lc)), 1.0e-12)
        return a_body, b_body, la, lb, lc

    def body_xy_to_world_point(self, xy_body, z=0.035):
        c = math.cos(float(self.ego_yaw))
        s = math.sin(float(self.ego_yaw))
        xy = np.asarray(xy_body, dtype=float).reshape(2)
        p = Point()
        p.x = float(self.ego_cart_x) + c * float(xy[0]) - s * float(xy[1])
        p.y = float(self.ego_cart_y) + s * float(xy[0]) + c * float(xy[1])
        p.z = float(z)
        return p

    def publish_ego_sector_markers(self, stamp):
        # print("fuck!!!!!!!!!!!!!")
        """Publish the designated vulnerable sector as an orange filled fan.

        The marker is emitted only when both show_ego_sector and
        mpc.enable_impact_normal_regulation are true.  A dedicated MarkerArray
        topic keeps sector visibility independently controllable in RViz.
        """
        marker_array = MarkerArray()
        frame_id = self.get_frame_id(self.latest_ego_msg) if self.latest_ego_msg is not None else "base_link"

        # Clear markers left by an earlier plotter process/configuration once.
        if not bool(self.ego_sector_marker_initialized):
            delete_all = Marker()
            delete_all.header.frame_id = frame_id
            delete_all.header.stamp = stamp
            delete_all.action = Marker.DELETEALL
            marker_array.markers.append(delete_all)
            self.ego_sector_marker_initialized = True

        enabled = bool(self.show_ego_sector) and bool(self.enable_impact_normal_regulation)
        if not enabled:
            self.ego_sector_pub.publish(marker_array)
            return

        a_body, b_body, la_body, lb_body, _lc_body = self.ego_sector_body_points()
        corner_radius = max(
            float(np.linalg.norm(a_body)),
            float(np.linalg.norm(b_body)),
            1.0e-3,
        )
        radius = max(float(self.ego_sector_radius_m), 1.05 * corner_radius)

        theta_a = math.atan2(float(la_body[1]), float(la_body[0]))
        theta_b = math.atan2(float(lb_body[1]), float(lb_body[0]))
        delta = math.atan2(math.sin(theta_b - theta_a), math.cos(theta_b - theta_a))
        arc_body = []
        for i in range(int(self.ego_sector_arc_steps) + 1):
            ratio = float(i) / float(self.ego_sector_arc_steps)
            theta = theta_a + delta * ratio
            arc_body.append(np.array([
                radius * math.cos(theta),
                radius * math.sin(theta),
            ], dtype=float))

        center = Point()
        center.x = float(self.ego_cart_x)
        center.y = float(self.ego_cart_y)
        center.z = 0.028

        fill = Marker()
        fill.header.frame_id = frame_id
        fill.header.stamp = stamp
        fill.ns = "ego_impact_sector"
        fill.id = 0
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = 1.0
        fill.scale.y = 1.0
        fill.scale.z = 1.0
        self.set_color(fill, self.ego_sector_fill_color)

        for i in range(len(arc_body) - 1):
            p1 = self.body_xy_to_world_point(arc_body[i], z=center.z)
            p2 = self.body_xy_to_world_point(arc_body[i + 1], z=center.z)
            # Add both winding orders so the filled region is visible from
            # either side of the RViz camera.
            fill.points.extend([center, p1, p2, center, p2, p1])
        marker_array.markers.append(fill)

        outline = Marker()
        outline.header.frame_id = frame_id
        outline.header.stamp = stamp
        outline.ns = "ego_impact_sector"
        outline.id = 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.035
        self.set_color(outline, self.ego_sector_outline_color)
        outline.points.append(center)
        for xy_body in arc_body:
            outline.points.append(self.body_xy_to_world_point(xy_body, z=center.z + 0.001))
        outline.points.append(center)
        marker_array.markers.append(outline)

        self.ego_sector_pub.publish(marker_array)


    @staticmethod
    def get_frame_id(msg):
        if hasattr(msg, "header") and msg.header.frame_id:
            return "base_link"
        return "base_link"

    @staticmethod
    def set_color(marker, rgba):
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])

    @staticmethod
    def time_to_vis_z(step_idx, dt=PRED_DT):
        # RViz z-axis is used only as a time axis.
        return float(TIME_SCALE) * float(dt) * float(step_idx)

    def active_prediction_dt(self):
        if self.best_cmd_stamp != rospy.Time(0):
            age = (rospy.Time.now() - self.best_cmd_stamp).to_sec()
            if age <= float(self.best_cmd_timeout_s):
                return max(float(self.best_traj_dt), 1.0e-6)
        return max(float(self.pred_dt), 1.0e-6)

    def time_to_vis_z_active(self, step_idx):
        return float(TIME_SCALE) * self.active_prediction_dt() * float(step_idx)

    def normalized_boundary_buffer_m(self, configured_value=None):
        """Return only non-vehicle baseline clearance in world meters."""
        initial = max(0.0, float(getattr(self, "initial_safe_margin_m", 0.0)))
        try:
            configured = float(configured_value)
        except Exception:
            configured = float("nan")
        if math.isfinite(configured) and configured > 0.0:
            return max(0.0, configured)
        return initial

    def base_safe_margin(self):
        # Vehicle geometry is added direction-by-direction in
        # safe_boundary_vertices_for_ego_state(); do not add a radius here.
        return max(0.0, float(self.boundary_buffer_m)) + \
            self.smooth_extra_buffer_for_safe_boundary() + \
            max(0.0, float(self.halfspace_buffer_m))


    def configured_collision_probability(self):
        """Return collision probability from /setup, accepting common JSON key names."""
        value = self.normal_cfg.get(
            "probability_collision",
            self.normal_cfg.get("pc", self.normal_cfg.get("p_c", 0.8)),
        )
        try:
            value = float(value)
        except Exception:
            value = 0.8
        return min(max(value, 1.0e-12), 1.0 - 1.0e-12)

    def probability_safe_margin(self, t_pred):
        mpc_cfg = self.raw_cfg.get("mpc", {}) if isinstance(self.raw_cfg.get("mpc", {}), dict) else {}
        use_prob = _as_bool(
            self.normal_cfg.get("use_probability_propagation_margin", True),
            True,
        )
        t = max(float(t_pred), 0.0)
        if use_prob:
            sigma_vx = max(float(self.normal_cfg.get("sigma_vx", 0.0)), 0.0)
            sigma_vy = max(float(self.normal_cfg.get("sigma_vy", 0.0)), 0.0)
            p_col = self.configured_collision_probability()
            # Match the MPC C++/CUDA collision field exactly: the current
            # implementation uses the RSS/isotropic velocity uncertainty
            # sigma_iso = hypot(sigma_vx, sigma_vy).
            sigma_iso = math.hypot(sigma_vx, sigma_vy)
            return t * sigma_iso * math.sqrt(max(-2.0 * math.log(1.0 - p_col), 0.0))
        rate = self.normal_cfg.get("pred_safe_margin_rate_mps", None)
        if rate is None:
            step_ref = float(self.normal_cfg.get("pred_safe_margin_step_ref_m", 0.025))
            dt_ref = max(float(self.normal_cfg.get("pred_safe_margin_step_ref_dt_s", self.pred_dt)), 1.0e-12)
            rate = step_ref / dt_ref
        return max(float(rate), 0.0) * t

    def probability_safe_margin_along_normal(self, t_pred, normal_world):
        """pc confidence-ellipse support along one world-frame unit normal."""
        mpc_cfg = self.raw_cfg.get("mpc", {}) if isinstance(self.raw_cfg.get("mpc", {}), dict) else {}
        use_prob = _as_bool(
            self.normal_cfg.get("use_probability_propagation_margin", True),
            True,
        )
        if not use_prob:
            return 0.0
        n = np.asarray(normal_world, dtype=float).reshape(2)
        nrm = max(float(np.linalg.norm(n)), 1.0e-12)
        n = n / nrm
        sigma_vx = max(float(self.normal_cfg.get("sigma_vx", 0.0)), 0.0)
        sigma_vy = max(float(self.normal_cfg.get("sigma_vy", 0.0)), 0.0)
        p_col = self.configured_collision_probability()
        gamma = math.sqrt(max(-2.0 * math.log(1.0 - p_col), 0.0))
        sigma_n = math.sqrt((sigma_vx * float(n[0])) ** 2 + (sigma_vy * float(n[1])) ** 2)
        return gamma * max(float(t_pred), 0.0) * sigma_n

    def ego_oriented_ellipse_support(self, normal_world, ego_yaw):
        """Support radius of the rectangle-circumscribing oriented ellipse."""
        n = np.asarray(normal_world, dtype=float).reshape(2)
        nrm = max(float(np.linalg.norm(n)), 1.0e-12)
        n = n / nrm
        yaw = float(ego_yaw)
        c = math.cos(yaw)
        s = math.sin(yaw)
        # body-frame normal R(yaw)^T n
        nbx = c * float(n[0]) + s * float(n[1])
        nby = -s * float(n[0]) + c * float(n[1])
        a = max(abs(float(self.ego_L)), 1.0e-12) / math.sqrt(2.0)
        b = max(abs(float(self.ego_W)), 1.0e-12) / math.sqrt(2.0)
        return math.sqrt((a * nbx) ** 2 + (b * nby) ** 2)

    def inflate_polygon_vertices(self, vertices, margin):
        # Small polygons are processed much faster with plain Python than with
        # creating temporary NumPy arrays on every RViz callback.
        n = len(vertices)
        if n < 3 or margin <= 0.0:
            return list(vertices)

        cx = 0.0
        cy = 0.0
        for x, y in vertices:
            cx += float(x)
            cy += float(y)
        inv_n = 1.0 / float(n)
        cx *= inv_n
        cy *= inv_n

        out = []
        m = float(margin)
        for x, y in vertices:
            x = float(x)
            y = float(y)
            dx = x - cx
            dy = y - cy
            norm = math.sqrt(dx * dx + dy * dy)
            if norm < 1.0e-9:
                out.append((x, y))
            else:
                scale = m / norm
                out.append((x + scale * dx, y + scale * dy))
        return out

    def ego_scaled_local_transform_matrices(self, ego_yaw):
        """Return T=S^-1 R^-1 and T_inv=R S for the ego ellipse frame."""
        yaw = float(ego_yaw)
        c = math.cos(yaw)
        s = math.sin(yaw)
        _major, minor, axis_scale = _ellipse_axes_from_lengths(self.ego_L, self.ego_W)
        self.ego_major_axis_m = _major
        self.ego_minor_axis_m = minor
        self.ego_axis_scale = axis_scale
        sx = max(float(axis_scale[0]), 1.0e-12)
        sy = max(float(axis_scale[1]), 1.0e-12)
        # T = S^-1 R^T, expanded to avoid allocating R/S and inverting T later.
        T = np.array([[c / sx, s / sx], [-s / sy, c / sy]], dtype=float)
        # T_inv = R S.
        T_inv = np.array([[c * sx, -s * sy], [s * sx, c * sy]], dtype=float)
        return T, T_inv

    def ego_scaled_local_transform_matrix(self, ego_yaw):
        T, _ = self.ego_scaled_local_transform_matrices(ego_yaw)
        return T

    def transform_world_to_ego_scaled_local(self, vertices, ego_x, ego_y, ego_yaw):
        V = np.asarray(vertices, dtype=float).reshape(-1, 2)
        T, T_inv = self.ego_scaled_local_transform_matrices(ego_yaw)
        ego = np.array([float(ego_x), float(ego_y)], dtype=float).reshape(1, 2)
        return (V - ego).dot(T.T), T, T_inv, ego

    def transform_ego_scaled_local_to_world(self, vertices_local, T_or_inv, ego, already_inverse=False):
        Z = np.asarray(vertices_local, dtype=float).reshape(-1, 2)
        M = np.asarray(T_or_inv, dtype=float).reshape(2, 2)
        if not already_inverse:
            try:
                M = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M = np.eye(2, dtype=float)
        P = Z.dot(M.T) + np.asarray(ego, dtype=float).reshape(1, 2)
        return [(float(x), float(y)) for x, y in P]

    def smooth_extra_buffer_for_safe_boundary(self):
        try:
            return max(float(self.normal_cfg.get("smooth_extra_buffer_m", 0.0)), 0.0)
        except Exception:
            return 0.0

    def matlab_style_pseudo_boundary_vertices(self, vertices, margin):
        """Return an ordered safe-boundary polyline using MPC pseudo features.

        The MPC collision model uses one edge-midpoint pseudo vertex and one
        shifted original-vertex feature per polygon edge/vertex.  For RViz line
        drawing, connect them in boundary order:
            vertex_i -> pseudo_edge_i -> vertex_{i+1}.
        This makes /obs_polygons_horizon show the same support-feature geometry
        that the collision-avoidance constraint sees, instead of a radial
        centroid inflation of only the original polygon vertices.
        """
        V = np.asarray(vertices, dtype=float).reshape(-1, 2)
        n = int(V.shape[0])
        if n < 3:
            return [(float(x), float(y)) for x, y in V]

        if self.polygon_signed_area(V) < 0.0:
            V = np.flipud(V).copy()

        total_buffer = max(float(margin), 0.0) + self.smooth_extra_buffer_for_safe_boundary()
        center = np.mean(V, axis=0)
        edge_normals_out = np.zeros((n, 2), dtype=float)
        pseudo_points = np.zeros((n, 2), dtype=float)

        for i in range(n):
            a = V[i]
            b = V[(i + 1) % n]
            e = b - a
            L = max(float(np.linalg.norm(e)), 1.0e-12)
            n_out = np.array([e[1], -e[0]], dtype=float) / L
            edge_normals_out[i] = n_out
            pseudo_points[i] = 0.5 * (a + b) + total_buffer * n_out

        vertex_normals_out = np.zeros((n, 2), dtype=float)
        for i in range(n):
            ip = (i - 1) % n
            n_vertex = edge_normals_out[ip] + edge_normals_out[i]
            n_vertex_norm = float(np.linalg.norm(n_vertex))
            if n_vertex_norm < 1.0e-12:
                radial = V[i] - center
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm < 1.0e-12:
                    n_vertex = edge_normals_out[i]
                    n_vertex_norm = float(np.linalg.norm(n_vertex))
                else:
                    n_vertex = radial
                    n_vertex_norm = radial_norm
            vertex_normals_out[i] = n_vertex / max(n_vertex_norm, 1.0e-12)

        vertex_points = V + total_buffer * vertex_normals_out

        ordered = []
        for i in range(n):
            ordered.append(vertex_points[i])
            ordered.append(pseudo_points[i])

        return [(float(p[0]), float(p[1])) for p in ordered]

    @staticmethod
    def convex_hull_2d(points):
        """Return the CCW convex hull of 2-D points as [(x, y), ...]."""
        pts = []
        for p in points:
            try:
                x = float(p[0])
                y = float(p[1])
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))

        # Remove duplicates before the monotone-chain hull.
        pts = sorted(set(pts))
        if len(pts) <= 2:
            return list(pts)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for pt in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 1.0e-12:
                lower.pop()
            lower.append(pt)

        upper = []
        for pt in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 1.0e-12:
                upper.pop()
            upper.append(pt)

        hull = lower[:-1] + upper[:-1]
        return [(float(x), float(y)) for x, y in hull]

    def ego_local_transform_matrices(self, ego_yaw):
        """Return T=R^T and T_inv=R for the unscaled ego local frame."""
        yaw = float(ego_yaw)
        c = math.cos(yaw)
        s = math.sin(yaw)
        T = np.array([[c, s], [-s, c]], dtype=float)
        T_inv = np.array([[c, -s], [s, c]], dtype=float)
        return T, T_inv

    def polygon_original_vertex_outward_normals(self, vertices):
        """Return vertices and outward vertex normals in one-to-one order."""
        V = np.asarray(vertices, dtype=float).reshape(-1, 2)
        n = int(V.shape[0])
        if n < 3:
            return V, np.zeros((0, 2), dtype=float)

        # Use CCW order so the edge outward normal is [dy, -dx].
        if self.polygon_signed_area(V) < 0.0:
            V = np.flipud(V).copy()

        center = np.mean(V, axis=0)
        edge_normals_out = np.zeros((n, 2), dtype=float)
        for i in range(n):
            a = V[i]
            b = V[(i + 1) % n]
            e = b - a
            L = max(float(np.linalg.norm(e)), 1.0e-12)
            edge_normals_out[i] = np.array([e[1], -e[0]], dtype=float) / L

        vertex_normals_out = np.zeros((n, 2), dtype=float)
        for i in range(n):
            ip = (i - 1) % n
            n_vertex = edge_normals_out[ip] + edge_normals_out[i]
            n_vertex_norm = float(np.linalg.norm(n_vertex))
            if n_vertex_norm < 1.0e-12:
                radial = V[i] - center
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm < 1.0e-12:
                    n_vertex = edge_normals_out[i]
                    n_vertex_norm = float(np.linalg.norm(n_vertex))
                else:
                    n_vertex = radial
                    n_vertex_norm = radial_norm
            vertex_normals_out[i] = n_vertex / max(n_vertex_norm, 1.0e-12)

        return V, vertex_normals_out

    def safe_boundary_vertices_for_ego_state(self, vertices, margin, ego_state=None, t_pred=0.0, is_dynamic=False):
        """Inflate obstacle by oriented ego ellipse + optional uncertainty.

        Each original-vertex outward normal receives the Minkowski support sum
            r_vehicle(n, yaw) + margin + r_uncertainty(n, t).
        This preserves the vehicle heading explicitly instead of scaling a single
        isotropic radius by the ellipse aspect ratio.
        """
        if len(vertices) < 3:
            return list(vertices)

        if ego_state is None:
            ego_yaw = float(self.ego_yaw)
        else:
            ego_yaw = float(ego_state.get("yaw", self.ego_yaw))

        V_world, normals_world = self.polygon_original_vertex_outward_normals(vertices)
        if V_world.shape[0] < 3 or normals_world.shape[0] != V_world.shape[0]:
            return list(vertices)

        extra = max(float(margin), 0.0)
        inflated = np.empty_like(V_world)
        for i in range(V_world.shape[0]):
            n = normals_world[i]
            vehicle_r = self.ego_oriented_ellipse_support(n, ego_yaw) if bool(self.use_ego_scaled_local_frame) else max(float(self.ego_R), 0.0)
            uncertainty_r = self.probability_safe_margin_along_normal(t_pred, n) if bool(is_dynamic) else 0.0
            total_r = vehicle_r + extra + uncertainty_r
            inflated[i, :] = V_world[i, :] + total_r * n

        hull = self.convex_hull_2d(inflated)
        if len(hull) >= 3:
            return hull
        return [(float(x), float(y)) for x, y in inflated]

    def active_planning_states(self):
        if self.best_cmd_stamp != rospy.Time(0):
            age = (rospy.Time.now() - self.best_cmd_stamp).to_sec()
            if age <= float(self.best_cmd_timeout_s) and len(self.best_traj_states) > 0:
                return list(self.best_traj_states)
        return [{"x": float(self.ego_cart_x), "y": float(self.ego_cart_y), "yaw": float(self.ego_yaw), "t": 0.0}]

    def active_planning_state_for_step(self, step_idx, fallback_t):
        states = self.active_planning_states()
        if len(states) <= 0:
            return {"x": float(self.ego_cart_x), "y": float(self.ego_cart_y), "yaw": float(self.ego_yaw), "t": float(fallback_t)}
        idx = min(max(int(step_idx), 0), len(states) - 1)
        state = dict(states[idx])
        if "t" not in state:
            state["t"] = float(fallback_t)
        return state

    def get_obstacle_type(self, obstacle, obs_idx):
        if hasattr(obstacle, "obs_types") and obs_idx < len(obstacle.obs_types):
            return 1 if int(obstacle.obs_types[obs_idx]) == 1 else 0
        vx = float(obstacle.obs_vxs[obs_idx]) if hasattr(obstacle, "obs_vxs") and obs_idx < len(obstacle.obs_vxs) else 0.0
        vy = float(obstacle.obs_vys[obs_idx]) if hasattr(obstacle, "obs_vys") and obs_idx < len(obstacle.obs_vys) else 0.0
        return 1 if abs(vx) + abs(vy) > 1.0e-12 else 0

    def get_obstacle_radius(self, obstacle, obs_idx):
        if hasattr(obstacle, "obs_rs") and obs_idx < len(obstacle.obs_rs):
            r = float(obstacle.obs_rs[obs_idx])
            if r > 1.0e-6:
                return r
        return float(self.moving_obstacle_R)

    def get_obstacle_polygon_vertices(self, obstacle, obs_idx):
        """Return 2-D vertices [(x, y), ...].

        Any obs_poly_zs field is intentionally ignored because this scenario is
        2-D. Marker Point.z is assigned later from the prediction time index.
        """
        if obs_idx < 0 or obs_idx >= int(obstacle.obs_cnt):
            return []

        if not hasattr(obstacle, "obs_vertex_counts"):
            return []
        if not hasattr(obstacle, "obs_poly_xs") or not hasattr(obstacle, "obs_poly_ys"):
            return []

        nv = int(obstacle.obs_vertex_counts[obs_idx])
        nv = max(0, min(nv, MAX_VERTICES))

        if nv < 3:
            return []

        vertices = []
        for j in range(nv):
            flat_idx = obs_idx * MAX_VERTICES + j

            if flat_idx >= len(obstacle.obs_poly_xs) or flat_idx >= len(obstacle.obs_poly_ys):
                break

            x = float(obstacle.obs_poly_xs[flat_idx])
            y = float(obstacle.obs_poly_ys[flat_idx])
            vertices.append((x, y))

        if len(vertices) < 3:
            return []

        return vertices

    def actual_polygon_color(self, obs_type, time_idx):
        # Current obstacle polygons are rendered as light gray regardless of
        # static/dynamic type.  Dynamic prediction is shown on the separate
        # boundary-only horizon topic.
        alpha = 0.62 if int(time_idx) == 0 else 0.90
        return (0.78, 0.78, 0.78, alpha)

    def perceived_polygon_color(self, obs_type, time_idx):
        # Keep perceived obstacles visually consistent with /obstacle, but a
        # little more transparent so the two streams can be overlaid in RViz.
        alpha = 0.48 if int(time_idx) == 0 else 0.75
        return (0.78, 0.78, 0.78, alpha)

    def safe_margin_boundary_color(self, obs_type):
        # Boundary-only planning horizon uses the same light-gray obstacle color.
        return (0.0, 0.0, 0.0, 1.0)

    def publish_obstacle_positions(
        self,
        obstacle,
        pub,
        ns,
        marker_id,
        rgba,
        radius_offset=0.0,
        draw_prediction=True,
    ):
        marker = Marker()
        marker.header.frame_id = self.get_frame_id(obstacle)
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        radius = max(float(self.moving_obstacle_R) + radius_offset, 1.0e-3)
        marker.scale.x = radius * 2.0
        marker.scale.y = radius * 2.0
        marker.scale.z = radius * 2.0
        self.set_color(marker, rgba)

        obs_cnt = min(int(obstacle.obs_cnt), MAX_OBS)
        time_indices = list(range(0, int(self.pred_horizon) + 1, int(self.pred_vis_step))) if draw_prediction else [0]
        dt = self.active_prediction_dt()

        for i in range(obs_cnt):
            vx = float(obstacle.obs_vxs[i])
            vy = float(obstacle.obs_vys[i])

            for j in time_indices:
                p = Point()
                p.x = float(obstacle.obs_xs[i]) + j * vx * dt
                p.y = float(obstacle.obs_ys[i]) + j * vy * dt
                p.z = float(TIME_SCALE) * dt * float(j)
                marker.points.append(p)

        pub.publish(marker)

    def safe_margin_prediction_indices(self):
        horizon = max(0, int(self.pred_horizon))
        interval = max(1, int(self.safe_margin_pred_step_interval))
        indices = list(range(0, horizon + 1, interval))
        if not indices or indices[-1] != horizon:
            indices.append(horizon)
        # Keep deterministic order and avoid duplicates when horizon is zero.
        return sorted(set(int(i) for i in indices))

    def point_in_ego_fov(self, x, y):
        # Fast FOV test: squared radius check + dot-product angle check.
        # This avoids atan2/sin/cos wrapping for every obstacle vertex.
        dx = float(x) - float(self.ego_cart_x)
        dy = float(y) - float(self.ego_cart_y)
        r = max(float(self.ego_FOV), 0.0)
        dist_sq = dx * dx + dy * dy
        if dist_sq > r * r:
            return False
        if dist_sq <= 1.0e-12:
            return True

        half_angle = math.radians(abs(float(self.obs_fov_deg)))
        if half_angle >= math.pi:
            return True

        yaw = float(getattr(self, "ego_yaw", 0.0))
        heading_x = math.cos(yaw)
        heading_y = math.sin(yaw)
        cos_rel = (dx * heading_x + dy * heading_y) / math.sqrt(dist_sq)
        return cos_rel >= math.cos(half_angle)

    def obstacle_has_vertex_in_ego_fov(self, vertices):
        for x, y in vertices:
            if self.point_in_ego_fov(x, y):
                return True
        return False

    def obstacle_intersects_ego_fov_fast(self, vertices):
        """Fast current-FOV test used to cull heavy horizon safe-boundary drawing."""
        n = len(vertices)
        if n <= 0:
            return False

        ex = float(self.ego_cart_x)
        ey = float(self.ego_cart_y)
        yaw = float(self.ego_yaw)
        r = max(float(self.ego_FOV), 0.0)
        if r <= 1.0e-9:
            return False
        r2 = r * r
        half_angle = math.radians(abs(float(self.obs_fov_deg)))
        if half_angle >= math.pi:
            cos_half = -1.0
        else:
            cos_half = math.cos(half_angle)
        fx = math.cos(yaw)
        fy = math.sin(yaw)

        # Vertex-in-FOV catches the normal case and is cheaper than polygon tests.
        cx = 0.0
        cy = 0.0
        min_d2 = float("inf")
        for x, y in vertices:
            x = float(x)
            y = float(y)
            cx += x
            cy += y
            dx = x - ex
            dy = y - ey
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
            if d2 <= 1.0e-18:
                return True
            if d2 <= r2 and (fx * dx + fy * dy) / math.sqrt(d2) >= cos_half:
                return True

        # Center-in-FOV catches small polygons whose vertices are just outside
        # the frustum edge, without doing expensive polygon-sector intersection.
        inv_n = 1.0 / float(n)
        cx = cx * inv_n - ex
        cy = cy * inv_n - ey
        d2c = cx * cx + cy * cy
        if d2c <= 1.0e-18:
            return True
        if d2c <= r2 and (fx * cx + fy * cy) / math.sqrt(d2c) >= cos_half:
            return True

        return False

    def add_filled_polygon_marker(self, marker_array, frame_id, stamp, ns, marker_id, vertices, rgba):
        if len(vertices) < 3:
            return marker_id

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.set_color(marker, rgba)

        center_xy = np.mean(np.asarray(vertices, dtype=float).reshape(-1, 2), axis=0)
        center = Point()
        center.x = float(center_xy[0])
        center.y = float(center_xy[1])
        center.z = 0.0

        for j in range(len(vertices)):
            x1, y1 = vertices[j]
            x2, y2 = vertices[(j + 1) % len(vertices)]

            p1 = Point()
            p1.x = float(x1)
            p1.y = float(y1)
            p1.z = 0.0

            p2 = Point()
            p2.x = float(x2)
            p2.y = float(y2)
            p2.z = 0.0

            # Add both windings so the transparent fill is stable in RViz.
            marker.points.append(center)
            marker.points.append(p1)
            marker.points.append(p2)
            marker.points.append(center)
            marker.points.append(p2)
            marker.points.append(p1)

        marker_array.markers.append(marker)
        return marker_id + 1

    def add_polygon_line_marker(self, marker_array, frame_id, stamp, ns, marker_id, vertices, rgba, line_width, z_vis):
        if len(vertices) < 3:
            return marker_id

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(line_width)
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.set_color(marker, rgba)

        for x, y in vertices:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z_vis)
            marker.points.append(p)
        if marker.points:
            marker.points.append(marker.points[0])

        marker_array.markers.append(marker)
        return marker_id + 1

    def append_polygon_triangles(self, marker, vertices, z_vis=0.0):
        n = len(vertices)
        if n < 3:
            return
        cx = 0.0
        cy = 0.0
        for x, y in vertices:
            cx += float(x)
            cy += float(y)
        cx /= float(n)
        cy /= float(n)

        for j in range(n):
            x1, y1 = vertices[j]
            x2, y2 = vertices[(j + 1) % n]

            p0 = Point()
            p0.x = cx
            p0.y = cy
            p0.z = float(z_vis)
            p1 = Point()
            p1.x = float(x1)
            p1.y = float(y1)
            p1.z = float(z_vis)
            p2 = Point()
            p2.x = float(x2)
            p2.y = float(y2)
            p2.z = float(z_vis)

            # Both windings keep transparent fill stable in RViz.
            marker.points.append(p0)
            marker.points.append(p1)
            marker.points.append(p2)
            marker.points.append(p0)
            marker.points.append(p2)
            marker.points.append(p1)

    def append_polygon_line_segments(self, marker, vertices, z_vis):
        n = len(vertices)
        if n < 3:
            return
        z = float(z_vis)
        for j in range(n):
            x1, y1 = vertices[j]
            x2, y2 = vertices[(j + 1) % n]
            p1 = Point()
            p1.x = float(x1)
            p1.y = float(y1)
            p1.z = z
            p2 = Point()
            p2.x = float(x2)
            p2.y = float(y2)
            p2.z = z
            marker.points.append(p1)
            marker.points.append(p2)

    def publish_obstacle_polygons(
        self,
        obstacle,
        pub,
        ns,
        prev_cnt_attr,
        color_fn,
        line_width=0.035,
        draw_prediction=True,
        apply_safe_margin=False,
        draw_current_fill=True,
    ):
        marker_array = MarkerArray()
        frame_id = self.get_frame_id(obstacle)
        stamp = rospy.Time.now()
        obs_cnt = min(int(obstacle.obs_cnt), MAX_OBS)
        dt = self.active_prediction_dt()
        base_margin = self.base_safe_margin() if bool(apply_safe_margin) else 0.0
        pred_indices = self.safe_margin_prediction_indices() if (draw_prediction and apply_safe_margin) else []

        marker_id = 0
        fill_marker = None
        line_marker = None

        # Aggregate all filled polygons into one TRIANGLE_LIST marker and all
        # horizon safe boundaries into one LINE_LIST marker.  This keeps RViz
        # from receiving O(num_obs * horizon) individual markers.
        if bool(draw_current_fill):
            fill_marker = Marker()
            fill_marker.header.frame_id = frame_id
            fill_marker.header.stamp = stamp
            fill_marker.ns = ns
            fill_marker.id = marker_id
            fill_marker.type = Marker.TRIANGLE_LIST
            fill_marker.action = Marker.ADD
            fill_marker.pose.orientation.w = 1.0
            fill_marker.scale.x = 1.0
            fill_marker.scale.y = 1.0
            fill_marker.scale.z = 1.0
            self.set_color(fill_marker, color_fn(0, 0))
            marker_id += 1

        if bool(apply_safe_margin):
            line_marker = Marker()
            line_marker.header.frame_id = frame_id
            line_marker.header.stamp = stamp
            line_marker.ns = ns
            line_marker.id = marker_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = float(line_width)
            line_marker.scale.y = 1.0
            line_marker.scale.z = 1.0
            self.set_color(line_marker, self.safe_margin_boundary_color(0))
            marker_id += 1

        # Cache active MPC states once.  Previously this was re-read for every
        # obstacle and horizon step, which creates avoidable Python overhead.
        active_states = self.active_planning_states() if bool(apply_safe_margin) else []

        def state_for_step(step_idx, fallback_t):
            if not active_states:
                return {"x": float(self.ego_cart_x), "y": float(self.ego_cart_y), "yaw": float(self.ego_yaw), "t": float(fallback_t)}
            idx = min(max(int(step_idx), 0), len(active_states) - 1)
            state = active_states[idx]
            if "t" in state:
                return state
            out = dict(state)
            out["t"] = float(fallback_t)
            return out

        # Precompute the horizon state/time table once per MarkerArray.
        pred_state_table = []
        if line_marker is not None and draw_prediction:
            for t_idx in pred_indices:
                if int(t_idx) == 0:
                    continue
                t_pred_default = float(t_idx) * dt
                ego_state_k = state_for_step(int(t_idx), t_pred_default)
                t_pred = float(ego_state_k.get("t", t_pred_default))
                pred_state_table.append((int(t_idx), ego_state_k, t_pred, float(TIME_SCALE) * t_pred))

        horizon_fov_only = bool(apply_safe_margin) and bool(getattr(self, "safe_margin_horizon_fov_only", True))
        draw_current_boundary = bool(getattr(self, "safe_margin_draw_current_boundary", True))

        for i in range(obs_cnt):
            base_vertices = self.get_obstacle_polygon_vertices(obstacle, i)
            if len(base_vertices) < 3:
                continue

            if fill_marker is not None:
                self.append_polygon_triangles(fill_marker, base_vertices, z_vis=0.0)

            if line_marker is None:
                continue

            # Heavy planning-horizon safe boundaries are only useful for the
            # obstacles currently visible to the ego.  Keep current filled
            # obstacle polygons untouched, but cull /obs_polygons_horizon here.
            if horizon_fov_only and not self.obstacle_intersects_ego_fov_fast(base_vertices):
                continue

            obs_type = self.get_obstacle_type(obstacle, i)

            if draw_current_boundary:
                state0 = state_for_step(0, 0.0)
                baseline_vertices = self.safe_boundary_vertices_for_ego_state(
                    base_vertices, base_margin, state0, t_pred=0.0, is_dynamic=(int(obs_type) == 1)
                )
                self.append_polygon_line_segments(line_marker, baseline_vertices, z_vis=0.0)

            if not draw_prediction:
                continue

            vx = float(obstacle.obs_vxs[i]) if hasattr(obstacle, "obs_vxs") and i < len(obstacle.obs_vxs) else 0.0
            vy = float(obstacle.obs_vys[i]) if hasattr(obstacle, "obs_vys") and i < len(obstacle.obs_vys) else 0.0
            is_dynamic = int(obs_type) == 1
            for _t_idx, ego_state_k, t_pred, z_vis in pred_state_table:
                dx = t_pred * vx if is_dynamic else 0.0
                dy = t_pred * vy if is_dynamic else 0.0
                # Direction-dependent uncertainty is added inside
                # safe_boundary_vertices_for_ego_state() along each support normal.
                r_buffer = base_margin
                if dx != 0.0 or dy != 0.0:
                    shifted_vertices = [(float(x) + dx, float(y) + dy) for x, y in base_vertices]
                else:
                    shifted_vertices = base_vertices
                draw_vertices = self.safe_boundary_vertices_for_ego_state(
                    shifted_vertices, r_buffer, ego_state_k, t_pred=t_pred, is_dynamic=is_dynamic
                )
                self.append_polygon_line_segments(line_marker, draw_vertices, z_vis=z_vis)

        if fill_marker is not None:
            marker_array.markers.append(fill_marker)
        if line_marker is not None:
            marker_array.markers.append(line_marker)

        if apply_safe_margin and self.safe_margin_text_enabled:
            terminal_margin = self.probability_safe_margin(float(self.pred_horizon) * dt)
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = ns
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(self.x_min)
            text_marker.pose.position.y = float(self.y_max)
            text_marker.pose.position.z = float(TIME_SCALE) * dt * float(self.pred_horizon)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.28
            mpc_cfg = self.raw_cfg.get("mpc", {}) if isinstance(self.raw_cfg.get("mpc", {}), dict) else {}
            sigma_vx = float(self.normal_cfg.get("sigma_vx", 0.0))
            sigma_vy = float(self.normal_cfg.get("sigma_vy", 0.0))
            p_col = float(self.normal_cfg.get("probability_collision", 0.0))
            fov_text = "FOV" if horizon_fov_only else "all obs"
            text_marker.text = "%s base=%.3f  dyn extra: dt=%.3f  sigma=(%.2f, %.2f)  p=%.2f  r_N=%.3f  total_N=%.3f  interval=%d" % (
                fov_text,
                base_margin,
                dt,
                sigma_vx,
                sigma_vy,
                p_col,
                terminal_margin,
                base_margin + terminal_margin,
                int(self.safe_margin_pred_step_interval),
            )
            self.set_color(text_marker, (0.72, 0.72, 0.72, 1.0))
            marker_array.markers.append(text_marker)
            marker_id += 1

        prev_cnt = int(getattr(self, prev_cnt_attr))
        for stale_id in range(marker_id, prev_cnt):
            mdel = Marker()
            mdel.header.frame_id = frame_id
            mdel.header.stamp = stamp
            mdel.ns = ns
            mdel.id = stale_id
            mdel.action = Marker.DELETE
            marker_array.markers.append(mdel)

        setattr(self, prev_cnt_attr, marker_id)
        if pub is self.obs_polygon_pub:
            self._cached_obs_polygon_marker_array = marker_array
            self._obs_polygon_dirty = False
        pub.publish(marker_array)

    @staticmethod
    def _publisher_has_subscribers(pub):
        """Cheaply skip visualization work when nobody subscribes to the topic."""
        try:
            return int(pub.get_num_connections()) > 0
        except Exception:
            return True

    def should_publish_visualization_now(self, stamp_attr, hz):
        hz = float(hz)
        if hz <= 0.0:
            return True

        now = rospy.Time.now()
        # When a visualization requests the same (or a higher) rate as the common
        # loop, publish on every loop tick. A second wall-time throttle at exactly
        # 20 Hz can otherwise turn 19.9~20 Hz scheduler jitter into accidental
        # 10 Hz output by rejecting every other tick.
        common_hz = max(float(getattr(self, "visualization_pub_hz", 0.0)), 0.0)
        if common_hz > 0.0 and hz >= common_hz * 0.999:
            setattr(self, stamp_attr, now)
            return True

        last = getattr(self, stamp_attr, rospy.Time(0))
        if last != rospy.Time(0) and (now - last).to_sec() < 1.0 / max(hz, 1.0e-9):
            return False

        setattr(self, stamp_attr, now)
        return True

    def dynamic_obstacles_callback(self, msg):
        self.latest_dynamic_obstacles_msg = msg
        self._selector_horizon_dirty = True

    def obs_callback(self, obstacle):
        # Keep only the newest ground-truth obstacle message.  This stream is
        # used for visualization only; SDF evaluation uses perceived geometry.
        self.latest_obstacle_msg = obstacle
        self._obs_polygon_dirty = True

    def percepted_obstacle_callback(self, obstacle):
        # vertex_selector publishes the obstacle polygons that survived
        # perception/FOV processing.  Keep that geometry separate from
        # /obstacle so the SDF never sees unperceived ground-truth obstacles.
        self.latest_percepted_obstacle_msg = obstacle
        self._perceived_field_arrays_cache_msg = None
        self._perceived_field_arrays_cache = None

    def _extract_control_time_intervals(self, msg, n, default_dt):
        intervals = []
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
        elif len(intervals) == 1 and int(n) > 1:
            intervals = [float(intervals[0])] * int(n)
        elif len(intervals) < int(n):
            intervals = list(intervals) + [float(intervals[-1])] * (int(n) - len(intervals))

        fallback = float(default_dt) if np.isfinite(default_dt) and default_dt > 1.0e-9 else float(self.pred_dt)
        out = []
        for value in intervals[:int(n)]:
            dt_i = float(value)
            if not np.isfinite(dt_i) or dt_i <= 1.0e-9:
                dt_i = fallback
            out.append(dt_i)
        return out

    @staticmethod
    def _wrap_yaw(yaw):
        return math.atan2(math.sin(float(yaw)), math.cos(float(yaw)))

    def _rollout_mpc_control_seq_outputs(self, msg):
        """Roll out one control sequence once and return both states and points.

        The previous implementation performed two independent NumPy RK4
        rollouts for the same message: one for best_traj_states and one for the
        RViz points. This scalar RK4 implementation preserves the dynamics but
        avoids temporary arrays and allows both outputs to share one pass.
        """
        try:
            yawrates = msg.yawrates
            accs = msg.accs
            n = min(len(yawrates), len(accs))
        except Exception:
            return [], []

        if n <= 0:
            return [], []

        dt = float(getattr(msg, "dt", self.pred_dt))
        if not np.isfinite(dt) or dt <= 1.0e-9:
            dt = float(self.pred_dt)
        dts = self._extract_control_time_intervals(msg, n, dt)

        x = float(getattr(self, "ego_cart_x", 0.0))
        y = float(getattr(self, "ego_cart_y", 0.0))
        v = float(getattr(self, "ego_v", 0.0))
        yaw = float(getattr(self, "ego_yaw", 0.0))
        if not math.isfinite(v):
            v = 0.0
        if not math.isfinite(yaw):
            yaw = 0.0
        yaw = self._wrap_yaw(yaw)

        states = [{"x": x, "y": y, "v": v, "yaw": yaw, "t": 0.0}]
        points = []
        p0 = Point()
        p0.x = x
        p0.y = y
        p0.z = 0.0
        points.append(p0)

        t_acc = 0.0
        for k in range(n):
            dt_i = float(dts[k])
            try:
                yawrate = float(yawrates[k])
            except Exception:
                yawrate = 0.0
            try:
                acc = float(accs[k])
            except Exception:
                acc = 0.0
            if not math.isfinite(yawrate):
                yawrate = 0.0
            if not math.isfinite(acc):
                acc = 0.0

            half_dt = 0.5 * dt_i
            v2 = v + half_dt * acc
            yaw2 = yaw + half_dt * yawrate
            v4 = v + dt_i * acc
            yaw4 = yaw + dt_i * yawrate

            k1x = v * math.cos(yaw)
            k1y = v * math.sin(yaw)
            k2x = v2 * math.cos(yaw2)
            k2y = v2 * math.sin(yaw2)
            # k3 evaluates at the same v/yaw midpoint for constant controls.
            k3x = k2x
            k3y = k2y
            k4x = v4 * math.cos(yaw4)
            k4y = v4 * math.sin(yaw4)

            sixth_dt = dt_i / 6.0
            x += sixth_dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            y += sixth_dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            v = v4
            yaw = self._wrap_yaw(yaw4)
            t_acc += dt_i

            states.append({"x": x, "y": y, "v": v, "yaw": yaw, "t": float(t_acc)})
            p = Point()
            p.x = x
            p.y = y
            p.z = float(TIME_SCALE) * t_acc
            points.append(p)

        return states, points

    def _rollout_mpc_control_seq_states(self, msg):
        states, _ = self._rollout_mpc_control_seq_outputs(msg)
        return states

    def _rollout_mpc_control_seq(self, msg):
        _, points = self._rollout_mpc_control_seq_outputs(msg)
        return points

    def mpc_pred_color_for_idx(self, planner_idx):
        idx = max(1, int(planner_idx))
        if idx <= len(self.mpc_pred_colors):
            return self.mpc_pred_colors[idx - 1]
        return self.mpc_pred_colors[-1]

    def initial_guidance_color_for_idx(self, guidance_idx):
        """Return the visibility-guidance color assigned to /initial_guidanceN.

        This intentionally uses the Visibility-PRM guidance palette, not the
        MPC prediction palette.  In the split forward/backward setup, the
        default color cycle is 3, so 1/4 share red, 2/5 share green, and
        3/6 share blue.  Set plotter.initial_guidance_color_cycle to 12 if
        every absolute /initial_guidanceN topic should use a unique global
        color.
        """
        colors = getattr(self, "initial_guidance_colors", None)
        if not colors:
            return self.mpc_pred_color_for_idx(guidance_idx)
        try:
            idx = max(1, int(guidance_idx))
        except Exception:
            idx = 1
        cycle = max(1, int(getattr(self, "initial_guidance_color_cycle", len(colors))))
        local_idx = (idx - 1) % cycle
        color_idx = local_idx % len(colors)
        return colors[color_idx]

    def planner_visual_color_for_idx(self, planner_idx):
        if bool(getattr(self, "use_initial_guidance_colors_for_mpc_pred", True)):
            return self.initial_guidance_color_for_idx(planner_idx)
        return self.mpc_pred_color_for_idx(planner_idx)

    @staticmethod
    def _planner_idx_from_control_msg(msg, fallback_idx=1):
        """Recover planner id encoded by MPC_plannerN in seq_id.

        Every planner writes seq_id = planner_id * 1_000_000 + local_seq.
        This keeps plotter correct even when multiple planners share one
        /mpc_control_seq topic and therefore one ROS subscriber.
        """
        try:
            seq_id = int(getattr(msg, "seq_id"))
            encoded_idx = seq_id // 1000000
            if encoded_idx > 0:
                return int(encoded_idx)
        except Exception:
            pass
        try:
            return max(1, int(fallback_idx))
        except Exception:
            return 1

    @staticmethod
    def _trajectory_type_from_control_msg(msg):
        """Normalize planner stages: 1=avoidance, 2=mitigation."""
        try:
            trajectory_type = int(getattr(msg, "trajectory_type"))
        except Exception:
            trajectory_type = 1
        return 2 if trajectory_type == 2 else 1

    @staticmethod
    def _control_msg_order_key(msg):
        """Return a per-planner monotonic key used to reject stale Stage 2."""
        try:
            return int(getattr(msg, "seq_id"))
        except Exception:
            pass
        try:
            stamp = getattr(getattr(msg, "header"), "stamp")
            return int(stamp.to_nsec())
        except Exception:
            return int(rospy.Time.now().to_nsec())

    def _select_mpc_pred_markers_locked(self):
        """Select one prediction per planner with mitigation priority.

        Stage 2 is selected only when it is at least as new as the latest
        Stage 1 from the same planner.  Therefore a mitigation from the
        previous MPC cycle cannot mask a newly published avoidance trajectory.
        """
        selected = {}
        for planner_idx in sorted(self.mpc_pred_candidates):
            candidates = self.mpc_pred_candidates.get(planner_idx, {})
            avoid = candidates.get(1)
            mitigation = candidates.get(2)

            chosen = None
            if mitigation is not None and (avoid is None or mitigation[0] >= avoid[0]):
                chosen = mitigation
            elif avoid is not None:
                chosen = avoid
            elif mitigation is not None:
                chosen = mitigation

            if chosen is not None:
                selected[int(planner_idx)] = chosen[1]

        self.mpc_pred_markers = selected
        return [selected[k] for k in sorted(selected)]

    def publish_mpc_pred_traj_from_control(self, msg, planner_idx, points=None):
        if points is None:
            points = self._rollout_mpc_control_seq(msg)
        if not points:
            return

        planner_idx = self._planner_idx_from_control_msg(msg, planner_idx)
        trajectory_type = self._trajectory_type_from_control_msg(msg)
        order_key = self._control_msg_order_key(msg)

        marker = Marker()
        marker.header.frame_id = "base_link"
        if hasattr(msg, "header") and getattr(msg.header, "frame_id", ""):
            marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "mpc_pred_traj_all"
        marker.id = int(planner_idx)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(self.mpc_pred_marker_width), 1.0e-4)
        self.set_color(marker, self.planner_visual_color_for_idx(planner_idx))
        marker.points = points
        marker.lifetime = rospy.Duration(max(float(self.best_cmd_timeout_s), 0.05))

        with self.fast_marker_lock:
            planner_cache = self.mpc_pred_candidates.setdefault(int(planner_idx), {})
            planner_cache[int(trajectory_type)] = (order_key, marker)
            self.mpc_pred_markers_dirty = True

    def flush_fast_marker_arrays(self, _event=None, force=False):
        """Publish cached planner marker groups.

        With force=True, every non-empty group is republished on every common
        visualization tick even when no new planner callback arrived.
        """
        with self.fast_marker_lock:
            publish_mpc = bool(self.mpc_pred_candidates) if force else bool(self.mpc_pred_markers_dirty)
            publish_guidance = bool(self.initial_guidance_markers) if force else bool(self.initial_guidance_markers_dirty)
            publish_initial = bool(self.initial_traj_markers) if force else bool(self.initial_traj_markers_dirty)

            # Publish one trajectory per planner in one MarkerArray:
            # current mitigation if available, otherwise current avoidance.
            mpc_markers = self._select_mpc_pred_markers_locked() if publish_mpc else []
            guidance_markers = [self.initial_guidance_markers[k] for k in sorted(self.initial_guidance_markers)] if publish_guidance else []
            initial_markers = [self.initial_traj_markers[k] for k in sorted(self.initial_traj_markers)] if publish_initial else []

            self.mpc_pred_markers_dirty = False
            self.initial_guidance_markers_dirty = False
            self.initial_traj_markers_dirty = False

        stamp = rospy.Time.now()
        if publish_mpc:
            # Rebuild /mpc_pred_trajs from a clean RViz marker state every flush.
            # DELETEALL is inserted first so any trajectory left from the previous
            # publish is removed before the current planner trajectories are added.
            arr = MarkerArray()

            delete_all = Marker()
            delete_all.header.frame_id = "base_link"
            delete_all.header.stamp = stamp
            delete_all.action = Marker.DELETEALL
            arr.markers.append(delete_all)

            for marker in mpc_markers:
                marker.header.stamp = stamp
                arr.markers.append(marker)

            self.mpc_pred_traj_pubs.publish(arr)
        if publish_guidance:
            for marker in guidance_markers:
                marker.header.stamp = stamp
            arr = MarkerArray()
            arr.markers = guidance_markers
            self.initial_guidance_traj_pub.publish(arr)
        if publish_initial:
            for marker in initial_markers:
                marker.header.stamp = stamp
            arr = MarkerArray()
            arr.markers = initial_markers
            self.initial_traj_pub.publish(arr)

    @staticmethod
    def _trajectories_from_guidance_msg(msg):
        """Decode flattened Guidance arrays into [(x, y, t), ...] groups."""
        lengths = getattr(msg, "lengths", [])
        xs = getattr(msg, "xs", [])
        ys = getattr(msg, "ys", [])
        ts = getattr(msg, "ts", [])
        group_count = int(getattr(msg, "P", len(lengths)))
        group_count = min(max(group_count, 0), len(lengths))

        groups = []
        offset = 0
        total = min(len(xs), len(ys), len(ts))
        for i in range(group_count):
            n = max(0, int(lengths[i]))
            stop = min(offset + n, total)
            group = []
            for j in range(offset, stop):
                try:
                    x = float(xs[j])
                    y = float(ys[j])
                    t = float(ts[j])
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(t):
                    group.append((x, y, t))
            groups.append(group)
            offset += n
        return groups

    def _visibility_point(self, xyz):
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(self.visibility_z_scale) * float(xyz[2])
        return p

    def publish_visibility_prm_graph_from_data(self, msg, stamp=None, pub=None):
        groups = self._trajectories_from_guidance_msg(msg)
        while len(groups) < 4:
            groups.append([])
        start_nodes, goal_nodes, sample_nodes, edge_points = groups[:4]

        frame_id = self.get_frame_id(msg)
        if stamp is None:
            stamp = rospy.Time.now()
        arr = MarkerArray()

        nodes = Marker()
        nodes.header.frame_id = frame_id
        nodes.header.stamp = stamp
        nodes.ns = "prm_nodes"
        nodes.id = 0
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD
        nodes.pose.orientation.w = 1.0
        nodes.scale.x = 0.12
        nodes.scale.y = 0.12
        nodes.scale.z = 0.12
        self.set_color(nodes, (1.0, 1.0, 1.0, 1.0))

        for group, rgba in (
            (start_nodes, (0.1, 1.0, 0.1, 1.0)),
            (goal_nodes, (1.0, 0.4, 0.1, 1.0)),
            (sample_nodes, (0.9, 0.9, 0.9, 1.0)),
        ):
            for xyz in group:
                nodes.points.append(self._visibility_point(xyz))
                color = nodes.color.__class__()
                color.r, color.g, color.b, color.a = [float(v) for v in rgba]
                nodes.colors.append(color)
        arr.markers.append(nodes)

        edges = Marker()
        edges.header.frame_id = frame_id
        edges.header.stamp = stamp
        edges.ns = "prm_edges"
        edges.id = 1
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.pose.orientation.w = 1.0
        edges.scale.x = 0.03
        self.set_color(edges, (0.6, 0.6, 0.6, 0.8))
        edges.points = [self._visibility_point(xyz) for xyz in edge_points]
        arr.markers.append(edges)

        if pub is None:
            pub = self.visibility_prm_pub
        pub.publish(arr)

    def visibility_prm_graph_callback(self, msg):
        self.latest_visibility_prm_graph_msg = msg
        self._visibility_prm_dirty = True

    def visibility_backward_prm_graph_callback(self, msg):
        self.latest_visibility_backward_prm_graph_msg = msg
        self._visibility_backward_prm_dirty = True

    def _make_visibility_guide_markers(
        self, guidances, frame_id, stamp, guidance_slot_start=1,
        initialized_attr="visibility_guide_marker_initialized",
        prev_count_attr="prev_visibility_guide_count",
    ):
        arr = MarkerArray()

        # Reset the topic once after startup. Repeating DELETEALL every frame
        # forces RViz to destroy and recreate every guide/FOV marker, which is
        # substantially slower than stable-ID replacement.
        if not bool(getattr(self, initialized_attr, False)):
            delete_all = Marker()
            delete_all.header.frame_id = frame_id
            delete_all.header.stamp = stamp
            delete_all.action = Marker.DELETEALL
            arr.markers.append(delete_all)
            setattr(self, initialized_attr, True)

        fov = Marker()
        fov.header.frame_id = frame_id
        fov.header.stamp = stamp
        fov.ns = "guide_fov"
        fov.id = 10
        fov.type = Marker.LINE_STRIP
        fov.action = Marker.ADD
        fov.pose.orientation.w = 1.0
        fov.scale.x = 0.035
        self.set_color(fov, (1.0, 1.0, 1.0, 0.8))
        half_deg = float(self.obs_fov_deg)
        steps = 32
        th = float(self.ego_yaw) - math.radians(half_deg)
        fov.points.append(self._visibility_point((self.ego_cart_x, self.ego_cart_y, 0.0)))
        fov.points.append(self._visibility_point((
            self.ego_cart_x + self.ego_FOV * math.cos(th),
            self.ego_cart_y + self.ego_FOV * math.sin(th),
            0.0,
        )))
        for j in range(steps + 1):
            th_deg = -half_deg + 2.0 * half_deg * float(j) / float(max(1, steps))
            th = float(self.ego_yaw) + math.radians(th_deg)
            fov.points.append(self._visibility_point((
                self.ego_cart_x + self.ego_FOV * math.cos(th),
                self.ego_cart_y + self.ego_FOV * math.sin(th),
                0.0,
            )))
        th = float(self.ego_yaw) + math.radians(half_deg)
        fov.points.append(self._visibility_point((
            self.ego_cart_x + self.ego_FOV * math.cos(th),
            self.ego_cart_y + self.ego_FOV * math.sin(th),
            0.0,
        )))
        fov.points.append(self._visibility_point((self.ego_cart_x, self.ego_cart_y, 0.0)))
        arr.markers.append(fov)

        active_guide_indices = set()
        for i, traj in enumerate(guidances):
            if len(traj) < 2:
                continue
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "guide_paths"
            marker.id = 1000 + i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.06
            self.set_color(marker, self.initial_guidance_color_for_idx(int(guidance_slot_start) + i))
            marker.points = [self._visibility_point(xyz) for xyz in traj]
            arr.markers.append(marker)

            # Also show the actual MPC-grid waypoints associated with this
            # candidate.  visibility_prm_forward uses slots 1..3 and the
            # backward branch uses slots 4..6.  The cached Point.z values are
            # already TIME_SCALE * t, matching the rest of this plotter.
            slot_idx = int(guidance_slot_start) + i
            with self.fast_marker_lock:
                cached_waypoints = self.visibility_initial_guidance_waypoints.get(slot_idx, ())
                waypoint_points = [self._copy_point(p) for p in cached_waypoints]
            if waypoint_points:
                waypoint_marker = Marker()
                waypoint_marker.header.frame_id = frame_id
                waypoint_marker.header.stamp = stamp
                waypoint_marker.ns = "guide_waypoints"
                waypoint_marker.id = 2000 + i
                waypoint_marker.type = Marker.SPHERE_LIST
                waypoint_marker.action = Marker.ADD
                waypoint_marker.pose.orientation.w = 1.0
                waypoint_marker.scale.x = float(self.visibility_guide_waypoint_scale)
                waypoint_marker.scale.y = float(self.visibility_guide_waypoint_scale)
                waypoint_marker.scale.z = float(self.visibility_guide_waypoint_scale)
                self.set_color(
                    waypoint_marker,
                    self.initial_guidance_color_for_idx(slot_idx),
                )
                waypoint_marker.points = waypoint_points
                arr.markers.append(waypoint_marker)

            active_guide_indices.add(i)

        # Delete only paths/waypoints that disappeared or became empty.
        previous_count = int(getattr(self, prev_count_attr, 0))
        for i in range(max(len(guidances), previous_count)):
            if i in active_guide_indices:
                continue
            stale = Marker()
            stale.header.frame_id = frame_id
            stale.header.stamp = stamp
            stale.ns = "guide_paths"
            stale.id = 1000 + i
            stale.action = Marker.DELETE
            arr.markers.append(stale)

            stale_waypoint = Marker()
            stale_waypoint.header.frame_id = frame_id
            stale_waypoint.header.stamp = stamp
            stale_waypoint.ns = "guide_waypoints"
            stale_waypoint.id = 2000 + i
            stale_waypoint.action = Marker.DELETE
            arr.markers.append(stale_waypoint)
        setattr(self, prev_count_attr, len(guidances))
        return arr

    def _publish_visibility_guide_paths(self, guidances, frame_id, stamp, pubs=None):
        if pubs is None:
            pubs = self.visibility_guide_path_pubs
        for i, pub in enumerate(pubs):
            path = Path()
            path.header.frame_id = frame_id
            path.header.stamp = stamp
            if i < len(guidances):
                for xyz in guidances[i]:
                    ps = PoseStamped()
                    ps.header.frame_id = frame_id
                    ps.header.stamp = stamp
                    ps.pose.position = self._visibility_point(xyz)
                    ps.pose.orientation.w = 1.0
                    path.poses.append(ps)
            pub.publish(path)

    def publish_visibility_guidance_from_data(
        self, msg, stamp=None, marker_pub=None, path_pubs=None, guidance_slot_start=1,
        initialized_attr="visibility_guide_marker_initialized",
        prev_count_attr="prev_visibility_guide_count",
    ):
        guidances = self._trajectories_from_guidance_msg(msg)
        frame_id = self.get_frame_id(msg)
        if stamp is None:
            stamp = rospy.Time.now()
        if marker_pub is None:
            marker_pub = self.visibility_guide_marker_pub
        marker_pub.publish(
            self._make_visibility_guide_markers(
                guidances, frame_id, stamp,
                guidance_slot_start=guidance_slot_start,
                initialized_attr=initialized_attr,
                prev_count_attr=prev_count_attr,
            )
        )
        self._publish_visibility_guide_paths(guidances, frame_id, stamp, pubs=path_pubs)

    def visibility_guidance_callback(self, msg):
        self.latest_visibility_guidance_msg = msg
        self._visibility_guidance_dirty = True

    def visibility_backward_guidance_callback(self, msg):
        self.latest_visibility_backward_guidance_msg = msg
        self._visibility_backward_guidance_dirty = True

    def _points_from_control_seq_reference(self, msg):
        xs = getattr(msg, "xs", None) if hasattr(msg, "xs") else None
        ys = getattr(msg, "ys", None) if hasattr(msg, "ys") else None
        if xs is not None and ys is not None:
            try:
                n = min(len(xs), len(ys))
            except Exception:
                n = 0
            if n > 0:
                dt = float(getattr(msg, "dt", self.pred_dt))
                if not np.isfinite(dt) or dt <= 1.0e-9:
                    dt = float(self.pred_dt)
                ts = getattr(msg, "ts", None) if hasattr(msg, "ts") else None
                points = []
                for i in range(n):
                    try:
                        x = float(xs[i])
                        y = float(ys[i])
                    except Exception:
                        continue
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    if ts is not None and i < len(ts):
                        try:
                            t_i = float(ts[i])
                        except Exception:
                            t_i = float(i) * dt
                    else:
                        t_i = float(i) * dt
                    if not np.isfinite(t_i):
                        t_i = float(i) * dt
                    p = Point()
                    p.x = x
                    p.y = y
                    p.z = float(TIME_SCALE) * float(t_i)
                    points.append(p)
                if len(points) > 0:
                    return points
        return self._rollout_mpc_control_seq(msg)

    @staticmethod
    def _copy_point(p):
        q = Point()
        q.x = float(p.x)
        q.y = float(p.y)
        q.z = float(p.z)
        return q

    def publish_initial_guidance_from_control(self, msg, planner_idx):
        points = self._points_from_control_seq_reference(msg)
        if len(points) < 2:
            return

        marker = Marker()
        marker.header.frame_id = "base_link"
        if hasattr(msg, "header") and getattr(msg.header, "frame_id", ""):
            marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "initial_guidance_dotted"
        marker.id = int(planner_idx)
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(self.initial_guidance_marker_width), 1.0e-4)
        self.set_color(marker, self.initial_guidance_color_for_idx(planner_idx))
        marker.lifetime = rospy.Duration(max(float(self.initial_guidance_lifetime_s), 0.05))

        # RViz has no native dotted LINE_STRIP.  Draw every other short segment
        # as a LINE_LIST so the raw visibility guidance appears dotted/dashed.
        for i in range(len(points) - 1):
            if i % 2 != 0:
                continue
            marker.points.append(self._copy_point(points[i]))
            marker.points.append(self._copy_point(points[i + 1]))

        with self.fast_marker_lock:
            self.initial_guidance_markers[int(planner_idx)] = marker
            # Keep the full N+1 reference points for guide_markers SPHERE_LIST.
            self.visibility_initial_guidance_waypoints[int(planner_idx)] = [
                self._copy_point(p) for p in points
            ]
            self.initial_guidance_markers_dirty = True

    def initial_guidance_callback(self, msg, planner_idx=None):
        if planner_idx is None:
            planner_idx = 1
        self.publish_initial_guidance_from_control(msg, int(planner_idx))

    def initial_traj_callback(self, msg, planner_idx=None):
        if planner_idx is None:
            planner_idx = 1
        idx = int(planner_idx)
        if not hasattr(msg, "points") or len(msg.points) < 2:
            return

        marker = Marker()
        marker.header.frame_id = self.get_frame_id(msg)
        marker.header.stamp = rospy.Time.now()
        marker.ns = "initial_traj_solid"
        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(self.initial_traj_marker_width), 1.0e-4)
        self.set_color(marker, self.initial_guidance_color_for_idx(idx))
        marker.points = [self._copy_point(p) for p in msg.points]
        marker.lifetime = rospy.Duration(max(float(self.initial_traj_lifetime_s), 0.05))

        with self.fast_marker_lock:
            self.initial_traj_markers[idx] = marker
            self.initial_traj_markers_dirty = True

    def mpc_control_callback(self, msg, planner_idx=None):
        dt = float(msg.dt) if hasattr(msg, "dt") else self.pred_dt
        if not np.isfinite(dt) or dt <= 1.0e-9:
            dt = self.pred_dt
        cost = float(getattr(msg, "cost", 1.0e100))
        if not np.isfinite(cost):
            cost = 1.0e100
        now = rospy.Time.now()
        if self.best_cmd_stamp == rospy.Time(0):
            stale = True
        else:
            stale = (now - self.best_cmd_stamp).to_sec() > float(self.best_cmd_timeout_s)
        # One rollout supplies both the best-state cache and the RViz path.
        states, points = self._rollout_mpc_control_seq_outputs(msg)
        if stale or cost <= float(self.best_traj_cost):
            self.best_traj_dt = dt
            self.best_traj_cost = cost
            self.best_cmd_stamp = now
            self.best_traj_states = states
            self._selector_horizon_dirty = True

        if planner_idx is None:
            planner_idx = 1
        # Prefer the planner id encoded by MPC_plannerN in seq_id.  The callback
        # argument remains a fallback for legacy messages without seq_id.
        planner_idx = self._planner_idx_from_control_msg(msg, planner_idx)
        self.publish_mpc_pred_traj_from_control(msg, int(planner_idx), points=points)

    def polygon_signed_area(self, vertices):
        arr = np.asarray(vertices, dtype=float).reshape(-1, 2)
        if arr.shape[0] < 3:
            return 0.0
        x = arr[:, 0]
        y = arr[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def publish_selected_vertex_markers_from_data(self, msg):
        """Draw the exact feature positions/normals published by vertex_selector."""
        frame_id = self.get_frame_id(msg)
        stamp = rospy.Time.now()

        point_marker = Marker()
        point_marker.header.frame_id = frame_id
        point_marker.header.stamp = stamp
        point_marker.ns = "selected_vertex_set"
        point_marker.id = 0
        point_marker.type = Marker.POINTS
        point_marker.action = Marker.ADD
        point_marker.pose.orientation.w = 1.0
        point_marker.scale.x = max(float(self.selected_vertex_marker_width), 1.0e-4)
        point_marker.scale.y = max(float(self.selected_vertex_marker_width), 1.0e-4)
        self.set_color(point_marker, self.selected_vertex_color)

        normal_array = MarkerArray()
        z_vis = 0.06
        arrow_len = max(float(self.selected_vertex_normal_arrow_length), 1.0e-6)
        n_msg = min(
            max(0, int(getattr(msg, "obs_cnt", 0))),
            len(getattr(msg, "obs_xs", [])),
            len(getattr(msg, "obs_ys", [])),
            len(getattr(msg, "obs_vxs", [])),
            len(getattr(msg, "obs_vys", [])),
        )

        valid_features = []
        for i in range(n_msg):
            x = float(msg.obs_xs[i])
            y = float(msg.obs_ys[i])
            nx = float(msg.obs_vxs[i])
            ny = float(msg.obs_vys[i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            p = Point()
            p.x = x
            p.y = y
            p.z = z_vis
            point_marker.points.append(p)

            norm = math.hypot(nx, ny)
            if not np.isfinite(norm) or norm <= 1.0e-12:
                continue
            valid_features.append((x, y, nx / norm, ny / norm))

        if self.selected_vertex_normal_marker_mode != "arrows":
            line_marker = Marker()
            line_marker.header.frame_id = frame_id
            line_marker.header.stamp = stamp
            line_marker.ns = "selected_vertex_normal"
            line_marker.id = 0
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = max(float(self.selected_vertex_normal_shaft_diameter), 1.0e-4)
            self.set_color(line_marker, self.selected_vertex_normal_color)

            for x, y, nx, ny in valid_features:
                tail = Point()
                tail.x = x
                tail.y = y
                tail.z = z_vis
                head = Point()
                head.x = x + arrow_len * nx
                head.y = y + arrow_len * ny
                head.z = z_vis
                line_marker.points.append(tail)
                line_marker.points.append(head)
            normal_array.markers.append(line_marker)

            # If a previous configuration used one Marker.ARROW per normal,
            # delete the now-unused arrow IDs while retaining stable line ID 0.
            for stale_id in range(1, int(self.prev_selected_vertex_normal_arrow_cnt)):
                marker_delete = Marker()
                marker_delete.header.frame_id = frame_id
                marker_delete.header.stamp = stamp
                marker_delete.ns = "selected_vertex_normal"
                marker_delete.id = int(stale_id)
                marker_delete.action = Marker.DELETE
                normal_array.markers.append(marker_delete)
            self.prev_selected_vertex_normal_arrow_cnt = 1
        else:
            arrow_count = 0
            for x, y, nx, ny in valid_features:
                tail = Point()
                tail.x = x
                tail.y = y
                tail.z = z_vis
                head = Point()
                head.x = x + arrow_len * nx
                head.y = y + arrow_len * ny
                head.z = z_vis

                arrow = Marker()
                arrow.header.frame_id = frame_id
                arrow.header.stamp = stamp
                arrow.ns = "selected_vertex_normal"
                arrow.id = int(arrow_count)
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                arrow.scale.x = max(float(self.selected_vertex_normal_shaft_diameter), 1.0e-4)
                arrow.scale.y = max(float(self.selected_vertex_normal_head_diameter), 1.0e-4)
                arrow.scale.z = max(float(self.selected_vertex_normal_head_length), 1.0e-4)
                self.set_color(arrow, self.selected_vertex_normal_color)
                arrow.points.append(tail)
                arrow.points.append(head)
                normal_array.markers.append(arrow)
                arrow_count += 1

            for stale_id in range(arrow_count, int(self.prev_selected_vertex_normal_arrow_cnt)):
                marker_delete = Marker()
                marker_delete.header.frame_id = frame_id
                marker_delete.header.stamp = stamp
                marker_delete.ns = "selected_vertex_normal"
                marker_delete.id = int(stale_id)
                marker_delete.action = Marker.DELETE
                normal_array.markers.append(marker_delete)
            self.prev_selected_vertex_normal_arrow_cnt = int(arrow_count)

        self._cached_selected_vertex_point_marker = point_marker
        self._cached_selected_vertex_normal_array = normal_array
        self._selected_vertex_dirty = False
        self.selected_vertex_set_pub.publish(point_marker)
        self.selected_vertex_normal_pub.publish(normal_array)

    def selected_vertex_data_callback(self, msg):
        self.latest_selected_vertex_data_msg = msg
        self._selected_vertex_field_arrays_cache_msg = None
        self._selected_vertex_field_arrays_cache = None
        self._selected_vertex_dirty = True
        # h=0 of /obs_polygons_horizon comes from this message.
        self._selector_horizon_dirty = True

    def selected_vertex_horizon_data_callback(self, msg):
        # This is the selector-owned stage-wise geometry used by MPC.  Keep only
        # the newest payload; RViz rendering happens in the common publish loop.
        self.latest_selected_vertex_horizon_data_msg = msg
        self._selector_horizon_dirty = True

    def separating_hyperplane_callback(self, msg):
        self.latest_separating_hyperplane_msg = msg
        self._separating_hyperplane_dirty = True

    def perceived_polygon_field_arrays(self, msg):
        """Flatten perceived raw polygons in the format expected by utils_linearized.

        ``utils_linearized`` constructs the field from the perceived augmented
        obstacle geometry.  For field_flag=0 these are the ordered raw boundary
        vertices of the complete occluded-region-augmented obstacle.  In particular, field_flag=4
        builds edge midpoints/normals and the configured pseudo-vertex normals
        internally, so the SDF plotter must not feed already-constructed selector
        features into that path.

        Returns:
            (vertices_world, None, vertex_types, obstacle_ids), where each polygon
            keeps its message order and every vertex carries the polygon's type and
            group ID.  ``None`` normals are intentional: the unified evaluator does
            not need caller-supplied normals for raw-polygon field construction.
        """
        if msg is None:
            return None
        if (msg is self._perceived_field_arrays_cache_msg and
                self._perceived_field_arrays_cache is not None):
            return self._perceived_field_arrays_cache

        try:
            obs_cnt = min(max(0, int(getattr(msg, "obs_cnt", 0))), MAX_OBS)
        except Exception:
            obs_cnt = 0
        if obs_cnt <= 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                None,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        vertices = []
        vertex_types = []
        obstacle_ids = []
        for obs_idx in range(obs_cnt):
            polygon = self.get_obstacle_polygon_vertices(msg, obs_idx)
            if len(polygon) < 3:
                continue
            obs_type = int(self.get_obstacle_type(msg, obs_idx))
            for x, y in polygon:
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                vertices.append((float(x), float(y)))
                vertex_types.append(obs_type)
                obstacle_ids.append(int(obs_idx))

        if not vertices:
            # A nonempty perceived-obstacle message without polygon vertices is
            # a format/producer mismatch, not free space.
            return None

        result = (
            np.ascontiguousarray(vertices, dtype=np.float64),
            None,
            np.ascontiguousarray(vertex_types, dtype=np.int32),
            np.ascontiguousarray(obstacle_ids, dtype=np.int32),
        )
        self._perceived_field_arrays_cache_msg = msg
        self._perceived_field_arrays_cache = result
        return result

    def selected_vertex_field_arrays(self, msg):
        """Decode selector features for the collision-field evaluator.

        ``obstacle_polygon`` carries selected raw boundary-vertex positions in
        ``obs_xs/ys``, dynamic/static labels in ``obs_types``, and field geometry
        group IDs in ``obs_rs``.  Normals remain zero because the downstream
        evaluator constructs every method in the ego body-fixed frame.  This is
        the SDF geometry source for fields 1--4, matching the MPC convex groups.
        """
        if msg is None:
            return None
        if (msg is self._selected_vertex_field_arrays_cache_msg and
                self._selected_vertex_field_arrays_cache is not None):
            return self._selected_vertex_field_arrays_cache
        n = min(
            max(0, int(getattr(msg, "obs_cnt", 0))),
            len(getattr(msg, "obs_xs", [])),
            len(getattr(msg, "obs_ys", [])),
            len(getattr(msg, "obs_vxs", [])),
            len(getattr(msg, "obs_vys", [])),
        )
        if n <= 0:
            return None

        points = []
        normals = []
        vertex_types = []
        obstacle_ids = []
        for i in range(n):
            row = [
                float(msg.obs_xs[i]), float(msg.obs_ys[i]),
                float(msg.obs_vxs[i]), float(msg.obs_vys[i]),
            ]
            if not np.all(np.isfinite(row)):
                continue
            points.append(row[:2])
            normals.append(row[2:])
            vertex_types.append(
                1 if hasattr(msg, "obs_types") and i < len(msg.obs_types)
                and int(msg.obs_types[i]) == 1 else 0
            )
            obstacle_ids.append(
                int(round(float(msg.obs_rs[i])))
                if hasattr(msg, "obs_rs") and i < len(msg.obs_rs) else 0
            )

        if not points:
            return None
        result = (
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(normals, dtype=np.float64),
            np.ascontiguousarray(vertex_types, dtype=np.int32),
            np.ascontiguousarray(obstacle_ids, dtype=np.int32),
        )
        self._selected_vertex_field_arrays_cache_msg = msg
        self._selected_vertex_field_arrays_cache = result
        return result

    def publish_sdf_grid_axes(self):
        """Publish ego-relative x/y offsets used by /sdf.

        The origin (0, 0) of these axes is the current ego position.  The
        corresponding world coordinates are ego_position + (x_offset, y_offset).
        """
        x_msg = Float64MultiArray()
        x_msg.layout.dim = [
            MultiArrayDimension(
                label="x",
                size=int(self.sdf_grid_x.size),
                stride=int(self.sdf_grid_x.size),
            )
        ]
        x_msg.data = self.sdf_grid_x.astype(np.float64, copy=False).tolist()
        self.sdf_grid_x_pub.publish(x_msg)

        y_msg = Float64MultiArray()
        y_msg.layout.dim = [
            MultiArrayDimension(
                label="y",
                size=int(self.sdf_grid_y.size),
                stride=int(self.sdf_grid_y.size),
            )
        ]
        y_msg.data = self.sdf_grid_y.astype(np.float64, copy=False).tolist()
        self.sdf_grid_y_pub.publish(y_msg)

    def _sdf_rviz_color(self, value, clip_abs):
        """Map signed Phi to a diverging red-white-blue RViz color."""
        clip = max(float(clip_abs), 1.0e-12)
        u = min(abs(float(value)) / clip, 1.0)
        color = ColorRGBA()
        if float(value) < 0.0:
            # Negative: white near zero -> red at -clip.
            color.r = 1.0
            color.g = 1.0 - u
            color.b = 1.0 - u
        else:
            # Positive: white near zero -> blue at +clip.
            color.r = 1.0 - u
            color.g = 1.0 - u
            color.b = 1.0
        color.a = float(self.sdf_rviz_alpha)
        return color

    @staticmethod
    def _sdf_zero_crossing(p0, v0, p1, v1):
        """Linearly interpolate a Phi=0 crossing on one grid-cell edge."""
        v0 = float(v0)
        v1 = float(v1)
        denom = v0 - v1
        if abs(denom) <= 1.0e-12:
            t = 0.5
        else:
            t = v0 / denom
        t = min(1.0, max(0.0, t))
        return (
            float(p0[0]) + t * (float(p1[0]) - float(p0[0])),
            float(p0[1]) + t * (float(p1[1]) - float(p0[1])),
        )

    def publish_sdf_rviz_markers(
        self, sdf_grid, stamp=None, frame_id="base_link", ego_origin=None
    ):
        """Publish a 2-D SDF heat map and Phi=0 contour for RViz.

        Marker 0 is POINTS with one per-point color.  Negative Phi is red,
        positive Phi is blue, and values near zero are white.  Marker 1 is a
        black LINE_LIST approximating the zero level set using grid-edge
        interpolation (marching-squares style).
        """
        if not bool(self.publish_sdf_rviz):
            return
        grid = np.asarray(sdf_grid, dtype=np.float64)
        ny, nx = grid.shape if grid.ndim == 2 else (0, 0)
        if nx <= 0 or ny <= 0:
            return
        if stamp is None:
            stamp = rospy.Time.now()
        if ego_origin is None:
            ego_origin = (float(self.ego_cart_x), float(self.ego_cart_y))
        ego_origin_x = float(ego_origin[0])
        ego_origin_y = float(ego_origin[1])

        finite_abs = np.abs(grid[np.isfinite(grid)])
        if float(self.sdf_rviz_clip_abs) > 0.0:
            clip_abs = float(self.sdf_rviz_clip_abs)
        elif finite_abs.size > 0:
            clip_abs = max(float(np.max(finite_abs)), 1.0e-6)
        else:
            clip_abs = 1.0

        arr = MarkerArray()

        heat = Marker()
        heat.header.frame_id = str(frame_id) if frame_id else "base_link"
        heat.header.stamp = stamp
        heat.ns = "sdf_heatmap"
        heat.id = 0
        heat.type = Marker.POINTS
        heat.action = Marker.ADD
        # Marker points are expressed in the ego-centered local SDF grid.
        # Translate the whole marker so local (0, 0) coincides with ego_position.
        heat.pose.position.x = ego_origin_x
        heat.pose.position.y = ego_origin_y
        heat.pose.position.z = 0.0
        heat.pose.orientation.w = 1.0
        cell_size = max(
            float(self.sdf_grid_resolution) * float(self.sdf_rviz_stride) * 0.95,
            1.0e-4,
        )
        heat.scale.x = cell_size
        heat.scale.y = cell_size
        heat.scale.z = 1.0

        stride = max(1, int(self.sdf_rviz_stride))
        z = float(self.sdf_rviz_z)
        heat.points = self._sdf_rviz_heat_points
        heat.colors = [
            self._sdf_rviz_color(float(grid[iy, ix]), clip_abs)
            for iy, ix in self._sdf_rviz_heat_indices
        ]
        arr.markers.append(heat)

        contour = Marker()
        contour.header.frame_id = heat.header.frame_id
        contour.header.stamp = stamp
        contour.ns = "sdf_zero_contour"
        contour.id = 1
        contour.type = Marker.LINE_LIST
        contour.action = Marker.ADD
        # Use the same ego-centered local coordinate system as the heat map.
        contour.pose.position.x = ego_origin_x
        contour.pose.position.y = ego_origin_y
        contour.pose.position.z = 0.0
        contour.pose.orientation.w = 1.0
        contour.scale.x = float(self.sdf_zero_contour_width)
        self.set_color(contour, (0.0, 0.0, 0.0, 1.0))
        contour_z = z + 0.003

        # Edge order: bottom, right, top, left.  Each cell normally yields two
        # crossings.  Ambiguous four-crossing cells are represented as two
        # segments, which is sufficient for visualization of the zero set.
        for iy, iy1, ix, ix1, x0, x1, y0, y1 in self._sdf_rviz_contour_cells:
            vals = (
                float(grid[iy, ix]),
                float(grid[iy, ix1]),
                float(grid[iy1, ix1]),
                float(grid[iy1, ix]),
            )
            pts = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            crossings = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                va, vb = vals[a], vals[b]
                if va == 0.0 and vb == 0.0:
                    continue
                if va == 0.0 or vb == 0.0 or (va < 0.0) != (vb < 0.0):
                    crossings.append(self._sdf_zero_crossing(pts[a], va, pts[b], vb))
            if len(crossings) < 2:
                continue
            pair_count = len(crossings) // 2
            for pair_idx in range(pair_count):
                a = crossings[2 * pair_idx]
                b = crossings[2 * pair_idx + 1]
                pa = Point()
                pa.x, pa.y, pa.z = float(a[0]), float(a[1]), contour_z
                pb = Point()
                pb.x, pb.y, pb.z = float(b[0]), float(b[1]), contour_z
                contour.points.extend([pa, pb])
        arr.markers.append(contour)

        self.sdf_rviz_pub.publish(arr)

    def _bp_visualization_body_frame(self, reference_state, points_world, states_world=None):
        """Map BP visualization geometry/query states to ego body-fixed normalized coordinates.

        The transform matches the non-BP collision fields:
            q = S^{-1} R(psi_ref)^T (p - p_ref),
        where S contains the circumscribed ego-ellipse semi-axes.  BP fitting and
        signed-distance evaluation are performed entirely in this body-fixed
        frame.  RViz still draws the resulting scalar samples at their original
        world-grid positions.
        """
        ref = np.asarray(reference_state, dtype=np.float64).reshape(4)
        yaw = float(ref[3])
        c = math.cos(yaw)
        s = math.sin(yaw)

        axis_scale = getattr(self.sdf_cfg, "ego_axis_scale", None)
        if axis_scale is None:
            sqrt2 = math.sqrt(2.0)
            sx = max(abs(float(getattr(self.sdf_cfg, "ego_L", self.ego_L))) / sqrt2, 1.0e-12)
            sy = max(abs(float(getattr(self.sdf_cfg, "ego_W", self.ego_W))) / sqrt2, 1.0e-12)
        else:
            scale = np.asarray(axis_scale, dtype=np.float64).reshape(-1)
            if scale.size < 2:
                sqrt2 = math.sqrt(2.0)
                sx = max(abs(float(getattr(self.sdf_cfg, "ego_L", self.ego_L))) / sqrt2, 1.0e-12)
                sy = max(abs(float(getattr(self.sdf_cfg, "ego_W", self.ego_W))) / sqrt2, 1.0e-12)
            else:
                sx = max(abs(float(scale[0])), 1.0e-12)
                sy = max(abs(float(scale[1])), 1.0e-12)

        P = np.asarray(points_world, dtype=np.float64)
        p_shape = P.shape
        P2 = P.reshape(-1, 2)
        dx = P2[:, 0] - float(ref[0])
        dy = P2[:, 1] - float(ref[1])
        P_body = np.empty_like(P2)
        P_body[:, 0] = (c * dx + s * dy) / sx
        P_body[:, 1] = (-s * dx + c * dy) / sy
        P_body = np.ascontiguousarray(P_body.reshape(p_shape), dtype=np.float64)

        ref_body = np.array([0.0, 0.0, float(ref[2]), 0.0], dtype=np.float64)
        if states_world is None:
            return P_body, None, ref_body

        X = np.ascontiguousarray(np.asarray(states_world, dtype=np.float64), dtype=np.float64)
        X_body = X.copy()
        qdx = X[:, 0] - float(ref[0])
        qdy = X[:, 1] - float(ref[1])
        X_body[:, 0] = (c * qdx + s * qdy) / sx
        X_body[:, 1] = (-s * qdx + c * qdy) / sy
        # The fitted BP boundary is already expressed in the reference body frame;
        # query yaw is therefore zero in that chart. Velocity is irrelevant to BP.
        X_body[:, 3] = 0.0
        return P_body, np.ascontiguousarray(X_body, dtype=np.float64), ref_body


    def _bp_sdf_grid_world_meter_equivalent(self, sdf_grid_E, reference_state):
        """Convert a normalized body-fixed BP-SDF grid to local world meters.

        This matches the field-comparison conversion used for normalized-E
        fields. For q_E = D_E^{-1} R^T q_w and unit E-frame level-set normal
        n_E, one normalized E unit corresponds locally to

            scale_m_per_E = 1 / ||D_E^{-T} n_E||.

        The E-frame normal is recovered from the sampled BP field gradient. A
        strictly positive scale preserves both the sign and the zero level set.
        Impact-normal visualization continues to use the raw normalized-E BP
        field, because the angular comparison is defined in that frame.
        """
        F_E = np.asarray(sdf_grid_E, dtype=np.float64)
        if F_E.ndim != 2 or F_E.size == 0:
            return np.asarray(F_E, dtype=np.float64)

        ref = np.asarray(reference_state, dtype=np.float64).reshape(4)
        yaw = float(ref[3])
        c = math.cos(yaw)
        s = math.sin(yaw)

        axis_scale = getattr(self.sdf_cfg, "ego_axis_scale", None)
        if axis_scale is None:
            sqrt2 = math.sqrt(2.0)
            sx = max(abs(float(getattr(self.sdf_cfg, "ego_L", self.ego_L))) / sqrt2, 1.0e-12)
            sy = max(abs(float(getattr(self.sdf_cfg, "ego_W", self.ego_W))) / sqrt2, 1.0e-12)
        else:
            scale = np.asarray(axis_scale, dtype=np.float64).reshape(-1)
            if scale.size < 2:
                sqrt2 = math.sqrt(2.0)
                sx = max(abs(float(getattr(self.sdf_cfg, "ego_L", self.ego_L))) / sqrt2, 1.0e-12)
                sy = max(abs(float(getattr(self.sdf_cfg, "ego_W", self.ego_W))) / sqrt2, 1.0e-12)
            else:
                sx = max(abs(float(scale[0])), 1.0e-12)
                sy = max(abs(float(scale[1])), 1.0e-12)

        ny, nx = F_E.shape
        if nx < 2 or ny < 2:
            return F_E * math.sqrt(sx * sy)

        # np.gradient returns derivatives in axis order: y first, then x.
        # sdf_grid_x/y are world-meter coordinates, so these are world-frame
        # derivatives of the normalized-E scalar field.
        edge_order = 2 if nx >= 3 and ny >= 3 else 1
        dF_dy_w, dF_dx_w = np.gradient(
            F_E,
            np.asarray(self.sdf_grid_y, dtype=np.float64),
            np.asarray(self.sdf_grid_x, dtype=np.float64),
            edge_order=edge_order,
        )

        # g_E = D_E R^T g_w because q_E = D_E^{-1} R^T q_w.
        g_body_x = c * dF_dx_w + s * dF_dy_w
        g_body_y = -s * dF_dx_w + c * dF_dy_w
        gE_x = sx * g_body_x
        gE_y = sy * g_body_y
        grad_norm_E = np.hypot(gE_x, gE_y)

        finite = np.isfinite(F_E) & np.isfinite(gE_x) & np.isfinite(gE_y)
        valid = finite & (grad_norm_E > 1.0e-12)

        scale_m_per_E = np.full(F_E.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            nE_x = np.zeros(F_E.shape, dtype=np.float64)
            nE_y = np.zeros(F_E.shape, dtype=np.float64)
            nE_x[valid] = gE_x[valid] / grad_norm_E[valid]
            nE_y[valid] = gE_y[valid] / grad_norm_E[valid]
            metric_norm = np.hypot(nE_x / sx, nE_y / sy)
            good_metric = valid & np.isfinite(metric_norm) & (metric_norm > 1.0e-12)
            scale_m_per_E[good_metric] = 1.0 / metric_norm[good_metric]
            valid = good_metric

        # Same finite fallback as the MATLAB comparison helper when the local
        # field gradient vanishes: geometric mean of the ellipse radii.
        fallback = np.isfinite(F_E) & ~valid
        scale_m_per_E[fallback] = math.sqrt(sx * sy)

        return F_E * scale_m_per_E

    def _evaluate_sdf_grid_batch(
        self, x_state, points, normals, vertex_types, obstacle_ids,
        reference_state=None, bp_model=None
    ):
        """Evaluate the configured Phi at world-frame query states.

        For field_flag=2, visualization first maps obstacle geometry and query states
        into the current ego body-fixed normalized frame. collision_lin_cpp then
        fits the Bernstein boundary and returns signed distance in that frame.
        The samples are associated back with the original world-grid positions.
        Other fields keep using the unified MPC evaluator.
        """
        query = np.ascontiguousarray(x_state, dtype=np.float64)
        count = int(query.shape[0])
        if count <= 0:
            return np.zeros((0,), dtype=np.float64)

        field_flag = int(_utils_linearized.collision_field_flag(self.sdf_cfg))
        if field_flag == 2:
            # Match the other fields' visualization convention: first transform
            # obstacle geometry and world query positions into the current ego
            # body-fixed normalized frame q=S^{-1}R^T(p-p_ego), then fit/evaluate
            # BP-SDF there. The returned samples are later drawn at the original
            # world-grid positions by publish_sdf_rviz_markers().
            if reference_state is None:
                reference_state = query[0]
            points_body, query_body, ref_body = self._bp_visualization_body_frame(
                reference_state, points, query
            )
            if bp_model is None:
                bp_model = _utils_linearized.build_body_fixed_bp_sdf_model(
                    ref_body.reshape(1, 4), points_body, vertex_types, obstacle_ids,
                    np.zeros(1, dtype=np.float64),
                    np.full(1, max(float(self.pred_dt), 1.0e-9), dtype=np.float64),
                    self.sdf_cfg, use_cache=True,
                )
            values = _utils_linearized.evaluate_body_fixed_bp_sdf_world(
                query_body, bp_model, self.sdf_cfg
            )
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if values.size != count:
                raise ValueError(
                    "Bernstein SDF evaluator returned %d values for %d world grid states"
                    % (int(values.size), count)
                )
            return values

        eval_states = query
        eval_count = int(eval_states.shape[0])
        # Dense current-time RViz grids use one shared obstacle geometry.  The
        # dedicated helper generates proposed edge/pseudo features only once
        # (H=1) and broadcasts them inside collision_lin_cpp while evaluating
        # the complete query grid in one C++ call.
        if field_flag != 2 and hasattr(_utils_linearized, "evaluate_external_phi_grid_cpp"):
            phi = _utils_linearized.evaluate_external_phi_grid_cpp(
                eval_states, points, normals, vertex_types, obstacle_ids, self.sdf_cfg
            )
        else:
            h_query = np.zeros(eval_count, dtype=np.float64)
            dt_query = np.full(
                eval_count, max(float(self.pred_dt), 1.0e-9), dtype=np.float64
            )
            phi = _utils_linearized.evaluate_external_phi_sequence_fast(
                eval_states, points, normals, vertex_types, obstacle_ids,
                h_query, dt_query, self.sdf_cfg,
            )
        values = np.asarray(phi, dtype=np.float64).reshape(-1)
        if values.size != eval_count:
            raise ValueError(
                "SDF evaluator returned %d values for %d evaluation states"
                % (int(values.size), eval_count)
            )
        if values.size != count:
            raise ValueError(
                "SDF evaluator returned %d query values for %d grid states"
                % (int(values.size), count)
            )
        return values

    def publish_current_sdf(self, perceived_msg=None):
        """Publish the configured Phi_k(q) on an ego-centered world-meter grid.

        The visualization uses the same field implementation selected by the
        planner. Fields 0/1/3/4 use their configured collision-field evaluators;
        field 2 uses the same ego body-fixed normalized-frame visualization
        convention as the other fields:

          - 0: circle over-estimation of the COMPLETE occluded-region-augmented
               obstacle.  Center = mean(vertex points + edge points), radius =
               maximum distance to those points; then inflate by the ego unit disk
               and the prediction-uncertainty disk.
          - 1: DCSAT on the body-fixed convex groups,
          - 2: Geometry-Aware Bernstein boundary fitted independently per obstacle
               after q=S^{-1}R^T(P-p_ego); signed distance is recomputed from the
               fitted boundary in that body-fixed normalized frame,
          - 3: Wu et al. smooth polygon field on the body-fixed convex groups,
          - 4: proposed field, with edge and N_n pseudo-vertex features generated
               internally from the body-fixed convex groups.

        For visualization, field 0 consumes ``/percepted_obstacle`` so its circle is
        built from the complete augmented boundary; fields 1--4 consume
        ``/selected_vertex_data`` so their geometry matches the MPC convex groups.

        ``sdf_grid_x/y`` are relative world-meter offsets centered at the current
        ego position.  The published grid/markers remain in the world/map frame.
        For field 2, both obstacle vertices and grid queries are first transformed
        by q=S^{-1}R(psi)^T(p-p_ego). BP-SDF is evaluated in that body-fixed
        normalized frame. Before publishing, the BP scalar grid is converted
        to a local world-meter-equivalent value using the E-frame level-set
        normal, while the samples are drawn at the corresponding world-grid
        positions. Impact-normal estimation keeps the raw normalized-E field.
        h=0 geometry is used.
        """
        if not bool(self.publish_sdf) or self.sdf_cfg is None:
            return
        sdf_needed = self._publisher_has_subscribers(self.sdf_pub)
        rviz_needed = (bool(self.publish_sdf_rviz) and
                       self._publisher_has_subscribers(self.sdf_rviz_pub))
        if not sdf_needed and not rviz_needed:
            return

        now = rospy.Time.now()
        if float(self.sdf_grid_pub_hz) > 0.0 and self.sdf_grid_last_stamp != rospy.Time(0):
            min_dt = 1.0 / max(float(self.sdf_grid_pub_hz), 1.0e-9)
            if (now - self.sdf_grid_last_stamp).to_sec() < min_dt:
                return

        field_flag = int(_utils_linearized.collision_field_flag(self.sdf_cfg))

        # Geometry source is field-specific but always represents the same
        # perceived/occlusion-augmented obstacle before body-fixed evaluation.
        # field 0 needs the COMPLETE raw augmented boundary for one enclosing
        # circle per perceived obstacle.  Fields 1--4 use selector convex groups
        # so RViz and MPC evaluate identical polygon groups.
        if field_flag == 0:
            source_msg = perceived_msg if perceived_msg is not None else self.latest_percepted_obstacle_msg
            arrays = self.perceived_polygon_field_arrays(source_msg)
            source_name = self.percepted_obstacle_topic
        else:
            source_msg = self.latest_selected_vertex_data_msg
            arrays = self.selected_vertex_field_arrays(source_msg)
            source_name = self.selected_vertex_data_topic

        if source_msg is None:
            rospy.logwarn_throttle(
                5.0,
                "[plotter] SDF field_flag=%d (%s) is waiting for %s",
                field_flag,
                _utils_linearized.collision_field_name(self.sdf_cfg),
                source_name,
            )
            return

        if arrays is None:
            rospy.logwarn_throttle(
                5.0,
                "[plotter] SDF source %s has obstacles but no valid raw polygon vertices",
                source_name,
            )
            return

        points, normals, vertex_types, obstacle_ids = arrays
        no_geometry = int(points.shape[0]) == 0

        # max_vertex_num limits raw geometry consumed by utils_linearized.
        # Retain every available boundary vertex/group sample in visualization.
        self.sdf_cfg.max_vertex_num = max(
            int(getattr(self.sdf_cfg, "max_vertex_num", 0)),
            int(points.shape[0]),
        )

        nx = int(self.sdf_grid_x.size)
        ny = int(self.sdf_grid_y.size)
        if nx <= 0 or ny <= 0:
            return

        # Snapshot once so q=0, all grid states, and the BP reference use exactly
        # the same ego state even if /ego_state updates during evaluation.
        ego_x = float(self.ego_cart_x)
        ego_y = float(self.ego_cart_y)
        ego_v = float(self.ego_v)
        ego_yaw = float(self.ego_yaw)
        reference_state = np.array([ego_x, ego_y, ego_v, ego_yaw], dtype=np.float64)

        count = int(nx * ny)

        states = self.sdf_grid_state_buffer
        if states.shape[0] != count:
            states = np.empty((count, 4), dtype=np.float64)
            self.sdf_grid_state_buffer = states
        states[:, 0] = ego_x + self.sdf_grid_x_flat
        states[:, 1] = ego_y + self.sdf_grid_y_flat
        states[:, 2] = ego_v
        states[:, 3] = ego_yaw

        try:
            bp_model = None
            if field_flag == 2 and not no_geometry:
                # BP visualization is fitted in the same ego body-fixed normalized
                # frame used by the other collision fields, not directly in world.
                bp_points_body, _, bp_reference_body = self._bp_visualization_body_frame(
                    reference_state, points, None
                )
                bp_model = _utils_linearized.build_body_fixed_bp_sdf_model(
                    bp_reference_body.reshape(1, 4),
                    bp_points_body,
                    vertex_types,
                    obstacle_ids,
                    np.zeros(1, dtype=np.float64),
                    np.full(1, max(float(self.pred_dt), 1.0e-9), dtype=np.float64),
                    self.sdf_cfg,
                    use_cache=True,
                )
                self._last_sdf_bp_model = bp_model
                self._last_sdf_bp_reference = reference_state.copy()
                self._last_sdf_bp_points_obj = points
            else:
                self._last_sdf_bp_model = None
                self._last_sdf_bp_reference = None
                self._last_sdf_bp_points_obj = None

            if no_geometry:
                values = np.full(
                    count,
                    float(getattr(self.sdf_cfg, "h_upper_bound", 1.0e15)),
                    dtype=np.float64,
                )
            else:
                # One full-grid call.  For field_flag=4 this reaches
                # collision_lin_cpp with one H=1 shared feature tensor instead
                # of rebuilding/duplicating features for every Python batch.
                values = self._evaluate_sdf_grid_batch(
                    states,
                    points,
                    normals,
                    vertex_types,
                    obstacle_ids,
                    reference_state=reference_state,
                    bp_model=bp_model,
                )

            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite values found in SDF grid")

            sdf_grid = values.reshape(ny, nx)

            # BP is fitted/evaluated in the anisotropically normalized body-fixed
            # E frame. Convert only the published scalar grid to the same local
            # world-meter-equivalent convention used by the field-comparison
            # plots. This positive directional scaling preserves the sign and
            # zero contour. Impact normals use the raw E-frame evaluator below.
            if field_flag == 2 and not no_geometry:
                sdf_grid = self._bp_sdf_grid_world_meter_equivalent(
                    sdf_grid, reference_state
                )

            if sdf_needed:
                msg = Float64MultiArray()
                msg.layout.dim = [
                    MultiArrayDimension(
                        label="y",
                        size=ny,
                        stride=ny * nx,
                    ),
                    MultiArrayDimension(
                        label="x",
                        size=nx,
                        stride=nx,
                    ),
                ]
                msg.layout.data_offset = 0
                msg.data = sdf_grid.reshape(-1).tolist()
                self.sdf_pub.publish(msg)

            frame_id = "base_link"
            if source_msg is not None:
                frame_id = self.get_frame_id(source_msg)
            elif self.latest_ego_msg is not None:
                frame_id = self.get_frame_id(self.latest_ego_msg)
            if rviz_needed:
                self.publish_sdf_rviz_markers(
                    sdf_grid,
                    stamp=now,
                    frame_id=frame_id,
                    ego_origin=(ego_x, ego_y),
                )

            self.sdf_grid_last_stamp = now
            self.sdf_last_error = None

        except Exception as exc:
            error = repr(exc)
            if error != self.sdf_last_error:
                rospy.logerr(
                    "[plotter] SDF grid evaluation failed for field_flag=%d (%s), source=%s: %s",
                    field_flag,
                    _utils_linearized.collision_field_name(self.sdf_cfg),
                    source_name,
                    error,
                )
                self.sdf_last_error = error

    @staticmethod
    def _impact_normal_arrow(frame_id, stamp, ns, marker_id, ego_xy, normal_world,
                             length, color, shaft_diameter, head_diameter, head_length,
                             z=0.08):
        """Create one normalized RViz ARROW, analogous to a 2-D quiver vector."""
        n = np.asarray(normal_world, dtype=float).reshape(2)
        nrm = float(np.linalg.norm(n))
        if not np.isfinite(nrm) or nrm <= 1.0e-12:
            return None
        n = n / nrm

        marker = Marker()
        marker.header.frame_id = str(frame_id) if frame_id else "base_link"
        marker.header.stamp = stamp
        marker.ns = str(ns)
        marker.id = int(marker_id)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(shaft_diameter), 1.0e-4)
        marker.scale.y = max(float(head_diameter), marker.scale.x)
        marker.scale.z = max(float(head_length), 1.0e-4)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        p0 = Point()
        p0.x = float(ego_xy[0])
        p0.y = float(ego_xy[1])
        p0.z = float(z)
        p1 = Point()
        p1.x = float(ego_xy[0]) + float(length) * float(n[0])
        p1.y = float(ego_xy[1]) + float(length) * float(n[1])
        p1.z = float(z)
        marker.points.extend([p0, p1])
        return marker

    def _exact_current_impact_normal_world(self, reference_state, points, obstacle_ids):
        """Return the nearest exact h=0 obstacle-induced normal in world frame."""
        X = np.asarray(reference_state, dtype=float).reshape(-1)
        if X.size < 4:
            return None
        P = np.asarray(points, dtype=float).reshape(-1, 2)
        if P.shape[0] < 2:
            return None
        if obstacle_ids is None:
            gids = np.zeros(P.shape[0], dtype=np.int64)
        else:
            gids = np.asarray(obstacle_ids, dtype=np.int64).reshape(-1)
            if gids.size != P.shape[0]:
                return None

        pbar = X[:2].astype(float)
        yaw = float(X[3])
        a = max(abs(float(self.ego_L)) / math.sqrt(2.0), 1.0e-12)
        b = max(abs(float(self.ego_W)) / math.sqrt(2.0), 1.0e-12)
        c = math.cos(yaw)
        s = math.sin(yaw)
        T = np.array([[c / a, s / a], [-s / b, c / b]], dtype=float)

        best = None
        for gid in np.unique(gids):
            Vw = P[gids == gid]
            hull_world = self.convex_hull_2d(Vw)
            if len(hull_world) < 2:
                continue
            Vw = np.asarray(hull_world, dtype=float).reshape(-1, 2)
            Vl = (Vw - pbar.reshape(1, 2)).dot(T.T)
            hull_local = self.convex_hull_2d(Vl)
            if len(hull_local) < 2:
                continue
            closest = self._closest_point_from_origin_to_polygon_boundary(hull_local)
            if closest is None:
                continue
            q_local, radius = closest
            if not math.isfinite(radius) or radius <= 1.0e-10:
                continue
            inside = self._origin_inside_convex_polygon(hull_local) if len(hull_local) >= 3 else False
            sign = -1.0 if inside else 1.0
            phi_gt = sign * float(radius) - 1.0
            n_local = np.asarray(q_local, dtype=float).reshape(2) / float(radius)

            # Phi_GT derivative w.r.t. ego translation is -sign*T^T*n_local.
            # Thus the obstacle-induced impact normal is -grad_p(Phi_GT).
            n_world = sign * (T.T @ n_local)
            nrm = float(np.linalg.norm(n_world))
            if not math.isfinite(nrm) or nrm <= 1.0e-12:
                continue
            n_world = n_world / nrm

            if best is None or phi_gt < best["phi_gt"]:
                best = {
                    "group_id": int(gid),
                    "phi_gt": float(phi_gt),
                    "normal_world": n_world,
                }
        return best

    def _current_sdf_geometry(self, perceived_msg=None):
        """Return the h=0 geometry source used by the configured SDF."""
        if self.sdf_cfg is None:
            return None
        field_flag = int(_utils_linearized.collision_field_flag(self.sdf_cfg))
        if field_flag == 0:
            source_msg = perceived_msg if perceived_msg is not None else self.latest_percepted_obstacle_msg
            arrays = self.perceived_polygon_field_arrays(source_msg)
            source_name = self.percepted_obstacle_topic
        else:
            source_msg = self.latest_selected_vertex_data_msg
            arrays = self.selected_vertex_field_arrays(source_msg)
            source_name = self.selected_vertex_data_topic
        if source_msg is None or arrays is None:
            return None
        return field_flag, source_msg, arrays, source_name

    def _estimated_current_impact_normal_world(self, reference_state, points, normals,
                                               vertex_types, obstacle_ids):
        """Return -grad_p(Phi)/||grad_p(Phi)|| for the current differentiable field."""
        X0 = np.asarray(reference_state, dtype=np.float64).reshape(4)
        eps = max(float(self.impact_normal_fd_eps_m), 1.0e-5)
        query = np.repeat(X0.reshape(1, 4), 4, axis=0)
        query[0, 0] += eps
        query[1, 0] -= eps
        query[2, 1] += eps
        query[3, 1] -= eps

        field_flag = int(_utils_linearized.collision_field_flag(self.sdf_cfg))
        bp_model = None
        if field_flag == 2:
            if (self._last_sdf_bp_model is not None and
                    points is self._last_sdf_bp_points_obj):
                bp_model = self._last_sdf_bp_model
            else:
                # Let _evaluate_sdf_grid_batch build the model after converting
                # this world geometry to the reference ego body-fixed frame.
                bp_model = None

        values = self._evaluate_sdf_grid_batch(
            query, points, normals, vertex_types, obstacle_ids,
            reference_state=X0, bp_model=bp_model,
        )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != 4 or not np.all(np.isfinite(values)):
            return None

        grad_p = np.array([
            (float(values[0]) - float(values[1])) / (2.0 * eps),
            (float(values[2]) - float(values[3])) / (2.0 * eps),
        ], dtype=float)
        grad_norm = float(np.linalg.norm(grad_p))
        if not math.isfinite(grad_norm) or grad_norm <= 1.0e-10:
            return None
        return -grad_p / grad_norm

    def publish_current_impact_normal_comparison(self, perceived_msg=None, stamp=None):
        """Draw exact GT (green) and differentiable-SDF estimate (black) at ego."""
        if not bool(self.publish_impact_normal_comparison) or self.sdf_cfg is None:
            return
        if not self._publisher_has_subscribers(self.impact_normal_pub):
            return
        if stamp is None:
            stamp = rospy.Time.now()

        geometry = self._current_sdf_geometry(perceived_msg=perceived_msg)
        frame_id = "base_link"
        arr = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        if geometry is None:
            self.impact_normal_pub.publish(arr)
            return

        field_flag, source_msg, arrays, source_name = geometry
        frame_id = self.get_frame_id(source_msg)
        delete_all.header.frame_id = frame_id
        points, normals, vertex_types, obstacle_ids = arrays
        if int(points.shape[0]) < 2:
            self.impact_normal_pub.publish(arr)
            return

        self.sdf_cfg.max_vertex_num = max(
            int(getattr(self.sdf_cfg, "max_vertex_num", 0)), int(points.shape[0])
        )
        reference_state = np.array([
            float(self.ego_cart_x), float(self.ego_cart_y),
            float(self.ego_v), float(self.ego_yaw),
        ], dtype=np.float64)

        try:
            gt = self._exact_current_impact_normal_world(
                reference_state, points, obstacle_ids
            )
            pred = self._estimated_current_impact_normal_world(
                reference_state, points, normals, vertex_types, obstacle_ids
            )

            ego_xy = reference_state[:2]
            length = float(self.impact_normal_arrow_length)
            # Exact normal: thick green. SDF estimate: thinner black.
            if gt is not None:
                gt_arrow = self._impact_normal_arrow(
                    frame_id, stamp, "impact_normal_gt", 0, ego_xy,
                    gt["normal_world"], length,
                    (0.0, 1.0, 0.0, 1.0), 0.050, 0.140, 0.160, z=0.085,
                )
                if gt_arrow is not None:
                    arr.markers.append(gt_arrow)
            if pred is not None:
                pred_arrow = self._impact_normal_arrow(
                    frame_id, stamp, "impact_normal_sdf", 1, ego_xy,
                    pred, length,
                    (0.0, 0.0, 0.0, 1.0), 0.026, 0.085, 0.120, z=0.090,
                )
                if pred_arrow is not None:
                    arr.markers.append(pred_arrow)

            if gt is not None and pred is not None:
                dot = float(np.dot(np.asarray(gt["normal_world"]), np.asarray(pred)))
                dot = min(max(dot, -1.0), 1.0)
                angle_deg = math.degrees(math.acos(dot))
                rospy.loginfo_throttle(
                    1.0,
                    "[plotter] current impact-normal comparison: field=%d (%s) "
                    "GT_gid=%d Phi_GT=%.6f angle_error=%.3fdeg",
                    int(field_flag), _utils_linearized.collision_field_name(self.sdf_cfg),
                    int(gt["group_id"]), float(gt["phi_gt"]), float(angle_deg),
                )

            self.impact_normal_pub.publish(arr)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "[plotter] current impact-normal visualization failed for field_flag=%d, source=%s: %s",
                int(field_flag), str(source_name), repr(exc),
            )
            self.impact_normal_pub.publish(arr)

    def decode_selected_vertex_horizon_payload(self, msg):
        """Decode selector Float64MultiArray as H x K x 6.

        vertex_selector publishes fields
            [x, y, nx, ny, type, group_id]
        for stages h=1..N.  The flattened data order is authoritative here; the
        publisher's dim sizes are used only to recover H/K/F.
        """
        if msg is None:
            return np.zeros((0, 0, 6), dtype=float)

        dims = list(getattr(getattr(msg, "layout", None), "dim", []) or [])
        data = np.asarray(getattr(msg, "data", []), dtype=float).reshape(-1)
        offset = int(getattr(getattr(msg, "layout", None), "data_offset", 0) or 0)
        if offset > 0:
            data = data[offset:]

        H = K = F = 0
        if len(dims) >= 3:
            H = max(0, int(getattr(dims[0], "size", 0)))
            K = max(0, int(getattr(dims[1], "size", 0)))
            F = max(0, int(getattr(dims[2], "size", 0)))

        if H <= 0 or F <= 0:
            H = max(0, int(self.pred_horizon))
            F = 6
            if H > 0 and data.size % (H * F) == 0:
                K = int(data.size // (H * F))

        expected = int(H) * int(K) * int(F)
        if H <= 0 or K <= 0 or F < 6 or expected <= 0 or data.size < expected:
            return np.zeros((0, 0, 6), dtype=float)

        return data[:expected].reshape(int(H), int(K), int(F))[:, :, :6]

    def selector_feature_outward_normal(self, nx, ny):
        n = np.array([float(nx), float(ny)], dtype=float)
        norm = float(np.linalg.norm(n))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            return None
        n /= norm
        if str(self.selected_vertex_normal_orientation) == "inward":
            n = -n
        return n

    def selector_safe_feature_point(self, x, y, nx, ny, ego_state):
        """Add ego footprint + baseline clearance to selector geometry.

        This helper is used by visualizations that need the ego-expanded planning
        boundary.  The raw /obs_polygons_horizon path below does not call it.
        """
        n_out = self.selector_feature_outward_normal(nx, ny)
        if n_out is None:
            return None
        ego_yaw = float((ego_state or {}).get("yaw", self.ego_yaw))
        if bool(self.use_ego_scaled_local_frame):
            vehicle_r = self.ego_oriented_ellipse_support(n_out, ego_yaw)
        else:
            vehicle_r = max(float(self.ego_R), 0.0)
        total_r = vehicle_r + self.base_safe_margin()
        return (float(x) + total_r * float(n_out[0]),
                float(y) + total_r * float(n_out[1]))

    def inflate_selector_group_by_disk(self, points, margin, circle_steps=32):
        """Return convex-hull geometry inflated by an isotropic disk.

        /selected_vertex_horizon_data intentionally carries zero feature normals,
        so horizon visualization must not shift individual features along nx/ny.
        Instead reconstruct the group hull first and form an explicit polygonal
        approximation of the hull Minkowski-summed with D(margin).  The sampled circle radius is the
        exact MPC uncertainty radius r_h; increasing circle_steps only improves
        rendering resolution and does not change r_h itself.
        """
        hull = self.convex_hull_2d(points)
        r = max(float(margin), 0.0)
        if len(hull) == 0 or r <= 0.0:
            return hull

        steps = max(12, int(circle_steps))
        dirs = [
            (r * math.cos(2.0 * math.pi * float(i) / float(steps)),
             r * math.sin(2.0 * math.pi * float(i) / float(steps)))
            for i in range(steps)
        ]
        expanded = [
            (float(vx) + dx, float(vy) + dy)
            for vx, vy in hull
            for dx, dy in dirs
        ]
        return self.convex_hull_2d(expanded)

    def append_selector_group_boundary(self, marker, points, z_vis, inflation_radius=0.0):
        hull = self.inflate_selector_group_by_disk(points, inflation_radius)
        if len(hull) >= 3:
            self.append_polygon_line_segments(marker, hull, z_vis=z_vis)
        elif len(hull) == 2:
            p1 = Point()
            p1.x, p1.y, p1.z = float(hull[0][0]), float(hull[0][1]), float(z_vis)
            p2 = Point()
            p2.x, p2.y, p2.z = float(hull[1][0]), float(hull[1][1]), float(z_vis)
            marker.points.extend([p1, p2])

    def decode_separating_hyperplane_payload(self, msg):
        """Decode planner affine rows as [R,14]."""
        if msg is None:
            return np.zeros((0, 14), dtype=float)
        data = np.asarray(getattr(msg, "data", []), dtype=float).reshape(-1)
        offset = int(getattr(getattr(msg, "layout", None), "data_offset", 0) or 0)
        if offset > 0:
            data = data[offset:]
        fields = 14
        dims = list(getattr(getattr(msg, "layout", None), "dim", []) or [])
        if len(dims) >= 2:
            try:
                fields = int(getattr(dims[1], "size", fields))
            except Exception:
                fields = 14
        if fields < 14 or data.size < fields:
            return np.zeros((0, 14), dtype=float)
        rows = int(data.size // fields)
        if rows <= 0:
            return np.zeros((0, 14), dtype=float)
        return data[:rows * fields].reshape(rows, fields)[:, :14]

    def clip_world_line_to_bounds(self, a, b, c):
        """Clip a*x+b*y=c to the configured world x/y rectangle."""
        a = float(a); b = float(b); c = float(c)
        eps = 1.0e-12
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            return None
        if math.hypot(a, b) <= eps:
            return None

        xmin, xmax = sorted((float(self.x_min), float(self.x_max)))
        ymin, ymax = sorted((float(self.y_min), float(self.y_max)))
        tol = 1.0e-8 * max(1.0, xmax - xmin, ymax - ymin)
        pts = []

        if abs(b) > eps:
            for x in (xmin, xmax):
                y = (c - a * x) / b
                if ymin - tol <= y <= ymax + tol:
                    pts.append((x, min(max(y, ymin), ymax)))
        if abs(a) > eps:
            for y in (ymin, ymax):
                x = (c - b * y) / a
                if xmin - tol <= x <= xmax + tol:
                    pts.append((min(max(x, xmin), xmax), y))

        unique = []
        for p in pts:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= tol for q in unique):
                unique.append(p)
        if len(unique) < 2:
            return None

        best = None
        best_d2 = -1.0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dx = unique[i][0] - unique[j][0]
                dy = unique[i][1] - unique[j][1]
                d2 = dx * dx + dy * dy
                if d2 > best_d2:
                    best_d2 = d2
                    best = (unique[i], unique[j])
        return best

    def clip_polygon_to_halfplane(self, polygon, a, b, c, tol=1.0e-10):
        """Clip a convex polygon by the feasible half-plane a*x+b*y <= c."""
        poly = [(float(x), float(y)) for x, y in polygon]
        if len(poly) < 3:
            return []

        def value(pt):
            return float(a) * pt[0] + float(b) * pt[1] - float(c)

        out = []
        prev = poly[-1]
        prev_v = value(prev)
        prev_in = prev_v <= tol
        for cur in poly:
            cur_v = value(cur)
            cur_in = cur_v <= tol
            if cur_in != prev_in:
                dx = cur[0] - prev[0]
                dy = cur[1] - prev[1]
                denom = float(a) * dx + float(b) * dy
                if abs(denom) > 1.0e-14:
                    t = (float(c) - float(a) * prev[0] - float(b) * prev[1]) / denom
                    t = min(max(float(t), 0.0), 1.0)
                    out.append((prev[0] + t * dx, prev[1] + t * dy))
            if cur_in:
                out.append(cur)
            prev = cur
            prev_v = cur_v
            prev_in = cur_in
        return out

    def append_filled_polygon_triangles(self, marker, polygon, z_vis=0.0):
        """Triangulate a convex polygon as a fan into a TRIANGLE_LIST marker."""
        if polygon is None or len(polygon) < 3:
            return
        p0 = polygon[0]
        for i in range(1, len(polygon) - 1):
            for x, y in (p0, polygon[i], polygon[i + 1]):
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = float(z_vis)
                marker.points.append(p)

    @staticmethod
    def _origin_inside_convex_polygon(vertices):
        """Return True when the origin lies inside/on a convex polygon."""
        V = np.asarray(vertices, dtype=float).reshape(-1, 2)
        if V.shape[0] < 3:
            return False
        signs = []
        for i in range(V.shape[0]):
            a = V[i]
            b = V[(i + 1) % V.shape[0]]
            e = b - a
            # cross(edge, origin-a)
            cross = float(e[0] * (-a[1]) - e[1] * (-a[0]))
            if abs(cross) > 1.0e-10:
                signs.append(1 if cross > 0.0 else -1)
        return (not signs) or all(v == signs[0] for v in signs)

    @staticmethod
    def _closest_point_from_origin_to_polygon_boundary(vertices):
        """Exact closest boundary point to the origin for a polygon."""
        V = np.asarray(vertices, dtype=float).reshape(-1, 2)
        n = int(V.shape[0])
        if n <= 0:
            return None
        if n == 1:
            q = V[0].copy()
            return q, float(np.linalg.norm(q))

        best_q = None
        best_d2 = float("inf")
        edge_count = n if n >= 3 else 1
        for i in range(edge_count):
            j = (i + 1) % n if n >= 3 else 1
            a = V[i]
            b = V[j]
            d = b - a
            den = float(np.dot(d, d))
            if den <= 1.0e-24:
                q = a.copy()
            else:
                # Projection of the origin onto segment a+t d.
                t = -float(np.dot(a, d)) / den
                t = min(max(t, 0.0), 1.0)
                q = a + t * d
            d2 = float(np.dot(q, q))
            if d2 < best_d2:
                best_d2 = d2
                best_q = q.copy()
        if best_q is None:
            return None
        return best_q, math.sqrt(max(best_d2, 0.0))

    def _selector_groups_current_step(self):
        """Return [(group_id, world convex hull), ...] from h=0 selector data.

        This function intentionally uses only /selected_vertex_data.  It never
        falls back to /selected_vertex_horizon_data, whose first slice is h=1.
        """
        groups = []
        msg = self.latest_selected_vertex_data_msg
        if msg is None:
            return groups
        n = min(
            max(0, int(getattr(msg, "obs_cnt", 0))),
            len(getattr(msg, "obs_xs", [])),
            len(getattr(msg, "obs_ys", [])),
        )
        if n <= 0:
            return groups
        xs = np.asarray(msg.obs_xs[:n], dtype=float)
        ys = np.asarray(msg.obs_ys[:n], dtype=float)
        if hasattr(msg, "obs_rs") and len(msg.obs_rs) >= n:
            gids = np.rint(np.asarray(msg.obs_rs[:n], dtype=float)).astype(np.int64)
        else:
            gids = np.zeros(n, dtype=np.int64)
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(gids.astype(float))
        xs = xs[valid]
        ys = ys[valid]
        gids = gids[valid]
        for gid in np.unique(gids):
            pts = np.column_stack((xs[gids == gid], ys[gids == gid]))
            hull = self.convex_hull_2d(pts)
            if len(hull) >= 2:
                groups.append((int(gid), hull))
        return groups

    def exact_min_distance_hyperplane_for_row(self, row):
        """Build the exact h=0 nearest-obstacle tangent model.

        The row supplies the current ego linearization state and d_safe, while
        obstacle geometry comes only from /selected_vertex_data (h=0).  This
        computes the exact minimum signed distance from q=0 to each convex
        selector group in the MPC unit-disk frame

            z = diag(sqrt(2)/L, sqrt(2)/W) R(yaw)^T (P - p),

        then subtracts the ego unit radius.  The nearest group therefore gives

            Phi_GT = sd(0, O_local) - 1.

        For g_GT = d_safe - Phi_GT, the world-translation gradient is obtained
        analytically by the chain rule, and the comparable affine boundary is

            g_GT_bar + grad_g_GT^T (p-pbar) = s_k.

        Returns None when no valid selector group / unique closest direction is
        available.
        """
        row = np.asarray(row, dtype=float).reshape(-1)
        if row.size < 14 or not np.all(np.isfinite(row[:14])):
            return None

        pbar = np.array([float(row[3]), float(row[4])], dtype=float)
        yaw = float(row[6])
        g_bar_sur = float(row[7])
        phi_sur = float(row[13])
        s_k = max(float(row[12]), 0.0)
        d_safe = g_bar_sur + phi_sur
        if not math.isfinite(d_safe):
            return None

        # Match utils_linearized.normalize_ellipse_local_frame_config(): the
        # corner-passing ego ellipse has semi-axes L/sqrt(2), W/sqrt(2), and is
        # mapped exactly to the unit disk.
        a = max(abs(float(self.ego_L)) / math.sqrt(2.0), 1.0e-12)
        b = max(abs(float(self.ego_W)) / math.sqrt(2.0), 1.0e-12)
        c = math.cos(yaw)
        s = math.sin(yaw)
        T = np.array([[c / a, s / a], [-s / b, c / b]], dtype=float)

        best = None
        for gid, hull_world in self._selector_groups_current_step():
            Vw = np.asarray(hull_world, dtype=float).reshape(-1, 2)
            if Vw.shape[0] < 2:
                continue
            Vl = (Vw - pbar.reshape(1, 2)).dot(T.T)
            # Affine maps preserve convexity, but recomputing the hull also
            # removes repeated selector points and gives an ordered boundary.
            hull_local = self.convex_hull_2d(Vl)
            if len(hull_local) < 2:
                continue
            closest = self._closest_point_from_origin_to_polygon_boundary(hull_local)
            if closest is None:
                continue
            q_local, radius = closest
            if not math.isfinite(radius) or radius <= 1.0e-10:
                continue
            inside = self._origin_inside_convex_polygon(hull_local) if len(hull_local) >= 3 else False
            sign = -1.0 if inside else 1.0
            signed_polygon_distance = sign * float(radius)
            phi_gt = signed_polygon_distance - 1.0
            if best is None or phi_gt < best["phi_gt"]:
                n_local = np.asarray(q_local, dtype=float) / float(radius)
                # Phi_GT derivative wrt ego world translation is
                #   grad_p Phi_GT = -sign * T^T n_local,
                # hence grad_p g_GT = sign * T^T n_local.
                grad_g = sign * (T.T @ n_local.reshape(2))
                g_gt = float(d_safe) - float(phi_gt)
                rhs = float(s_k) - g_gt + float(np.dot(grad_g, pbar))
                q_world = pbar + np.linalg.solve(T, np.asarray(q_local, dtype=float))
                best = {
                    "group_id": int(gid),
                    "phi_gt": float(phi_gt),
                    "g_gt": float(g_gt),
                    "gx": float(grad_g[0]),
                    "gy": float(grad_g[1]),
                    "rhs": float(rhs),
                    "closest_world": (float(q_world[0]), float(q_world[1])),
                    "inside": bool(inside),
                    "d_safe": float(d_safe),
                }
        return best

    def publish_separating_hyperplanes(self, msg, stamp=None):
        """Render the k=0 smooth-Phi and exact-minimum-distance hyperplanes.

        MPC_planner0 publishes an additional diagnostic h=0 affine row computed
        directly from the current ego state and /selected_vertex_data.  Only
        that h=0 row is visualized here; h=1..H solver rows are ignored.

        Magenta:
            smooth Phi_0 affine boundary and its feasible half-plane.
        Cyan:
            exact minimum-distance tangent boundary from the same h=0 selector
            geometry, and its feasible half-plane.
        """
        if stamp is None:
            stamp = rospy.Time.now()
        rows = self.decode_separating_hyperplane_payload(msg)

        # Both affine equations are in world coordinates.  Prefer the selector
        # frame because the exact comparison is reconstructed from selector data.
        frame_id = "base_link"
        if self.latest_selected_vertex_data_msg is not None:
            frame_id = self.get_frame_id(self.latest_selected_vertex_data_msg)
        elif self.latest_obstacle_msg is not None:
            frame_id = self.get_frame_id(self.latest_obstacle_msg)

        arr = MarkerArray()

        # Clear markers from the former multi-horizon visualization.
        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        # Smooth-Phi boundary (magenta).
        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = "mpc_separating_hyperplane_k0"
        line.id = 0
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = float(self.separating_hyperplane_line_width)
        line.scale.y = 1.0
        line.scale.z = 1.0
        self.set_color(line, self.separating_hyperplane_color)

        fill = Marker()
        fill.header.frame_id = frame_id
        fill.header.stamp = stamp
        fill.ns = "mpc_separating_hyperplane_feasible_k0"
        fill.id = 1
        fill.type = Marker.TRIANGLE_LIST
        fill.action = Marker.ADD
        fill.pose.orientation.w = 1.0
        fill.scale.x = 1.0
        fill.scale.y = 1.0
        fill.scale.z = 1.0
        self.set_color(fill, self.separating_hyperplane_fill_color)

        # Exact minimum-distance boundary (cyan).
        exact_line = Marker()
        exact_line.header.frame_id = frame_id
        exact_line.header.stamp = stamp
        exact_line.ns = "exact_min_distance_separating_hyperplane_k0"
        exact_line.id = 2
        exact_line.type = Marker.LINE_LIST
        exact_line.action = Marker.ADD
        exact_line.pose.orientation.w = 1.0
        exact_line.scale.x = float(self.exact_min_distance_hyperplane_line_width)
        exact_line.scale.y = 1.0
        exact_line.scale.z = 1.0
        self.set_color(exact_line, self.exact_min_distance_hyperplane_color)

        exact_fill = Marker()
        exact_fill.header.frame_id = frame_id
        exact_fill.header.stamp = stamp
        exact_fill.ns = "exact_min_distance_separating_hyperplane_feasible_k0"
        exact_fill.id = 3
        exact_fill.type = Marker.TRIANGLE_LIST
        exact_fill.action = Marker.ADD
        exact_fill.pose.orientation.w = 1.0
        exact_fill.scale.x = 1.0
        exact_fill.scale.y = 1.0
        exact_fill.scale.z = 1.0
        self.set_color(exact_fill, self.exact_min_distance_hyperplane_fill_color)

        # Use only the genuine h=0 diagnostic row.  Never reinterpret h=1 as
        # current time; if h=0 is unavailable, publish empty markers instead.
        selected = None
        for row in rows:
            if not np.all(np.isfinite(row)):
                continue
            if int(round(float(row[1]))) != 0:
                continue
            gx, gy = float(row[8]), float(row[9])
            if math.hypot(gx, gy) <= 1.0e-10:
                continue
            selected = row
            break

        if selected is not None:
            row = selected
            pbar_x, pbar_y = float(row[3]), float(row[4])
            g_bar = float(row[7])
            gx, gy = float(row[8]), float(row[9])
            s_k = max(float(row[12]), 0.0)

            # Smooth-Phi affine feasible set:
            #   g_bar + grad_g^T(p-pbar) <= s_k
            # => gx*x + gy*y <= rhs.
            rhs = s_k - g_bar + gx * pbar_x + gy * pbar_y
            clipped_line = self.clip_world_line_to_bounds(gx, gy, rhs)
            if clipped_line is not None:
                for x, y in clipped_line:
                    p = Point()
                    p.x = float(x)
                    p.y = float(y)
                    p.z = 0.0
                    line.points.append(p)

            xmin, xmax = sorted((float(self.x_min), float(self.x_max)))
            ymin, ymax = sorted((float(self.y_min), float(self.y_max)))
            world_rect = [
                (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
            ]
            feasible_poly = self.clip_polygon_to_halfplane(
                world_rect, gx, gy, rhs
            )
            self.append_filled_polygon_triangles(fill, feasible_poly, z_vis=0.0)

            # Exact h=0 minimum-distance tangent, using /selected_vertex_data
            # only and replacing smooth Phi with exact polygon signed distance.
            exact = self.exact_min_distance_hyperplane_for_row(row)
            if exact is not None:
                egx = float(exact["gx"])
                egy = float(exact["gy"])
                erhs = float(exact["rhs"])

                clipped_exact = self.clip_world_line_to_bounds(egx, egy, erhs)
                if clipped_exact is not None:
                    for x, y in clipped_exact:
                        p = Point()
                        p.x = float(x)
                        p.y = float(y)
                        p.z = 0.003
                        exact_line.points.append(p)

                exact_feasible_poly = self.clip_polygon_to_halfplane(
                    world_rect, egx, egy, erhs
                )
                self.append_filled_polygon_triangles(
                    exact_fill, exact_feasible_poly, z_vis=0.0015
                )

                ng = math.hypot(gx, gy)
                neg = math.hypot(egx, egy)
                if ng > 1.0e-12 and neg > 1.0e-12:
                    dot = (gx * egx + gy * egy) / (ng * neg)
                    dot = min(max(dot, -1.0), 1.0)
                    angle_deg = math.degrees(math.acos(dot))
                    surrogate_shift = (rhs - gx * pbar_x - gy * pbar_y) / ng
                    exact_shift = (erhs - egx * pbar_x - egy * pbar_y) / neg
                    rospy.loginfo_throttle(
                        1.0,
                        "[plotter] k0 separating-hyperplane comparison: exact_gid=%d "
                        "Phi_smooth=%.6f Phi_exact=%.6f normal_angle=%.3fdeg "
                        "boundary_shift_smooth=%.6f boundary_shift_exact=%.6f",
                        int(exact["group_id"]), float(row[13]), float(exact["phi_gt"]),
                        float(angle_deg), float(surrogate_shift), float(exact_shift),
                    )

        arr.markers.append(fill)
        arr.markers.append(exact_fill)
        arr.markers.append(line)
        arr.markers.append(exact_line)
        self._cached_separating_hyperplane_marker_array = arr
        self._separating_hyperplane_dirty = False
        self.separating_hyperplane_pub.publish(arr)

    def publish_dynamic_circle_horizon(self, msg, stamp=None):
        """Draw only detected dynamic circles from /dynamic_obstacles.

        Input center/velocity/radius are assumed to be expressed in msg.header.frame_id
        (normally odom). Marker z is prediction time, as elsewhere in this plotter.
        """
        if stamp is None:
            stamp = rospy.Time.now()
        frame_id = str(getattr(getattr(msg, "header", None), "frame_id", "") or "odom").lstrip("/")
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "dynamic_circle_planning_horizon"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(self.safe_margin_line_width), 0.01)
        self.set_color(marker, (0.0, 0.85, 0.25, 1.0))
        n = max(0, min(int(msg.obs_cnt), len(msg.obs_xs), len(msg.obs_ys),
                       len(msg.obs_rs), len(msg.obs_vxs), len(msg.obs_vys)))
        steps = self.safe_margin_prediction_indices()
        if not self.safe_margin_draw_current_boundary:
            steps = [h for h in steps if h != 0]
        resolution = max(12, int(rospy.get_param("~dynamic_circle_horizon_resolution", 32)))
        angles = [2.0 * math.pi * j / resolution for j in range(resolution + 1)]
        unit = [(math.cos(a), math.sin(a)) for a in angles]
        valid_count = 0
        for i in range(n):
            cx, cy = float(msg.obs_xs[i]), float(msg.obs_ys[i])
            vx, vy = float(msg.obs_vxs[i]), float(msg.obs_vys[i])
            r = float(msg.obs_rs[i])
            if not all(math.isfinite(v) for v in (cx, cy, vx, vy, r)) or r <= 0.0:
                continue
            valid_count += 1
            for h in steps:
                t = max(0.0, float(h) * float(self.pred_dt))
                x, y = cx + vx * t, cy + vy * t
                radius = r + (self.probability_safe_margin(t) if t > 0.0 else 0.0)
                z = float(TIME_SCALE) * t
                for j in range(resolution):
                    ax, ay = unit[j]
                    bx, by = unit[j + 1]
                    marker.points.append(Point(x=x+radius*ax, y=y+radius*ay, z=z))
                    marker.points.append(Point(x=x+radius*bx, y=y+radius*by, z=z))
        out.markers.append(marker)
        self._cached_selector_horizon_marker_array = out
        self._selector_horizon_dirty = False
        self.obs_polygon_horizon_pub.publish(out)
        self.obs_planning_horizon_pub.publish(out)
        rospy.loginfo_throttle(3.0, "[plotter] dynamic circle horizon: input=%s circles=%d stages=%d frame=%s segments=%d",
                               self.dynamic_obstacles_topic, valid_count, len(steps), frame_id, len(marker.points)//2)

    def publish_selector_horizon_boundaries(self, horizon_msg, current_msg=None, stamp=None):
        """Publish selector-only obstacle horizon at 20 Hz with cached geometry.

        h=0 uses /selected_vertex_data. h=1..H uses
        /selected_vertex_horizon_data. Features are grouped by group_id and each
        group is reconstructed with a convex hull. Dynamic future features are
        inflated by the configured sigma_vx/sigma_vy/pc uncertainty. No
        /obstacle position, vx, vy, or polygon prediction is used.
        """
        if stamp is None:
            stamp = rospy.Time.now()

        payload = self.decode_selected_vertex_horizon_payload(horizon_msg)
        if payload.size == 0:
            rospy.logwarn_throttle(
                2.0,
                "[plotter] %s is empty/invalid; cannot build /obstacle_polygon_planning_horizon",
                self.selected_vertex_horizon_data_topic,
            )

        # vertex_selector publishes both /selected_vertex_data and
        # /selected_vertex_horizon_data in odom coordinates.  Their custom ROS
        # messages have no reliable frame header, so get_frame_id() (which
        # defaults to base_link) would label odom points as base_link.  The
        # RViz projector would then transform them a second time, making fixed
        # obstacles rotate/translate with the ego vehicle.
        frame_id = "odom"

        marker_array = MarkerArray()
        line_marker = Marker()
        line_marker.header.frame_id = frame_id
        line_marker.header.stamp = stamp
        line_marker.ns = "obstacle_polygons_horizon"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = float(self.safe_margin_line_width)
        line_marker.scale.y = 1.0
        line_marker.scale.z = 1.0
        self.set_color(line_marker, self.safe_margin_boundary_color(0))

        # h=0. Convert message arrays once and group by group_id with NumPy.
        if bool(self.safe_margin_draw_current_boundary) and current_msg is not None:
            n_msg = min(
                max(0, int(getattr(current_msg, "obs_cnt", 0))),
                len(getattr(current_msg, "obs_xs", [])),
                len(getattr(current_msg, "obs_ys", [])),
            )
            if n_msg > 0:
                xs0 = np.asarray(current_msg.obs_xs[:n_msg], dtype=np.float64)
                ys0 = np.asarray(current_msg.obs_ys[:n_msg], dtype=np.float64)
                if hasattr(current_msg, "obs_rs") and len(current_msg.obs_rs) >= n_msg:
                    gids0 = np.rint(np.asarray(current_msg.obs_rs[:n_msg], dtype=np.float64)).astype(np.int64)
                else:
                    gids0 = np.zeros(n_msg, dtype=np.int64)
                valid0 = np.isfinite(xs0) & np.isfinite(ys0)
                if np.any(valid0):
                    xs0 = xs0[valid0]
                    ys0 = ys0[valid0]
                    gids0 = gids0[valid0]
                    for gid in np.unique(gids0):
                        mask = gids0 == gid
                        pts = np.column_stack((xs0[mask], ys0[mask]))
                        self.append_selector_group_boundary(line_marker, pts, z_vis=0.0)

        H = int(payload.shape[0]) if payload.ndim == 3 else 0
        wanted_steps = self.safe_margin_prediction_indices()
        wanted_set = set(wanted_steps)
        terminal_wanted_step = max(wanted_steps) if wanted_steps else -1
        dt = max(float(self.pred_dt), 1.0e-9)

        # The obstacle horizon is indexed exactly like the MPC collision field:
        # stage h uses t_h = h * dt.  It does not depend on the selected ego trajectory.

        for h0 in range(H):
            h = h0 + 1
            if h not in wanted_set:
                continue

            t_pred = float(h) * dt

            stage = np.asarray(payload[h0], dtype=np.float64)
            if stage.ndim != 2 or stage.shape[1] < 6 or stage.shape[0] == 0:
                continue
            valid = np.all(np.isfinite(stage[:, :6]), axis=1)
            if not np.any(valid):
                continue
            stage = stage[valid, :6]

            px = stage[:, 0].copy()
            py = stage[:, 1].copy()
            dynamic = np.rint(stage[:, 4]).astype(np.int64) != 0
            gids = np.rint(stage[:, 5]).astype(np.int64)

            # The selector owns prediction only.  Reconstruct each predicted
            # obstacle/group from its stage-wise vertices, then apply the same
            # isotropic uncertainty radius r_h used by the MPC collision field.
            r_unc_stage = self.probability_safe_margin(t_pred)
            dynamic_group_count = 0
            inflated_group_count = 0

            z_vis = float(TIME_SCALE) * t_pred
            for gid in np.unique(gids):
                mask = gids == gid
                pts = np.column_stack((px[mask], py[mask]))
                group_dynamic = bool(np.any(dynamic[mask]))
                if group_dynamic:
                    dynamic_group_count += 1
                group_inflation = float(r_unc_stage) if (group_dynamic and t_pred > 0.0) else 0.0
                if group_inflation > 0.0:
                    inflated_group_count += 1
                self.append_selector_group_boundary(
                    line_marker, pts, z_vis=z_vis, inflation_radius=group_inflation
                )

            if h == terminal_wanted_step:
                rospy.loginfo_throttle(
                    2.0,
                    "[plotter] /obs_polygons_horizon uncertainty: h=%d t=%.3f sigma=(%.4f,%.4f) pc=%.4f r_unc=%.4f dynamic_groups=%d inflated_groups=%d",
                    int(h), float(t_pred),
                    float(self.normal_cfg.get("sigma_vx", 0.0)),
                    float(self.normal_cfg.get("sigma_vy", 0.0)),
                    float(self.configured_collision_probability()),
                    float(r_unc_stage), int(dynamic_group_count), int(inflated_group_count),
                )

        marker_array.markers.append(line_marker)
        marker_id = 1

        if bool(self.safe_margin_text_enabled):
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = "obstacle_polygons_horizon"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(self.x_min)
            text_marker.pose.position.y = float(self.y_max)
            text_marker.pose.position.z = float(TIME_SCALE) * dt * float(H)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.28
            sigma_vx = float(self.normal_cfg.get("sigma_vx", 0.0))
            sigma_vy = float(self.normal_cfg.get("sigma_vy", 0.0))
            p_col = float(self.configured_collision_probability())
            terminal_t = float(H) * dt
            terminal_r_unc = self.probability_safe_margin(terminal_t)
            text_marker.text = (
                "vertex-selector obstacle horizon: stages=%d features=%d interval=%d "
                "uncertainty sigma=(%.3f,%.3f) pc=%.3f r_unc(H)=%.3f m"
            ) % (
                H, int(payload.shape[1]) if payload.ndim == 3 else 0,
                int(self.safe_margin_pred_step_interval), sigma_vx, sigma_vy, p_col,
                terminal_r_unc,
            )
            self.set_color(text_marker, (0.72, 0.72, 0.72, 1.0))
            marker_array.markers.append(text_marker)
            marker_id += 1

        prev_cnt = max(
            int(self.prev_obs_polygon_horizon_marker_cnt),
            int(self.prev_obs_polygon_planning_horizon_marker_cnt),
        )
        for stale_id in range(marker_id, prev_cnt):
            mdel = Marker()
            mdel.header.frame_id = frame_id
            mdel.header.stamp = stamp
            mdel.ns = "obstacle_polygons_horizon"
            mdel.id = stale_id
            mdel.action = Marker.DELETE
            marker_array.markers.append(mdel)

        self.prev_obs_polygon_horizon_marker_cnt = marker_id
        self.prev_obs_polygon_planning_horizon_marker_cnt = marker_id
        self._cached_selector_horizon_marker_array = marker_array
        self._selector_horizon_dirty = False
        self.obs_polygon_horizon_pub.publish(marker_array)

    def publish_ego_vehicle_marker(self, stamp=None):
        # =========================
        # Ego visualization
        # =========================
        if stamp is None:
            stamp = rospy.Time.now()

        # ego_L/ego_W define the real rectangular ego body.
        # The dashed ellipse is a conservative over-estimation of that
        # rectangle.  For a rectangle with full length L and width W, the
        # ellipse with full axes sqrt(2)*L and sqrt(2)*W passes through the
        # four rectangle corners:
        #   (L/2)^2/(L/sqrt(2))^2 + (W/2)^2/(W/sqrt(2))^2 = 1.
        L = abs(float(self.ego_L))
        W = abs(float(self.ego_W))
        fallback_d = max(2.0 * abs(float(getattr(self, "ego_R", 0.0))), 1.0e-3)
        if L <= 1.0e-9:
            L = fallback_d
        if W <= 1.0e-9:
            W = fallback_d

        qx, qy, qz, qw = self.yaw_to_quaternion(self.ego_yaw)

        # Remove the old CYLINDER ellipse marker once, so RViz does not keep a
        # stale solid ellipse when this node is hot-restarted.
        if not bool(getattr(self, "ego_legacy_cylinder_deleted", False)):
            legacy_delete = Marker()
            legacy_delete.header.frame_id = "base_link"
            legacy_delete.header.stamp = stamp
            legacy_delete.ns = "ego_visualization"
            legacy_delete.id = 0
            legacy_delete.action = Marker.DELETE
            self.ego_pub.publish(legacy_delete)
            self.ego_legacy_cylinder_deleted = True

        # 1) Real ego vehicle body: rectangle with full size ego_L x ego_W.
        body_marker = Marker()
        body_marker.header.frame_id = "base_link"
        body_marker.header.stamp = stamp
        body_marker.ns = "ego_vehicle_rectangle"
        body_marker.id = 0
        body_marker.type = Marker.CUBE
        body_marker.action = Marker.ADD
        body_marker.pose.position.x = float(self.ego_cart_x)
        body_marker.pose.position.y = float(self.ego_cart_y)
        body_marker.pose.position.z = 0.01
        body_marker.pose.orientation.x = qx
        body_marker.pose.orientation.y = qy
        body_marker.pose.orientation.z = qz
        body_marker.pose.orientation.w = qw
        body_marker.scale.x = max(L, 1.0e-4)
        body_marker.scale.y = max(W, 1.0e-4)
        body_marker.scale.z = 0.02
        self.set_color(body_marker, (0.0, 0.0, 1.0, 0.70))
        self.ego_pub.publish(body_marker)

        # 2) Conservative ego ellipse: dashed line, over-estimating rectangle.
        ellipse_marker = Marker()
        ellipse_marker.header.frame_id = "base_link"
        ellipse_marker.header.stamp = stamp
        ellipse_marker.ns = "ego_overestimated_ellipse"
        ellipse_marker.id = 0
        ellipse_marker.type = Marker.LINE_LIST
        ellipse_marker.action = Marker.ADD
        ellipse_marker.pose.position.x = float(self.ego_cart_x)
        ellipse_marker.pose.position.y = float(self.ego_cart_y)
        ellipse_marker.pose.position.z = 0.035
        ellipse_marker.pose.orientation.x = qx
        ellipse_marker.pose.orientation.y = qy
        ellipse_marker.pose.orientation.z = qz
        ellipse_marker.pose.orientation.w = qw
        ellipse_marker.scale.x = float(rospy.get_param("~ego_ellipse_line_width", 0.035))
        ellipse_marker.scale.y = 1.0
        ellipse_marker.scale.z = 1.0
        self.set_color(ellipse_marker, (0.0, 0.0, 1.0, 1.0))

        a = L / math.sqrt(2.0)  # semi-major/semi-x axis of over-estimation
        b = W / math.sqrt(2.0)  # semi-minor/semi-y axis of over-estimation
        dash_count = max(16, int(rospy.get_param("~ego_ellipse_dash_count", 64)))
        for i in range(dash_count):
            # LINE_LIST has no native dashed style, so draw every other small
            # arc segment and leave the next one empty.
            if i % 2 != 0:
                continue
            th1 = 2.0 * math.pi * float(i) / float(dash_count)
            th2 = 2.0 * math.pi * float(i + 1) / float(dash_count)

            p1 = Point()
            p1.x = float(a * math.cos(th1))
            p1.y = float(b * math.sin(th1))
            p1.z = 0.0
            p2 = Point()
            p2.x = float(a * math.cos(th2))
            p2.y = float(b * math.sin(th2))
            p2.z = 0.0
            ellipse_marker.points.append(p1)
            ellipse_marker.points.append(p2)

        self.ego_pub.publish(ellipse_marker)


    def publish_ego_aux_markers(self, stamp=None):
        # Ego FOV and designated-sector visualization.  This method is called
        # only by the common fixed-rate publication loop.
        if stamp is None:
            stamp = rospy.Time.now()
        fov_ns = "ego_fov_visualization"

        if not self.fov_marker_initialized:
            for old_id in (0, 1):
                fov_delete = Marker()
                fov_delete.header.frame_id = "base_link"
                fov_delete.header.stamp = stamp
                fov_delete.ns = fov_ns
                fov_delete.id = old_id
                fov_delete.action = Marker.DELETE
                self.ego_FOV_pub.publish(fov_delete)
            self.fov_marker_initialized = True

        fov_marker = Marker()
        fov_marker.header.frame_id = "base_link"
        fov_marker.header.stamp = stamp
        fov_marker.ns = fov_ns
        fov_marker.id = 0
        fov_marker.type = Marker.TRIANGLE_LIST
        fov_marker.action = Marker.ADD
        fov_marker.pose.orientation.w = 1.0
        fov_marker.scale.x = 1.0
        fov_marker.scale.y = 1.0
        fov_marker.scale.z = 1.0
        self.set_color(fov_marker, (0.0, 1.0, 0.0, 0.18))

        fov_radius = max(float(self.ego_FOV), 1.0e-6)
        half_angle = math.radians(float(self.obs_fov_deg))
        yaw = float(self.ego_yaw)
        theta_min = yaw - half_angle
        theta_max = yaw + half_angle
        arc_steps = max(8, min(int(self.fov_arc_steps_max), int(abs(self.obs_fov_deg))))

        center = Point()
        center.x = self.ego_cart_x
        center.y = self.ego_cart_y
        center.z = 0.0

        for j in range(arc_steps):
            th1 = theta_min + (theta_max - theta_min) * float(j) / float(arc_steps)
            th2 = theta_min + (theta_max - theta_min) * float(j + 1) / float(arc_steps)

            p0 = Point()
            p0.x = center.x
            p0.y = center.y
            p0.z = center.z
            p1 = Point()
            p1.x = self.ego_cart_x + fov_radius * math.cos(th1)
            p1.y = self.ego_cart_y + fov_radius * math.sin(th1)
            p1.z = 0.0
            p2 = Point()
            p2.x = self.ego_cart_x + fov_radius * math.cos(th2)
            p2.y = self.ego_cart_y + fov_radius * math.sin(th2)
            p2.z = 0.0

            fov_marker.points.extend([p0, p1, p2, p0, p2, p1])

        self.ego_FOV_pub.publish(fov_marker)

        fov_outline = Marker()
        fov_outline.header.frame_id = "base_link"
        fov_outline.header.stamp = stamp
        fov_outline.ns = fov_ns
        fov_outline.id = 1
        fov_outline.type = Marker.LINE_STRIP
        fov_outline.action = Marker.ADD
        fov_outline.pose.orientation.w = 1.0
        fov_outline.scale.x = 0.03
        self.set_color(fov_outline, (0.0, 1.0, 0.0, 0.75))

        p_start = Point()
        p_start.x = self.ego_cart_x
        p_start.y = self.ego_cart_y
        p_start.z = 0.0
        fov_outline.points.append(p_start)

        for j in range(arc_steps + 1):
            th = theta_min + (theta_max - theta_min) * float(j) / float(arc_steps)
            p = Point()
            p.x = self.ego_cart_x + fov_radius * math.cos(th)
            p.y = self.ego_cart_y + fov_radius * math.sin(th)
            p.z = 0.0
            fov_outline.points.append(p)

        p_end = Point()
        p_end.x = self.ego_cart_x
        p_end.y = self.ego_cart_y
        p_end.z = 0.0
        fov_outline.points.append(p_end)
        self.ego_FOV_pub.publish(fov_outline)

        self.publish_ego_sector_markers(stamp)

    def update_ego_from_tf(self):
        """Obtain the robot-center pose in odom directly from TF.

        Matches the vertex_selector transform direction and yaw extraction.
        Time(0) selects the newest common transform for continuous RViz output.
        """
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.ego_tf_target, self.ego_tf_frame,
                rospy.Time(0), rospy.Duration(self.ego_tf_timeout_s))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0,
                "[plotter] ego TF %s->%s unavailable: %s",
                self.ego_tf_frame, self.ego_tf_target, str(exc))
            self.have_ego_tf = False
            return False
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        self.ego_cart_x = float(t.x)
        self.ego_cart_y = float(t.y)
        self.ego_yaw = _wrap_yaw_np(math.atan2(float(R[1, 0]), float(R[0, 0])))
        self.have_ego_tf = True
        return True

    def publish_goal_marker(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "goal_visualization"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # print("goal_x: ", self.goal_x)
        # print("goal_y: ", self.goal_y)
        marker.pose.position.x = float(self.goal_x)
        marker.pose.position.y = float(self.goal_y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        r = max(float(self.ego_R), 1.0e-3)
        marker.scale.x = r * 2.0
        marker.scale.y = r * 2.0
        marker.scale.z = r * 2.0
        self.set_color(marker, (1.0, 0.0, 0.0, 1.0))
        self.goal_pub.publish(marker)

    def goal_callback(self, data):
        """Update RViz goal from setter's /ego_goal_set (odom coordinates)."""
        gx = float(data.x_end)
        gy = float(data.y_end)
        if not (math.isfinite(gx) and math.isfinite(gy)):
            rospy.logwarn("[plotter] rejecting non-finite /ego_goal_set")
            return
        self.latest_goal_msg = data
        self.goal_x = gx
        self.goal_y = gy
        self.runtime_goal_received = True
        self.publish_goal_marker(rospy.Time.now())
        rospy.loginfo("[plotter] runtime goal updated: (%.3f, %.3f) frame=base_link", gx, gy)

    def publish_path_marker(self, data, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "trajectory_visualization"
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        self.set_color(marker, (0.0, 1.0, 0.0, 1.0))

        horizon = min(
            max(0, int(data.horizon)),
            len(getattr(data, "traj_xs", [])),
            len(getattr(data, "traj_ys", [])),
        )
        for i in range(horizon):
            p = Point()
            p.x = float(data.traj_xs[i])
            p.y = float(data.traj_ys[i])
            p.z = self.time_to_vis_z(i)
            marker.points.append(p)

        self.trajectory_pub.publish(marker)

    def path_callback(self, data):
        self.latest_path_msg = data
        self._path_dirty = True

    def publish_sampled_traj_markers(self, data, stamp=None):
        """Publish all sampled trajectories as one LINE_LIST Marker.

        Keeping one marker instead of up to hundreds of LINE_STRIP markers is
        substantially cheaper for rospy serialization and RViz scene updates,
        while drawing the exact same line segments at the same 20 Hz cadence.
        """
        K = int(data.plot_cnt)
        N = int(data.horizon)
        if K <= 0 or N <= 1:
            return

        total = K * N
        if len(data.sampled_traj_xs) < total or len(data.sampled_traj_ys) < total:
            rospy.logwarn_throttle(1.0, "Sampled_traj x/y array size is smaller than K*N")
            return

        max_markers = int(self.max_sampled_traj_markers)
        if max_markers <= 0:
            return
        if K <= max_markers:
            selected_indices = range(K)
        else:
            stride = int(math.ceil(float(K) / float(max_markers)))
            selected_indices = range(0, K, stride)

        if stamp is None:
            stamp = rospy.Time.now()
        dt = self.active_prediction_dt()

        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "sampled_trajs"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        self.set_color(marker, (1.0, 1.0, 1.0, 0.5))

        xs = data.sampled_traj_xs
        ys = data.sampled_traj_ys
        # Build segment endpoints directly.  LINE_LIST avoids one Marker object
        # per trajectory, which is the dominant cost when K is large.
        count = 0
        for k in selected_indices:
            if count >= max_markers:
                break
            rr = self.RRRs[k % len(self.RRRs)] / 255.0
            gg = self.GGGs[k % len(self.GGGs)] / 255.0
            bb = self.BBBs[k % len(self.BBBs)] / 255.0
            base = int(k) * int(N)
            for j in range(N - 1):
                p1 = Point()
                p1.x = float(xs[base + j])
                p1.y = float(ys[base + j])
                p1.z = float(TIME_SCALE) * dt * float(j)
                p2 = Point()
                p2.x = float(xs[base + j + 1])
                p2.y = float(ys[base + j + 1])
                p2.z = float(TIME_SCALE) * dt * float(j + 1)
                marker.points.append(p1)
                marker.points.append(p2)
                c1 = marker.color.__class__()
                c1.r, c1.g, c1.b, c1.a = rr, gg, bb, 0.5
                c2 = marker.color.__class__()
                c2.r, c2.g, c2.b, c2.a = rr, gg, bb, 0.5
                marker.colors.append(c1)
                marker.colors.append(c2)
            count += 1

        marker_array.markers.append(marker)

        # Clean up marker IDs left by the old one-Marker-per-trajectory layout.
        for stale_id in range(1, int(self.prev_sampled_marker_cnt)):
            mdel = Marker()
            mdel.header.frame_id = "base_link"
            mdel.header.stamp = stamp
            mdel.ns = "sampled_trajs"
            mdel.id = stale_id
            mdel.action = Marker.DELETE
            marker_array.markers.append(mdel)

        self.prev_sampled_marker_cnt = 1
        self._cached_sampled_traj_marker_array = marker_array
        self._sampled_traj_dirty = False
        self.sampled_traj_pub.publish(marker_array)

    def selected_traj_callback(self, msg):
        """Cache ego_mover's final selected trajectory for RViz."""
        self.latest_selected_traj_msg = msg
        self._selected_traj_dirty = True

    def publish_selected_traj_marker(self, msg, stamp=None):
        """Publish /selected_traj as one emphasized RViz LINE_STRIP.

        Expected payload is an (H+1) x 4 Float64MultiArray containing
        [x, y, v, yaw].  A flat-array fallback is accepted when the layout is
        absent, provided the payload length is divisible by four.
        """
        if stamp is None:
            stamp = rospy.Time.now()

        raw = np.asarray(getattr(msg, "data", []), dtype=float).reshape(-1)
        if raw.size < 8:
            return

        cols = 4
        dims = list(getattr(getattr(msg, "layout", None), "dim", []))
        if len(dims) >= 2:
            try:
                rows = int(dims[0].size)
                cols_msg = int(dims[1].size)
            except Exception:
                rows, cols_msg = 0, 0
            if rows > 0 and cols_msg >= 2 and raw.size >= rows * cols_msg:
                cols = cols_msg
                states = raw[:rows * cols].reshape(rows, cols)
            elif raw.size % 4 == 0:
                states = raw.reshape(-1, 4)
            else:
                rospy.logwarn_throttle(2.0, "[plotter] malformed /selected_traj layout")
                return
        elif raw.size % 4 == 0:
            states = raw.reshape(-1, 4)
        else:
            rospy.logwarn_throttle(2.0, "[plotter] /selected_traj payload is not Nx4")
            return

        if states.shape[0] < 2 or states.shape[1] < 2:
            return

        xy = np.asarray(states[:, :2], dtype=float)
        valid = np.all(np.isfinite(xy), axis=1)
        xy = xy[valid]
        if xy.shape[0] < 2:
            return

        marker = Marker()
        marker.header.frame_id = self.get_frame_id(self.latest_ego_msg) if self.latest_ego_msg is not None else "base_link"
        marker.header.stamp = stamp
        marker.ns = "selected_traj"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(self.selected_traj_marker_width), 1.0e-4)
        self.set_color(marker, self.selected_traj_color)

        z = float(self.selected_traj_marker_z)
        for x, y in xy:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = z
            marker.points.append(point)

        self._cached_selected_traj_marker = marker
        self._selected_traj_dirty = False
        self.selected_traj_marker_pub.publish(marker)

    def sampled_traj_callback(self, data):
        self.latest_sampled_traj_msg = data
        self._sampled_traj_dirty = True

    def publish_all_visualizations(self):
        """Publish RViz outputs at configured rates, reusing cached heavy geometry."""
        stamp = rospy.Time.now()
        # Refresh ego pose before generating any RViz geometry or SDF queries.
        # Avoid mixing the simulation /ego_state coordinates with TF odometry.
        if not self.update_ego_from_tf():
            return

        # All visualization groups can remain at 20 Hz. Heavy geometry below is
        # cached, so 20 Hz does not imply 20 Hz convex-hull/Point reconstruction.
        if self.should_publish_visualization_now("last_static_viz_stamp", self.static_viz_hz):
            self.publish_boundary(stamp)
            self.publish_goal_marker(stamp)

        # Keep the actual ego body responsive; throttle only the heavier FOV/sector.
        if self.should_publish_visualization_now("last_ego_vehicle_viz_stamp", self.ego_vehicle_viz_hz):
            self.publish_ego_vehicle_marker(stamp)
        if self.should_publish_visualization_now("last_ego_aux_viz_stamp", self.ego_aux_viz_hz):
            self.publish_ego_aux_markers(stamp)

        path_msg = self.latest_path_msg
        if (path_msg is not None and
                (not self.fast_viz_mode or self._path_dirty) and
                self._publisher_has_subscribers(self.trajectory_pub) and
                self.should_publish_visualization_now("last_path_viz_stamp", self.path_viz_hz)):
            self.publish_path_marker(path_msg, stamp)
            self._path_dirty = False

        obstacle = self.latest_obstacle_msg
        if (obstacle is not None and
                (not self.fast_viz_mode or self._obs_polygon_dirty) and
                self._publisher_has_subscribers(self.obs_polygon_pub) and
                self.should_publish_visualization_now(
                    "last_obs_polygon_viz_stamp", self.obs_polygon_viz_hz)):
            if bool(self.show_obstacle_positions):
                self.publish_obstacle_positions(
                    obstacle=obstacle,
                    pub=self.obs_pos_pub,
                    ns="obstacle_positions",
                    marker_id=1,
                    rgba=(0.0, 0.0, 0.0, 1.0),
                    radius_offset=0.0,
                    draw_prediction=False,
                )
            if self._obs_polygon_dirty or self._cached_obs_polygon_marker_array is None:
                self.publish_obstacle_polygons(
                    obstacle=obstacle,
                    pub=self.obs_polygon_pub,
                    ns="obstacle_polygons",
                    prev_cnt_attr="prev_obs_polygon_marker_cnt",
                    color_fn=self.actual_polygon_color,
                    line_width=self.safe_margin_line_width,
                    draw_prediction=False,
                    apply_safe_margin=False,
                    draw_current_fill=True,
                )
            elif not self.fast_viz_mode:
                self.obs_polygon_pub.publish(self._cached_obs_polygon_marker_array)

        # Dynamic circle horizon now comes exclusively from /dynamic_obstacles,
        # not the selector's polygon-vertex horizon or ground-truth /obstacle.
        dynamic_msg = self.latest_dynamic_obstacles_msg
        if dynamic_msg is not None:
            if ((not self.fast_viz_mode or self._selector_horizon_dirty) and
                    (self._publisher_has_subscribers(self.obs_polygon_horizon_pub) or
                     self._publisher_has_subscribers(self.obs_planning_horizon_pub)) and
                    self.should_publish_visualization_now(
                        "last_obs_polygon_horizon_viz_stamp", self.safe_margin_horizon_viz_hz)):
                self.publish_dynamic_circle_horizon(dynamic_msg, stamp=stamp)
            elif not self.fast_viz_mode and self._cached_selector_horizon_marker_array is not None:
                self.obs_polygon_horizon_pub.publish(self._cached_selector_horizon_marker_array)
                self.obs_planning_horizon_pub.publish(self._cached_selector_horizon_marker_array)
        else:
            rospy.logwarn_throttle(3.0, "[plotter] waiting for %s", self.dynamic_obstacles_topic)

        # Exact solver separating hyperplanes from MPC_planner0.  Reuse the same
        # visualization cadence and displayed horizon stages as /obs_polygons_horizon.
        sep_msg = self.latest_separating_hyperplane_msg
        if (sep_msg is not None and
                (not self.fast_viz_mode or self._separating_hyperplane_dirty) and
                self._publisher_has_subscribers(self.separating_hyperplane_pub) and
                self.should_publish_visualization_now(
                    "last_separating_hyperplane_viz_stamp", self.safe_margin_horizon_viz_hz)):
            if (self._separating_hyperplane_dirty or
                    self._cached_separating_hyperplane_marker_array is None):
                self.publish_separating_hyperplanes(sep_msg, stamp=stamp)
            elif not self.fast_viz_mode:
                self.separating_hyperplane_pub.publish(
                    self._cached_separating_hyperplane_marker_array
                )

        sampled_msg = self.latest_sampled_traj_msg
        if (sampled_msg is not None and
                (not self.fast_viz_mode or self._sampled_traj_dirty) and
                self._publisher_has_subscribers(self.sampled_traj_pub) and
                self.should_publish_visualization_now(
                    "last_sampled_traj_viz_stamp", self.sampled_traj_viz_hz)):
            if self._sampled_traj_dirty or self._cached_sampled_traj_marker_array is None:
                self.publish_sampled_traj_markers(sampled_msg, stamp)
            elif not self.fast_viz_mode:
                self.sampled_traj_pub.publish(self._cached_sampled_traj_marker_array)

        selected_traj_msg = self.latest_selected_traj_msg
        if (selected_traj_msg is not None and
                (not self.fast_viz_mode or self._selected_traj_dirty) and
                self._publisher_has_subscribers(self.selected_traj_marker_pub) and
                self.should_publish_visualization_now(
                    "last_selected_traj_viz_stamp", self.selected_traj_viz_hz)):
            if self._selected_traj_dirty or self._cached_selected_traj_marker is None:
                self.publish_selected_traj_marker(selected_traj_msg, stamp)
            elif not self.fast_viz_mode:
                self._cached_selected_traj_marker.header.stamp = stamp
                self.selected_traj_marker_pub.publish(self._cached_selected_traj_marker)

        # SDF has its own independent sdf_grid_pub_hz throttle.
        self.publish_current_sdf(perceived_msg=self.latest_percepted_obstacle_msg)

        impact_hz = float(self.impact_normal_viz_hz)
        try:
            # Bernstein fitting is the expensive case. Keep the normal diagnostic
            # synchronized with the SDF refresh so one fitted model serves both.
            if int(_utils_linearized.collision_field_flag(self.sdf_cfg)) == 2:
                impact_hz = min(impact_hz, max(float(self.sdf_grid_pub_hz), 1.0e-9))
        except Exception:
            pass
        if (self._publisher_has_subscribers(self.impact_normal_pub) and
                self.should_publish_visualization_now(
                    "last_impact_normal_viz_stamp", impact_hz)):
            self.publish_current_impact_normal_comparison(
                perceived_msg=self.latest_percepted_obstacle_msg, stamp=stamp
            )

        selected_msg = self.latest_selected_vertex_data_msg
        selected_needed = (
            self._publisher_has_subscribers(self.selected_vertex_set_pub) or
            self._publisher_has_subscribers(self.selected_vertex_normal_pub)
        )
        if (bool(self.publish_selected_vertex_markers) and selected_msg is not None and
                (not self.fast_viz_mode or self._selected_vertex_dirty) and selected_needed and
                self.should_publish_visualization_now(
                    "last_selected_vertex_viz_stamp", self.selected_vertex_viz_hz)):
            if (self._selected_vertex_dirty or self._cached_selected_vertex_point_marker is None or
                    self._cached_selected_vertex_normal_array is None):
                self.publish_selected_vertex_markers_from_data(selected_msg)
            elif not self.fast_viz_mode:
                self.selected_vertex_set_pub.publish(self._cached_selected_vertex_point_marker)
                self.selected_vertex_normal_pub.publish(self._cached_selected_vertex_normal_array)

        prm_msg = self.latest_visibility_prm_graph_msg
        if (prm_msg is not None and
                (not self.fast_viz_mode or self._visibility_prm_dirty) and
                self._publisher_has_subscribers(self.visibility_prm_pub) and
                self.should_publish_visualization_now(
                    "last_visibility_prm_viz_stamp", self.visibility_viz_hz)):
            self.publish_visibility_prm_graph_from_data(prm_msg, stamp)
            self._visibility_prm_dirty = False

        guidance_msg = self.latest_visibility_guidance_msg
        guidance_needed = (
            self._publisher_has_subscribers(self.visibility_guide_marker_pub) or
            any(self._publisher_has_subscribers(pub) for pub in self.visibility_guide_path_pubs)
        )
        if (guidance_msg is not None and
                (not self.fast_viz_mode or self._visibility_guidance_dirty) and guidance_needed and
                self.should_publish_visualization_now(
                    "last_visibility_guidance_viz_stamp", self.visibility_viz_hz)):
            self.publish_visibility_guidance_from_data(guidance_msg, stamp)
            self._visibility_guidance_dirty = False

        backward_prm_msg = self.latest_visibility_backward_prm_graph_msg
        if (backward_prm_msg is not None and
                (not self.fast_viz_mode or self._visibility_backward_prm_dirty) and
                self._publisher_has_subscribers(self.visibility_backward_prm_pub) and
                self.should_publish_visualization_now(
                    "last_visibility_backward_prm_viz_stamp", self.visibility_viz_hz)):
            self.publish_visibility_prm_graph_from_data(
                backward_prm_msg, stamp, pub=self.visibility_backward_prm_pub
            )
            self._visibility_backward_prm_dirty = False

        backward_guidance_msg = self.latest_visibility_backward_guidance_msg
        backward_guidance_needed = (
            self._publisher_has_subscribers(self.visibility_backward_guide_marker_pub) or
            any(self._publisher_has_subscribers(pub) for pub in self.visibility_backward_guide_path_pubs)
        )
        if (backward_guidance_msg is not None and
                (not self.fast_viz_mode or self._visibility_backward_guidance_dirty) and
                backward_guidance_needed and
                self.should_publish_visualization_now(
                    "last_visibility_backward_guidance_viz_stamp", self.visibility_viz_hz)):
            self.publish_visibility_guidance_from_data(
                backward_guidance_msg, stamp,
                marker_pub=self.visibility_backward_guide_marker_pub,
                path_pubs=self.visibility_backward_guide_path_pubs,
                guidance_slot_start=4,
                initialized_attr="visibility_backward_guide_marker_initialized",
                prev_count_attr="prev_visibility_backward_guide_count",
            )
            self._visibility_backward_guidance_dirty = False

        # Callbacks already mark these caches dirty.  Do not serialize the same
        # complete MarkerArrays again when no planner message changed.
        self.flush_fast_marker_arrays(force=False)

    def publish_boundary(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns = "boundary_visualization"
        marker.id = 10
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        self.set_color(marker, (1.0, 0.0, 0.0, 1.0))

        corners = [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
            (self.x_min, self.y_min),
        ]

        for x, y in corners:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.boundary_pub.publish(marker)

def main():
    rospy.init_node("Plotter", anonymous=False)
    fct = Frenet2CartesianTransformer()
    pub_hz = max(float(fct.visualization_pub_hz), 0.1)
    rate = rospy.Rate(pub_hz)
    rospy.loginfo(
        "[plotter] common RViz loop %.1f Hz; heavy rates: obs=%.1f horizon=%.1f sampled=%.1f visibility=%.1f sdf=%.1f Hz",
        pub_hz, float(fct.obs_polygon_viz_hz), float(fct.safe_margin_horizon_viz_hz),
        float(fct.sampled_traj_viz_hz), float(fct.visibility_viz_hz),
        float(fct.sdf_grid_pub_hz),
    )

    while not rospy.is_shutdown():
        fct.publish_all_visualizations()
        rate.sleep()


if __name__ == "__main__":
    main()
