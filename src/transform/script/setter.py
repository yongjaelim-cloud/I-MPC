#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of JSON configuration for the ROS stack.

Only this node reads ``mpc_planner_common.json``. The complete typed
configuration is published once on the latched ``/setup`` topic. All other
nodes subscribe to ``/setup`` and configure themselves from that message.
"""

import json
import os
import math
import threading

import rospy
import tf2_ros
import tf2_geometry_msgs  # Registers geometry_msgs transforms for tf2
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker

from drone_msgs.msg import Setup, Ego_goal_setting
from setup_config import raw_to_setup_msg


def _load_json_config():
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpc_planner_common.json")
    path = str(rospy.get_param("~config_path", default_path))
    if not os.path.isfile(path):
        raise IOError("setup JSON not found: %s" % path)
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("setup JSON must contain a top-level object: %s" % path)
    return path, raw


class SetupSetter(object):
    def __init__(self, source_path, raw):
        self.source_path = str(source_path)
        self.raw = raw
        self.setup_topic = "/setup"
        self.goal_topic = str(rospy.get_param("~goal_topic", "/ego_goal_set"))
        # RViz Fixed Frame=map; persist mouse clicks in world coordinates.
        # The frameless Ego_goal_setting used by MPC also uses map coordinates.
        self.goal_frame = "map"
        self.output_frame = "map"  # Frameless planner goals are map coordinates.
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.goal_lock = threading.RLock()
        self.current_goal = None
        self.pending_goal = None  # newest RViz click awaiting an available TF
        config = raw.get("config", {}) if isinstance(raw.get("config"), dict) else {}
        ego = raw.get("ego", {}) if isinstance(raw.get("ego"), dict) else {}
        self.goal_config = config
        self.goal_ego_config = ego
        self.goal_mpc_config = raw.get("mpc", {}) if isinstance(raw.get("mpc"), dict) else {}
        self.default_goal_z = float(ego.get("goal_z", config.get("z_end", 0.0)))
        self.publish_initial_goal = bool(rospy.get_param("~publish_initial_goal", False))
        self.goal_pub = rospy.Publisher(self.goal_topic, Ego_goal_setting, queue_size=1, latch=True)
        self.goal_marker_pub = rospy.Publisher("/ego_goal_marker", Marker, queue_size=1, latch=True)
        self.goal_fixed_pub = rospy.Publisher("/ego_goal_map", PointStamped, queue_size=1, latch=True)

        # latch=True is essential: nodes started after setter still receive setup.
        self.setup_pub = rospy.Publisher(self.setup_topic, Setup, queue_size=1, latch=True)
        self.setup_msg = raw_to_setup_msg(self.raw, self.source_path)
        self.setup_pub.publish(self.setup_msg)

        rospy.loginfo("[setter] published latched %s from %s", self.setup_topic, self.source_path)
        rospy.loginfo("[setter] /setup remains the only configuration output; %s is the interactive goal output", self.goal_topic)
        rospy.loginfo(
            "[setter] /setup seed order acc_order=%s yawrate_order=%s",
            str(list(getattr(self.setup_msg, "mpc_acc_order", []))),
            str(list(getattr(self.setup_msg, "mpc_yawrate_order", []))),
        )
        rospy.loginfo(
            "[setter] parameters theta=%s angle_margin=%s reg_margin=%s nonlinear_dyn_flag=%s delta_o=%s delta_f=%s M_max=%s F_max=%s epsilon_o=%s epsilon_f=%s epsilon_tot=%s N_n=%s lambda_f=lambda_o(auto)=%s lambda_r(config)=%s",
            str(getattr(self.setup_msg, "mpc_theta", "<missing>")),
            str(getattr(self.setup_msg, "mpc_angle_margin", "<missing>")),
            str(getattr(self.setup_msg, "mpc_reg_margin", "<missing>")),
            str(getattr(self.setup_msg, "mpc_nonlinear_dyn_flag", "<missing>")),
            str(getattr(self.setup_msg, "mpc_delta_o", "<missing>")),
            str(getattr(self.setup_msg, "mpc_delta_f", "<missing>")),
            str(getattr(self.setup_msg, "mpc_m_max", "<missing>")),
            str(getattr(self.setup_msg, "mpc_f_max", "<missing>")),
            str(getattr(self.setup_msg, "mpc_epsilon_o", "<missing>")),
            str(getattr(self.setup_msg, "mpc_epsilon_f", "<missing>")),
            str(getattr(self.setup_msg, "mpc_epsilon_tot", "<missing>")),
            str(getattr(self.setup_msg, "normal_cfg_n_n", "<missing>")),
            str(getattr(self.setup_msg, "mpc_lambda_f", "<missing>")),
            str(getattr(self.setup_msg, "mpc_lambda_r", "<missing>")),
        )
        rospy.loginfo(
            "[setter] CUDA proposed enabled=%s min_hf=%s threads=%s fallback_cpu=%s profile=%s compare_cpu=%s profile_every=%s",
            str(getattr(self.setup_msg, "mpc_use_native_cuda_proposed", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_min_hf", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_threads", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_fallback_to_cpu", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_profile", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_compare_cpu", "<missing>")),
            str(getattr(self.setup_msg, "mpc_native_cuda_profile_every", "<missing>")),
        )


        self.nav_goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self._nav_goal_callback, queue_size=1
        )
        self.clicked_point_sub = rospy.Subscriber(
            "/clicked_point", PointStamped, self._clicked_point_callback, queue_size=1
        )
        self.goal_timer = rospy.Timer(rospy.Duration(0.1), self._goal_timer_callback)
        rospy.loginfo("[setter] interactive goal input ready: /move_base_simple/goal, /clicked_point (stored=%s, output=%s)", self.goal_frame, self.output_frame)

        if self.publish_initial_goal:
            if "x_end" in config and "y_end" in config:
                self._publish_goal(float(config["x_end"]), float(config["y_end"]), self.default_goal_z, "initial_config")
            else:
                rospy.logwarn("[setter] ~publish_initial_goal=true but config.x_end/y_end missing")

    def _goal_in_map(self, point, source):
        """Anchor an RViz click to map using the newest map <- input TF.

        RViz fixed frame is map. RViz wall-time stamps must not be
        used to query transforms played from a rosbag with older timestamps.
        """
        frame = str(point.header.frame_id or "").lstrip("/")
        if not frame:
            rospy.logwarn("[setter] ignored %s: missing frame_id", source)
            return None
        if frame == self.goal_frame:
            return point
        try:
            transform = self.tf_buffer.lookup_transform(
                self.goal_frame, frame, rospy.Time(0), rospy.Duration(0.1))
            # Apply exactly the TF returned by the lookup (not RViz's stamp).
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as exc:
            rospy.logwarn_throttle(2.0, "[setter] awaiting map <- %s TF for %s: %s" % (frame, source, exc))
            return None

    def _receive_goal(self, point, source):
        """Keep the latest click even when rosbag TF is not ready yet."""
        with self.goal_lock:
            self.pending_goal = (point, source)
        self._process_pending_goal()

    def _process_pending_goal(self):
        with self.goal_lock:
            pending = self.pending_goal
        if pending is None:
            return
        point, source = pending
        fixed = self._goal_in_map(point, source)
        if fixed is None:
            return
        with self.goal_lock:
            # A newer click may have arrived during the TF lookup.
            if self.pending_goal is not pending:
                return
            self.pending_goal = None
        self._publish_goal(fixed.point.x, fixed.point.y, fixed.point.z, source)

    def _publish_goal(self, x, y, z, source):
        """Store and publish the stationary goal in map."""
        if not all(math.isfinite(float(v)) for v in (x, y, z)):
            rospy.logwarn("[setter] ignored nonfinite goal from %s", source)
            return
        with self.goal_lock:
            self.current_goal = (float(x), float(y), float(z))
        rospy.loginfo("[setter] fixed map goal from %s: (%.3f, %.3f, %.3f)", source, x, y, z)
        self._publish_fixed_goal()
        self._publish_current_goal()

    def _publish_fixed_goal(self):
        with self.goal_lock:
            goal = self.current_goal
        if goal is None:
            return
        point = PointStamped()
        point.header.frame_id = self.goal_frame
        point.header.stamp = rospy.Time(0)
        point.point.x, point.point.y, point.point.z = goal
        self.goal_fixed_pub.publish(point)  # Always map, stationary in RViz.

        marker = Marker()
        marker.header.frame_id = self.goal_frame
        marker.header.stamp = rospy.Time(0)
        marker.ns = "fixed_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal[0])
        marker.pose.position.y = float(goal[1])
        marker.pose.position.z = float(goal[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        self.goal_marker_pub.publish(marker)

    def _publish_current_goal(self):
        with self.goal_lock:
            goal = self.current_goal
        if goal is None:
            return
        p = PointStamped()
        p.header.frame_id = self.goal_frame
        # Latest TF: transform a fixed world point using the current robot pose.
        p.header.stamp = rospy.Time(0)
        p.point.x, p.point.y, p.point.z = goal
        if self.output_frame == self.goal_frame:
            local = p
        else:
            return  # Both frames are map by contract.
        msg = Ego_goal_setting()
        msg.x_end = float(local.point.x)
        msg.y_end = float(local.point.y)
        msg.z_end = float(local.point.z)
        self.goal_pub.publish(msg)

    def _goal_timer_callback(self, _event):
        self._process_pending_goal()
        # Fixed map goal is latched and only needs publishing when it changes.
        self._publish_current_goal()

    def _nav_goal_callback(self, msg):
        point = PointStamped()
        point.header = msg.header
        point.point = msg.pose.position
        self._receive_goal(point, "2D Nav Goal")

    def _clicked_point_callback(self, msg):
        self._receive_goal(msg, "Publish Point")


def main():
    rospy.init_node("setup_setter", anonymous=False)
    source_path, raw = _load_json_config()
    SetupSetter(source_path, raw)

    # Keep the latched publisher alive so late-starting nodes can receive /setup.
    rospy.spin()


if __name__ == "__main__":
    main()
