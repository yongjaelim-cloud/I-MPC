#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 tracker for the SELECTED RCMS (default) or Parseh trajectory.

Default: tracks the trajectory chosen by MPC_rcms_selector.py, i.e.
/mpc_rcms_control_seq (winner of /mpc_rcms_control_seq0..3) with mode
/mpc_rcms_mode (AVOIDANCE or MITIGATION). Run MPC_rcms0..3 + MPC_rcms_selector.

Usage (run only one /cmd_vel controller):
  python3 traj_tracker.py                     # RCMS selector output
  python3 traj_tracker.py _planner:=parseh
  python3 traj_tracker.py _trajectory_topic:=/mpc_rcms_control_seq

Requires rospy, numpy, geometry_msgs, std_msgs, nav_msgs and drone_msgs.
No setup_config, utils_linearized, tf2, or /odom dependency.
Input MPCControlSeq states: xs, ys, vs, yaws, scalar dt or per-stage dts.
All poses and trajectories must be in map (or ~frame_id). Empty trajectory
frame_id is treated as ~frame_id for legacy messages. Pose is base_link's pose;
apply any sensor-to-base extrinsic upstream if /tracked_pose is a sensor pose.

Reference epoch is reception time plus a bounded spatial projection onto the
first ~align_max_s of the trajectory. Selector header stamps do not consistently
represent solve start, so are NOT used as the trajectory epoch. Per-stage yaw
feedforward comes from the yaw differences. Reverse speeds are supported.
The latest buffered pose is reused at 100 Hz, without pose-age/jump gating.
Missing/invalid/expired trajectory, FAILED mode and shutdown command zero.
Timeouts use ROS time, so /use_sim_time follows rosbag playback.

Outputs: /cmd_vel, /selected_mpc_traj [N+1,4] (x,y,v,yaw),
         /traj_tracker/selected_path (nav_msgs/Path).
Do not run ego_mover_tracker or another /cmd_vel publisher simultaneously.
Motion limits below are ROS params; match them to your robot/planner.
"""
import math
import threading

import numpy as np
import rospy
from drone_msgs.msg import MPCControlSeq
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def decode_trajectory(msg, default_dt):
    """Reject malformed selected trajectories; never reselect worker candidates."""
    fields = [np.asarray(getattr(msg, name), dtype=float)
              for name in ('xs', 'ys', 'vs', 'yaws')]
    n = len(fields[0])
    if n < 2 or any(a.ndim != 1 or len(a) != n for a in fields):
        raise ValueError('xs/ys/vs/yaws must have equal length >= 2')
    states = np.column_stack(fields)
    if not np.isfinite(states).all():
        raise ValueError('non-finite trajectory state')
    intervals = None
    for name in ('time_intervals', 'time_interval', 'dts', 'dt_seq', 'dt_list', 'dt'):
        value = getattr(msg, name, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size:
            intervals = arr
            break
    if intervals is None:
        intervals = np.array([default_dt], dtype=float)
    if len(intervals) == 1:
        intervals = np.full(n - 1, intervals[0])
    if len(intervals) != n - 1 or not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError('dt must be positive scalar or N per-stage intervals')
    return states, np.r_[0., np.cumsum(intervals)]


def project_time(states, edges, pose, max_time):
    if pose is None or max_time <= 0:
        return 0.
    best_dist, best_t = float('inf'), 0.
    p = np.asarray(pose[:2])
    for i in range(len(states) - 1):
        if edges[i] >= max_time:
            break
        delta = states[i + 1, :2] - states[i, :2]
        norm = float(np.dot(delta, delta))
        upper = min(1., (max_time - edges[i]) / (edges[i + 1] - edges[i]))
        u = 0. if norm < 1.e-12 else float(np.clip(
            np.dot(p - states[i, :2], delta) / norm, 0., upper))
        dist = float(np.sum((states[i, :2] + u * delta - p) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_t = float(edges[i] + u * (edges[i + 1] - edges[i]))
    return best_t


def sample_reference(states, edges, elapsed):
    if elapsed < 0 or elapsed >= edges[-1]:
        return None
    i = max(0, min(int(np.searchsorted(edges, elapsed, side='right')) - 1,
                   len(states) - 2))
    dt = float(edges[i + 1] - edges[i])
    alpha = (elapsed - edges[i]) / dt
    a, b = states[i], states[i + 1]
    ref = (1. - alpha) * a + alpha * b
    dyaw = wrap(float(b[3] - a[3]))
    ref[3] = wrap(float(a[3]) + alpha * dyaw)
    return ref, dyaw / dt


class TrajectoryTracker:
    def __init__(self):
        self.lock = threading.RLock()
        self.pose = None
        self.speed = 0.  # latest /cmd_vel.linear.x; not measured wheel speed
        self.reference = None
        planner = str(rospy.get_param('~planner', 'rcms')).lower()
        if planner not in ('parseh', 'rcms'):
            raise ValueError('~planner must be parseh or rcms')
        self.frame = str(rospy.get_param('~frame_id', 'map')).lstrip('/')
        self.hz = self.param('hz', 100., positive=True)
        self.default_dt = self.param('default_dt', .2, positive=True)
        self.timeout = self.param('trajectory_timeout_s', 1., positive=True)
        self.align_max = self.param('align_max_s', .3)
        self.kx = self.param('tracking_omega_lon', .8, positive=True)
        self.omega_lat = self.param('tracking_omega_lat', .6, positive=True)
        self.v_soft = self.param('tracking_v_soft', .2, positive=True)
        self.v_min, self.v_max = self.param('v_min', -.2), self.param('v_max', .8)
        self.w_min, self.w_max = self.param('yawrate_min', -1.3), self.param('yawrate_max', 1.3)
        self.a_min, self.a_max = self.param('a_min', -.6), self.param('a_max', .6)
        if not (self.v_min <= 0 <= self.v_max and self.w_min <= 0 <= self.w_max
                and self.a_min < 0 < self.a_max):
            raise ValueError('velocity/yaw limits must include zero; a_min < 0 < a_max')
        self.planner = planner
        self.topic = str(rospy.get_param('~trajectory_topic', '/mpc_%s_control_seq' % planner))
        mode_topic = str(rospy.get_param('~mode_topic', '/mpc_%s_mode' % planner))
        cmd_topic = str(rospy.get_param('~cmd_vel_topic', '/cmd_vel'))
        self.pub_cmd = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self.pub_array = rospy.Publisher('/selected_mpc_traj', Float64MultiArray, queue_size=1)
        self.pub_path = rospy.Publisher('/traj_tracker/selected_path', Path, queue_size=1)
        self.subs = [
            rospy.Subscriber(str(rospy.get_param('~pose_topic', '/tracked_pose')),
                             PoseStamped, self.pose_cb, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber(cmd_topic, Twist, self.speed_cb, queue_size=1),
            rospy.Subscriber(self.topic, MPCControlSeq, self.trajectory_cb, queue_size=1,
                             tcp_nodelay=True),
            rospy.Subscriber(mode_topic, String, self.mode_cb, queue_size=1),
        ]
        rospy.on_shutdown(self.stop)
        rospy.loginfo('[traj_tracker] planner=%s selected=%s mode=%s frame=%s hz=%.1f',
                      planner, self.topic, mode_topic, self.frame, self.hz)

    @staticmethod
    def param(name, default, positive=False):
        value = float(rospy.get_param('~' + name, default))
        if not math.isfinite(value) or (positive and value <= 0):
            raise ValueError('invalid ~' + name)
        return value

    def pose_cb(self, msg):
        if msg.header.frame_id.lstrip('/') != self.frame:
            rospy.logwarn_throttle(2., '[traj_tracker] pose frame must be %s', self.frame)
            return
        q = msg.pose.orientation
        values = [msg.pose.position.x, msg.pose.position.y, q.x, q.y, q.z, q.w]
        if not all(math.isfinite(v) for v in values):
            return
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1.e-9:
            return
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        with self.lock:
            self.pose = (float(values[0]), float(values[1]), yaw)

    def speed_cb(self, msg):
        if math.isfinite(msg.linear.x):
            with self.lock:
                self.speed = float(msg.linear.x)

    def mode_cb(self, msg):
        # These selectors publish the sequence before its mode notification.
        # Successful modes must not clear or gate the just-received trajectory.
        if msg.data.strip().upper() in ('FAILED', 'MITIGATION_PATH_ONLY'):
            with self.lock:
                self.reference = None
                self.stop()

    def trajectory_cb(self, msg):
        try:
            frame = msg.header.frame_id.lstrip('/')
            if frame and frame != self.frame:
                raise ValueError('trajectory frame %s != %s' % (frame, self.frame))
            # RCMS: ModeAwarePublisher sets trajectory_type 1=avoidance,
            # 2=mitigation, 0=failed. The selector never forwards failed ones,
            # but reject them here too in case the topic is remapped.
            ttype = int(getattr(msg, 'trajectory_type', -1))
            if self.planner == 'rcms' and ttype == 0:
                raise ValueError('RCMS trajectory_type=0 (FAILED)')
            states, edges = decode_trajectory(msg, self.default_dt)
            with self.lock:
                offset = project_time(states, edges, self.pose, self.align_max)
                self.reference = (states, edges, rospy.Time.now().to_sec(), offset)
            self.publish_reference(states)
            rospy.loginfo_throttle(1., '[traj_tracker] %s seq=%d type=%d N=%d T=%.2fs',
                                   self.planner, int(getattr(msg, 'seq_id', -1)), ttype,
                                   len(states) - 1, float(edges[-1]))
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            with self.lock:
                self.reference = None
                self.stop()
            rospy.logwarn_throttle(1., '[traj_tracker] rejected selected trajectory: %s', str(exc))

    def publish_reference(self, states):
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label='stage', size=len(states), stride=4*len(states)),
                          MultiArrayDimension(label='state_xyvyaw', size=4, stride=4)]
        msg.data = states.ravel().tolist()
        self.pub_array.publish(msg)
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame
        for x, y, speed, yaw in states:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y = float(x), float(y)
            p.pose.orientation.z = math.sin(float(yaw) / 2.)
            p.pose.orientation.w = math.cos(float(yaw) / 2.)
            path.poses.append(p)
        self.pub_path.publish(path)

    def stop(self):
        self.pub_cmd.publish(Twist())
        with self.lock:
            self.speed = 0.

    def step(self, now, dt):
        # Serialize command publication with trajectory rejection/FAILED callbacks.
        with self.lock:
            if self.pose is None or self.reference is None:
                self.stop()
                return
            states, edges, received, offset = self.reference
            age = now - received
            sampled = sample_reference(states, edges, age + offset)
            if age < 0 or age > self.timeout or sampled is None:
                self.reference = None
                self.stop()
                return
            ref, yaw_ff = sampled
            x, y, yaw = self.pose
            dx, dy = ref[0] - x, ref[1] - y
            ex = math.cos(yaw)*dx + math.sin(yaw)*dy
            ey = -math.sin(yaw)*dx + math.cos(yaw)*dy
            epsi = wrap(float(ref[3]) - yaw)
            # P position feedback + reference speed/yawrate feedforward.
            # Locally critically damped lateral design; signed gain for reverse.
            vref = float(ref[2])
            gain = self.omega_lat**2 / max(abs(vref), abs(self.speed), self.v_soft)
            gain = math.copysign(gain, vref) if abs(vref) > .01 else 0.
            kpsi = 2.*self.omega_lat
            v_target = vref * math.cos(epsi) + self.kx * ex
            w_target = yaw_ff + kpsi*(epsi + math.atan(gain * ey / kpsi))
            base = float(np.clip(self.speed, self.v_min, self.v_max))
            v_cmd = float(np.clip(v_target, max(self.v_min, base + self.a_min*dt),
                                  min(self.v_max, base + self.a_max*dt)))
            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = float(np.clip(w_target, self.w_min, self.w_max))
            self.pub_cmd.publish(cmd)
            self.speed = v_cmd
            rospy.loginfo_throttle(1., '[traj_tracker] ex=%+.3f ey=%+.3f epsi=%+.3f v=%.3f w=%.3f',
                                  ex, ey, epsi, cmd.linear.x, cmd.angular.z)

    def run(self):
        rate = rospy.Rate(self.hz)
        previous = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt = min(.1, max(1.e-4, now - previous))
            previous = now
            self.step(now, dt)
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                with self.lock:
                    self.reference = None
                    self.stop()
                previous = rospy.Time.now().to_sec()
            except rospy.ROSInterruptException:
                break


def main():
    rospy.init_node('traj_tracker', anonymous=False)
    TrajectoryTracker().run()


if __name__ == '__main__':
    main()
