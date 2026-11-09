#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ego_mover_vanilla.py

Same as ego_mover_tracker.py (candidate selection, /selected_traj, /ego_state,
scenario reset/start, goal stop, actuator-lag model) EXCEPT the control law:
NO PID / pose feedback.  The command is generated purely from the selected
trajectory /selected_traj = [x, y, v, yaw] (time-parameterized by the MPC dts):

    v_cmd(t) = v_ref(t)                                (linear interpolation)
    w_cmd(t) = wrap(yaw[i+1] - yaw[i]) / dt[i]         (segment yaw rate)

Both are clipped only to the configured v / yawrate limits.

Usage (same params/topics as ego_mover_tracker.py):
  python3 ego_mover_vanilla.py
  python3 ego_mover_vanilla.py _tracking_align_max_s:=0.0   # pure time playback

~tracking_align_max_s (default 0.5 s, inherited): on each new selected
trajectory, t=0 is shifted to the ego's projection on the path to absorb
planning latency.  This is a one-shot time alignment, not feedback; set 0 to
play the trajectory strictly from reception time.
Do not run together with ego_mover_tracker.py or traj_tracker.py (/cmd_vel).
"""

import time

import numpy as np
import rospy

import ego_mover_tracker as base


class VanillaEgoMoverNode(base.EgoMoverNode):
    """EgoMoverNode with pure feedforward (open-loop) trajectory playback."""

    def compute_tracking_command(self, now, dt):
        sampled = self._reference_at_time(now)
        if sampled is None:
            self.reset_pi_tracker()
            return None
        ref, yaw_ff = sampled[0], sampled[1]

        v_min = float(getattr(self.cfg, "v_min", 0.0))
        v_max = float(getattr(self.cfg, "v_max", 0.6))
        w_min = float(getattr(self.cfg, "yawrate_min", -1.8))
        w_max = float(getattr(self.cfg, "yawrate_max", 1.8))
        v_cmd = float(np.clip(float(ref[2]), v_min, v_max))
        w_cmd = float(np.clip(float(yaw_ff), w_min, w_max))

        self.tracking_prev_u = (v_cmd, w_cmd)
        rospy.loginfo_throttle(
            1.0, "[ego_mover_vanilla] ff v_ref=%.3f w_ref=%+.3f -> v=%.3f w=%+.3f align=%.3fs src=%s",
            float(ref[2]), float(yaw_ff), v_cmd, w_cmd,
            float(getattr(self, "tracking_last_offset", 0.0)),
            str(getattr(self, "tracking_source", "")))
        return v_cmd, w_cmd


def main() -> None:
    rospy.init_node("ego_mover", anonymous=False)
    node = VanillaEgoMoverNode()
    rospy.loginfo("[ego_mover_vanilla] feedforward-only control from selected trajectory (no PID)")
    time.sleep(0.2)
    rate = rospy.Rate(node.hz)
    while not rospy.is_shutdown():
        node.step()
        rate.sleep()


if __name__ == "__main__":
    main()
