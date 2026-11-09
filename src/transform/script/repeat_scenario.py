#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeated two-goal lap scenario with runtime sector switching.

Lap definition
  GOAL0 -> GOAL1 -> GOAL0 is one lap. The robot first drives to GOAL0
  (approach, not counted); every lap then starts and ends at GOAL0.
  A goal is reached when the /tracked_pose position is within ~reach_dist
  (default 0.3 m) of it, and the next goal is published immediately.

Sector schedule
  For each sector_id in ~sectors (default [2, 1, 3, 4]) the scenario runs
  ~laps_per_sector laps (default 7).  ~laps_per_sector is either one integer
  applied to every sector, or a list with one count per ~sectors entry
  (e.g. _laps_per_sector:="[7,5,3,7]").  The sector is switched while the robot is
  at GOAL0, by republishing setter.py's latched /setup message with only
  config_sector_id changed.

Sector verification (before laps start in a new sector)
  1) every planner in ~planner_ids must publish /mpc_sector_ack<i> == sector_id
     AFTER the /setup republish (the planner ACKs only after cfg.sector_id and
     sector_rays_body -- the n(phi_c) passed to every OCP solve -- are updated);
  2) the planner log "runtime /setup sector update: a -> b, n(phi_c)_body=[x, y]"
     on /rosout must match l_c computed here independently from
     ego_L/ego_W/sector_dir (skipped when the sector did not change).
  If verification fails and ~abort_on_verify_fail is true, the scenario stops.

Goals are sent like goal_pub.py: PoseStamped on /move_base_simple/goal, which
setter.py echoes on /ego_goal_set (the planner goal input).

Run:  python3 repeat_scenario.py   (setter.py and MPC planners running)
      e.g. python3 repeat_scenario.py _laps_per_sector:=7 _sectors:="[2,1,3,4]"
           python3 repeat_scenario.py _sectors:="[2,1,3,4]" _laps_per_sector:="[7,5,3,7]"
"""

import copy
import csv
import json
import math
import os
import re
import threading
import time

import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Log
from std_msgs.msg import Int32, String

from drone_msgs.msg import Setup, Ego_goal_setting
from setup_config import setup_msg_to_raw

DEFAULT_GOALS = [[11.83, -4.338], [2.103, 2.158]]
SECTOR_NAMES = {1: "left", 2: "rear", 3: "right", 4: "front"}
SECTOR_LOG_RE = re.compile(
    r"\[(MPC_planner\d+)\] runtime /setup sector update: (\d+) -> (\d+), "
    r"n\(phi_c\)_body=\[([-+0-9.eE]+), ([-+0-9.eE]+)\]"
)


def _param_list(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        value = json.loads(value)
    return list(value)


def expected_lc_body(ego_L, ego_W, sector_id, sector_dir="outward"):
    """Independent copy of utils_linearized.sector_rays_body_from_id()[2]."""
    L, W = float(ego_L), float(ego_W)
    fl, fr = (L / 2.0, W / 2.0), (L / 2.0, -W / 2.0)
    rr, rl = (-L / 2.0, -W / 2.0), (-L / 2.0, W / 2.0)
    a, b = {1: (fl, rl), 2: (rl, rr), 3: (rr, fr), 4: (fr, fl)}[int(sector_id)]
    na, nb = math.hypot(*a), math.hypot(*b)
    c = (a[0] / na + b[0] / nb, a[1] / na + b[1] / nb)
    nc = max(math.hypot(*c), 1e-12)
    c = (c[0] / nc, c[1] / nc)
    if str(sector_dir).strip().lower() == "inward":
        c = (-c[0], -c[1])
    return c


class RepeatScenario(object):
    def __init__(self):
        self.frame = "map"
        self.goals = [tuple(map(float, g)) for g in _param_list("~goals", DEFAULT_GOALS)]
        if len(self.goals) != 2:
            raise ValueError("~goals must contain exactly two [x, y] goals")
        self.sectors = [int(s) for s in _param_list("~sectors", [2, 1, 3, 4])]
        if any(s not in SECTOR_NAMES for s in self.sectors):
            raise ValueError("~sectors entries must be 1..4")
        laps = rospy.get_param("~laps_per_sector", 7)
        if isinstance(laps, str):
            laps = json.loads(laps)
        if isinstance(laps, (list, tuple)):
            if len(laps) != len(self.sectors):
                raise ValueError("~laps_per_sector list length (%d) must match ~sectors length (%d)"
                                 % (len(laps), len(self.sectors)))
            self.laps_per_sector = [int(n) for n in laps]
        else:
            self.laps_per_sector = [int(laps)] * len(self.sectors)
        if any(n < 0 for n in self.laps_per_sector):
            raise ValueError("~laps_per_sector entries must be >= 0")
        self.reach_dist = float(rospy.get_param("~reach_dist", 0.3))
        self.planner_ids = [int(i) for i in _param_list("~planner_ids", [0, 1, 2, 3])]
        self.ack_timeout = float(rospy.get_param("~ack_timeout", 10.0))
        self.log_timeout = float(rospy.get_param("~log_timeout", 3.0))
        self.confirm_timeout = float(rospy.get_param("~confirm_timeout", 1.0))
        self.settle_time = float(rospy.get_param("~settle_time", 1.0))
        self.abort_on_verify_fail = bool(rospy.get_param("~abort_on_verify_fail", True))
        self.goal_z = float(rospy.get_param("~goal_z", 0.0))
        self.pose_topic = str(rospy.get_param("~pose_topic", "/tracked_pose"))
        self.goal_out_topic = str(rospy.get_param("~goal_out_topic", "/move_base_simple/goal"))
        self.ego_goal_topic = str(rospy.get_param("~ego_goal_topic", "/ego_goal_set"))
        default_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "repeat_scenario_logs",
            time.strftime("repeat_scenario_%Y%m%d_%H%M%S.csv"))
        self.csv_path = str(rospy.get_param("~csv_path", default_csv))

        self.lock = threading.RLock()
        self.ego = None
        self.active_goal = None          # goal echoed by setter on /ego_goal_set
        self.acks = {}                   # planner_id -> (sector_id, wall time)
        self.sector_logs = {}            # planner name -> (old, new, lc, wall time)

        rospy.loginfo("[repeat_scenario] waiting for latched /setup from setter.py ...")
        self.base_setup = rospy.wait_for_message("/setup", Setup)
        if not hasattr(self.base_setup, "config_sector_id"):
            raise RuntimeError("Setup.msg has no config_sector_id field")
        raw = setup_msg_to_raw(self.base_setup)
        self.ego_L = float(raw.get("ego", {}).get("ego_L"))
        self.ego_W = float(raw.get("ego", {}).get("ego_W"))
        self.sector_dir = str(raw.get("config", {}).get("sector_dir", "outward"))
        self.initial_sector = int(self.base_setup.config_sector_id)

        self.setup_pub = rospy.Publisher("/setup", Setup, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher(self.goal_out_topic, PoseStamped, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("/repeat_scenario/status", String, queue_size=10, latch=True)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.ego_goal_topic, Ego_goal_setting, self._ego_goal_cb, queue_size=1)
        # /rosout_agg drops planner lines under cartographer log load; read /rosout directly.
        rospy.Subscriber("/rosout", Log, self._rosout_cb, queue_size=1000)
        for pid in self.planner_ids:
            rospy.Subscriber("/mpc_sector_ack%d" % pid, Int32,
                             lambda m, p=pid: self._ack_cb(p, m), queue_size=5)

        self._warn_other_goal_publishers()
        self._open_csv()
        rospy.loginfo(
            "[repeat_scenario] goals G0=%s G1=%s reach=%.2f m sectors=%s laps/sector=%s "
            "planners=%s ego L/W=%.3f/%.3f sector_dir=%s current sector=%d csv=%s",
            self.goals[0], self.goals[1], self.reach_dist, self.sectors, self.laps_per_sector,
            self.planner_ids, self.ego_L, self.ego_W, self.sector_dir, self.initial_sector,
            self.csv_path)

    # ------------------------------------------------------------ callbacks
    def _pose_cb(self, msg):
        if str(msg.header.frame_id).lstrip("/") != self.frame:
            rospy.logwarn_throttle(2.0, "[repeat_scenario] %s frame=%s, expected %s"
                                   % (self.pose_topic, msg.header.frame_id, self.frame))
            return
        x, y = float(msg.pose.position.x), float(msg.pose.position.y)
        if math.isfinite(x) and math.isfinite(y):
            with self.lock:
                self.ego = (x, y)

    def _ego_goal_cb(self, msg):
        with self.lock:
            self.active_goal = (float(msg.x_end), float(msg.y_end))

    def _ack_cb(self, pid, msg):
        with self.lock:
            self.acks[pid] = (int(msg.data), time.time())

    def _rosout_cb(self, msg):
        m = SECTOR_LOG_RE.search(msg.msg)
        if m is None:
            return
        with self.lock:
            self.sector_logs[m.group(1)] = (
                int(m.group(2)), int(m.group(3)),
                (float(m.group(4)), float(m.group(5))), time.time())

    # ------------------------------------------------------------ helpers
    def _warn_other_goal_publishers(self):
        try:
            state = rosgraph.Master(rospy.get_name()).getSystemState()
            pubs = [n for t, nodes in state[0] if t == self.goal_out_topic for n in nodes]
            others = [n for n in pubs if n != rospy.get_name()]
            if others:
                rospy.logwarn("[repeat_scenario] other nodes also publish %s: %s -- stop them "
                              "(e.g. goal_pub.py) or they will override the scenario goals",
                              self.goal_out_topic, others)
        except Exception as exc:
            rospy.logwarn("[repeat_scenario] could not query master: %s", exc)

    def _open_csv(self):
        d = os.path.dirname(self.csv_path)
        if d:
            if os.path.exists(d) and not os.path.isdir(d):
                raise IOError("CSV directory path exists but is not a directory: %s "
                              "(set _csv_path:=...)" % d)
            if not os.path.isdir(d):
                os.makedirs(d)
        self.csv_file = open(self.csv_path, "w")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["wall_time", "ros_time", "event", "sector_id", "lap",
                           "goal_idx", "ego_x", "ego_y", "detail"])
        self.csv_file.flush()

    def _event(self, event, sector, lap, goal_idx, detail=""):
        with self.lock:
            ego = self.ego
        ex, ey = (ego if ego is not None else (float("nan"), float("nan")))
        self.csv.writerow(["%.3f" % time.time(), "%.3f" % rospy.get_time(), event, sector,
                           lap, goal_idx, "%.3f" % ex, "%.3f" % ey, detail])
        self.csv_file.flush()
        text = "%s sector=%s lap=%s goal=%s %s" % (event, sector, lap, goal_idx, detail)
        self.status_pub.publish(String(data=text))
        rospy.loginfo("[repeat_scenario] %s", text)

    def _send_goal(self, idx):
        g = self.goals[idx]
        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = g[0], g[1], self.goal_z
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _goal_confirmed(self, idx):
        g = self.goals[idx]
        with self.lock:
            a = self.active_goal
        return a is not None and math.hypot(a[0] - g[0], a[1] - g[1]) < 1e-3

    def _drive_to(self, idx, sector, lap):
        """Publish goal idx and block until the ego is within reach_dist."""
        self._send_goal(idx)
        sent = time.time()
        self._event("goal_sent", sector, lap, idx, "(%.3f, %.3f)" % self.goals[idx])
        rate = rospy.Rate(20)
        t0 = rospy.get_time()
        while not rospy.is_shutdown():
            if not self._goal_confirmed(idx):
                if time.time() - sent > self.confirm_timeout:
                    rospy.logwarn_throttle(2.0, "[repeat_scenario] setter has not echoed goal %d; resending" % idx)
                    self._send_goal(idx)
                    sent = time.time()
            else:
                with self.lock:
                    ego = self.ego
                if ego is not None:
                    d = math.hypot(ego[0] - self.goals[idx][0], ego[1] - self.goals[idx][1])
                    if d <= self.reach_dist:
                        self._event("goal_reached", sector, lap, idx,
                                    "dist=%.3f elapsed=%.2fs" % (d, rospy.get_time() - t0))
                        return True
            rate.sleep()
        return False

    def _switch_sector(self, sector):
        """Republish /setup with config_sector_id=sector and verify planner application."""
        with self.lock:
            prev_acks = {p: v[0] for p, v in self.acks.items()}
        msg = copy.deepcopy(self.base_setup)
        msg.config_sector_id = int(sector)
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
        t_pub = time.time()
        self.setup_pub.publish(msg)
        self._event("setup_published", sector, 0, -1,
                    "sector %d (%s), previous acks=%s" % (sector, SECTOR_NAMES[sector], prev_acks))

        # 1) ACK from every planner received after the republish.
        deadline = t_pub + self.ack_timeout
        missing = list(self.planner_ids)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                missing = [p for p in self.planner_ids
                           if not (p in self.acks and self.acks[p][0] == sector and self.acks[p][1] >= t_pub)]
            if not missing:
                break
            time.sleep(0.05)
        ok = not missing
        details = ["ack_missing=%s" % missing] if missing else ["ack_ok=%s" % self.planner_ids]

        # 2) n(phi_c) logged by each planner matches the expected sector center ray.
        lc_exp = expected_lc_body(self.ego_L, self.ego_W, sector, self.sector_dir)
        changed = [p for p in self.planner_ids if prev_acks.get(p) != sector]
        deadline = time.time() + self.log_timeout
        pending = list(changed)
        while pending and not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                logs = dict(self.sector_logs)
            for p in list(pending):
                entry = logs.get("MPC_planner%d" % p)
                if entry is not None and entry[3] >= t_pub and entry[1] == sector:
                    pending.remove(p)
            time.sleep(0.05)
        with self.lock:
            logs = dict(self.sector_logs)
        for p in changed:
            entry = logs.get("MPC_planner%d" % p)
            if entry is None or entry[3] < t_pub or entry[1] != sector:
                ok = False
                details.append("P%d:no_sector_log" % p)
                continue
            lc = entry[2]
            err = math.hypot(lc[0] - lc_exp[0], lc[1] - lc_exp[1])
            if err > 1e-4:
                ok = False
            details.append("P%d:%d->%d lc=[%.4f,%.4f] err=%.1e" % (p, entry[0], entry[1], lc[0], lc[1], err))
        details.append("lc_expected=[%.4f,%.4f]" % lc_exp)
        self._event("sector_verified" if ok else "sector_verify_FAILED", sector, 0, -1, " ".join(details))
        if not ok:
            (rospy.logerr if self.abort_on_verify_fail else rospy.logwarn)(
                "[repeat_scenario] sector %d was NOT confirmed in all planner OCPs", sector)
        return ok

    # ------------------------------------------------------------ main loop
    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            with self.lock:
                ego = self.ego
            if ego is not None:
                break
            rospy.logwarn_throttle(5.0, "[repeat_scenario] waiting for %s" % self.pose_topic)
            rate.sleep()
        if rospy.is_shutdown():
            return

        self._event("scenario_start", self.initial_sector, 0, 0, "approach G0")
        if not self._drive_to(0, self.initial_sector, 0):
            return
        total = 0
        for sector, n_laps in zip(self.sectors, self.laps_per_sector):
            if not self._switch_sector(sector) and self.abort_on_verify_fail:
                self._event("scenario_aborted", sector, 0, -1, "sector verification failed")
                return
            if self.settle_time > 0.0:
                rospy.sleep(self.settle_time)
            for lap in range(1, n_laps + 1):
                t_lap = rospy.get_time()
                self._event("lap_start", sector, lap, 1, "")
                if not self._drive_to(1, sector, lap) or not self._drive_to(0, sector, lap):
                    return
                total += 1
                self._event("lap_done", sector, lap, 0,
                            "lap_time=%.2fs total_laps=%d" % (rospy.get_time() - t_lap, total))
        self._event("scenario_done", self.sectors[-1], 0, 0, "total_laps=%d" % total)

    def close(self):
        try:
            self.csv_file.close()
        except Exception:
            pass


def main():
    rospy.init_node("repeat_scenario", anonymous=False)
    node = RepeatScenario()
    try:
        node.run()
    finally:
        node.close()


if __name__ == "__main__":
    main()
