#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random goal generator compatible with setter.py.

Publishes map-frame goals as PoseStamped on /move_base_simple/goal, the same
input as RViz "2D Nav Goal". setter.py stores the goal and republishes it as
Ego_goal_setting on /ego_goal_set, so the planner path is unchanged.

Behaviour
  * Goals are sampled uniformly inside the rectangle (convex quad) given by
    ~rect_corners in the map frame, shrunk by ~rect_margin.
  * When the ego (/tracked_pose) comes within ~reach_dist (0.5 m) of the
    current goal, a new goal is sampled within ~max_goal_dist (5 m) of the ego,
    inside the rectangle and in free space.
  * Free space = OccupancyGrid (~map_topic) cells within ~clearance are free,
    and no obstacle point from ~cloud_topic is within ~clearance.
  * The active goal is read back from /ego_goal_set, so a manual RViz click
    (via setter.py) becomes the goal that must be reached next.

Run:  python3 goal_pub.py   (after setter.py)
"""

import json
import math
import random
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from drone_msgs.msg import Ego_goal_setting

DEFAULT_CORNERS = [
    [4.64849853515625, 4.78348970413208],
    [0.36292552947998047, 1.2759346961975098],
    [8.812450408935547, -8.220586776733398],
    [12.06702995300293, -2.409409999847412],
]


def _param_list(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        value = json.loads(value)
    return value


def _order_ccw(corners):
    c = np.asarray(corners, dtype=float)
    center = c.mean(axis=0)
    ang = np.arctan2(c[:, 1] - center[1], c[:, 0] - center[0])
    return c[np.argsort(ang)]


class RandomGoalPublisher(object):
    def __init__(self):
        self.frame = "map"
        self.out_topic = str(rospy.get_param("~goal_out_topic", "/move_base_simple/goal"))
        self.ego_goal_topic = str(rospy.get_param("~ego_goal_topic", "/ego_goal_set"))
        self.pose_topic = str(rospy.get_param("~pose_topic", "/tracked_pose"))
        self.map_topic = str(rospy.get_param("~map_topic", "/map"))
        self.cloud_topic = str(rospy.get_param("~cloud_topic", "/velodyne_points_interest_map"))

        self.reach_dist = float(rospy.get_param("~reach_dist", 0.5))
        self.max_goal_dist = float(rospy.get_param("~max_goal_dist", 5.0))
        self.min_goal_dist = float(rospy.get_param("~min_goal_dist", 1.5))
        self.clearance = float(rospy.get_param("~clearance", 0.5))
        self.rect_margin = float(rospy.get_param("~rect_margin", 0.3))
        self.goal_z = float(rospy.get_param("~goal_z", 0.0))
        self.occupied_thresh = int(rospy.get_param("~occupied_thresh", 50))
        self.unknown_is_free = bool(rospy.get_param("~unknown_is_free", False))
        self.cloud_z_min = float(rospy.get_param("~cloud_z_min", -1e9))
        self.cloud_z_max = float(rospy.get_param("~cloud_z_max", 1e9))
        self.cloud_timeout = float(rospy.get_param("~cloud_timeout", 1.0))
        self.require_free_check = bool(rospy.get_param("~require_free_check", True))
        self.max_samples = int(rospy.get_param("~max_samples", 3000))
        self.confirm_timeout = float(rospy.get_param("~confirm_timeout", 1.0))
        seed = rospy.get_param("~seed", None)
        self.rng = random.Random(seed)

        corners = _param_list("~rect_corners", DEFAULT_CORNERS)
        if len(corners) < 3:
            raise ValueError("~rect_corners needs at least 3 points")
        self.poly = _order_ccw(corners)
        self.poly_min = self.poly.min(axis=0)
        self.poly_max = self.poly.max(axis=0)

        self.lock = threading.RLock()
        self.ego = None            # (x, y)
        self.goal = None           # active goal (x, y), confirmed by setter
        self.sent_goal = None      # last goal we sent, awaiting confirmation
        self.sent_time = None
        self.grid = None           # (data2d, res, ox, oy, w, h)
        self.cloud_xy = None
        self.cloud_stamp = None

        self.pub = rospy.Publisher(self.out_topic, PoseStamped, queue_size=1, latch=True)
        self.rect_pub = rospy.Publisher("/goal_pub/rectangle", Marker, queue_size=1, latch=True)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.ego_goal_topic, Ego_goal_setting, self._ego_goal_cb, queue_size=1)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_cb, queue_size=1)
        if self.cloud_topic:
            rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb,
                             queue_size=1, buff_size=2 ** 24)
        self._publish_rect_marker()
        self.timer = rospy.Timer(rospy.Duration(0.1), self._tick)
        rospy.loginfo("[goal_pub] rect=%s reach=%.2f max_dist=%.2f clearance=%.2f -> %s",
                      self.poly.round(3).tolist(), self.reach_dist, self.max_goal_dist,
                      self.clearance, self.out_topic)

    # ------------------------------------------------------------ callbacks
    def _pose_cb(self, msg):
        if str(msg.header.frame_id).lstrip("/") != self.frame:
            rospy.logwarn_throttle(2.0, "[goal_pub] %s frame=%s, expected %s",
                                   self.pose_topic, msg.header.frame_id, self.frame)
            return
        x, y = float(msg.pose.position.x), float(msg.pose.position.y)
        if math.isfinite(x) and math.isfinite(y):
            with self.lock:
                self.ego = (x, y)

    def _ego_goal_cb(self, msg):
        g = (float(msg.x_end), float(msg.y_end))
        with self.lock:
            if self.goal is None or math.hypot(g[0] - self.goal[0], g[1] - self.goal[1]) > 1e-3:
                rospy.loginfo("[goal_pub] active goal from setter: (%.3f, %.3f)", g[0], g[1])
            self.goal = g
            if self.sent_goal is not None and \
                    math.hypot(g[0] - self.sent_goal[0], g[1] - self.sent_goal[1]) < 1e-3:
                self.sent_goal = None

    def _map_cb(self, msg):
        if str(msg.header.frame_id).lstrip("/") != self.frame:
            rospy.logwarn_throttle(5.0, "[goal_pub] map frame=%s, expected %s",
                                   msg.header.frame_id, self.frame)
            return
        q = msg.info.origin.orientation
        if abs(q.z) > 1e-3 or abs(q.x) > 1e-3 or abs(q.y) > 1e-3:
            rospy.logwarn_throttle(5.0, "[goal_pub] rotated map origin not supported")
            return
        w, h = msg.info.width, msg.info.height
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        with self.lock:
            self.grid = (data, float(msg.info.resolution),
                         float(msg.info.origin.position.x), float(msg.info.origin.position.y), w, h)

    def _cloud_cb(self, msg):
        if str(msg.header.frame_id).lstrip("/") != self.frame:
            rospy.logwarn_throttle(5.0, "[goal_pub] cloud frame=%s, expected %s",
                                   msg.header.frame_id, self.frame)
            return
        off = {f.name: f.offset for f in msg.fields}
        if not all(k in off for k in ("x", "y", "z")) or msg.width * msg.height == 0:
            with self.lock:
                self.cloud_xy = np.zeros((0, 2))
                self.cloud_stamp = rospy.get_time()
            return
        n = msg.width * msg.height
        dt = np.dtype({"names": ["x", "y", "z"],
                       "formats": ["<f4"] * 3 if not msg.is_bigendian else [">f4"] * 3,
                       "offsets": [off["x"], off["y"], off["z"]],
                       "itemsize": msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dt, count=n)
        x, y, z = arr["x"].astype(float), arr["y"].astype(float), arr["z"].astype(float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) \
            & (z >= self.cloud_z_min) & (z <= self.cloud_z_max)
        xy = np.stack([x[ok], y[ok]], axis=1)
        if len(xy):  # 5 cm voxel downsample
            _, idx = np.unique(np.round(xy / 0.05).astype(np.int64), axis=0, return_index=True)
            xy = xy[idx]
        with self.lock:
            self.cloud_xy = xy
            self.cloud_stamp = rospy.get_time()

    # ------------------------------------------------------------ geometry
    def _inside_rect(self, x, y):
        p = self.poly
        n = len(p)
        for i in range(n):
            ax, ay = p[i]
            bx, by = p[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            L = math.hypot(ex, ey)
            # CCW polygon: interior is on the left; signed distance must be >= margin
            if (ex * (y - ay) - ey * (x - ax)) / L < self.rect_margin:
                return False
        return True

    def _free(self, x, y):
        with self.lock:
            grid = self.grid
            cloud = self.cloud_xy
            cloud_ok = self.cloud_stamp is not None and \
                (rospy.get_time() - self.cloud_stamp) <= self.cloud_timeout
        checked = False
        if grid is not None:
            data, res, ox, oy, w, h = grid
            r = int(math.ceil(self.clearance / res))
            ci = int(math.floor((x - ox) / res))
            cj = int(math.floor((y - oy) / res))
            if ci - r < 0 or cj - r < 0 or ci + r >= w or cj + r >= h:
                if not self.unknown_is_free:
                    return False
            else:
                patch = data[cj - r:cj + r + 1, ci - r:ci + r + 1]
                jj, ii = np.mgrid[-r:r + 1, -r:r + 1]
                disk = (ii * ii + jj * jj) * res * res <= self.clearance ** 2
                vals = patch[disk]
                if np.any(vals >= self.occupied_thresh):
                    return False
                if not self.unknown_is_free and np.any(vals < 0):
                    return False
                checked = True
        if cloud_ok and cloud is not None:
            if len(cloud):
                d2 = (cloud[:, 0] - x) ** 2 + (cloud[:, 1] - y) ** 2
                if float(d2.min()) < self.clearance ** 2:
                    return False
            checked = True
        return checked or not self.require_free_check

    def _sample_goal(self, ego, max_dist):
        ex, ey = ego
        lo = np.maximum(self.poly_min, [ex - max_dist, ey - max_dist])
        hi = np.minimum(self.poly_max, [ex + max_dist, ey + max_dist])
        if np.any(lo >= hi):
            return None
        for _ in range(self.max_samples):
            x = self.rng.uniform(lo[0], hi[0])
            y = self.rng.uniform(lo[1], hi[1])
            d = math.hypot(x - ex, y - ey)
            if d > max_dist or d < self.min_goal_dist:
                continue
            if not self._inside_rect(x, y):
                continue
            if self._free(x, y):
                return (x, y)
        return None

    # ------------------------------------------------------------ output
    def _send_goal(self, g, ego):
        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = g[0], g[1], self.goal_z
        yaw = math.atan2(g[1] - ego[1], g[0] - ego[0])
        msg.pose.orientation.z = math.sin(0.5 * yaw)
        msg.pose.orientation.w = math.cos(0.5 * yaw)
        self.pub.publish(msg)
        with self.lock:
            self.sent_goal = g
            self.sent_time = rospy.get_time()
        rospy.loginfo("[goal_pub] new goal (%.3f, %.3f), %.2f m from ego (%.3f, %.3f)",
                      g[0], g[1], math.hypot(g[0] - ego[0], g[1] - ego[1]), ego[0], ego[1])

    def _publish_rect_marker(self):
        m = Marker()
        m.header.frame_id = self.frame
        m.ns = "goal_pub_rect"
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 1.0
        for x, y in list(self.poly) + [self.poly[0]]:
            m.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.rect_pub.publish(m)

    def _tick(self, _event):
        with self.lock:
            ego, goal = self.ego, self.goal
            sent, sent_time = self.sent_goal, self.sent_time
        if ego is None:
            rospy.logwarn_throttle(5.0, "[goal_pub] waiting for %s" % self.pose_topic)
            return
        if sent is not None:
            # Waiting for setter to echo the goal on /ego_goal_set; resend if lost.
            if rospy.get_time() - sent_time > self.confirm_timeout:
                rospy.logwarn_throttle(2.0, "[goal_pub] setter has not confirmed goal; resending")
                self._send_goal(sent, ego)
            return
        if goal is not None and math.hypot(goal[0] - ego[0], goal[1] - ego[1]) > self.reach_dist:
            return
        if goal is not None:
            rospy.loginfo("[goal_pub] reached goal (%.3f, %.3f)", goal[0], goal[1])
        g = self._sample_goal(ego, self.max_goal_dist)
        if g is None:
            rospy.logwarn_throttle(2.0, "[goal_pub] no free goal within %.1f m of ego (%.2f, %.2f) "
                                   "inside rectangle; retrying" % (self.max_goal_dist, ego[0], ego[1]))
            return
        self._send_goal(g, ego)


def main():
    rospy.init_node("goal_pub", anonymous=False)
    RandomGoalPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
