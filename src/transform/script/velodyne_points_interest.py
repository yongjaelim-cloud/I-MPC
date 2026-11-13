#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter PointCloud2 by horizontal radius and z height in its input frame.

Vectorized with NumPy: the raw PointCloud2 byte buffer is viewed as a
structured array, filtered with a boolean mask and re-serialized directly,
so all fields (intensity/ring/time...) are preserved without per-point loops.
"""
import time
import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2, PointField

_DTYPES = {
    PointField.INT8: 'i1', PointField.UINT8: 'u1',
    PointField.INT16: 'i2', PointField.UINT16: 'u2',
    PointField.INT32: 'i4', PointField.UINT32: 'u4',
    PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8',
}


def _cloud_dtype(msg):
    endian = '>' if msg.is_bigendian else '<'
    names, formats, offsets = [], [], []
    for f in msg.fields:
        base = np.dtype(endian + _DTYPES[f.datatype])
        names.append(f.name)
        formats.append(base if f.count == 1 else (base, (f.count,)))
        offsets.append(f.offset)
    return np.dtype({'names': names, 'formats': formats,
                     'offsets': offsets, 'itemsize': msg.point_step})


def _cloud_to_array(msg, dtype):
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.height > 1 and msg.row_step != msg.width * msg.point_step:
        raw = raw.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
        raw = np.ascontiguousarray(raw).reshape(-1)
    n = msg.width * msg.height
    return raw[:n * msg.point_step].view(dtype)


def _make_cloud(header, template, points):
    out = PointCloud2()
    out.header = header
    out.height = 1
    out.width = points.shape[0]
    out.fields = template.fields
    out.is_bigendian = template.is_bigendian
    out.point_step = template.point_step
    out.row_step = template.point_step * out.width
    out.is_dense = True
    out.data = points.tobytes()
    return out


def _points_in_polygon(px, py, poly):
    """Vectorized even-odd test; poly is (N, 2) vertices in order."""
    inside = np.zeros(px.shape[0], dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cross = (yi > py) != (yj > py)
        x_at = (xj - xi) * (py - yi) / ((yj - yi) if yj != yi else 1e-12) + xi
        inside ^= cross & (px < x_at)
        xj, yj = xi, yi
    return inside


# Region of interest in the map frame (2D Nav Goal clicks, 2026-11-12).
DEFAULT_MAP_POLYGON = [[-0.561, -0.456], [9.444, -10.508], [13.689, -2.018], [2.244, 7.907]]


class VelodyneInterest:
    def __init__(self):
        self.input_topic = rospy.get_param('~input_topic', '/velodyne_points')
        self.output_topic = rospy.get_param('~output_topic', '/velodyne_ponints_interest')
        self.map_frame = "map"
        self.map_output_topic = rospy.get_param('~map_output_topic', '/velodyne_points_interest_map')
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_pub = rospy.Publisher(self.map_output_topic, PointCloud2, queue_size=1)
        self.radius = float(rospy.get_param('~radius', 9.0))
        self.z_min = float(rospy.get_param('~z_min', -0.3))
        self.z_max = float(rospy.get_param('~z_max', 1.2))
        if not (self.radius > 0.0 and self.z_min <= self.z_max):
            raise ValueError('Invalid radius or z bounds')
        self.r2 = self.radius * self.radius
        self.max_tf_age = float(rospy.get_param('~max_tf_age', 0.2))
        # Points outside this map-frame polygon are removed. Set ~map_polygon:=[]
        # to disable.
        poly = rospy.get_param('~map_polygon', DEFAULT_MAP_POLYGON)
        if isinstance(poly, str):
            import ast
            poly = ast.literal_eval(poly)
        self.polygon = np.asarray(poly, dtype=float).reshape(-1, 2) if len(poly) else None
        if self.polygon is not None and self.polygon.shape[0] < 3:
            raise ValueError('~map_polygon needs at least 3 vertices')
        self._stat_n = 0
        self._stat_proc = 0.0
        self._stat_t0 = None
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.callback,
                                    queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
        self._dtype_key = None
        self._dtype = None
        rospy.loginfo('Filtering %s -> %s: 360 deg, radius <= %.1f m, %.2f <= z <= %.2f (input frame)',
                      self.input_topic, self.output_topic, self.radius,
                      self.z_min, self.z_max)
        if self.polygon is not None:
            rospy.loginfo('Map polygon filter (%s): %s', self.map_frame,
                          np.round(self.polygon, 3).tolist())

    def _get_dtype(self, msg):
        key = (tuple((f.name, f.offset, f.datatype, f.count) for f in msg.fields),
               msg.point_step, msg.is_bigendian)
        if key != self._dtype_key:
            self._dtype = _cloud_dtype(msg)
            self._dtype_key = key
        return self._dtype

    def callback(self, msg):
        t_start = time.time()
        try:
            self._process(msg)
        finally:
            self._report(t_start)

    def _report(self, t_start):
        now = time.time()
        if self._stat_t0 is None:
            self._stat_t0 = now
        self._stat_n += 1
        self._stat_proc += now - t_start
        if now - self._stat_t0 >= 5.0:
            rospy.loginfo('input %.1f Hz, processing %.2f ms/frame',
                          self._stat_n / (now - self._stat_t0),
                          1000.0 * self._stat_proc / self._stat_n)
            self._stat_n, self._stat_proc, self._stat_t0 = 0, 0.0, now

    def _lookup_map_tf(self, msg, source):
        """(R 3x3, t 3) map <- source at the cloud stamp, or None. Never blocks."""
        if source == self.map_frame:
            return np.eye(3), np.zeros(3)
        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, source, msg.header.stamp)
        except tf2_ros.ExtrapolationException:
            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame, source, rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                rospy.logwarn_throttle(2.0, 'Map TF unavailable: %s', str(exc))
                return None
            age = (msg.header.stamp - transform.header.stamp).to_sec()
            if abs(age) > self.max_tf_age:
                rospy.logwarn_throttle(2.0, 'Map TF too old (%.3f s)', age)
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as exc:
            rospy.logwarn_throttle(2.0, 'Map TF unavailable: %s', str(exc))
            return None
        q = transform.transform.rotation
        t = transform.transform.translation
        return quaternion_matrix((q.x, q.y, q.z, q.w))[:3, :3], np.array([t.x, t.y, t.z])

    def _process(self, msg):
        names = [f.name for f in msg.fields]
        if not all(n in names for n in ('x', 'y', 'z')):
            rospy.logwarn_throttle(5.0, 'Cloud is missing x/y/z fields')
            return
        pts = _cloud_to_array(msg, self._get_dtype(msg))
        x = pts['x']
        y = pts['y']
        z = pts['z']
        # NaN comparisons are False, so NaN points are dropped (skip_nans).
        mask = (x * x + y * y <= self.r2) & (z >= self.z_min) & (z <= self.z_max)
        filtered = pts[mask]  # copy, keeps all fields

        source = str(msg.header.frame_id).lstrip('/')
        tf_map = None
        if source and msg.header.stamp.to_sec() > 0.0:
            tf_map = self._lookup_map_tf(msg, source)
        else:
            rospy.logwarn_throttle(2.0, 'Map TF requires a nonempty frame and sensor timestamp')

        xyz_map = None
        if tf_map is not None and filtered.shape[0]:
            rot, trans = tf_map
            xyz = np.stack((filtered['x'], filtered['y'], filtered['z']), axis=1).astype(np.float64)
            xyz_map = xyz @ rot.T + trans
            if self.polygon is not None:
                inside = _points_in_polygon(xyz_map[:, 0], xyz_map[:, 1], self.polygon)
                filtered = filtered[inside]
                xyz_map = xyz_map[inside]
        elif tf_map is None and self.polygon is not None:
            # Without map TF the region test is impossible; keep all points
            # (conservative for obstacle avoidance) rather than dropping the scan.
            rospy.logwarn_throttle(2.0, 'Map polygon filter skipped (no map TF)')

        self.pub.publish(_make_cloud(msg.header, msg, filtered))  # sensor frame

        if tf_map is None or self.map_pub.get_num_connections() == 0:
            return
        if xyz_map is not None:
            filtered = filtered.copy()
            filtered['x'] = xyz_map[:, 0]
            filtered['y'] = xyz_map[:, 1]
            filtered['z'] = xyz_map[:, 2]
        header = type(msg.header)()
        header.seq = msg.header.seq
        header.stamp = msg.header.stamp
        header.frame_id = self.map_frame
        self.map_pub.publish(_make_cloud(header, msg, filtered))

if __name__ == '__main__':
    rospy.init_node('velodyne_points_interest')
    VelodyneInterest()
    rospy.spin()
