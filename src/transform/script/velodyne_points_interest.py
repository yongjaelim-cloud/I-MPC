#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter PointCloud2 by horizontal radius and z height in its input frame."""
import copy
import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


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
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.callback,
                                    queue_size=1, buff_size=2 ** 24)
        rospy.loginfo('Filtering %s -> %s: 360 deg, radius <= %.1f m, %.2f <= z <= %.2f (input frame)',
                      self.input_topic, self.output_topic, self.radius,
                      self.z_min, self.z_max)

    def callback(self, msg):
        # Keep ALL input fields (including intensity/ring/time if present).
        fields = [field.name for field in msg.fields]
        if not all(field in fields for field in ('x', 'y', 'z')):
            rospy.logwarn_throttle(5.0, 'Cloud is missing x/y/z fields')
            return
        offsets = {}
        offset = 0
        for field in msg.fields:
            offsets[field.name] = offset
            offset += field.count
        ix, iy, iz = (offsets[n] for n in ('x', 'y', 'z'))
        filtered = []
        for row in point_cloud2.read_points(msg, field_names=fields, skip_nans=True):
            x, y, z = float(row[ix]), float(row[iy]), float(row[iz])
            if x * x + y * y <= self.radius * self.radius \
                    and self.z_min <= z <= self.z_max:
                filtered.append(row)
        output = point_cloud2.create_cloud(msg.header, msg.fields, filtered)
        output.is_dense = True
        self.pub.publish(output)  # Keep sensor coordinates for height-filter consumers.
        source = str(msg.header.frame_id).lstrip('/')
        if not source or msg.header.stamp.to_sec() <= 0.0:
            rospy.logwarn_throttle(2.0, 'Map cloud requires a nonempty frame and sensor timestamp')
            return
        header = copy.deepcopy(msg.header)
        header.frame_id = self.map_frame
        map_rows = filtered
        if source != self.map_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, source, msg.header.stamp, rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                rospy.logwarn_throttle(2.0, 'Map cloud TF unavailable: %s', str(exc))
                return
            q = transform.transform.rotation
            t = transform.transform.translation
            rotation = quaternion_matrix((q.x, q.y, q.z, q.w))[:3, :3]
            if filtered:
                xyz = np.array([[r[ix], r[iy], r[iz]] for r in filtered], dtype=float)
                xyz = xyz @ rotation.T + np.array([t.x, t.y, t.z])
                map_rows = []
                for row, position in zip(filtered, xyz):
                    transformed = list(row)
                    transformed[ix], transformed[iy], transformed[iz] = position
                    map_rows.append(transformed)
        mapped = point_cloud2.create_cloud(header, msg.fields, map_rows)
        mapped.is_dense = True
        self.map_pub.publish(mapped)


if __name__ == '__main__':
    rospy.init_node('velodyne_points_interest')
    VelodyneInterest()
    rospy.spin()
