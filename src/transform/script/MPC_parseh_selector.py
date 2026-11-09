#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Avoidance aggregation and single on-demand Parseh mitigation.

Worker results: /mpc_parseh_control_seq{0..3}; status topics are optional.
Mitigation surfaces are reconstructed directly from vertex_selector_exp:
  /selected_vertex_data obstacle_polygon: obs_xs/ys = vertex coordinates,
    obs_rs = group_id, obs_vertex_counts = LOCAL VERTEX INDEX (not count),
    obs_vxs/vys = vertex velocity, obs_types = static/dynamic.
  /selected_vertex_horizon_data Float64MultiArray [H,K,6]:
    [x,y,unused,unused,type,group_id].
Group vertices are ordered by local index; each group is a convex polygon,
segment, or point supported by the native collision evaluator.
/setup supplies configuration; /tracked_pose and /cmd_vel supply ego state.
Mitigation output: /mpc_parseh_mitigation_path nav_msgs/Path,
 /mpc_parseh_mitigation_states Float32MultiArray [H+1,4],
 /mpc_parseh_mitigation_diagnostics Float64MultiArray.

IMPORTANT: MPCControlSeq serialization for mitigated paths depends on the
original custom drone_msgs definition. This script supports a small set of
explicitly detected sequence schemas; otherwise it publishes a Path and
refuses to fabricate a misleading MPCControlSeq.
"""
import copy
import importlib
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String, Float64MultiArray, Float32MultiArray
from drone_msgs.msg import MPCControlSeq, obstacle_polygon, Setup, Ego_goal_setting
from parseh_accel import MitigationEvaluator, control_grid_speed


def _reshape_msg(msg, shape, dtype=np.float32):
    arr = np.asarray(msg.data, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError('message element count %d != %d for shape %s' % (arr.size, expected, shape))
    return arr.reshape(shape)


def _dims(msg):
    return tuple(int(d.size) for d in msg.layout.dim)


def _yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def _finite_cost(msg):
    for attr in ('global_cost', 'cost'):
        if hasattr(msg, attr):
            val = float(getattr(msg, attr))
            if math.isfinite(val):
                return val
    return math.inf


def _safe_avoidance(msg):
    return bool(getattr(msg,'zero_slack_feasible',False)) and \
           int(getattr(msg,'trajectory_type',0))==1 and \
           len(getattr(msg,'xs',[]))>=2 and \
           len(getattr(msg,'accs',[]))>=1


def _path_from_states(states, stamp, frame_id='map'):
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    for x, y, speed, yaw in states:
        p = PoseStamped()
        p.header = path.header
        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.orientation.z = math.sin(float(yaw) * .5)
        p.pose.orientation.w = math.cos(float(yaw) * .5)
        path.poses.append(p)
    return path


def _serialize_mitigation(template, states, controls, dt, v_min, v_max):
    """Encode actual MPCControlSeq.msg: xs/ys/vs/yaws and accs/yawrates."""
    if template is None:
        msg = MPCControlSeq()
    else:
        msg = copy.deepcopy(template)
    required=('xs','ys','vs','yaws','accs','yawrates','start_ego_x',
              'start_ego_y','start_ego_v','start_ego_yaw')
    if not all(hasattr(msg,n) for n in required):
        return None
    if not (np.isfinite(states).all() and np.isfinite(controls).all()):
        return None
    msg.xs=list(map(float,states[:,0]))
    msg.ys=list(map(float,states[:,1]))
    msg.vs=list(map(float,states[:,2]))
    msg.yaws=list(map(float,states[:,3]))
    msg.accs=list(map(float,controls[:,0]))
    msg.yawrates=list(map(float,controls[:,1]))
    msg.normal_slacks=[0.0]*len(controls)
    msg.start_ego_x=float(states[0,0]);msg.start_ego_y=float(states[0,1])
    msg.start_ego_v=float(states[0,2]);msg.start_ego_yaw=float(states[0,3])
    msg.dt=float(dt)
    # Stage 2 isn't necessarily supported by downstream controller. Explicitly
    # mark infeasible; consumers must opt into mitigation trajectory_type=3.
    msg.trajectory_type=3
    for name in ('zero_slack_feasible', 'nonlinear_zero_slack_feasible',
                 'original_ocp_feasible', 'ocp_feasible', 'collision_free',
                 'paper_feasible'):
        if hasattr(msg, name):
            setattr(msg, name, False)
    if hasattr(msg, 'selection_eligible'):
        msg.selection_eligible = True
    if hasattr(msg, 'trajectory_stage'):
        msg.trajectory_stage = 'parseh_mitigation'
    msg.cost=1.e9
    msg.header.stamp=rospy.Time.now();msg.header.frame_id='map'
    if template is None: msg.seq_id=0
    return msg


class ParsehSelector:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {}
        self.avoidance = {}
        self.last_signature = None
        self.last_key = None
        self.max_age = float(rospy.get_param('~max_age_s', 0.45))
        self.hz = float(rospy.get_param('~hz', 10.0))
        self.frame = str(rospy.get_param('~frame_id', 'map'))
        self.backend_name = str(rospy.get_param('~backend', 'auto'))
        self.horizon = int(rospy.get_param('~mitigation_horizon', 15))
        self.dt = float(rospy.get_param('~mitigation_dt', 0.2))
        self.steering_samples = int(rospy.get_param('~steering_samples', 500))
        self.speed_samples = int(rospy.get_param('~speed_samples', 40))
        self.omega_min = float(rospy.get_param('~omega_min', -1.3))
        self.omega_max = float(rospy.get_param('~omega_max', 1.3))
        self.v_min = float(rospy.get_param('~v_min', -.2))
        self.v_max = float(rospy.get_param('~v_max', .8))
        self.length = float(rospy.get_param('~ego_length', .5))
        self.width = float(rospy.get_param('~ego_width', .5))
        self.margin = float(rospy.get_param('~collision_margin', 0.0))
        self.substeps = int(rospy.get_param('~collision_substeps', 3))
        self.sector_id = None  # Set exclusively by /setup; no ROS mask override.
        self.vulnerable_penalty = float(rospy.get_param('~vulnerable_penalty', 11.))
        self.impact_weight = float(rospy.get_param('~impact_weight', 0.02))
        self.goal_weight = float(rospy.get_param('~goal_weight', 0.))
        self.min_mitigation_gap = float(rospy.get_param('~mitigation_min_interval_s', .05))
        self.max_geometry_age = float(rospy.get_param('~max_geometry_age_s', .3))
        self.last_mitigation_at = -1.e12
        self.selected_vertices = None
        self.selected_stamp = 0.
        self.selected_horizon = None
        self.horizon_stamp = 0.
        self.geometry_source = 'none'

        self.pose = None
        self.speed = None
        self.goal = None
        self.geometry = None
        self.counts = None
        self.velocities = None
        self.pose_stamp = 0.
        self.geometry_stamp = 0.
        self.counts_stamp = 0.
        self.vel_stamp = 0.

        self.pub = rospy.Publisher('/mpc_parseh_control_seq', MPCControlSeq, queue_size=4)
        self.mode_pub = rospy.Publisher('/mpc_parseh_mode', String, queue_size=4)
        self.path_pub = rospy.Publisher('/mpc_parseh_mitigation_path', Path, queue_size=2)
        self.states_pub = rospy.Publisher('/mpc_parseh_mitigation_states', Float32MultiArray, queue_size=2)
        self.diag_pub = rospy.Publisher('/mpc_parseh_mitigation_diagnostics', Float64MultiArray, queue_size=4)
        self.backend = MitigationEvaluator()
        self.controls = None  # rebuild acceleration library from current /cmd_vel speed
        self.subs = []
        for i in range(4):
            self.subs.append(rospy.Subscriber('/mpc_parseh_control_seq%d' % i,
                MPCControlSeq, self._avoidance_cb, callback_args=i, queue_size=2))
            self.subs.append(rospy.Subscriber('/mpc_parseh_status%d' % i,
                Float64MultiArray, self._status_cb, callback_args=i, queue_size=3))
        self.subs.extend([
            rospy.Subscriber('/setup', Setup, self._setup_cb, queue_size=1),
            rospy.Subscriber('/ego_goal_set', Ego_goal_setting, self._runtime_goal_cb, queue_size=1),
            rospy.Subscriber('/selected_vertex_data', obstacle_polygon, self._selected_cb, queue_size=1),
            rospy.Subscriber('/selected_vertex_horizon_data', Float64MultiArray, self._selected_horizon_cb, queue_size=1),
            rospy.Subscriber('/tracked_pose', PoseStamped, self._pose_cb, queue_size=1),
            rospy.Subscriber('/cmd_vel', Twist, self._speed_cb, queue_size=1),
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self._goal_cb, queue_size=1),
        ])
        rospy.loginfo('[parseh_selector] avoidance-first, single C++/CUDA mitigation backend=%s', self.backend_name)

    def _setup_cb(self, msg):
        """Use the same typed /setup conversion/config loader as the workers."""
        try:
            from setup_config import setup_msg_to_raw
            raw = setup_msg_to_raw(msg)
            base = importlib.import_module('MPC_planner0')
            cfg, _, _ = base.load_planner_config(raw_config=raw)
            goal = (float(raw['config']['x_end']), float(raw['config']['y_end']))
            if not np.all(np.isfinite(goal)):
                raise ValueError('non-finite /setup goal')
            sector = int(base.normalize_sector_config(cfg))
            if sector not in (1, 2, 3, 4):
                raise ValueError('/setup sector_id must be in 1..4')
            with self.lock:
                self.goal = goal
                self.sector_id = sector
                # Defaults must match the actual MPC vehicle and time grid.
                for target, param, source in (
                    ('horizon', '~mitigation_horizon', 'N'),
                    ('dt', '~mitigation_dt', 'dt_init'),
                    ('length', '~ego_length', 'ego_L'),
                    ('width', '~ego_width', 'ego_W'),
                    ('v_min', '~v_min', 'v_min'), ('v_max', '~v_max', 'v_max'),
                    ('omega_min', '~omega_min', 'yawrate_min'),
                    ('omega_max', '~omega_max', 'yawrate_max')):
                    if not rospy.has_param(param) and hasattr(cfg, source):
                        value = getattr(cfg, source)
                        setattr(self, target, int(value) if target == 'horizon' else float(value))
            rospy.loginfo('[parseh_selector] setup sector_id=%d goal=%s horizon=%d dt=%.3f ego=%.3fx%.3f',
                          sector, str(goal), self.horizon, self.dt, self.length, self.width)
        except Exception as exc:
            rospy.logerr_throttle(2., '[parseh_selector] /setup conversion failed: %r', exc)

    def _runtime_goal_cb(self, msg):
        goal = (float(msg.x_end), float(msg.y_end))
        if np.all(np.isfinite(goal)):
            with self.lock:
                self.goal = goal

    def _selected_cb(self, msg):
        """Decode raw selected vertices, not the full-polygon message schema."""
        frame = getattr(getattr(msg, 'header', None), 'frame_id', '')
        if frame and frame != self.frame:
            rospy.logwarn_throttle(2., '[parseh_selector] vertex frame=%s expected=%s', frame, self.frame)
            return
        try:
            K = int(msg.obs_cnt)
            fields = ('obs_xs', 'obs_ys', 'obs_vxs', 'obs_vys',
                      'obs_types', 'obs_rs', 'obs_vertex_counts')
            if K < 0 or any(len(getattr(msg, field)) < K for field in fields):
                raise ValueError('invalid selected vertex count')
            values = [np.asarray(getattr(msg, field)[:K], dtype=float) for field in fields]
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError('non-finite selected vertex data')
            xy = np.column_stack(values[:2])
            velocity = np.column_stack(values[2:4])
            types, groups, local = [np.rint(value).astype(np.int32) for value in values[4:]]
            if any(not np.allclose(value, np.rint(value), rtol=0, atol=1.e-6) for value in values[4:]):
                raise ValueError('nonintegral type/group/local index')
            if np.any(groups < 0) or np.any(local < 0):
                raise ValueError('negative group/local index')
            velocity[types != 1] = 0.
            for group in np.unique(groups):
                indices = local[groups == group]
                if not np.array_equal(np.sort(indices), np.arange(len(indices))):
                    raise ValueError('group %d local indices are incomplete or repeated' % group)
            with self.lock:
                self.selected_vertices = (xy, velocity, types, groups, local)
                self.selected_stamp = time.monotonic()
        except (ValueError, TypeError, AttributeError, IndexError) as exc:
            rospy.logwarn_throttle(2., '[parseh_selector] /selected_vertex_data rejected: %s', str(exc))

    def _selected_horizon_cb(self, msg):
        try:
            dims = _dims(msg)
            if len(dims) != 3 or dims[0] < 1 or dims[1] < 0 or dims[2] < 6:
                raise ValueError('expected [H,K,6] selected vertex horizon')
            offset = int(getattr(msg.layout, 'data_offset', 0))
            data = np.asarray(msg.data, dtype=float)[offset:]
            if data.size != int(np.prod(dims)):
                raise ValueError('selected horizon payload size mismatch')
            tensor = data.reshape(dims).copy()
            if not np.isfinite(tensor).all():
                raise ValueError('non-finite selected horizon')
            if not np.allclose(tensor[:, :, 4:6], np.rint(tensor[:, :, 4:6]), rtol=0, atol=1.e-6):
                raise ValueError('nonintegral horizon type/group')
            if not np.all(tensor[:, :, 4:6] == tensor[0:1, :, 4:6]):
                raise ValueError('horizon group/type changes across stages')
            with self.lock:
                self.selected_horizon = tensor
                self.horizon_stamp = time.monotonic()
        except (ValueError, TypeError, AttributeError) as exc:
            rospy.logwarn_throttle(2., '[parseh_selector] /selected_vertex_horizon_data rejected: %s', str(exc))

    def _selected_geometry(self, now):
        """Reconstruct surfaces from current group IDs and ordered vertices.

        Called with lock held. A paired horizon supplies stage geometry;
        otherwise use the same constant-velocity prediction as vertex_selector_exp.
        Never pair unrelated current/horizon packets by arrival time alone.
        """
        selected = getattr(self, 'selected_vertices', None)
        stamp = getattr(self, 'selected_stamp', 0.)
        if selected is None or now - stamp > self.max_geometry_age:
            self.geometry = self.counts = self.velocities = None
            return
        xy, velocity, types, groups, local = selected
        K = len(xy)
        if K == 0:
            self.geometry = self.counts = self.velocities = None
            rospy.logwarn_throttle(2., '[parseh_selector] WAIT selected vertex set is empty; no mitigation surfaces')
            return
        H = self.horizon
        times = (np.arange(H) + 1) * self.dt
        positions = xy[None, :, :] + (times[:, None, None] + now - stamp) * velocity[None, :, :]
        source = 'selected_vertices+velocity'
        horizon = getattr(self, 'selected_horizon', None)
        if (horizon is not None and horizon.shape[:2] == (H, K)
                and now - self.horizon_stamp <= self.max_geometry_age
                and np.array_equal(horizon[0, :, 4].astype(np.int32), types)
                and np.array_equal(horizon[0, :, 5].astype(np.int32), groups)
                and np.allclose(horizon[0, :, :2], xy + self.dt * velocity, rtol=0., atol=1.e-5)):
            positions = horizon[:, :, :2] + (now - stamp) * velocity[None, :, :]
            source = 'selected_vertex_horizon'
        group_ids = np.unique(groups)
        indices = [np.flatnonzero(groups == group) for group in group_ids]
        indices = [ids[np.argsort(local[ids], kind='stable')] for ids in indices]
        counts = np.array([len(ids) for ids in indices], dtype=np.int32)
        V = int(counts.max())
        vertices = np.zeros((H, len(indices), V, 2), dtype=np.float32)
        velocities = np.zeros((H, len(indices), 2), dtype=np.float32)
        for i, ids in enumerate(indices):
            vertices[:, i, :len(ids)] = positions[:, ids]
            velocities[:, i] = velocity[ids].mean(axis=0)
        self.geometry, self.counts, self.velocities = vertices, counts, velocities
        self.geometry_stamp = self.counts_stamp = self.vel_stamp = stamp
        self.geometry_source = source

    def _avoidance_cb(self, msg, i):
        with self.lock:
            now = time.monotonic()
            seq = int(getattr(msg, 'seq_id', -1))
            if seq < 0:
                rospy.logwarn_throttle(2., '[parseh_selector] worker=%d invalid seq_id=%d', i, seq)
                return
            self.avoidance[i] = (now, msg)
            # The trajectory itself also reports success/failure. Do not make
            # selection depend on the arrival of a redundant status topic.
            previous = self.status.get(i)
            solve_ms = previous[3] if previous is not None and seq == previous[1] else float('nan')
            self.status[i] = (now, seq, _safe_avoidance(msg), solve_ms)

    def _status_cb(self, msg, i):
        if len(msg.data) < 5 or not np.all(np.isfinite(msg.data[:3])) or int(msg.data[1]) < 0:
            return
        with self.lock:
            record = (time.monotonic(), int(msg.data[1]), bool(msg.data[2] > .5), float(msg.data[4]))
            self.status[i] = record

    def _pose_cb(self, msg):
        if msg.header.frame_id != self.frame:
            rospy.logwarn_throttle(2., '[parseh_selector] pose frame %s != %s', msg.header.frame_id, self.frame)
            return
        with self.lock:
            self.pose = np.array([msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation)], dtype=np.float32)
            self.pose_stamp = time.monotonic()

    def _speed_cb(self, msg):
        with self.lock:
            self.speed = float(msg.linear.x)

    def _goal_cb(self, msg):
        if msg.header.frame_id != self.frame:
            rospy.logwarn_throttle(2., '[parseh_selector] goal frame mismatch')
            return
        with self.lock:
            self.goal = (msg.pose.position.x, msg.pose.position.y)

    def _collect(self):
        with self.lock:
            now = time.monotonic()
            self._selected_geometry(now)
            snapshot = {i:record for i,record in self.status.items()
                        if now - record[0] <= self.max_age}
            trajectories = self.avoidance
            missing = [i for i in range(4) if i not in snapshot]
            options = []
            for i in snapshot:
                if not snapshot[i][2]:
                    continue
                record = trajectories.get(i)
                if record and now - record[0] <= self.max_age and _safe_avoidance(record[1]):
                    # Separate ROS topics may arrive in either order. Match
                    # planning IDs, never infer identity from arrival times.
                    if int(getattr(record[1], 'seq_id', -1)) == snapshot[i][1]:
                        options.append((i,record[1]))
            signature = tuple((i,snapshot[i][1]) for i in sorted(snapshot))
            state = None if self.pose is None or self.speed is None else np.array(
                [self.pose[0],self.pose[1],self.speed,self.pose[2]], dtype=np.float32)
            # seq_id is worker-local in this deployment. It only pairs a
            # trajectory with THAT worker's status, never different workers.
            all_failed = not missing and all(not record[2] for record in snapshot.values())
            if not options and not all_failed:
                rospy.logwarn_throttle(2.,
                    '[parseh_selector] WAIT workers_missing_or_stale=%s worker_seq_ids=%s zero_slack_workers=%s; '
                    'need /mpc_parseh_control_seq0..3 (or /mpc_parseh_status0..3). '
                    'Mitigation needs four fresh failed results; worker IDs need not match.',
                    str(missing), str(dict(signature)),
                    str([i for i,r in snapshot.items() if r[2]]))
            return (signature, options, all_failed,
                state, self.goal, self.geometry, self.counts, self.velocities,
                self.pose_stamp, self.geometry_stamp, self.counts_stamp, self.vel_stamp,
                next((self.avoidance[i][1] for i in range(4) if i in self.avoidance), None))

    def step(self):
        cycle_started = time.perf_counter()
        snap = self._collect()
        if snap is None:
            return
        (signature,options,all_failed,state,goal,vertices,counts,velocities,
         pose_stamp,geometry_stamp,counts_stamp,vel_stamp,template) = snap
        if signature == self.last_signature:
            rospy.loginfo_throttle(2., '[parseh_selector] WAIT no new worker results; last=%s', str(signature))
            return
        if options:
            planner_id,msg = min(options, key=lambda pair: (_finite_cost(pair[1]),pair[0]))
            key = ('AVOIDANCE',planner_id,int(getattr(msg,'seq_id',-1)))
            if key != self.last_key:
                self.pub.publish(msg)
                self.mode_pub.publish(String(data='AVOIDANCE'))
                self.last_key = key
            self.last_signature = signature
            worker_ms = self.status[planner_id][3] if planner_id in self.status else float('nan')
            selector_ms = (time.perf_counter() - cycle_started) * 1000.0
            rospy.loginfo('[parseh_selector] mode=AVOIDANCE planner=%d seq=%d worker_ms=%.3f selector_ms=%.3f cost=%.6f',
                          planner_id, int(getattr(msg, 'seq_id', -1)), worker_ms, selector_ms, _finite_cost(msg))
            return
        if not all_failed:
            return  # Wait for the successful worker's corresponding trajectory.
        with self.lock:
            sector_id = self.sector_id
        if sector_id is None:
            rospy.logwarn_throttle(2., '[parseh_selector] WAIT /setup sector_id before mitigation')
            return
        now = time.monotonic()
        if now - self.last_mitigation_at < self.min_mitigation_gap:
            return
        self.last_mitigation_at = now
        if (state is None or goal is None or vertices is None or counts is None or velocities is None
            or now - pose_stamp > self.max_geometry_age
            or now - min(geometry_stamp, counts_stamp, vel_stamp) > self.max_geometry_age):
            rospy.logwarn_throttle(2.,
                '[parseh_selector] WAIT mitigation inputs: pose_and_speed=%s goal=%s '
                'vertices=%s counts=%s velocities=%s ages_s(pose,vertices,counts,velocities)=%s; '
                'surface_topics=/selected_vertex_data,/selected_vertex_horizon_data '
                'goal_topics=/setup,/ego_goal_set,/move_base_simple/goal',
                state is not None, goal is not None, vertices is not None,
                counts is not None, velocities is not None,
                str([round(now-t,3) if t else None for t in
                     (pose_stamp,geometry_stamp,counts_stamp,vel_stamp)]))
            self.mode_pub.publish(String(data='FAILED'))
            return
        if (vertices.shape[0] != self.horizon or vertices.shape[1] != len(counts)
            or velocities.shape != (self.horizon,len(counts),2)):
            rospy.logwarn('[parseh_selector] mode=FAILED reason=geometry_shape_mismatch selector_ms=%.3f',
                          1000.0 * (time.perf_counter() - cycle_started))
            self.mode_pub.publish(String(data='FAILED'))
            return
        started = time.perf_counter()
        self.last_signature = signature
        try:
            rospy.loginfo('[parseh_selector] computing MITIGATION worker_seq_ids=%s surfaces=%d vertices=%d geometry=%s',
                          str(dict(signature)), len(counts), int(counts.sum()), self.geometry_source)
            controls = control_grid_speed(self.horizon, float(state[2]),
                yaw_samples=self.steering_samples, speed_samples=self.speed_samples,
                omega_min=self.omega_min, omega_max=self.omega_max,
                v_min=self.v_min, v_max=self.v_max, dt=self.dt)
            backend_started = time.perf_counter()
            result = self.backend.evaluate(state, controls, vertices, counts, velocities,
                dt=self.dt,length=self.length,width=self.width,margin=self.margin,
                collision_substeps=self.substeps,goal=goal,goal_weight=self.goal_weight,
                sector_id=sector_id,vulnerable_penalty=self.vulnerable_penalty,
                impact_weight=self.impact_weight,backend=self.backend_name,
                v_min=self.v_min,v_max=self.v_max,selected_only=True)
            backend_ms = 1000.0 * (time.perf_counter() - backend_started)
            best = int(result.best_index)
            if best < 0 or not np.all(np.isfinite(result.trajectories[0])):
                raise RuntimeError('native backend returned no finite trajectory')
            states = result.trajectories[0]
            stamp = rospy.Time.now()
            self.path_pub.publish(_path_from_states(states,stamp,self.frame))
            output = Float32MultiArray()
            from std_msgs.msg import MultiArrayDimension
            output.layout.dim = [MultiArrayDimension(label='stage',size=len(states),stride=len(states)*4),
                                 MultiArrayDimension(label='state_xyvyaw',size=4,stride=4)]
            output.data = [float(v) for v in states.ravel()]
            self.states_pub.publish(output)
            serialized = _serialize_mitigation(template,states,controls[best],self.dt,self.v_min,self.v_max)
            if serialized is not None:
                serialized.seq_id = int(signature[0][1])
                serialized.header.frame_id = self.frame
                self.pub.publish(serialized)
                self.mode_pub.publish(String(data='MITIGATION'))
            else:
                self.mode_pub.publish(String(data='MITIGATION_PATH_ONLY'))
                rospy.logwarn_throttle(3., '[parseh_selector] custom MPCControlSeq schema unknown; '
                    'publishing Path only, not potentially stale control sequence')
            ms = 1000. * (time.perf_counter()-started)
            score = result.scores[0]
            self.diag_pub.publish(Float64MultiArray(data=[float(ms),float(best),float(self.steering_samples*self.speed_samples),
                float(score[0]),float(score[1]),float(score[2]),float(score[3]),float(score[4]),
                float(score[5]),float(score[6]),float(result.backend == 'cuda')]))
            total_ms = 1000.0 * (time.perf_counter() - cycle_started)
            mode = 'MITIGATION' if serialized is not None else 'MITIGATION_PATH_ONLY'
            rospy.loginfo('[parseh_selector] mode=%s candidates=%d best=%d backend=%s backend_ms=%.3f mitigation_ms=%.3f selector_ms=%.3f',
                          mode, self.steering_samples*self.speed_samples, best, result.backend, backend_ms, ms, total_ms)
        except Exception as e:
            self.mode_pub.publish(String(data='FAILED'))
            rospy.logerr('[parseh_selector] mode=FAILED reason=mitigation_exception selector_ms=%.3f error=%r',
                         1000.0 * (time.perf_counter() - cycle_started), e)


def main():
    rospy.init_node('MPC_parseh_selector', anonymous=False)
    selector = ParsehSelector()
    rate = rospy.Rate(selector.hz)
    while not rospy.is_shutdown():
        selector.step()
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break

if __name__ == '__main__':
    main()
