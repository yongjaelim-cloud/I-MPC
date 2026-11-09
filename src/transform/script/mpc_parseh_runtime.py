#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Avoidance-only I-MPC workers. Mitigation belongs to the selector.

The original MPC_planner{0..3} owns seeds, snapshots and serialization.
Status layout: [planner_id, seq_id, zero_slack_ok, slack_max, solve_ms,
                avoidance_residual_max]. Every published result has a status,
including nonzero-slack results and solver fallbacks.
"""
import importlib
import time

import numpy as np
import rospy
from std_msgs.msg import String, Float64MultiArray
from drone_msgs.msg import MPCControlSeq


def avoidance_metrics(plan, tol):
    """Check avoidance slack and unrelaxed avoidance residual, not s_r.

    Missing diagnostics fail closed. A disabled normal constraint's zero s_r
    cannot establish that a trajectory avoids obstacles.
    """
    if plan is None or bool(getattr(plan, 'is_order_fallback', False)):
        return False, float('inf'), float('inf')
    try:
        slack = np.asarray(getattr(plan, 'slack', None), dtype=float).reshape(-1)
        slack_max = float(np.max(np.abs(slack))) if slack.size else float('inf')
        residual = float(getattr(plan, 'zero_slack_avoidance_residual_max', float('inf')))
        ok = (np.all(np.isfinite(slack)) and np.isfinite(slack_max)
              and slack_max <= tol and np.isfinite(residual) and residual <= tol)
        return bool(ok), slack_max, residual
    except (TypeError, ValueError):
        return False, float('inf'), float('inf')


class AvoidancePublisher:
    def __init__(self, delegate, status_pub, mode_pub, node, planner_id, tol):
        self.delegate = delegate
        self.status_pub = status_pub
        self.mode_pub = mode_pub
        self.node = node
        self.planner_id = planner_id
        self.tol = tol
        self.solve_ms = 0.0

    def publish(self, msg):
        plan = getattr(self.node, 'latest_plan', None)
        ok, slack_max, residual = avoidance_metrics(plan, self.tol)
        # This is the exact plan serialized by the original node. Do not use
        # planner.last_result, which can differ from the published SQP iterate.
        for name in ('xs', 'ys', 'vs', 'yaws', 'accs', 'yawrates'):
            values = np.asarray(getattr(msg, name, []), dtype=float)
            ok = bool(ok and values.size > 0 and np.all(np.isfinite(values)))
        ok = bool(ok and len(getattr(msg, 'xs', [])) >= 2)
        updates = dict(trajectory_stage='parseh_avoidance', trajectory_type=1,
                       zero_slack_feasible=ok, nonlinear_zero_slack_feasible=ok,
                       original_ocp_feasible=ok, ocp_feasible=ok,
                       collision_free=ok, selection_eligible=ok, paper_feasible=ok,
                       avoidance_slack_max=slack_max,
                       zero_slack_avoidance_residual_max=residual)
        for name, value in updates.items():
            if hasattr(msg, name):
                setattr(msg, name, value)
        # Publish failures as well: they are diagnostics, never executable
        # avoidance candidates. Only the selector chooses the maneuver.
        self.delegate.publish(msg)
        status = Float64MultiArray()
        status.data = [float(self.planner_id), float(getattr(msg, 'seq_id', -1)),
                       float(ok), slack_max, self.solve_ms, residual]
        self.status_pub.publish(status)
        self.mode_pub.publish(String(data='AVOIDANCE' if ok else 'NO_ZERO_SLACK'))
        rospy.loginfo_throttle(1., '[MPC_parseh%d] avoidance_only seq=%d zero_slack=%d slack=%.3g residual=%.3g solve_ms=%.3f',
                               self.planner_id, int(getattr(msg, 'seq_id', -1)),
                               int(ok), slack_max, residual, self.solve_ms)

    def unregister(self):
        self.delegate.unregister()


def run(planner_id):
    planner_id = int(planner_id)
    if planner_id not in (0, 1, 2, 3):
        raise ValueError('planner_id must be in 0..3')
    rospy.init_node('MPC_parseh%d' % planner_id, anonymous=False)
    base = importlib.import_module('MPC_planner%d' % planner_id)
    original_factory = base.AcadosMPCPlanner

    def create_avoidance(cfg):
        cfg.mpc_flag = 4
        cfg.enable_impact_normal_constraint = False
        cfg.rho_impact = 0.0
        return original_factory(cfg)

    # Retain the factory for /setup-triggered solver rebuilds too.
    base.AcadosMPCPlanner = create_avoidance
    node = base.MPCPlannerTrajectoryNode()
    tol = float(rospy.get_param('~avoidance_slack_tol', 1.e-3))
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('avoidance_slack_tol must be finite and nonnegative')
    node.pub_mpc_control_seq.unregister()
    output = '/mpc_parseh_control_seq%d' % planner_id
    publisher = AvoidancePublisher(
        rospy.Publisher(output, MPCControlSeq, queue_size=max(10, getattr(node, 'mpc_candidate_queue_size', 10))),
        rospy.Publisher('/mpc_parseh_status%d' % planner_id, Float64MultiArray, queue_size=10),
        rospy.Publisher('/mpc_parseh_mode%d' % planner_id, String, queue_size=10),
        node, planner_id, tol)
    node.pub_mpc_control_seq = publisher
    # Measure only the ordinary I-MPC solve. No hierarchy or library fallback.
    original_solve = node.planner.solve

    def timed_solve(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_solve(*args, **kwargs)
        finally:
            publisher.solve_ms = 1000. * (time.perf_counter() - started)

    node.planner.solve = timed_solve
    rospy.loginfo('[MPC_parseh%d] avoidance-only I-MPC flag4 redirection=OFF; mitigation=selector; out=%s',
                  planner_id, output)
    rate = rospy.Rate(node.hz)
    while not rospy.is_shutdown():
        try:
            if not bool(getattr(node, 'have_selected_vertices', False)):
                rate.sleep()
                continue
            types = np.asarray(getattr(node, 'latest_vertex_types', []), dtype=int)
            if np.any(types == 1) and not bool(getattr(node, 'have_horizon_vertices', False)):
                rospy.logwarn_throttle(2., '[MPC_parseh%d] no dynamic vertex horizon data', planner_id)
                rate.sleep()
                continue
            node.step()
        except Exception as exc:
            rospy.logerr_throttle(1., '[MPC_parseh%d] planning cycle error: %r', planner_id, exc)
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
