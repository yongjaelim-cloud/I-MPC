#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS adapters for MPC_rcms0..3.

Requires the original MPC_planner0.py..MPC_planner3.py (unchanged) and the
original utils_linearized.py, setup_config.py, and drone_msgs package on PYTHONPATH.
Uses /setup exactly as those wrappers do and preserves their initial seed logic.
"""

import importlib
import time

import rospy
import numpy as np
from std_msgs.msg import String, Float64MultiArray
from drone_msgs.msg import MPCControlSeq

from rcms_hierarchy import RCMSHierarchy, RiskParameters


class ModeAwarePublisher:
    """Prevent an executable mitigation maneuver being mislabeled collision-free.

    The original MPC_planner publisher infers collision-free flags from the
    I-MPC normal_slack vector. This is *not* a valid RCMS mitigation criterion.
    Reset those fields at the output boundary while preserving serialization.
    """

    def __init__(self, delegate, mode_pub, diagnostic_pub, hierarchy, planner_id):
        self.delegate = delegate
        self.mode_pub = mode_pub
        self.diagnostic_pub = diagnostic_pub
        self.hierarchy = hierarchy
        self.planner_id = planner_id

    def publish(self, msg):
        mode = str(getattr(self.hierarchy,'mode','FAILED'))
        if mode == 'MITIGATION':
            # One controller may accept mitigation as a fallback but must NOT
            # treat a predicted collision as a safety/avoidance success.
            modifications = dict(
                original_ocp_feasible=False,
                ocp_feasible=False,
                collision_free=False,
                paper_feasible=False,
                zero_slack_feasible=False,
                nonlinear_zero_slack_feasible=False,
                selection_eligible=True,
                trajectory_type=2,
                trajectory_stage='rcms_mitigation',
            )
        elif mode == 'AVOIDANCE':
            modifications = dict(
                original_ocp_feasible=True,
                ocp_feasible=True,
                collision_free=True,
                paper_feasible=True,
                zero_slack_feasible=True,
                nonlinear_zero_slack_feasible=True,
                selection_eligible=True,
                trajectory_type=1,
                trajectory_stage='rcms_avoidance',
            )
        else:
            modifications = dict(
                original_ocp_feasible=False,
                ocp_feasible=False,
                collision_free=False,
                paper_feasible=False,
                zero_slack_feasible=False,
                nonlinear_zero_slack_feasible=False,
                selection_eligible=False,
                trajectory_type=0,
                trajectory_stage='rcms_failed',
            )
        for key,value in modifications.items():
            if hasattr(msg,key):
                setattr(msg,key,value)
        if mode != 'FAILED':
            self.delegate.publish(msg)
        else:
            rospy.logerr_throttle(1.,'[MPC_rcms%d] no validated avoidance or mitigation control: suppressing output',self.planner_id)
        self.mode_pub.publish(String(data=mode))
        st=self.hierarchy.stats
        diag=Float64MultiArray()
        diag.data=[float(self.planner_id),
                   float(getattr(msg,'seq_id',-1)),
                   float({'FAILED':-1,'AVOIDANCE':0,'MITIGATION':1}[mode]),
                   float(st.get('clearance',float('-inf'))),
                   float(st.get('avoidance_candidates',0)),
                   float(st.get('optimization_ms',0.))]
        self.diagnostic_pub.publish(diag)
        rospy.loginfo_throttle(1.,'[MPC_rcms%d] %s clearance=%.3f candidate_count=%d rcms_ms=%.2f',
                               self.planner_id,mode,
                               float(st.get('clearance',float('-inf'))),
                               int(st.get('avoidance_candidates',0)),
                               float(st.get('optimization_ms',0.)))

    def unregister(self):
        self.delegate.unregister()



class PlanningTimers:
    """Accumulate wall time without changing planner inputs or outputs."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.ms = {}
        self.calls = {}

    def wrap(self, key, function):
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.ms[key] = self.ms.get(key, 0.0) + (time.perf_counter() - started) * 1000.0
                self.calls[key] = self.calls.get(key, 0) + 1
        return measured


def run(planner_id):
    planner_id=int(planner_id)
    if planner_id not in (0,1,2,3):
        raise ValueError('RCMS planner id must be 0,1,2,3')
    rospy.init_node('MPC_rcms%d'%planner_id,anonymous=False)

    # Import only the matching original node. It remains the authority for
    # /tracked_pose, /cmd_vel, selector data, /setup, and initial seed creation.
    base=importlib.import_module('MPC_planner%d'%planner_id)
    # Use the original I-MPC minimum-distance OCP with redirection disabled.
    # /setup remains the source for all other MPC configuration values.
    # Set these BEFORE AcadosMPCPlanner is constructed: the OCP dimensions
    # depend on enable_impact_normal_constraint at build time.
    acados_factory=base.AcadosMPCPlanner
    def make_avoidance_solver(cfg):
        cfg.mpc_flag=4
        cfg.enable_impact_normal_constraint=False
        # Also remove impact-alignment cost from post-solve J_global ranking.
        cfg.rho_impact=0.0
        return acados_factory(cfg)
    base.AcadosMPCPlanner=make_avoidance_solver
    try:
        node=base.MPCPlannerTrajectoryNode()
    finally:
        base.AcadosMPCPlanner=acados_factory

    rospy.loginfo('[MPC_rcms%d] I-MPC avoidance OCP: mpc_flag=%d, impact_normal_constraint=%s, rho_impact=%.3g',
                  planner_id, int(node.cfg.mpc_flag),
                  str(bool(node.cfg.enable_impact_normal_constraint)), float(node.cfg.rho_impact))

    # Keep the same /setup parameterization as I-MPC. No new required Setup.msg
    # fields; RCMS-specific computational tuning is isolated here.
    pars=RiskParameters()
    # Optional one-time process-specific override, NOT a JSON read.
    pars.max_opt_ms=float(rospy.get_param('~rcms_max_opt_ms',pars.max_opt_ms))
    pars.max_iterations=2 #int(rospy.get_param('~rcms_max_iterations',pars.max_iterations))
    pars.require_regulation_zero_slack=bool(rospy.get_param(
        '~require_regulation_zero_slack',pars.require_regulation_zero_slack))
    hierarchy=RCMSHierarchy(node,node.planner.solve,pars)
    hierarchy.recovery_cycles=max(1,int(rospy.get_param('~rcms_recovery_cycles',1)))
    timers = PlanningTimers()
    hierarchy.original_solve = timers.wrap('mpc', hierarchy.original_solve)
    hierarchy._candidate_safe = timers.wrap('validation', hierarchy._candidate_safe)
    hierarchy._make_mitigation = timers.wrap('mitigation', hierarchy._make_mitigation)
    node.planner.solve = timers.wrap('hierarchy', hierarchy.solve)

    # Unregister the base /mpc_control_seq publisher. The only control stream
    # from this node is /mpc_rcms_control_seq{0..3} (as requested).
    old_publisher=node.pub_mpc_control_seq
    old_publisher.unregister()
    out_topic='/mpc_rcms_control_seq%d'%planner_id
    delegate=rospy.Publisher(out_topic,MPCControlSeq,queue_size=max(10,node.mpc_candidate_queue_size))
    mode_pub=rospy.Publisher('/mpc_rcms_mode%d'%planner_id,String,queue_size=10)
    diag_pub=rospy.Publisher('/mpc_rcms_diagnostics%d'%planner_id,Float64MultiArray,queue_size=10)
    node.pub_mpc_control_seq=ModeAwarePublisher(delegate,mode_pub,diag_pub,hierarchy,planner_id)

    rospy.loginfo('[MPC_rcms%d] avoidance via MPC_planner%d, mitigation via RCMS-risk planner; output=%s',
                  planner_id,planner_id,out_topic)
    rate=rospy.Rate(node.hz)
    while not rospy.is_shutdown():
        timers.reset()
        started=time.perf_counter()
        try:
            # Never certify collision-free motion before receiving a selector
            # snapshot. For dynamic vertices, require horizon predictions.
            if not bool(getattr(node,'have_selected_vertices',False)):
                rate.sleep()
                continue
            types=np.asarray(getattr(node,'latest_vertex_types',[]),dtype=int)
            if np.any(types==1) and not bool(getattr(node,'have_horizon_vertices',False)):
                rospy.logwarn_throttle(2.,'[MPC_rcms%d] dynamic vertices need horizon predictions; no plan published',planner_id)
                rate.sleep()
                continue
            node.step()
        except Exception as exc:
            rospy.logerr_throttle(1.,'[MPC_rcms%d] failed planning cycle: %r',planner_id,exc)
        elapsed_ms=(time.perf_counter()-started)*1000.
        mpc_ms = timers.ms.get('mpc', 0.0)
        validation_ms = timers.ms.get('validation', 0.0)
        mitigation_ms = timers.ms.get('mitigation', 0.0)
        hierarchy_ms = timers.ms.get('hierarchy', 0.0)
        hierarchy_other_ms = max(0.0, hierarchy_ms - mpc_ms - validation_ms - mitigation_ms)
        step_other_ms = max(0.0, elapsed_ms - hierarchy_ms)
        rospy.loginfo(
            '[MPC_rcms%d] mode=%s planning_ms=%.2f | mpc_ms=%.2f '
            'validation_ms=%.2f mitigation_ms=%.2f hierarchy_other_ms=%.2f '
            'step_other_ms=%.2f checks=%d solve_calls=%d',
            planner_id, str(getattr(hierarchy, 'mode', 'FAILED')),
            elapsed_ms, mpc_ms, validation_ms, mitigation_ms,
            hierarchy_other_ms, step_other_ms,
            timers.calls.get('validation', 0), timers.calls.get('mpc', 0))
        if elapsed_ms>1000./max(node.hz,1.e-6):
            rospy.logwarn_throttle(2.,'[MPC_rcms%d] planning deadline exceeded: %.2f ms (target %.2f ms)',
                                   planner_id,elapsed_ms,1000./max(node.hz,1.e-6))
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
