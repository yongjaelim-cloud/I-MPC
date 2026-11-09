#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hierarchical avoidance -> Parseh-style impact-severity trajectory library.

Based on the user-supplied MATLAB solve_parseh.m and shared_vehicle_utils.m.
Its simplified ORFS surrogate is W1 * relative_speed + F(impact_sector),
where F is 1/12 for allowable/disallowed impact locations. It is NOT the
full ORFS injury database, nor the original paper's Pacejka vehicle model.

Stage 1 reuses the original MPC_planner{0..3} acados solver/initial seeds with
mpc_flag=0, then verifies zero slacks and geometric collision avoidance.
Stage 2 enumerates 9 yaw-rate samples x 5 acceleration samples, predicts
first contact with selector obstacles, scores the impact, and publishes the
best candidate in the same I-MPC dynamics/control convention. A constant
*yaw-rate command* replaces the MATLAB *steering-rate* due to robot interface.

The sector_semantics='vulnerable' default matches the user's I-MPC sector_id:
inside the vulnerable sector => F_high. To reproduce supplied MATLAB exactly,
choose sector_semantics='allowable': inside sector => F_low.
"""

from dataclasses import dataclass
import math
import time

import numpy as np

from rcms_hierarchy import (
    RCMSHierarchy, RiskParameters,
    _ego_polygon, _polygon_clearance, build_obstacle_stages,
    minimum_clearance,
)


@dataclass
class ParsehParameters(RiskParameters):
    # Corresponds to solve_parseh.m's W1, F_low, F_high, 9 x 5 library.
    W1: float = 0.02
    F_low: float = 1.0
    F_high: float = 12.0
    steering_samples: int = 9
    acceleration_samples: int = 5
    # MATLAB's sector=allowable. I-MPC's sector_id=vulnerable.
    sector_semantics: str = 'vulnerable'
    # Source MATLAB computes collision at each discrete horizon step.
    collision_substeps: int = 1
    # Used only when NO collision occurs. No-collision scores use Vrel at end.
    # Leave 0 for literal MATLAB equivalence; tie-breaking by goal is optional.
    collision_free_goal_tiebreak: float = 0.0
    # Upper bound for *library* evaluation, not the complete planning cycle.
    max_library_ms: float = 100.0
    # If true, include the original guidance/control warm start as additional
    # library candidates. Default false keeps MATLAB's 45 constant controls.
    include_mpc_seeds: bool = False
    allow_partial_library: bool = False


def _surface_from_angle(theta, ego_length, ego_width):
    """Ego body-frame surface ID for a center-to-obstacle approach angle.

    Uses the exact corner-ray geometry of the provided MATLAB helper and
    utils_linearized.sector_rays_body_from_id: 1 left, 2 rear, 3 right, 4 front.
    """
    L, W = float(ego_length), float(ego_width)
    corner_ang = math.atan2(W, L)  # front-left corner, in [0, pi/2]
    a = (float(theta) + 2.0*math.pi) % (2.0*math.pi)
    if corner_ang <= a < math.pi - corner_ang:
        return 1
    if math.pi-corner_ang <= a < math.pi+corner_ang:
        return 2
    if math.pi+corner_ang <= a < 2.0*math.pi-corner_ang:
        return 3
    return 4


def _impact_info(state, obstacle_polygon, sector_id, pars, ego_L, ego_W):
    """Classify impact based on *obstacle centroid*, as in solve_parseh.m.

    Contact location is NOT inferred from physical contact normal; the MATLAB
    Parseh port uses direction to other vehicle's center in ego-body frame.
    """
    poly = np.asarray(obstacle_polygon, dtype=float).reshape(-1, 2)
    ctr = poly.mean(axis=0)
    dx, dy = ctr - np.asarray(state[:2], dtype=float)
    yaw = float(state[3])
    bx = math.cos(yaw)*dx + math.sin(yaw)*dy
    by = -math.sin(yaw)*dx + math.cos(yaw)*dy
    angle = math.atan2(by, bx)
    location_id = _surface_from_angle(angle, ego_L, ego_W)
    semantics = str(pars.sector_semantics).strip().lower()
    if semantics not in ('allowable', 'vulnerable'):
        raise ValueError('sector_semantics must be allowable or vulnerable')
    in_sector = location_id == int(sector_id)
    preferred = in_sector if semantics == 'allowable' else not in_sector
    severity = float(pars.F_low if preferred else pars.F_high)
    return severity, location_id, angle, ctr


def _stage_cache(stages):
    """Precompute obstacle broadphase info to keep 45 x N library tractable."""
    caches = []
    for polys in stages:
        rec = []
        for group, poly in polys:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            center = np.mean(pts, axis=0)
            radius = float(np.max(np.linalg.norm(pts-center, axis=1)))
            rec.append((int(group), pts, center, radius))
        caches.append(rec)
    return caches


def _ego_vel(state):
    return np.asarray([float(state[2])*math.cos(float(state[3])),
                       float(state[2])*math.sin(float(state[3]))], dtype=float)


def _obstacle_velocities(caches, dt):
    """Finite-difference group centroid from selector horizon; never normals."""
    n = len(caches)
    positions = [{g:c for g,p,c,r in stage} for stage in caches]
    velocities = []
    for h in range(n):
        speeds = {}
        for g,c in positions[h].items():
            if h < n-1 and g in positions[h+1]:
                speeds[g] = (positions[h+1][g]-c)/max(float(dt),1.e-6)
            elif h > 0 and g in positions[h-1]:
                speeds[g] = (c-positions[h-1][g])/max(float(dt),1.e-6)
            else:
                speeds[g] = np.zeros(2, dtype=float)
        velocities.append(speeds)
    return velocities


def _interp_state(x_a, x_b, alpha):
    x = (1.0-alpha)*np.asarray(x_a, dtype=float)+alpha*np.asarray(x_b, dtype=float)
    angle = math.atan2(math.sin(float(x_b[3]-x_a[3])),
                       math.cos(float(x_b[3]-x_a[3])))
    x[3] = float(x_a[3])+alpha*angle
    return x


def first_contact(X, caches, dt, ego_L, ego_W, obstacle_vel, substeps=1):
    """Find the first time-aligned footprint contact with predicted polygons.

    The source MATLAB baseline checks polygon intersection at discrete times.
    Here optional intermediate samples reduce tunneling. At intermediate
    times, each predicted polygon is translated with the vertex-selector
    group velocity; no future obstacle polygon is applied at an earlier time.
    """
    trajectory = np.asarray(X, dtype=float)
    if trajectory.shape[0] != 4:
        trajectory = trajectory.T
    n = min(len(caches), trajectory.shape[1]-1)
    if n <= 0:
        return None
    ego_bound = math.hypot(float(ego_L), float(ego_W))*0.5
    subdivisions = max(1,int(substeps))
    for k in range(n):
        for s in range(subdivisions+1):
            if k>0 and s==0:
                continue # end of previous interval already checked
            alpha = float(s)/float(subdivisions)
            state = _interp_state(trajectory[:,k],trajectory[:,k+1],alpha)
            ego_poly = _ego_polygon(state,ego_L,ego_W)
            # Exactly use the next stage's geometry at the end of the
            # interval; otherwise move current polygon using its velocity.
            next_available = k+1<n
            if alpha==1. and next_available:
                obstacle_set = caches[k+1]
                velocity_map = obstacle_vel[k+1]
                shift_t = 0.0
            else:
                obstacle_set = caches[k]
                velocity_map = obstacle_vel[k]
                shift_t = alpha*float(dt)
            for g,poly,center,r in obstacle_set:
                v = velocity_map.get(g,np.zeros(2))
                center_at_t = center+v*shift_t
                if np.sum((np.asarray(state[:2])-center_at_t)**2)>(ego_bound+r)**2:
                    continue
                poly_at_t = poly+v*shift_t if shift_t>0. else poly
                if _polygon_clearance(ego_poly,poly_at_t)<=0.:
                    return dict(step=k,fraction=alpha,time=(k+alpha)*float(dt),
                                state=state,group=g,polygon=poly_at_t,
                                center=center_at_t,obstacle_velocity=v)
    return None


def _terminal_vrel(X, cache_last, obstacle_vel_last):
    """MATLAB no-collision cost uses obstacle relative speed at terminal step.

    Multiple obstacles: use the nearest obstacle to ego's terminal position.
    The supplied MATLAB source has only one obstacle, so this is an extension.
    """
    if not cache_last:
        return 0.0
    state=np.asarray(X,dtype=float)[:,-1]
    nearest=min(cache_last,key=lambda a:float(np.sum((a[2]-state[:2])**2)))
    return float(np.linalg.norm(_ego_vel(state)-obstacle_vel_last.get(nearest[0],np.zeros(2))))


def score_candidate(X,caches,obstacle_vel,dt,cfg,pars):
    """Parseh MATLAB's J=W1*Vrel+F(impact location), using first collision."""
    c=first_contact(X,caches,dt,float(cfg.ego_L),float(cfg.ego_W),
                    obstacle_vel,pars.collision_substeps)
    if c is None:
        vrel=_terminal_vrel(X,caches[-1] if caches else [],
                             obstacle_vel[-1] if obstacle_vel else {})
        # Literal MATLAB's no-collision cost: W1*Vrel_end.
        goal_term=0.0
        if pars.collision_free_goal_tiebreak>0:
            goal_term=float(pars.collision_free_goal_tiebreak)*float(
                np.sum((np.asarray(X,dtype=float)[:2,-1]-np.asarray([cfg.goal_x,cfg.goal_y]))**2))
        return float(pars.W1)*vrel+goal_term,dict(collided=False,
                      impact_surface=0, impact_severity=0.0,
                      relative_speed=vrel, collision_time=float('inf'),
                      impact_group=-1, impact_angle=float('nan'))
    F, surface, angle, center = _impact_info(
        c['state'],c['polygon'],int(getattr(cfg,'sector_id',1)),pars,
        float(cfg.ego_L),float(cfg.ego_W))
    vrel=float(np.linalg.norm(_ego_vel(c['state'])-c['obstacle_velocity']))
    score=float(pars.W1)*vrel+F
    return score,dict(collided=True,impact_surface=surface,
                      impact_severity=F,relative_speed=vrel,
                      collision_time=float(c['time']),impact_group=c['group'],
                      impact_angle=angle,impact_centroid=center)


def make_parseh_library(cfg,n,pars,initial_U=None,fallback_U=None):
    """Enumerate the MATLAB 9x5 constant controls in accel-major order.

    Note: MATLAB controls = steering_rate, acceleration_request; the robot's
    controls = commanded yaw_rate, acceleration. Numerical sample structure
    matches; the physical plant is not an exact dynamic bicycle reproduction.
    """
    n=int(n)
    if n<1:
        raise ValueError('planning horizon must be positive')
    a_lo=float(cfg.a_min); a_hi=float(cfg.a_max)
    if bool(getattr(cfg,'no_acc_flag',False)):
        a_lo=a_hi=0.0
    wlo=float(cfg.yawrate_min); whi=float(cfg.yawrate_max)
    if a_lo>a_hi or wlo>whi:
        raise ValueError('invalid control bounds')
    acc=np.linspace(a_lo,a_hi,max(1,int(pars.acceleration_samples)))
    steering=np.linspace(wlo,whi,max(1,int(pars.steering_samples)))
    candidates=[]
    for a in acc:
        for w in steering:
            u=np.zeros((2,n),dtype=float)
            u[0,:]=w
            u[1,:]=a
            candidates.append(u)
    if pars.include_mpc_seeds:
        for seed in (initial_U,fallback_U):
            if seed is None:
                continue
            try:
                u=np.asarray(seed,dtype=float).reshape(2,n).copy()
            except (ValueError,TypeError):
                continue
            if not np.isfinite(u).all():
                continue
            u[0,:]=np.clip(u[0,:],wlo,whi)
            u[1,:]=np.clip(u[1,:],a_lo,a_hi)
            candidates.append(u)
    return candidates


class ParsehHierarchy(RCMSHierarchy):
    """Keeps the existing avoidance validation and replaces mitigation stage."""
    def __init__(self,node,original_solve,pars=None):
        super(ParsehHierarchy,self).__init__(node,original_solve,pars or ParsehParameters())
        self.pars = pars or ParsehParameters()

    def _make_mitigation(self,x0,goal,kwargs,stages,error=None):
        from utils_linearized import PlanResult,rollout_ego
        cfg=self.node.cfg
        n=int(cfg.N)
        dt=float(cfg.dt_init)
        if not stages or not any(stage for stage in stages):
            raise RuntimeError('Parseh cannot evaluate a library with no perceived obstacles')
        pars=self.pars
        caches=_stage_cache(stages)
        velocities=_obstacle_velocities(caches,dt)
        library=make_parseh_library(cfg,n,pars,kwargs.get('initial_U'),kwargs.get('fallback_U'))
        t0=time.perf_counter()
        best=None
        scored=0
        failures=0
        total=len(library)
        for index,U in enumerate(library):
            if (time.perf_counter()-t0)*1000.0 > float(pars.max_library_ms):
                break
            try:
                # Use the actual I-MPC RK4+lag model, identical to output model.
                X=rollout_ego(x0,U,dt,cfg)
                score,detail=score_candidate(X,caches,velocities,dt,cfg,pars)
            except Exception:
                failures+=1
                continue
            if not np.isfinite(score):
                failures+=1
                continue
            scored+=1
            if best is None or score < best[0]:
                best=(float(score),index,U.copy(),X.copy(),detail)
        if best is None:
            raise RuntimeError('no finite Parseh candidate was evaluated')
        # A truncated library biases maneuver selection. Require the complete
        # sample set by default rather than claiming Parseh optimality.
        if scored != total and not bool(getattr(pars,'allow_partial_library',False)):
            raise RuntimeError('Parseh library incomplete: %d/%d (budget %.1f ms, failures=%d)'%(
                scored,total,float(pars.max_library_ms),failures))
        cost,index,U,X,detail=best
        clearance=minimum_clearance(X,stages,float(cfg.ego_L),float(cfg.ego_W))
        # Match the existing MPCControlSeq writer without falsely claiming
        # an avoidance result. The mode-aware publisher restores flags.
        plan=PlanResult(U=U,X=X,dt=dt,exitflag=0,cost=1.e9+cost,
                        vertex_set=kwargs.get('vertex_set'),
                        halfspaces=kwargs.get('halfspaces'),
                        slack=np.maximum(0.,max(float(getattr(cfg,'d_safe',0.)),0.)-clearance)*np.ones(n),
                        normal_slack=np.zeros(n),
                        vertex_types=kwargs.get('vertex_types'),
                        dynamic_halfspace_mask=kwargs.get('dynamic_halfspace_mask'))
        plan.sqp_iteration=0
        plan.trajectory_stage='parseh_mitigation'
        plan.J_global=1.e9+cost
        plan.global_cost=1.e9+cost
        plan.phi_min=clearance
        plan.min_clearance=clearance
        plan.original_ocp_feasible=False
        plan.zero_slack_feasible=False
        plan.selection_eligible=True
        plan.parseh_mode='MITIGATION'
        plan.parseh_library_cost=cost
        plan.parseh_best_index=index
        plan.parseh_library_evaluated=scored
        plan.parseh_library_total=total
        plan.parseh_first_contact=detail
        plan.parseh_compute_ms=(time.perf_counter()-t0)*1000.
        # Base hierarchy expects this key for its timing diagnostics.
        plan.rcms_compute_ms=plan.parseh_compute_ms
        return plan

    def solve(self, **kwargs):
        plan=super(ParsehHierarchy,self).solve(**kwargs)
        if self.mode=='AVOIDANCE':
            plan.trajectory_stage='parseh_avoidance'
            plan.parseh_mode='AVOIDANCE'
        elif self.mode=='MITIGATION':
            plan.trajectory_stage='parseh_mitigation'
            plan.parseh_mode='MITIGATION'
        else:
            plan.trajectory_stage='parseh_failed'
            plan.parseh_mode='FAILED'
        return plan
