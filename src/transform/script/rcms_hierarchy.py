#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hierarchical avoidance -> RCMS-inspired risk mitigation for I-MPC.

The normal AcadosMPCPlanner is stage 1. Stage 2 is invoked ONLY if no
successful collision-free, zero-avoidance-slack candidate is found.

This module deliberately does NOT use designated vulnerable body sectors:
RCMS is a sector-agnostic collision mitigation baseline.

Input: vertex_selector_exp /selected_vertex_data and horizon tensor, already
parsed by MPC_planner0..3 into stage_vertex_set / feature_obs_indices.
Output: utils_linearized.PlanResult with U[0,:]=yaw-rate, U[1,:]=acceleration.

Approximation disclosure: risk formulation (Gaussian spatial risk, closing/TTCE
and adaptive control cost) is RCMS-inspired, not an exact port of the paper's
8-state car model or of the user's MATLAB baseline. This uses the deployed
I-MPC five-state yaw-rate-lag model for controls and trajectory publication.
"""

import math
import time
from dataclasses import dataclass
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # still retain deterministic emergency-search fallback
    minimize = None


@dataclass
class RiskParameters:
    alpha_g: float = 0.3
    alpha_s: float = 4.0
    beta_l: float = 1.0
    beta_w: float = 1.5
    radius_floor: float = 0.15
    risk_scale: float = 12.0
    impact_speed_scale: float = 4.0
    terminal_speed_scale: float = 1.5
    goal_scale: float = 0.012
    control_yaw_scale: float = 0.03
    control_acc_scale: float = 0.008
    smooth_scale: float = 0.03
    max_opt_ms: float = 40.0
    max_iterations: int = 14
    knots: int = 4
    slack_tol: float = 1.0e-3
    clearance_tol: float = 1.0e-3
    extra_clearance: float = 0.0
    require_regulation_zero_slack: bool = False


def _hull(points):
    a = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(a) < 3:
        return a
    a = np.unique(a, axis=0)
    if len(a) < 3:
        return a
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    def cross(o, p, q):
        return (p[0]-o[0])*(q[1]-o[1])-(p[1]-o[1])*(q[0]-o[0])
    low, high = [], []
    for p in a:
        while len(low) >= 2 and cross(low[-2], low[-1], p) <= 1.e-12:
            low.pop()
        low.append(p)
    for p in a[::-1]:
        while len(high) >= 2 and cross(high[-2], high[-1], p) <= 1.e-12:
            high.pop()
        high.append(p)
    return np.asarray(low[:-1] + high[:-1], dtype=float)


def _valid_polygon(points, radius_floor):
    p = _hull(points)
    if len(p) >= 3:
        return p
    if len(p) == 2:
        d = p[1] - p[0]
        length = max(float(np.linalg.norm(d)), 1.e-9)
        t = radius_floor * np.array([-d[1], d[0]]) / length
        return np.array([p[0]-t, p[1]-t, p[1]+t, p[0]+t])
    if len(p) == 1:
        x, y = p[0]
        r = radius_floor
        return np.array([[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]])
    return np.zeros((0,2), dtype=float)


def _point_to_edges_dist(a, edges):
    if len(a) == 0 or len(edges) == 0:
        return float('inf')
    p = np.asarray(a, dtype=float)[:, None, :]
    s = edges[None, :, :]
    e = np.roll(edges, -1, axis=0)[None, :, :]
    d = e - s
    z = np.sum(d*d, axis=2)
    t = np.clip(np.sum((p-s)*d, axis=2) / np.maximum(z, 1.e-15), 0., 1.)
    projected = s + t[:, :, None]*d
    return float(np.sqrt(np.min(np.sum((p-projected)**2, axis=2))))


def _polygon_clearance(a, b):
    """Signed clearance for convex polygons: <=0 if touching/overlapping."""
    if len(a) < 3 or len(b) < 3:
        return float('inf')
    # Conservative penetration detection via SAT, exact Euclidean distance when disjoint.
    for poly in (a,b):
        edges = np.roll(poly, -1, axis=0)-poly
        axes = np.column_stack((-edges[:,1], edges[:,0]))
        axes /= np.maximum(np.linalg.norm(axes,axis=1,keepdims=True), 1.e-12)
        projection_a = a @ axes.T
        projection_b = b @ axes.T
        sep = np.maximum(np.min(projection_b,axis=0)-np.max(projection_a,axis=0),
                         np.min(projection_a,axis=0)-np.max(projection_b,axis=0))
        if np.any(sep > 0.):
            return min(_point_to_edges_dist(a,b),_point_to_edges_dist(b,a))
    return -1.e-6


def _ego_polygon(state, length, width):
    x, y, _, yaw = state[:4]
    c,s = math.cos(float(yaw)), math.sin(float(yaw))
    local = np.array([[length/2.,width/2.],[-length/2.,width/2.],
                      [-length/2.,-width/2.],[length/2.,-width/2.]])
    return local @ np.array([[c,s],[-s,c]]) + np.array([x,y])


def build_obstacle_stages(stage_vertex_set, group_ids, n, radius_floor=0.15):
    """Group selector vertices by convex-obstacle group id at each stage.

    The selector's obs_vxs/obs_vys are feature NORMALS, not obstacle speeds.
    Future obstacle motion comes ONLY from selected_vertex_horizon_data.
    """
    v = np.asarray(stage_vertex_set, dtype=float)
    if v.ndim == 2:
        v = np.repeat(v[None,:,:],n,axis=0)
    if v.ndim != 3 or v.shape[2] != 2:
        raise ValueError('stage_vertex_set must have shape H x K x 2')
    if v.shape[0] < n:
        v = np.concatenate((v,np.repeat(v[-1:,:,:],n-v.shape[0],axis=0)),axis=0)
    groups = np.asarray(group_ids,dtype=int).ravel()
    k = min(v.shape[1],len(groups))
    v = v[:n,:k,:]
    groups = groups[:k]
    good_indices = np.where(groups>=0)[0]
    unique_groups = np.unique(groups[good_indices])
    stages = []
    for h in range(n):
        stage = []
        for g in unique_groups:
            pts = v[h,(groups==g),:]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] == 0:
                continue
            polygon = _valid_polygon(pts,radius_floor)
            if len(polygon) >= 3:
                stage.append((int(g),polygon))
        stages.append(stage)
    return stages


def minimum_clearance(states, stages, ego_L, ego_W):
    """Conservative swept footprint check between horizon states.

    Each interval checks start/mid/end ego states against both the last and
    predicted obstacle polygons, covering translations missed by one sample.
    This is conservative; a continuously deforming obstacle is not modeled.
    """
    X = np.asarray(states,dtype=float)
    if X.shape[0] != 4:
        X = X.T
    worst = float('inf')
    for h, stage in enumerate(stages):
        if h+1 >= X.shape[1]:
            break
        # Include last and next obstacle shapes to account for sensor motion.
        polygons=[poly for _,poly in stage]
        if h>0:
            polygons.extend(poly for _,poly in stages[h-1])
        if not polygons:
            continue
        for alpha in (0.,0.5,1.):
            state=(1.-alpha)*X[:,h]+alpha*X[:,h+1]
            # Correct angular wrap across +/- pi.
            dyaw=math.atan2(math.sin(X[3,h+1]-X[3,h]),
                            math.cos(X[3,h+1]-X[3,h]))
            state[3]=X[3,h]+alpha*dyaw
            ego=_ego_polygon(state,ego_L,ego_W)
            for poly in polygons:
                worst=min(worst,_polygon_clearance(ego,poly))
                if worst <= 0.:
                    return worst
    return worst


def risk_tensors(stages, dt, ego_L, ego_W, cfg, pars):
    """Padded arrays H x O for vectorized RCMS-like risk evaluation."""
    n = len(stages)
    keys = sorted(set(g for stage in stages for g,_ in stage))
    oi = {g:i for i,g in enumerate(keys)}
    m = len(keys)
    center = np.zeros((n,m,2))
    sigma = np.ones((n,m,2))
    mask = np.zeros((n,m),dtype=bool)
    for h,stage in enumerate(stages):
        for g,p in stage:
            j = oi[g]
            c = np.mean(p,axis=0)
            extent = np.max(p,axis=0)-np.min(p,axis=0)
            center[h,j]=c
            sigma[h,j,0]=max(pars.beta_l*(0.5*extent[0]+0.5*ego_L),pars.radius_floor)
            sigma[h,j,1]=max(pars.beta_w*(0.5*extent[1]+0.5*ego_W),pars.radius_floor)
            mask[h,j]=True
    vel=np.zeros_like(center)
    if n > 1:
        delta = (center[1:,:,:]-center[:-1,:,:])/max(dt,1.e-9)
        valid = mask[1:,:] & mask[:-1,:]
        vel[1:,:,:]=np.where(valid[:,:,None],delta,0.)
        vel[0,:,:]=vel[1,:,:]
    return center,sigma,vel,mask


def rollout_risk_model(x0,U,dt,cfg):
    """Fast approximate yaw-lag rollout for optimizing, not for publishing.

    Publication always uses utils_linearized.rollout_ego (native RK4).
    """
    U=np.asarray(U,dtype=float).reshape(2,-1)
    N=U.shape[1]
    X=np.zeros((4,N+1),dtype=float)
    X[:,0]=np.asarray(x0,dtype=float).reshape(-1)[:4]
    w=float(getattr(cfg,'ego_yawrate0',0.) or 0.)
    tau=max(float(getattr(cfg,'yawrate_time_constant',0.7) or 0.7),1.e-3)
    a=math.exp(-dt/tau)
    vmin=float(getattr(cfg,'v_min',0.)); vmax=float(getattr(cfg,'v_max',1.))
    for k in range(N):
        x,y,v,yaw=X[:,k]
        wnext=a*w+(1.-a)*float(U[0,k])
        yaw_next=yaw+0.5*dt*(w+wnext)
        vnext=float(np.clip(v+dt*float(U[1,k]),vmin,vmax))
        X[0,k+1]=x+0.5*dt*(v*math.cos(yaw)+vnext*math.cos(yaw_next))
        X[1,k+1]=y+0.5*dt*(v*math.sin(yaw)+vnext*math.sin(yaw_next))
        X[2,k+1]=vnext
        X[3,k+1]=yaw_next
        w=wnext
    return X


def _sigmoid(x):
    return 1. / (1.+np.exp(-np.clip(x,-60.,60.)))


def evaluate_risk(U,x0,goal,dt,cfg,risk_data,pars,ref_U=None):
    """RCMS-inspired risk with closing-velocity, TTCE, and impact-speed cost."""
    X=rollout_risk_model(x0,U,dt,cfg)
    ctr,sig,obs_vel,active=risk_data
    n=U.shape[1]
    if active.size == 0 or not np.any(active):
        risk=0.; contact=0.; kappa=0.; ttce=float('inf')
    else:
        ego_xy = X[:2,1:].T[:,None,:]
        diff = ctr-ego_xy  # obstacle relative to ego
        mahal = np.sum((diff/np.maximum(sig,1.e-3))**2,axis=2)
        psi = 1. / (pars.alpha_g+mahal)
        ego_vel = np.stack((X[2,1:]*np.cos(X[3,1:]),
                            X[2,1:]*np.sin(X[3,1:])),axis=1)[:,None,:]
        rel_v = obs_vel-ego_vel
        closing = -np.sum(diff*rel_v,axis=2)/np.maximum(np.linalg.norm(diff,axis=2),0.1)
        # Fix the sign convention: closing > 0 means agents approach.
        direction = _sigmoid(pars.alpha_s*closing)
        density = np.exp(-0.5*np.minimum(mahal,120.))
        rho = psi*(0.3+0.7*direction)
        risk = float(np.sum(np.where(active,rho,0.)))
        closing_sq = np.maximum(closing,0.)**2
        contact = float(np.sum(np.where(active,density*closing_sq,0.)))
        kappa = float(np.sum(np.where(active[:1],density[:1],0.)))
        r0=diff[0]
        v0=rel_v[0]
        approach=-np.sum(r0*v0,axis=1)
        dtc=np.where(approach>0., approach/np.maximum(np.sum(v0*v0,axis=1),1.e-8),np.inf)
        ttce=float(np.min(np.where(active[0],dtc,np.inf)))
    u=np.asarray(U,dtype=float)
    adaptive = 1./max(0.10, kappa + (0.0 if not np.isfinite(ttce) else 1./(1.+ttce)))
    controls = float(pars.control_yaw_scale*np.sum(u[0]**2)+
                     pars.control_acc_scale*np.sum(u[1]**2))*adaptive
    smooth = float(pars.smooth_scale*np.sum(np.diff(u,axis=1)**2))
    speed = pars.terminal_speed_scale*float(X[2,-1]**2)
    goal_term = pars.goal_scale*float(np.sum((X[:2,-1]-goal)**2))
    ref = 0. if ref_U is None else .001*float(np.sum((u-ref_U)**2))
    return (pars.risk_scale*risk+pars.impact_speed_scale*contact+
            controls+smooth+speed+goal_term+ref),X


class RCMSHierarchy:
    def __init__(self,node,original_solve,pars=None):
        self.node=node
        self.original_solve=original_solve
        self.pars=pars or RiskParameters()
        self.previous_rcms_U=None
        self.mode='AVOIDANCE'
        self.stats={}
        self.recovery_count=0
        self.recovery_cycles=1

    def _candidate_safe(self,cand,stages,x0):
        """Accept avoidance by solver validity and avoidance slack only.

        stages/x0 remain in the signature for compatibility with runtime timers.
        No extra rollout, polygon distance, or impact-normal check is performed.
        """
        cfg=self.node.cfg
        if bool(getattr(cand,'is_order_fallback',False)):
            return False,float('-inf')
        if int(getattr(cand,'exitflag',-1)) < 0:
            return False,float('-inf')
        U=np.asarray(getattr(cand,'U',[]),dtype=float)
        if U.shape != (2,int(cfg.N)) or not np.all(np.isfinite(U)):
            return False,float('-inf')
        sl=np.asarray(getattr(cand,'slack',[]),dtype=float).ravel()
        if sl.size != int(cfg.N) or not np.all(np.isfinite(sl)):
            return False,float('-inf')
        if np.max(sl)>self.pars.slack_tol:
            return False,float('-inf')
        if int(getattr(cfg,'mpc_flag',4)) in (0,1,2,3):
            surface_slack=np.asarray(getattr(cand,'surface_slack',[]),dtype=float)
            if not surface_slack.size or not np.all(np.isfinite(surface_slack)):
                return False,float('-inf')
            if np.max(surface_slack)>self.pars.slack_tol:
                return False,float('-inf')
        # Preserve existing MPC field diagnostics; do not substitute polygon distance.
        return True,float(getattr(cand,'phi_min',float('nan')))

    def _make_mitigation(self,x0,goal,kwargs,stages,error=None):
        from utils_linearized import PlanResult,rollout_ego
        n=int(self.node.cfg.N); cfg=self.node.cfg; dt=float(cfg.dt_init)
        pars=self.pars
        data=risk_tensors(stages,dt,float(cfg.ego_L),float(cfg.ego_W),cfg,pars)
        def bound_controls(U):
            U=np.asarray(U,dtype=float).reshape(2,n).copy()
            U[0]=np.clip(U[0],float(cfg.yawrate_min),float(cfg.yawrate_max))
            U[1]=np.clip(U[1],float(cfg.a_min),float(cfg.a_max))
            if bool(getattr(cfg,'no_acc_flag',False)):
                U[1,:]=0.
            return U
        yaw_max=float(cfg.yawrate_max);yaw_min=float(cfg.yawrate_min)
        acc_min=float(cfg.a_min)
        Ubase=kwargs.get('initial_U')
        seeds=[]
        if Ubase is not None:
            try: seeds.append(bound_controls(Ubase))
            except Exception: pass
        fallback=kwargs.get('fallback_U')
        if fallback is not None:
            try: seeds.append(bound_controls(fallback))
            except Exception: pass
        if self.previous_rcms_U is not None:
            p=self.previous_rcms_U
            if p.shape == (2,n):
                seeds.append(bound_controls(np.column_stack((p[:,1:],p[:,-1]))))
        for yawrate in (0.,yaw_max,yaw_min):
            U=np.zeros((2,n))
            U[0,:]=yawrate;U[1,:]=acc_min
            seeds.append(bound_controls(U))
        # Rate-and-velocity constrained backup: no arbitrary unsafe seed.
        best_U=None;best_val=float('inf')
        for seed in seeds:
            val,_=evaluate_risk(seed,x0,goal,dt,cfg,data,pars)
            if np.isfinite(val) and val<best_val:
                best_val,best_U=float(val),seed.copy()
        if best_U is None:
            raise RuntimeError('No finite mitigation candidate')
        time0=time.perf_counter()
        used_optimizer=False
        nknots=max(1,min(pars.knots,n))
        split=np.array_split(np.arange(n),nknots)
        def expand(z):
            u=np.empty((2,n),dtype=float)
            for j,cols in enumerate(split):
                u[0,cols]=z[j]
                u[1,cols]=z[j+nknots]
            return bound_controls(u)
        def compress(U):
            return np.array([np.mean(U[0,cols]) for cols in split]+
                            [np.mean(U[1,cols]) for cols in split],dtype=float)
        z0=compress(best_U)
        bounds=[(yaw_min,yaw_max)]*nknots + [
            (float(cfg.a_min),float(cfg.a_max)) if not bool(getattr(cfg,'no_acc_flag',False)) else (0.,0.)
        ]*nknots
        def objective(z):
            nonlocal best_val,best_U
            if (time.perf_counter()-time0)*1000.>pars.max_opt_ms:
                raise TimeoutError('RCMS mitigation optimization budget exceeded')
            candidate=expand(z)
            val,_=evaluate_risk(candidate,x0,goal,dt,cfg,data,pars)
            if np.isfinite(val) and val<best_val:
                best_val,best_U=float(val),candidate.copy()
            return val
        if minimize is not None and pars.max_iterations>0:
            try:
                res=minimize(objective,z0,method='L-BFGS-B',bounds=bounds,
                             options={'maxiter':int(pars.max_iterations),'maxfun':150,
                                      'ftol':1.e-5})
                used_optimizer=bool(res.success)
                objective(res.x)
            except (TimeoutError,ValueError,RuntimeError,OverflowError):
                pass
        # True C++ RK4 yaw-lag rollout is authoritative for publication.
        X=rollout_ego(x0,best_U,dt,cfg)
        final_clearance=minimum_clearance(X,stages,float(cfg.ego_L),float(cfg.ego_W))
        self.previous_rcms_U=best_U.copy()
        # Use a high outer-hierarchy rank: across four planners EVERY verified
        # avoidance trajectory should outrank EVERY mitigation trajectory.
        plan=PlanResult(U=best_U,X=X,dt=dt,exitflag=0,cost=1.e9+best_val,
                        vertex_set=kwargs.get('vertex_set'),
                        halfspaces=kwargs.get('halfspaces'),
                        slack=np.maximum(0.,max(float(getattr(cfg,'d_safe',0.)),0.)-final_clearance)*np.ones(n),
                        normal_slack=np.zeros(n),
                        vertex_types=kwargs.get('vertex_types'),
                        dynamic_halfspace_mask=kwargs.get('dynamic_halfspace_mask'))
        plan.sqp_iteration=0
        plan.trajectory_stage='rcms_mitigation'
        plan.J_global=1.e9+best_val
        plan.global_cost=1.e9+best_val
        plan.rcms_raw_risk_cost=float(best_val)
        plan.phi_min=final_clearance
        plan.min_clearance=final_clearance
        plan.original_ocp_feasible=False
        plan.zero_slack_feasible=False
        plan.selection_eligible=True  # Executable fallback, NOT a collision-free claim.
        plan.rcms_mode='MITIGATION'
        plan.rcms_solver_success=used_optimizer
        plan.rcms_compute_ms=(time.perf_counter()-time0)*1000.
        return plan

    def solve(self, **kwargs):
        """Called instead of the normal AcadosMPCPlanner.solve by the ROS node."""
        previous_mode=self.mode
        self.mode='FAILED'
        self.stats={'mode':'FAILED','clearance':float('-inf'),
                    'avoidance_candidates':0,'optimization_ms':0.}
        self.node.planner.last_iteration_results=[]
        cfg=self.node.cfg; n=int(cfg.N)
        x0=np.asarray(kwargs['x0'],dtype=float).reshape(4)
        goal=np.asarray(kwargs.get('original_goal',kwargs['goal']),dtype=float).reshape(2)
        groups=np.asarray(kwargs.get('feature_obs_indices',np.zeros(0)),dtype=int).ravel()
        vertices=kwargs.get('stage_vertex_set')
        if vertices is None:
            vertices=kwargs.get('vertex_set',np.zeros((0,2)))
        stages=None  # Polygon geometry is only needed if mitigation is invoked.
        plan=None;avoidance_error=None
        try:
            plan=self.original_solve(**kwargs)
        except Exception as exc:
            avoidance_error=exc
            self.node.planner.last_iteration_results=[]
        iteration_plans=list(getattr(self.node.planner,'last_iteration_results',[]) or [])
        if plan is not None and not iteration_plans:
            iteration_plans=[plan]
        feasible=[]
        for cand in iteration_plans:
            ok,clearance=self._candidate_safe(cand,stages,x0)
            if ok:
                cand.rcms_mode='AVOIDANCE'
                cand.trajectory_stage='rcms_avoidance'
                cand.phi_min=clearance
                feasible.append(cand)
        # Enter mitigation immediately upon avoidance failure. Optional
        # recovery hysteresis is disabled by default (recovery_cycles=1).
        if feasible and previous_mode=='MITIGATION':
            self.recovery_count+=1
            if self.recovery_count<self.recovery_cycles:
                feasible=[]
        elif feasible:
            self.recovery_count=self.recovery_cycles
        else:
            self.recovery_count=0
        if feasible:
            # Select lowest ranking cost among zero-avoidance-slack candidates.
            def rank(p):
                global_cost=float(getattr(p,'J_global',float('inf')))
                if not np.isfinite(global_cost):
                    global_cost=float(getattr(p,'cost',float('inf')))
                return global_cost
            winner=min(feasible,key=rank)
            self.mode='AVOIDANCE'
            self.recovery_count=self.recovery_cycles
            self.stats={'mode':self.mode,'clearance':float(winner.phi_min),
                        'avoidance_candidates':len(feasible),'optimization_ms':0.}
        else:
            try:
                stages=build_obstacle_stages(vertices,groups,n,self.pars.radius_floor)
                winner=self._make_mitigation(x0,goal,kwargs,stages,avoidance_error)
                self.mode='MITIGATION'
            except Exception as exc:
                # In no circumstances call a trajectory with unknown validity 'safe'.
                # Return source planner's fallback when its compilation/runtime fails.
                if plan is None:
                    raise RuntimeError('RCMS mitigation failed: {} (avoidance: {})'.format(exc,avoidance_error))
                winner=plan
                winner.is_order_fallback=True
                winner.rcms_mode='FAILED'
                self.mode='FAILED'
            self.stats={'mode':self.mode,'clearance':float(getattr(winner,'phi_min',-np.inf)),
                        'avoidance_candidates':0,
                        'optimization_ms':float(getattr(winner,'rcms_compute_ms',0.))}
        # Base MPC_planner.step selects the LAST successful iteration. Install
        # only the chosen hierarchical plan so that it cannot be overwritten.
        self.node.planner.last_iteration_results=[winner]
        self.node.planner.last_feasible_candidates=feasible
        self.node.planner.last_result=winner
        return winner
