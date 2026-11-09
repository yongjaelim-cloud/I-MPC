"""ctypes adapter for the standalone Parseh candidate evaluator.

IMPORTANT: Dynamics, severity scoring and surface convention MUST be checked
against the original parseh_hierarchy.py before replacing a robot controller.
"""
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

class _PHConfig(ct.Structure):
    _fields_ = [
        ('C',ct.c_int),('H',ct.c_int),('O',ct.c_int),('V',ct.c_int),('substeps',ct.c_int),
        ('dt',ct.c_float),('length',ct.c_float),('width',ct.c_float),
        ('margin',ct.c_float),('goal_x',ct.c_float),('goal_y',ct.c_float),
        ('goal_weight',ct.c_float),('vulnerable_penalty',ct.c_float),
        ('impact_weight',ct.c_float),('vulnerable_mask',ct.c_uint32),
        ('v_min',ct.c_float),('v_max',ct.c_float)]

@dataclass
class Evaluation:
    best_index: int
    scores: np.ndarray  # C x 8: cost,collision,time,surface,severity,closing_speed,obstacle,goal_cost
    trajectories: np.ndarray  # C x (H+1) x 4: x,y,speed,yaw
    backend: str

class MitigationEvaluator:
    def __init__(self, library=None):
        library = Path(library) if library else Path(__file__).with_name('libparseh_accel.so')
        self.lib = ct.CDLL(str(library.resolve()))
        f = self.lib.ph_run
        f.argtypes = [_PHConfig,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
                      ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_int)]
        f.restype = ct.c_int
        self._run = f
        best = self.lib.ph_run_best
        best.argtypes = [_PHConfig,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
                         ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_int)]
        best.restype = ct.c_int
        self._best = best
        self.lib.ph_cuda_available.restype=ct.c_int

    @property
    def cuda_compiled(self):
        return bool(self.lib.ph_cuda_available())

    def evaluate(self, initial, controls, vertices, counts, velocities, *,
                 dt, length=.6, width=.35, margin=0.0, collision_substeps=2,
                 goal=(0.,0.), goal_weight=1.0, sector_id,
                 vulnerable_penalty=1000., impact_weight=1., backend='auto',
                 v_min=-.2, v_max=.8, selected_only=True):
        """Evaluate candidate controls: [a, yaw_rate] in SI units.

        sector_id: required /setup sector (1..4); converted to native surface bits
        initial: [x, y, speed, yaw]
        controls: [candidates, horizon, 2]
        vertices: [horizon, obstacles, max_vertices, 2] world frame, padded
        counts: [obstacles], identical for all horizon stages
        velocities: [horizon, obstacles, 2] world-frame obstacle velocity
        backend: 'cpu', 'cuda', or 'auto'
        """
        # Native ABI retains its bitmask field; public configuration uses sector_id only.
        if sector_id not in (1, 2, 3, 4):
            raise ValueError('sector_id must be in 1..4')
        vulnerable_mask = {1: 2, 2: 4, 3: 8, 4: 1}[sector_id]
        initial=np.ascontiguousarray(initial,dtype=np.float32)
        controls=np.ascontiguousarray(controls,dtype=np.float32)
        vertices=np.ascontiguousarray(vertices,dtype=np.float32)
        counts=np.ascontiguousarray(counts,dtype=np.int32)
        velocities=np.ascontiguousarray(velocities,dtype=np.float32)
        if initial.shape!=(4,) or controls.ndim!=3 or controls.shape[-1]!=2:
            raise ValueError('initial must be (4,) and controls must be (C,H,2)')
        C,H,_=controls.shape
        if vertices.ndim!=4 or vertices.shape[0]!=H or vertices.shape[3]!=2:
            raise ValueError('vertices must have shape (H,O,V,2)')
        _,O,V,_=vertices.shape
        if counts.shape!=(O,) or velocities.shape!=(H,O,2) or np.any(counts<0) or np.any(counts>V):
            raise ValueError('counts=(O,) and velocities=(H,O,2) required')
        if C<1 or H<1 or O<1 or V<1 or collision_substeps<1:
            raise ValueError('nonempty candidates/horizon/obstacles and substeps>=1 required')
        if not all(np.all(np.isfinite(x)) for x in (initial,controls,vertices,velocities)):
            raise ValueError('inputs must be finite')
        if backend not in ('cpu','cuda','auto'):
            raise ValueError('backend must be cpu, cuda or auto')
        cfg=_PHConfig(C,H,O,V,int(collision_substeps),float(dt),float(length),float(width),
                      float(margin),float(goal[0]),float(goal[1]),float(goal_weight),
                      float(vulnerable_penalty),float(impact_weight),int(vulnerable_mask),float(v_min),float(v_max))
        if not float(v_min)<float(v_max):
            raise ValueError('v_min must be less than v_max')
        if selected_only:
            score=np.empty(8,dtype=np.float32)
            states=np.empty((H+1,4),dtype=np.float32)
            idx=ct.c_int(-1)
            code=self._best(cfg,initial.ctypes.data,controls.ctypes.data,vertices.ctypes.data,
                            counts.ctypes.data,velocities.ctypes.data,
                            {'cpu':0,'cuda':1,'auto':2}[backend],score.ctypes.data,
                            states.ctypes.data,ct.byref(idx))
            if code<0:
                raise RuntimeError('native best-only evaluation failed: %s' % code)
            return Evaluation(idx.value,score.reshape(1,8),states.reshape(1,H+1,4),
                              'cuda' if code==1 else 'cpu')
        scores=np.empty((C,8),dtype=np.float32)
        trajectories=np.empty((C,H+1,4),dtype=np.float32)
        idx=ct.c_int(-1)
        code=self._run(cfg,initial.ctypes.data,controls.ctypes.data,vertices.ctypes.data,
                       counts.ctypes.data,velocities.ctypes.data,
                       {'cpu':0,'cuda':1,'auto':2}[backend],scores.ctypes.data,
                       trajectories.ctypes.data,ct.byref(idx))
        if code<0:raise RuntimeError('native evaluation failed (code %s); CUDA may be unavailable'%code)
        return Evaluation(idx.value,scores,trajectories,'cuda' if code==1 else 'cpu')


def control_grid(horizon, steering_rates, accelerations):
    """Constant-control candidate library; output (C,H,2)."""
    return np.asarray([np.tile([a,w],(horizon,1))
                       for w in steering_rates for a in accelerations],dtype=np.float32)


def control_grid_speed(horizon, initial_speed, yaw_samples=100, speed_samples=50,
                       omega_min=-1.3, omega_max=1.3, v_min=-.2, v_max=.8, dt=.2):
    """5000 constant (acceleration, yaw-rate) candidates from target terminal speed.

    Acceleration = (v_target - initial_speed)/(horizon*dt). Velocity is
    clipped to [v_min,v_max] in the native rollout. This is not acceleration
    sampling; it honors the user's speed range.
    """
    omega=np.linspace(omega_min,omega_max,int(yaw_samples),dtype=np.float32)
    speed=np.linspace(v_min,v_max,int(speed_samples),dtype=np.float32)
    # acceleration-major convention: speed varies slowly across yaw rates.
    pair=np.stack(np.meshgrid((speed-np.clip(initial_speed,v_min,v_max))/(horizon*dt),
                               omega,indexing='ij'),axis=-1).reshape(-1,2)
    return np.ascontiguousarray(np.broadcast_to(pair[:,None,:],
                      (len(pair),horizon,2)),dtype=np.float32)
