# Avoidance-first Parseh Selector v2

## What changed

- `MPC_parseh0.py` ... `MPC_parseh3.py`: **avoidance-only**. Each instantiates the existing `MPC_planner{0..3}.AcadosMPCPlanner` with `cfg.mpc_flag=0`. No `ParsehHierarchy` imported or instantiated; no per-worker mitigation library.
- `mpc_parseh_runtime.py`: publishes `/mpc_parseh_control_seq{i}` only if the original solver explicitly reports BOTH `zero_slack_feasible=True` and `collision_free=True`; always emits `/mpc_parseh_status{i}` including success/failure and elapsed time. **If your original planner uses different status names, edit `_validated()` to map the real solver fields.** It deliberately rejects unverified results.
- `MPC_parseh_selector.py`: selects lowest-cost fresh validated avoidance. If all four workers have *fresh, completed failure statuses*, performs **one** Parseh C++/CUDA library evaluation. A missing or stale worker does NOT count as failure.
- `parseh_accel.py` + native C++/CUDA source: batched candidate rollout, polygon collision, first sampled contact and a **provisional** impact score. Only a CPU build was tested in this environment.

## Requirements / limits

**The provided files do not include original `MPC_planner0..3.py`, `parseh_hierarchy.py`, `MPCControlSeq.msg`, or the data layout of `/selected_vertex_horizon_data`.** Therefore it is NOT possible to verify the exact original Parseh/RCMS severity model, avoidance success flags, input grouping, or tracker message serialization. In particular:

1. The current C++/CUDA scorer is a **heuristic placeholder** (relative closing-speed squared, vulnerable face penalty, goal distance). This is **not a verified RCMS/Parseh equation**.
2. The native rollout assumes `[x,y,v,yaw]` and controls `[acceleration,yaw_rate]`; confirm the original MPC discretization/bounds.
3. Collision checking uses rectangular ego shape, convex obstacle SAT, sampled substeps and stagewise geometry; no continuous collision guarantee.
4. Mitigation always publishes `/mpc_parseh_mitigation_path` (`nav_msgs/Path`) and `/mpc_parseh_mitigation_states` (`Float32MultiArray`). The final `/mpc_parseh_control_seq` (`drone_msgs/MPCControlSeq`) is only published if an explicit, known flat-vector trajectory schema is detected in a cached worker message. **This does NOT yet guarantee compatibility with your tracker**. Add a concrete serializer after supplying `rosmsg show drone_msgs/MPCControlSeq`.
5. No automatic conversion from `/selected_vertex_horizon_data` is possible until its layout, obstacle group indices and object velocity interpretation are known. Publish the normalized topics below with your existing perception node or add a bridge.
6. `collision_free` is never marked true for a mitigation result on an unverified model.

## Install

Copy all `.py` files, `.cpp`, `.hpp`, `.cu`, and `build.sh` into the existing ROS script directory where `MPC_planner0.py` ... `MPC_planner3.py` are importable.

```bash
cd ~/ILL_MPC_workspace/src/transform/script
chmod +x build.sh MPC_parseh*.py
./build.sh
python3 test_backend.py
```

`build.sh` uses CUDA if `nvcc` is found; use `PARSEH_DISABLE_CUDA=1 ./build.sh` to force CPU. The compiled `.so` must remain alongside `parseh_accel.py`.

Start the four workers and selector separately with their normal ROS environment:

```bash
python3 MPC_parseh0.py &
python3 MPC_parseh1.py &
python3 MPC_parseh2.py &
python3 MPC_parseh3.py &
python3 MPC_parseh_selector.py
```

Avoidance reports:

```bash
rostopic echo /mpc_parseh_status0
rostopic echo /mpc_parseh_status3
rostopic echo /mpc_parseh_mode
```

Status message format (data vector): `[planner_id, cycle_number, avoidance_success, Unix_time_seconds, wall_time_ms]`.

## Mandatory normalized mitigation obstacle input

The selector expects world-frame `map` geometry on these three topics:

- `/parseh/obstacle_vertices`, `std_msgs/Float32MultiArray`: `layout.dim.size = [H,O,V,2]`, flatten `vertices[h,o,v,xy]`. `O` = number of convex obstacles; `V` = padded maximum vertices per polygon; `H` must match `~mitigation_horizon` (default 20).
- `/parseh/obstacle_counts`, `std_msgs/Int32MultiArray`: length `O`, number of valid vertices for each obstacle (>=1, <=V). Convex vertices ordered along the polygon.
- `/parseh/obstacle_velocities`, `std_msgs/Float32MultiArray`: `layout.dim.size = [H,O,2]`, flatten world `vx,vy` for each predicted obstacle at each stage.

For example, in the perception bridge (where `vertices` and `velocities` are already calculated in `map`):

```python
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension

def array_message(a, labels):
    msg = Float32MultiArray()
    msg.layout.dim = [MultiArrayDimension(label=label, size=int(n), stride=int(a.size // max(1, int(__import__('numpy').prod(a.shape[:idx])))))
                      for idx, (label, n) in enumerate(zip(labels, a.shape))]
    msg.data = a.astype('float32').ravel().tolist()
    return msg

vertices_pub.publish(array_message(vertices_HOV2, ['horizon','obstacles','vertices','xy']))
counts_pub.publish(Int32MultiArray(data=counts_O.tolist()))
velocities_pub.publish(array_message(velocities_HO2, ['horizon','obstacles','xy']))
```

Pose: `/tracked_pose` `geometry_msgs/PoseStamped` in `map` frame. Speed: `/cmd_vel` `geometry_msgs/Twist.linear.x`. Goal: `/move_base_simple/goal` `geometry_msgs/PoseStamped` in `map` frame. If your actual message types differ, update the specific callback/subscriber; ROS will not deliver mismatched message types.

## Selector parameters

- `~max_age_s`: 0.45; freshness requirement for all worker statuses.
- `~max_geometry_age_s`: 0.3.
- `~backend`: `auto`/`cpu`/`cuda`; in auto mode CUDA is used only above the native workload threshold.
- `~mitigation_horizon`: 20, `~mitigation_dt`: 0.1.
- `~steering_samples`, `~acceleration_samples`: 15 each.
- `~omega_min/max`: -.6/.6, `~accel_min/max`: -.5/.5.
- `~ego_length/width`: .6/.35.
- `~collision_substeps`: 3.
- `~vulnerable_mask`: 1 (bit 0 front, bit 1 left, bit 2 rear, bit 3 right, **not necessarily your original sector numbering**).

## Verification sequence

1. Run **one avoidance worker** first; confirm it publishes a validated trajectory and a status success=1 under a known collision-free scenario. If not, inspect `node.planner.last_result` and map its actual flags in `_validated()`.
2. Run four workers and selector; confirm selector mode `AVOIDANCE` and no mitigation diagnostics messages.
3. Force four **fresh** failure statuses while providing normalized obstacle arrays, pose, velocity and goal; confirm one batch evaluation and publication to `/mpc_parseh_mitigation_path`.
4. Verify scores/first contact and trajectories against original Parseh using recorded rosbag before vehicle deployment. Then map concrete MPCControlSeq fields and perception layout.

Do not execute on the physical platform until the geometry, prediction, dynamics, ranking and tracker message contracts are verified.
