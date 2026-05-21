# Configurations

All configs are YAML files in `configs/`, loaded by `src/utils/config.py::load_config(name)`.

---

## `configs/env.yaml` — Environment & Task Constants

### Board Geometry
```yaml
table_center_xy: [0.88, 0.2641]   # World XY of table center (matches floor/robot Y alignment)
table_half_x: 0.28                 # Half-width in X (56cm total)
table_half_y: 0.28                 # Half-width in Y (56cm total)
table_surface_z: 0.400             # Z of table top surface (meters)
edge_margin: 0.04                  # Min distance from any edge for object/goal sampling
min_goal_dist: 0.10                # Min distance between object and goal positions
```

### Z-Levels
```yaml
cube_height: 0.030                 # Cube side length (30mm)
cube_z: 0.415                      # Cube CoM Z when on table: TABLE_SURFACE_Z + cube_height/2
grasp_z: 0.425                     # Grip site Z during grasping: ~6.5mm above cube top, 78% overlap
hover_z: 0.460                     # RL stop height above cube: safe min given RL variance + finger clearance
safe_z: 0.550                      # Transit height: high enough to clear any piece on the table
```

**Z-level rationale:**
- `cube_z = 0.400 + 0.015 = 0.415`: Cube sits on table with CoM at half-height.
- `grasp_z = 0.425`: Grip site at 25mm above table (cube top at 0.430, grip site 5mm below top = 78% finger overlap with cube height).
- `hover_z = 0.460`: RL descend target. 60mm above cube top = safe stopping distance before scripted grasp. Chosen to accommodate RL variance of ±15mm.
- `safe_z = 0.550`: 15cm above table surface. Enough clearance to transit over any chess piece (even queens ~8cm tall).

### Success & Reward
```yaml
success_threshold: 0.010           # 10mm radius for success detection
success_bonus: 500.0               # Reward for reaching goal
crash_penalty: -500.0              # Penalty for collision/drift/fault
drift_limit_start: 0.100           # Initial tube radius for descend/ascend (curriculum start)
drift_limit_end: 0.010             # Final tube radius after curriculum (1cm)
floor_limit: 0.400                 # Minimum Z for transit (= table surface Z)
```

### Dense Reward Weights
```yaml
dist_reward_weight: 1.0            # Distance-to-goal multiplier
z_reward_weight: 1.5               # Extra penalty for height error (vertical precision)
xy_reward_weight: 2.0              # Extra penalty for XY error
jitter_penalty_weight: 0.003       # Penalty for large action magnitudes (prevents jitter)
floor_penalty: -0.5                # Constant penalty when too close to floor/table
braking_reward_weight: 0.15        # Penalty for speed when near goal (encourages stopping)
braking_dist: 0.010                # Distance at which braking penalty activates
floor_proximity_threshold: 0.025   # Z-above-floor to trigger floor penalty
stability_vel_threshold: 0.05      # Max m/s for arm to be considered stable at goal
```

### Gripper Constants
```yaml
finger_open_joint: 0.0181          # Joint position = fully open (18.1mm from center each side = 36.2mm total gap)
finger_closed_joint: 0.0000        # Joint position = fully closed (commanded; stalls at ~0.0141 with cube)
finger_outer_offset: 0.033         # Distance from grip center to outer finger edge (for drift footprint check)
max_gripper_width: 0.05            # Max joint range (not the gap itself)
```

> **Note**: The actual "closed" stall position when grasping a 30mm cube is j≈0.0141 (not 0.0), because the cube resists at Kp×j = 150,000×0.0141 ≈ 2,115 N of contact force. The controller commands 0.0 but the cube stiffness prevents further closure.

### Grasp Stage Parameters
```yaml
grasp_contact_approach_tolerance: 0.001   # 1mm fine descent tolerance
grasp_close_steps: 150                    # Steps for linear finger ramp (OPEN → CLOSED)
grasp_hold_steps: 50                      # Hold steps after close to settle physics
grasp_verify_xy_threshold: 0.015          # Max cube-to-grip XY error to confirm hold (15mm)
grasp_verify_z_threshold: 0.020           # Max cube-to-grip Z error to confirm hold (20mm)
grasp_verify_finger_threshold: 0.016      # Must be ABOVE stall j≈0.0141 — not fully closed empty
cube_held_xy_limit: 0.030                 # Drop detection: >30mm XY drift = dropped
cube_held_z_limit: 0.020                  # Drop detection: >20mm Z deviation = dropped
```

### Pose & Misc
```yaml
hidden_object_pos: [2.0, 2.0, 0.015]     # Off-board position for "pure movement" (RL without cube)
home_position_xy: [0.680, 0.2641]         # Default transit start position (within arm reach, at table Y center)
halt_vel_threshold: 0.0005                # 0.5 mm/s — arm is "stopped" if below this speed
eval_drift_limit: 0.010                   # Stricter 1cm drift limit for evaluation (vs. curriculum end)
sample_debug_freq: 50                     # Enable verbose debug logs every 50 episodes
```

---

## `configs/physics.yaml` — Simulation & Settling Parameters

```yaml
max_settle_steps: 500              # Max steps to wait for arm to settle to start position
settle_tolerance: 0.003            # 3mm convergence threshold for _move_mocap_to (default)
settle_gain: 0.3                   # (Unused in current code — was a proportional gain for an older method)
vertical_quat: [1.0, 0.0, 1.0, 0.0]  # Pre-normalization quaternion for vertical orientation
initial_qpos: [-0.05, 0.00]        # Initial slide0 (X) and slide1 (Y) offsets for robot base
env_setup_steps: 10                # Steps to run in _env_setup before first episode
max_goal_retries: 100              # Goal sampling retries to find MIN_GOAL_DIST separation
pos_ctrl_scale: 0.015              # Max m per RL step (scales RL action to mocap delta)
settle_steps_final: 25             # (Unused in current code — was a post-settle step count)
```

### Vertical Quaternion

`[1.0, 0.0, 1.0, 0.0]` is normalized to `[0.707, 0, 0.707, 0]` (w=0.707, x=0, y=0.707, z=0). This represents a 90° rotation around the Y axis, which when applied to the gripper's default orientation (pointing forward in +X) rotates the grip site to point straight down (-Z). This is the "crane mode" orientation.

### `initial_qpos`

The two values `[-0.05, 0.00]` are applied to `self.initial_qpos[0]` and `self.initial_qpos[1]` in `_env_setup`. These are the initial joint positions of `robot0:slide0` (X slide) and `robot0:slide1` (Y slide). With `-0.05` in X, the robot base effectively starts 5cm in the -X direction relative to its XML-defined position, shifting the arm's reachable workspace slightly away from the table's near edge.

---

## `configs/training.yaml` — SAC Hyperparameters

```yaml
total_timesteps: 1000000           # Total training steps
num_envs: 8                        # Parallel training environments (SubprocVecEnv)
base_model: "models/sac-FetchPickAndPlace-v4.zip"  # Pretrained FetchPickAndPlace weights (starting point)
target_curriculum_total: 500000    # Steps over which drift curriculum tightens from start to end
learning_rate: 0.00005             # SAC learning rate (very low — fine-tuning existing weights)
batch_size: 512                    # SAC replay buffer batch size
target_entropy: -4.0               # SAC target entropy (action dim = 4)
learning_starts: 10000             # Steps before first gradient update
initial_ent_coef: 0.1              # Reset entropy coefficient at training start (jumpstart exploration)
ent_coef_lr: 0.001                 # Learning rate for adaptive entropy coefficient
eval_freq: 10000                   # Total steps between evaluations
n_eval_episodes: 50                # Episodes per scenario per evaluation
log_freq: 2000                     # Steps between console log lines
moving_avg_window: 100             # Episodes for rolling reward/success average
scenario_log_window: 30            # Episodes for per-scenario success rate log
scenario_min_episodes: 10          # Min episodes before logging scenario success
```

### Why These Values

- **base_model**: SAC was pre-trained by the Hugging Face community on standard `FetchPickAndPlace-v4`. It provides kinematic knowledge (inverse kinematics via the policy network). We fine-tune it on the new board size and 3-scenario setup.
- **learning_rate=0.00005**: Fine-tuning rate. Too high would destroy the pretrained weights.
- **initial_ent_coef=0.1**: The pretrained model has very low entropy (≈0.003), which means nearly deterministic behavior. Resetting to 0.1 jumpstarts exploration on the new reward surface.
- **target_entropy=-4.0**: For a 4D action space, a common heuristic is `-dim(action) = -4`.
- **pos_ctrl_scale=0.015**: The RL model outputs actions in [-1, 1]; at scale 0.015, the maximum displacement per step is 15mm. This limits movement speed to prevent physics instability.

---

## Config Dependencies

Many Python constants are derived at `__init__` time:
```python
self.SAFE_Z = self.env_cfg["safe_z"]                    # from env.yaml
self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)       # from env.yaml (fallback to 0.460)
self.GRASP_Z = self.env_cfg["grasp_z"]                  # from env.yaml
self.VERTICAL_QUAT = normalize(physics_cfg["vertical_quat"])  # from physics.yaml
```

The `get()` with fallback for `hover_z` suggests this value was added to the config later — the default ensures backward compatibility with older YAML files that don't have this key.
