# MuJoCo Environment

## Class Structure

`ChessFetchTask-v0` is registered by `src/chess_env/__init__.py` and implemented
by:

```text
MujocoFetchPickAndPlaceEnv
  -> ChessSimulationEnv
    -> ChessTaskEnv(
         GraspPlaceMixin,
         TaskStateMixin,
         TaskRuntimeMixin,
         ChessSimulationEnv
       )
```

`ChessSimulationEnv` owns low-level MuJoCo setup. `ChessTaskEnv` and its mixins
own scenario state, observations, rewards, resets, active pieces, grasp/place,
and runtime transitions.

## XML Loading

`ChessSimulationEnv` loads the checked-in scene:

```text
chess_env/assets/pick_and_place.xml
```

Gymnasium Robotics' Fetch env expects a module-level XML path. The constructor
temporarily replaces `gymnasium_robotics.envs.fetch.pick_and_place.MODEL_XML_PATH`
under a lock, calls the parent constructor, then restores the original path.

This keeps the installed dependency unmodified while allowing the project to own
its full scene XML.

## Important Config-Backed Constants

| Constant | Config | Value |
|---|---|---:|
| `TABLE_CENTER_XY` | `env.table_center_xy` | `[0.88, 0.2641]` |
| `TABLE_SURFACE_Z` | `env.table_surface_z` | `0.400` |
| `PIECE_HEIGHT` | `env.piece_height` | `0.030` |
| `GRASP_Z` | `env.grasp_z` | `0.430` |
| `HOVER_Z` | `env.hover_z` | `0.460` |
| `SAFE_Z` | `env.safe_z` | `0.530` |
| `SUCCESS_THRESHOLD` | `env.success_threshold` | `0.010` |
| `POS_CTRL_SCALE` | `physics.pos_ctrl_scale` | `0.015` |
| `VERTICAL_QUAT` | `physics.vertical_quat` | normalized `[0.7071068, 0, 0.7071068, 0]` |

## Action Interface

The RL action is a four-vector:

```text
[dx, dy, dz, gripper]
```

`ChessSimulationEnv._set_action()`:

1. Scales position deltas by `POS_CTRL_SCALE`.
2. Uses zero rotational delta.
3. Applies MuJoCo mocap movement.
4. Enforces gripper state from `finger_target_joint`.

There are two gripper modes:

| Mode | Condition | Behavior |
|---|---|---|
| Teleport mode | `grasp_mode=False` | Finger joint positions and velocities are set directly |
| Actuator mode | `grasp_mode=True` | Only actuator controls are set; contact physics can block the fingers |

Actuator mode is required while a piece is being held.

## Observations

The native observation is used for scripted control and status:

```text
observation:   7-D [grip_pos(3), grip_vel(3), l_finger(1)]
achieved_goal: 3-D grip position
desired_goal:  3-D goal position
grip_pos, grip_vel, l_finger, goal_pos
scenario_id:   0=None, 1=transit, 2=descend, 3=ascend
```

The transfer observation is used by SAC models:

```text
observation:   25-D FetchPickAndPlace-compatible vector
achieved_goal: 3-D grip position
desired_goal:  3-D goal position
```

The transfer trick sets `object_pos = grip_pos`, so the pretrained policy behaves
as if the gripper is already holding the object and needs to carry it to the goal.

`src/chess_env/transfer_obs.py` provides:

- `TRANSFER_OBS_SPACE`
- `transfer_obs_enabled(env)` for temporary load/inference contexts

## Scenarios

| Scenario | Start | Goal | Constraint | Finger target |
|---|---|---|---|---|
| `transit` | `SAFE_Z` | `SAFE_Z` | Must stay above floor/table | closed |
| `descend` | `SAFE_Z` | `HOVER_Z` | XY tube and table surface | open |
| `ascend` | `HOVER_Z` | `SAFE_Z` | XY tube and table surface | closed |

`force_scenario` locks training/evaluation to one scenario. Without it, reset
samples one of the three.

## Reset Behavior

`TaskRuntimeMixin._reset_sim()`:

1. Chooses scenario and start/goal positions.
2. Resets parent Fetch simulation.
3. Hides or places legacy `object0`.
4. Resets all chess piece freejoint bodies.
5. Sets torso height.
6. Settles the arm at the scenario start.
7. Performs scripted finger transition for the scenario.
8. Validates final finger state.
9. Captures the home posture when reset starts at `HOME_POS`.

In play mode, `show_chess_pieces=True` places active and reserve chess bodies in
their board/reserve positions. When false, pieces are hidden off-board for
training-style runs.

## Soft Reset / Stage Transition

`soft_reset(new_scenario, new_goal_pos, nominal_exit_pos, nominal_xy)` transitions
between movement stages without teleporting the arm:

1. Halt robot velocity.
2. Align to the nominal exit waypoint.
3. Update scenario, goal, and tube center.
4. Re-enforce vertical orientation.
5. Open or close fingers if needed and not in `grasp_mode`.
6. Return a new observation and transition diagnostics.

The controller resets wrapper elapsed steps after each transition.

## Reward and Termination

`step()` is used for RL training, not for runtime model inference. Runtime
inference calls `_set_action()` and `_mujoco_step()` directly.

Training termination checks:

- Finger joint deviates from target by more than 3 mm.
- Transit gripper hits the floor/table limit.
- Descend/ascend drift outside the current tube limit.
- Descend/ascend gripper drops below table surface.
- Success means near goal and stable below `stability_vel_threshold`.

Reward components:

```text
- dist_reward_weight * ||grip - goal||
- z_reward_weight    * |grip_z - goal_z|
- xy_reward_weight   * ||grip_xy - goal_xy||
- braking_weight     * speed near goal
- jitter_penalty     * ||action_xyz||^2
- floor_penalty      near table  (floor_penalty = 0.5, so this subtracts)
+ success_bonus      on success
crash_penalty        on crash
```

Descend uses a tighter Z success threshold from
`env.descend_success_threshold`.

