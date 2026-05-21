import gymnasium as gym
import numpy as np
import src.chess_env
import mujoco

env = gym.make('ChessFetchTask-v0')
env.reset()
inner = env.unwrapped
inner.model.body_pos[inner.model.body('robot0:base_link').id][0] = 0.48
mujoco.mj_forward(inner.model, inner.data)


target_xy = np.array([1.19, 0.2641])
target = np.array([target_xy[0], target_xy[1], inner.SAFE_Z])

print(f"Torso height: {inner._utils.get_joint_qpos(inner.model, inner.data, 'robot0:torso_lift_joint')}")
print(f"Base pos: {inner.model.body_pos[inner.model.body('robot0:base_link').id]}")

print(f"Moving to {target}...")
success = inner._move_mocap_to(target, inner.VERTICAL_QUAT, max_steps=200, tolerance=0.001)

grip_pos = inner._utils.get_site_xpos(inner.model, inner.data, "robot0:grip")
err = np.linalg.norm(target - grip_pos)
print(f"Success: {success}, Error: {err*1000:.2f}mm")

env.close()
