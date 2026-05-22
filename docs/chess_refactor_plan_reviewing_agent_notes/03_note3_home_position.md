# Note 3: HOME_POSITION = Board Center

## Goal
Change HOME_POSITION from `[0.680, 0.2641]` (arm base area) to `[0.88, 0.2641]` (board center).
The arm should rest over the center of the board at SAFE_Z between moves.

## Rules
- Home at start of game (reset)
- Home at end of each turn (after place + release)
- Do NOT teleport to home position; always move via scripted transit
- No home position during a move sequence (pick→place is one atomic operation)

## Changes

### 3.1 Update env.yaml
`home_position_xy: [0.88, 0.2641]`

### 3.2 Update chess.yaml
`arm_home_xy: [0.88, 0.2641]`

## No Code Changes Needed
`task.py` already reads `home_position_xy` from env_cfg at line 106:
```python
home_xy = self.env_cfg.get("home_position_xy", [0.680, 0.2641])
self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])
```
Updating the config is sufficient.

## Validation
After reset, `self.HOME_POS` should be `[0.88, 0.2641, 0.550]`.
