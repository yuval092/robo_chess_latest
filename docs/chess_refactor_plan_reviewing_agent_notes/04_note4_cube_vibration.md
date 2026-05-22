# Note 4: Reduce Cube Vibration

## Problem
Cube vibrates significantly during grasp. Two causes:
1. Low mass (0.05kg) — light pieces respond poorly to impulses from gripper contact
2. Low freejoint damping (0.5) — joint oscillation not damped quickly

## Changes

### 4.1 Increase piece mass: 0.05 → 0.15 kg
`scripts/generate_pieces_xml.py`: change `mass="0.05"` to `mass="0.15"` on cube geom
Then regenerate: `python scripts/generate_pieces_xml.py --write`

### 4.2 Increase freejoint_damping: 0.5 → 2.0
`configs/chess.yaml`: `freejoint_damping: 2.0`
Then regenerate: `python scripts/generate_pieces_xml.py --write`

## Interaction with Note 2 (Speed)
Higher mass + higher damping lets us close fingers faster (fewer grasp_close_steps) since:
- More mass = more inertia = less violent movement per impulse
- More damping = oscillations die faster = can release hold sooner

## Validation
`python scripts/test_grasp_physics.py` — visually confirm no oscillation after grasp.
