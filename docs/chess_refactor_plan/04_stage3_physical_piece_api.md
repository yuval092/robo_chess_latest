# Stage 3: Physical Piece API

## Goal

Introduce a physical layer that can move any chess piece cube by ID, teleport pieces to non-reachable locations, and report physical occupancy without exposing raw MuJoCo details to chess logic.

## Required Interfaces

### MovementExecutor

Create `src/physical/movement_executor.py`.

```python
class MovementExecutor:
    def __init__(self, env, controller, board_mapper, piece_state): ...

    def move_piece_between_squares(
        self,
        piece_id: str,
        src_square: str,
        dst_square: str,
    ) -> PhysicalMoveResult: ...

    def move_piece_xy(
        self,
        piece_id: str,
        src_xy: np.ndarray,
        dst_xy: np.ndarray,
    ) -> PhysicalMoveResult: ...
```

`move_piece_between_squares` must:

1. Validate `piece_id` is physically located at `src_square`.
2. Teleport or mark all non-target pieces as normal active bodies, not hidden.
3. Set environment context so `execute_grasp()` reads the selected piece, not `object0`.
4. Run the existing waypoint full move:
   - transit to source
   - descend
   - grasp selected cube
   - ascend
   - transit to destination
   - descend
   - place selected cube
   - ascend
5. Verify final selected piece XY is within 20mm of destination center and Z is table-resting.
6. Update physical piece occupancy only after verification succeeds.

### PieceTeleporter

Create `src/physical/piece_teleport.py`.

```python
class PieceTeleporter:
    def teleport_piece_to_xyz(self, piece_id: str, xyz: np.ndarray) -> None: ...
    def teleport_piece_to_square(self, piece_id: str, square: str) -> None: ...
    def teleport_piece_to_graveyard(self, piece_id: str, slot_id: str) -> None: ...
    def teleport_piece_to_promotion_reserve(self, piece_id: str, slot_id: str) -> None: ...
```

Requirements:

- Teleportation must set freejoint position, quaternion, qvel, and qacc.
- Teleportation must call `mujoco.mj_forward`.
- Teleportation is allowed only for graveyard/promotion reserve operations and reset/setup. Normal board-to-board moves must use the arm unless the operation is explicitly a recovery tool.

### PhysicalOccupancy

Create `src/physical/occupancy.py`.

```python
class PhysicalOccupancy:
    def piece_at_square(self, square: str) -> str | None: ...
    def square_of_piece(self, piece_id: str) -> str | None: ...
    def set_piece_square(self, piece_id: str, square: str | None) -> None: ...
    def assert_square_empty(self, square: str) -> None: ...
    def assert_piece_at(self, piece_id: str, square: str) -> None: ...
```

This layer tracks expected physical occupancy. It is not the chess source of truth, but it must match after every completed move.

## Active Piece Contract — Critical

`ChessTaskEnv` currently hardcodes `object0` in several places. After Stage 3 refactor, the following contract must hold:

**`set_active_piece(piece_id: str)` must be called before any of:**
- `execute_grasp()`
- `execute_place(dst_xy)`
- `_check_cube_held(grip_pos)`

The `MovementExecutor.move_piece_between_squares` is responsible for calling `set_active_piece` before initiating the arm move sequence. `execute_grasp` and `execute_place` must never contain hardcoded `"object0"` after this stage.

**Affected internal methods that must respect `active_piece_joint_name`:**

| Method | Change |
|--------|--------|
| `get_cube_position()` | Returns active piece position |
| `get_cube_quat()` | Returns active piece quaternion |
| `_check_cube_held(grip_pos)` | Checks active piece vs grip position, not object0 |
| `execute_grasp` Phase 4–6 | Live cube tracking uses `get_cube_position()`, which now routes to active piece |
| `_reset_sim` | Still resets object0 for legacy tests; add `reset_active_piece(piece_id)` for chess |

**The backward-compat alias pattern is correct but add an explicit guard:**

```python
def get_cube_position(self):
    if self.active_piece_body_name is None:
        return self._get_body_position("object0")  # legacy fallback
    return self.get_active_piece_position()
```

If `set_active_piece` is never called before `execute_grasp`, the system should log a warning rather than silently tracking the wrong piece.

## Refactor Existing Object-Specific Code

Current code hardcodes `object0` in:

- `ChessTaskEnv.get_cube_position`
- `ChessTaskEnv.get_cube_quat`
- `_reset_sim`
- `execute_grasp`
- `execute_place`
- `_check_cube_held`
- scripts and tests

Refactor to support a selected active piece:

```python
self.active_piece_joint_name = "piece_white_pawn_e:joint"
self.active_piece_body_name = "piece_white_pawn_e"
```

Add generic helpers:

```python
def get_active_piece_position(self) -> np.ndarray: ...
def get_active_piece_quat(self) -> np.ndarray: ...
def set_active_piece(self, piece_id: str) -> None: ...
```

Keep backward-compatible aliases temporarily:

```python
def get_cube_position(self):
    return self.get_active_piece_position()
```

This lets current tests continue while chess-specific tests are added.

## Crowded Descent Finger Clearance

The existing 2mm neighbor displacement test checks that adjacent pieces are not knocked over. A separate check is needed for whether the gripper FINGERS physically fit in the descent corridor.

At `GRASP_Z = 0.430m`, the gripper fingers extend outward by `finger_outer_offset = 0.033m` from the grip center. Adjacent cube faces at 8cm center-to-center spacing are 50mm apart. The finger outer edge extends 33mm from center, leaving `50 - 33 = 17mm` clearance between the finger tip and the adjacent cube face.

This is adequate for axis-aligned descents (picking from a1 with no piece at b1), but when picking pieces surrounded on all 4 sides (center board), the clearance is equal in all directions. The gripper must approach perfectly centered on the target square. The existing `GRASP_VERIFY_XY_THRESHOLD = 0.015m` (15mm) could allow descent with only 2mm finger clearance on the tight side.

**Required actions:**
1. In crowded-board integration tests, add an explicit clearance check: verify no neighbor piece is displaced while the gripper is descending and ascending (not just after the move).
2. For center-board crowded picks, consider tightening `TRANSIT_TOLERANCE_M` before descent from 4mm to 2mm if clearance tests show displacement.
3. Document that gripper is always oriented with the same yaw (VERTICAL_QUAT does not control gripper yaw, only wrist pitch). Verify the gripper yaw does not cause a finger to align toward a neighboring piece rather than between them.

## Graveyard And Promotion Reserve Geometry

Add floor locations in `configs/chess.yaml`.

Example:

```yaml
graveyards:
  white:
    origin_xyz: [0.30, -0.35, 0.015]
    rows: 4
    cols: 4
  black:
    origin_xyz: [1.38, 0.88, 0.015]
    rows: 4
    cols: 4
promotion_reserve:
  origin_xyz: [0.88, 0.95, 0.015]
  rows: 2
  cols: 4
```

Placement constraints:

- White graveyard near white side of board.
- Black graveyard near black side of board.
- Promotion reserve on the opposite side of the arm.
- All reserve/graveyard coordinates are on the floor and not reachable by the arm.
- They are visual/logical locations. Moving to/from them always uses teleportation.

Add visual floor markers as non-colliding geoms if useful:

```xml
<geom name="white_graveyard_slot_00" type="box" size="0.018 0.018 0.001"
      pos="..." rgba="..." contype="0" conaffinity="0"/>
```

## Physical Transaction Behavior

A physical operation returns:

```python
@dataclass
class PhysicalMoveResult:
    success: bool
    piece_id: str
    src_square: str | None
    dst_square: str | None
    stage_results: list
    error: str | None = None
```

If any stage fails:

- Do not update physical occupancy.
- Return the failed stage result.
- Leave the chess layer unchanged.
- Emit enough debug state for recovery:
  - piece position
  - grip position
  - current scenario
  - move being executed

## Post-Move Reconciliation

After each physical move, `MovementExecutor` must perform an explicit reconciliation step before returning `PhysicalMoveResult`:

1. Read the placed piece's actual MuJoCo position via `get_active_piece_position()`.
2. Compare to expected square center XY from `BoardMapper`.
3. If XY error > 20mm: mark result as failed, do not update occupancy.
4. If Z error > 10mm from `table_z + cube_height/2`: mark result as unstable, do not update occupancy.
5. Snap to identity quat (orientation reset) only after successful reconciliation.

This prevents the chess layer from accepting a move where the piece actually ended up on the wrong square due to physics settling.

## Tests

Add `tests/physical/test_piece_registry.py` and `tests/physical/test_piece_teleport.py`.

Add integration tests:

- Move `white_pawn_e` from `e2` to `e4`.
- Move `white_knight_g` from `g1` to `f3`.
- Attempt to move from an empty square and verify rejection.
- Teleport captured piece to a graveyard slot.
- Execute a physical move from a crowded starting position and assert all non-moving pieces remain within 2mm of their expected square centers.
- Attempt to move into an occupied square through `MovementExecutor` without a capture-removal command and verify it rejects before arm motion.

## Validation

Run:

```bash
pytest tests/physical -v
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
python scripts/eval_chess_piece_move.py --piece white_pawn_e --src e2 --dst e4
```

Pass criteria:

- Any piece can be selected as active piece.
- Existing waypoint behavior remains unchanged.
- Teleportation works for off-board locations.
- Physical occupancy updates only after verified success.
- Normal board-to-board physical moves require the destination square to be physically empty before descent.
