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
