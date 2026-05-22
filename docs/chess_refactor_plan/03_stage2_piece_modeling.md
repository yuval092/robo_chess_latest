# Stage 2: Chess Piece Modeling

## Goal

Replace the single cube object model with 32 chess pieces, each physically graspable as a 30mm cube and visually identifiable as a chess piece through a non-colliding STL mounted above the cube.

## Piece Representation

Each chess piece has:

- Stable logical ID, e.g. `white_king`, `black_queen`, `white_pawn_1`.
- Color: `white` or `black`.
- Type: `king`, `queen`, `rook`, `bishop`, `knight`, `pawn`.
- Physical MuJoCo freejoint body.
- Collision cube geom:
  - `type="box"`
  - `size="0.015 0.015 0.015"`
  - `mass="0.05"`
  - `condim="6"`
  - friction and contact settings matching the current `object0`.
- Visual STL geom:
  - mounted above cube.
  - `contype="0"`
  - `conaffinity="0"`
  - `mass="0"`
  - visual material only.

The cube remains the only part the gripper interacts with.

## STL Strategy

Preferred approach — generate STLs with a Python script using simple geometric primitives:

Create `scripts/generate_chess_stls.py`.

Use `numpy-stl` or hand-coded triangle meshes. Piece approximations:

| Piece | Geometric approximation |
|-------|------------------------|
| Pawn | Short cylinder (r=6mm, h=18mm) |
| Rook | Cylinder (r=7mm, h=20mm) with flat top |
| Knight | Cylinder (r=6mm, h=16mm) with angled top flat |
| Bishop | Tall narrow cylinder (r=5mm, h=24mm) with pointed top sphere |
| Queen | Medium cylinder (r=6mm, h=22mm) with crown indent |
| King | Tall cylinder (r=6mm, h=26mm) with cross top |

All shapes should be centered at origin with base at z=0 and fit within a 16mm radius, 30mm height envelope. Do not import from external STL repositories unless needed.

If using downloaded assets:

- Record source URL and license in `chess_env/stls/chess/LICENSES.md`.
- Normalize scale and orientation before committing.
- Verify mesh triangle count is under 2000 per mesh; MuJoCo loads large meshes slowly.
- Run `python -c "import mujoco; ..."` load test before committing.

Store generated STLs under:

```text
chess_env/stls/chess/pawn.stl
chess_env/stls/chess/rook.stl
chess_env/stls/chess/knight.stl
chess_env/stls/chess/bishop.stl
chess_env/stls/chess/queen.stl
chess_env/stls/chess/king.stl
```

The STLs are visual-only, so approximate silhouettes are acceptable as long as each type is visually distinct.

## Visual Mesh Constraints

The chess STL must not change physics, but it can still make the simulator visually confusing if it clips heavily through the gripper or neighboring pieces.

Use these constraints unless visual testing proves a better value:

- Visual mesh geoms must always have `contype="0"` and `conaffinity="0"`.
- Visual mesh footprint should stay inside the 30mm cube footprint or only slightly exceed it.
- Visual mesh height should be large enough to identify the piece but not so large that `HOVER_Z` and `SAFE_Z` movements are visually obscured.
- If a visually accurate STL would clip through the gripper during grasp, keep the collision cube authoritative and document the visual clipping as acceptable, or implement a render-only hide/fade of the selected piece's STL during the grasp/place phases.
- Do not add collision to the STL to solve visual clipping; that would violate the physics design.

## Reserve Piece Count

In the worst case, all 8 pawns of one color promote to the same piece type (e.g., all 8 to queens). Pre-create the following reserve pieces per color:

```yaml
reserves:
  per_color:
    queen: 9    # 1 starting + 8 possible promotions = 9; but starting queen is on board, so 8 in reserve
    rook: 8
    bishop: 8
    knight: 8
```

In practice, promotion to the same type multiple times in one game is rare, but the model must support it. Park all reserve pieces at their off-board `promotion_reserve` slots defined in `configs/chess.yaml`. Their freejoints start at those floor positions. Total additional bodies: 32 reserve bodies (8 queen + 8 rook + 8 bishop + 8 knight) × 2 colors = 64 reserve bodies. This brings total piece-related bodies to 96.

**Important**: All 64 reserve bodies exist in XML at all times. They are never removed. When a pawn promotes, the selected reserve piece is teleported from its floor slot to the promotion square. When a promoted piece is captured, it is teleported back to a reserve/graveyard slot.

## XML Generator Script

Do not write 96 piece XML bodies by hand. Create `scripts/generate_pieces_xml.py`.

The script must:

1. Load `configs/chess.yaml` for piece parameters and reserve geometry.
2. For each of the 32 active pieces: generate body XML with correct starting position, freejoint, collision geom, visual geom.
3. For each of the 64 reserve bodies: generate body XML at the designated floor reserve position.
4. Write `<!-- generated chess pieces -->` and `<!-- generated reserve pieces -->` XML fragments.

Starting piece positions use `BoardMapper.square_to_piece_xyz(square)` which returns `[x, y, table_z + cube_height/2]`. These are world coordinates. Since piece bodies are declared at world level (not inside `table0`), use world coordinates directly.

## XML Implementation

Add mesh declarations in `shared.xml`:

```xml
<mesh name="chess_pawn_mesh" file="../stls/chess/pawn.stl" scale="0.001 0.001 0.001"/>
```

Add materials:

```xml
<material name="white_piece_cube_mat" rgba="0.92 0.88 0.78 1"/>
<material name="black_piece_cube_mat" rgba="0.08 0.08 0.08 1"/>
<material name="white_piece_visual_mat" rgba="0.98 0.95 0.85 1"/>
<material name="black_piece_visual_mat" rgba="0.02 0.02 0.02 1"/>
```

Body template:

```xml
<body name="piece_white_king" pos="0 0 0">
  <joint name="piece_white_king:joint" type="free" damping="0.1"/>
  <geom name="piece_white_king_cube" type="box" size="0.015 0.015 0.015"
        material="white_piece_cube_mat" mass="0.05" condim="6"
        friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>
  <geom name="piece_white_king_visual" type="mesh" mesh="chess_king_mesh"
        pos="0 0 0.020" material="white_piece_visual_mat"
        contype="0" conaffinity="0" mass="0"/>
  <site name="piece_white_king_site" pos="0 0 0" size="0.005"/>
</body>
```

The visual mesh must not extend below the cube top in a way that touches the gripper fingers.

## Naming Convention

Use deterministic body and joint names:

```text
piece_white_king
piece_white_queen
piece_white_rook_a
piece_white_rook_h
piece_white_bishop_c
piece_white_bishop_f
piece_white_knight_b
piece_white_knight_g
piece_white_pawn_a
...
piece_black_pawn_h
```

Use starting file/rank suffixes for rooks, bishops, knights, and pawns. This makes move history and captures easy to debug.

## Initial Placement

Create `src/physical/piece_registry.py`.

Required API:

```python
@dataclass(frozen=True)
class PhysicalPiece:
    piece_id: str
    color: str
    piece_type: str
    body_name: str
    joint_name: str
    cube_geom_name: str
    visual_geom_name: str
    initial_square: str

class PieceRegistry:
    def all_pieces(self) -> list[PhysicalPiece]: ...
    def by_id(self, piece_id: str) -> PhysicalPiece: ...
    def starting_square_map(self) -> dict[str, str]: ...
    def ids_for_color(self, color: str) -> list[str]: ...
```

Starting layout:

```text
White: a1 rook, b1 knight, c1 bishop, d1 queen, e1 king, f1 bishop, g1 knight, h1 rook
White pawns: a2-h2
Black: a8 rook, b8 knight, c8 bishop, d8 queen, e8 king, f8 bishop, g8 knight, h8 rook
Black pawns: a7-h7
```

## Freejoint Damping

Use `damping="0.5"` for all piece freejoints instead of `object0`'s `damping="0.1"`. With 32 pieces on the table simultaneously, low damping leads to visible physics drift between moves. Higher damping also reduces the energy of minor piece impacts from gripper contact.

The value 0.5 is not validated against MuJoCo unit conventions. Add a `verify_physics.py` check that confirms all 32 pieces stay within 1mm of their initial squares over 5 simulated seconds (no arm interaction). Adjust upward if drift still occurs.

## Reset Behavior

Replace `object0`-specific reset with piece reset:

- At game reset, teleport every active piece to its starting square center.
- Set each piece quaternion to identity `[1, 0, 0, 0]`.
- Zero its 6 freejoint velocities.
- Set graveyard and promotion-reserve pieces/slots as inactive or parked.
- Call `mujoco.mj_forward`.

During this stage, it is acceptable to keep `object0` for backward-compatibility scripts, but new chess code must use the piece registry and not hardcode `object0`.

## Post-Place Piece Orientation Reset

After `execute_place` the piece cube is physically stable but may have a non-identity yaw relative to its starting orientation. The current arm uses `VERTICAL_QUAT` throughout, so yaw is not controlled. Over successive moves this accumulates: pieces drift from their "upright" visual orientation.

After each successful physical move, the `MovementExecutor` must snap the placed piece back to identity quaternion using a freejoint teleport:

```python
# After successful execute_place and arm retract
placed_xyz = board_mapper.square_to_piece_xyz(dst_square)
piece_teleporter.teleport_piece_to_xyz(piece_id, placed_xyz, quat=IDENTITY_QUAT)
```

This is safe because:
- The gripper has already retracted to SAFE_Z before this call.
- The piece is physically resting on the table; snapping its quat to identity with `mj_forward` is no different from a teleport.
- The XY correction also allows fine re-centering if placement error was within tolerance but not exact.

## Crowded Board Clearance Risk

The current project has mostly validated moving one cube on an otherwise open tabletop. A chess starting position is different: many cubes are adjacent on exact 8cm centers.

This must be treated as a first-class risk because:

- Cell half-width is 40mm.
- Piece collision cube half-width is 15mm.
- Neighboring cube faces can sit only 25mm from the selected piece center.
- Current gripper geometry and `finger_outer_offset` may place physical finger geometry close to a neighboring square during descent or grasp, depending on gripper orientation.

Requirements:

- Do not assume the single-cube pick tests prove chess starting-position safety.
- Add crowded-board grasp tests before the full chess setup is accepted.
- Test picking pawns from the starting rank with neighboring pawns present on both sides.
- Test picking back-rank pieces with adjacent pieces present.
- Test center-board captures where neighboring pieces occupy orthogonally and diagonally adjacent squares.

If collisions occur, evaluate mitigations in this order:

1. Verify actual gripper collision orientation and whether rotating the physical board mapping 90 degrees improves clearance while preserving exact 8cm cells.
2. Tighten pre-grasp XY alignment for crowded picks.
3. Use teleportation only for pieces that chess rules remove or introduce, not to clear normal neighboring blockers.
4. Adjust physical cube size only as a last resort and only with explicit approval, because the user requested cube-based pieces.

The normal chess move pipeline must not teleport adjacent blockers out of the way.

## Tests

Add tests:

- All 32 pieces exist in XML model.
- Every piece has one freejoint.
- Every collision cube has the exact current cube dimensions.
- Every visual STL geom is non-colliding.
- Visual STL footprint and height stay within documented visual constraints.
- Initial reset places all pieces on correct square centers.
- No two pieces share the same starting square.
- Crowded setup tests show neighboring pieces are not displaced by a selected-piece pick.

## Validation

Run:

```bash
pytest tests/ -v
python scripts/verify_physics.py
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Add a visual script command:

```bash
python scripts/visualize_chess_setup.py
```

Pass criteria:

- Board shows all 32 pieces in correct cells.
- Pieces rest stably on the table.
- The arm can still pick and place a selected piece cube.
- Crowded-board picks do not collide with or displace neighboring pieces beyond 2mm.
