# Scripts

This directory contains all runnable scripts for RoboChess: scene generation, evaluation, and visualization.

## Generation Scripts

These scripts produce or update project assets. Run them whenever the relevant configs change.

| Script | What it does | Run command |
|--------|-------------|-------------|
| `generate_chess_stls.py` | Generates STL meshes for all 6 chess piece types. STLs have a 30×30mm cube-covering base and a distinctive shape per type. | `python scripts/generate_chess_stls.py` |
| `generate_board_xml.py` | Generates the 64 chess board square geoms into `pick_and_place.xml`. | `python scripts/generate_board_xml.py --write` |
| `generate_pieces_xml.py` | Generates all piece body XML (active + promotion reserve) into `pick_and_place.xml`. Re-run after changing `chess.yaml`. | `python scripts/generate_pieces_xml.py --write` |
| `generate_zones_xml.py` | Generates flat zone marker geoms (graveyard + reserve areas) into `pick_and_place.xml`. | `python scripts/generate_zones_xml.py --write` |

### Typical regen workflow

```bash
python scripts/generate_chess_stls.py
python scripts/generate_pieces_xml.py --write
# if zone positions changed:
python scripts/generate_zones_xml.py --write
```

## Verification Scripts

| Script | What it does | Run command |
|--------|-------------|-------------|
| `verify_physics.py` | Loads the MuJoCo scene and checks that the arm can reach all 64 squares. Reports HEALTHY / UNHEALTHY. | `python scripts/verify_physics.py` |
| `eval_chess_reachability.py` | Validates board geometry constants (corner positions, spacing) against expected values from `chess.yaml`. | `python scripts/eval_chess_reachability.py` |

## Evaluation Scripts

All eval scripts support `--visualize` (human render) and `--delay <seconds>` for slow-motion playback.

| Script | What it does | Key flags |
|--------|-------------|-----------|
| `eval_chess_piece_move.py` | Evaluates moving a single piece from a given square. Useful for debugging a specific move. | `--from e2 --to e4` |
| `eval_chess_game_flow.py` | Evaluates complete game flow: opening moves, captures, castling, en passant, promotion. | `--verify-agreement` |
| `eval_sequence.py` | Runs a scripted sequence of moves and reports per-move success rate. | `--moves e2e3,e7e6,...` |
| `eval_special_moves.py` | Evaluates all special moves: castling (both sides), en passant, pawn promotion. | `--all` |
| `eval_stages.py` | Evaluates individual controller stages (transit, ascend, descend, grasp, release) in isolation. | `--stages transit,grasp` |
| `eval_stress.py` | Stress test: N random legal moves, reports success rate. | `--n-moves 50` |
| `eval_draw_conditions.py` | Verifies draw detection: 50-move rule, stalemate, threefold repetition. | (no flags needed) |

## Visualization Scripts

| Script | What it does | Run command |
|--------|-------------|-------------|
| `visualize.py` | Interactive MuJoCo viewer with ScriptedController. | `python scripts/visualize.py` |
| `visualize_chess_setup.py` | Static MuJoCo viewer with pieces on starting squares. | `python scripts/visualize_chess_setup.py` |

## Application

| Script | What it does | Run command |
|--------|-------------|-------------|
| `run_chess_ui.py` | Starts the Flask web server. Open `http://localhost:5000` in a browser. | `python scripts/run_chess_ui.py` |

## Physics Testing

| Script | What it does | Run command |
|--------|-------------|-------------|
| `test_grasp_physics.py` | Interactive test of grasp physics on a single piece. Run after XML or physics param changes to verify the cube can be held without vibration. | `python scripts/test_grasp_physics.py` |
