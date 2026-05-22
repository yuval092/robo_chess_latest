# Note 1: Scripts Folder README

## Goal
Add a README explaining every script in `scripts/`, verify each script runs, and fix any broken ones.

## Scripts inventory

| Script | Purpose |
|--------|---------|
| `generate_board_xml.py` | Generates chess board square geoms into pick_and_place.xml |
| `generate_chess_stls.py` | Generates STL meshes for all 6 chess piece types |
| `generate_pieces_xml.py` | Generates piece body XML into pick_and_place.xml |
| `generate_zones_xml.py` | Generates zone marker geoms (graveyard/reserve) into pick_and_place.xml |
| `verify_physics.py` | Verifies arm can reach all 64 squares; reports HEALTHY/UNHEALTHY |
| `visualize.py` | Opens MuJoCo viewer with the chess scene (no pieces) |
| `visualize_chess_setup.py` | Opens MuJoCo viewer with pieces on starting squares |
| `run_chess_ui.py` | Starts the Flask web UI chess server |
| `eval_chess_reachability.py` | Validates board geometry constants match expected positions |
| `eval_chess_piece_move.py` | Evaluates a single piece move from a given square |
| `eval_chess_game_flow.py` | Evaluates full game flow including captures, castling, promotion |
| `eval_sequence.py` | Evaluates a scripted sequence of moves for success rate |
| `eval_special_moves.py` | Evaluates special moves: castling, en passant, promotion |
| `eval_stages.py` | Evaluates individual scripted controller stages (transit/ascend/etc.) |
| `eval_stress.py` | Stress test: random moves repeated N times |
| `eval_draw_conditions.py` | Evaluates draw detection (50-move rule, stalemate, repetition) |
| `test_grasp_physics.py` | Interactive test of grasp physics on a single piece |

## Implementation Tasks

1. Create `scripts/README.md` with the table above plus usage examples
2. Verify each generation script runs without error
3. Verify each eval script runs in `--no-gui` / dry-run mode where applicable
