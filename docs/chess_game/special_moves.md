# Special Moves

Special moves are represented as ordinary legal chess moves plus explicit physical commands.

## Castling

The king moves first, then the rook moves to its castled square. Both are board-to-board arm moves.

## Captures

The captured piece is teleported to the next graveyard slot before the moving piece is picked and placed.

## En Passant

The captured pawn is removed from the passed-over square, not from the destination square. The moving pawn then moves normally.

## Promotion

The pawn is moved to the promotion square, teleported to the promotion reserve, and a reserve piece of the chosen type is teleported onto the promotion square.

## Validation

Run:

```bash
python scripts/eval_special_moves.py --all
```

Mocked special-move validation is the default. A selected case can be attempted with real physics:

```bash
python scripts/eval_special_moves.py --case castling --real-physics
```
