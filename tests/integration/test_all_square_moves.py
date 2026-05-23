import os

import pytest

from scripts.eval_all_square_moves import run_all_square_moves


@pytest.mark.skipif(
    os.environ.get("RUN_EXHAUSTIVE_PHYSICAL_MOVES") != "1",
    reason="Set RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 to run the 64x63 physical move sweep.",
)
def test_all_square_to_all_square_physical_moves():
    summary = run_all_square_moves(piece_id="black_rook_a", stop_on_failure=True)
    assert not summary.failures
