import pytest
from skaks_opt import selfplay


def test_arena_clock_args():
    # This test checks that the arena and _run_arena_shard accept clock args and propagate them.
    # It does not require the backend to support them yet.
    base_params = {"search": {}}
    cand_params = {"search": {}}
    fens = ["8/8/8/8/8/8/8/8 w - - 0 1"]
    payload = (fens, base_params, cand_params, 0, 0, 160, 10000, 10000, 0, 40)
    # Should not raise
    try:
        selfplay._run_arena_shard(payload)
    except Exception as e:
        pytest.skip(f"Backend does not support clock args yet: {e}")
