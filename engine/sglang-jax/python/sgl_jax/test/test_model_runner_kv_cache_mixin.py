import types
from unittest.mock import patch

import pytest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
)
from sgl_jax.srt.server_args import ServerArgs


def test_recurrent_state_legacy_disable_radix_cache_passes():
    sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
    _enforce_recurrent_state_server_constraints(sa)  # no raise


def test_recurrent_state_radix_cache_requires_unified_radix_tree():
    sa = ServerArgs(
        model_path="dummy",
        disable_radix_cache=False,
        enable_unified_radix_tree=False,
    )
    with pytest.raises(AssertionError, match="unified-radix-tree"):
        _enforce_recurrent_state_server_constraints(sa)


def _fake_runner(max_recurrent_state_size, dp_size):
    sa = ServerArgs(
        model_path="dummy",
        enable_unified_radix_tree=True,
        disable_radix_cache=False,
        dp_size=dp_size,
        max_recurrent_state_size=max_recurrent_state_size,
    )
    return types.SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=None,
        linear_recurrent_config=object(),
        server_args=sa,
        dp_size=dp_size,
    )


def test_recurrent_admission_cap_is_dp_aligned():
    # factor=2: 12 // 2 = 6 is off the dp_size=4 grid, so the cap rounds down to 4.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(12, 4), 1000)
    assert cap == 4
    assert cap % 4 == 0


def test_recurrent_admission_cap_already_aligned_unchanged():
    # 16 // 2 = 8 is already a dp_size=4 multiple.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(16, 4), 1000)
    assert cap == 8


def test_disable_radix_max_running_is_operator_sized_recurrent_pool():
    server_args = ServerArgs(
        model_path="dummy",
        disable_radix_cache=True,
        max_running_requests=16,
    )
    runner = types.SimpleNamespace(
        attention_tp_size=8,
        dp_size=1,
        linear_recurrent_config=object(),
        server_args=server_args,
    )
    with patch(
        "sgl_jax.srt.model_executor.model_runner_kv_cache_mixin."
        "_per_req_state_bytes_from_config",
        return_value=1024,
    ):
        remaining = ModelRunnerKVCacheMixin.handle_recurrent_cache(runner, 1 << 20)

    assert remaining == (1 << 20) - 16 * 1024
    assert runner.recurrent_size_user_supplied is True
    assert server_args.max_recurrent_state_size == 16
