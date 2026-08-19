from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.speculative.dspark_worker import (
    DSparkDraftInput,
    build_dspark_cache_loc,
    choose_confidence_chain_length,
    compact_dspark_verified_ids,
)
from sgl_jax.srt.speculative.overlap_utils import resolve_spec_decode_token_ids


def test_fixed_chain_uses_checkpoint_block_plus_anchor() -> None:
    decision = choose_confidence_chain_length(
        jnp.zeros((3, 7), dtype=jnp.float32),
        threshold=None,
    )

    assert decision.verify_tokens == 8
    np.testing.assert_array_equal(
        decision.per_request_verify_tokens,
        np.full((3,), 8, dtype=np.int32),
    )


def test_confidence_chain_uses_conservative_batch_prefix() -> None:
    logits = jnp.asarray(
        [
            [4.0, 4.0, 4.0, 4.0],
            [4.0, 0.0, -4.0, -4.0],
        ],
        dtype=jnp.float32,
    )

    decision = choose_confidence_chain_length(logits, threshold=0.40)

    np.testing.assert_array_equal(
        decision.per_request_verify_tokens,
        np.asarray([5, 3], dtype=np.int32),
    )
    assert decision.verify_tokens == 3


def test_confidence_chain_always_verifies_anchor_and_one_proposal() -> None:
    decision = choose_confidence_chain_length(
        jnp.full((2, 7), -20.0, dtype=jnp.float32),
        threshold=0.99,
    )

    assert decision.verify_tokens == 2
    np.testing.assert_array_equal(
        decision.per_request_verify_tokens,
        np.full((2,), 2, dtype=np.int32),
    )


def test_dspark_cross_round_state_splits_without_eagle_draft_fields() -> None:
    state = DSparkDraftInput(
        verified_id=np.asarray([11, 12, 13], dtype=np.int32),
        allocate_lens=np.asarray([20, 21, 22], dtype=np.int32),
        new_seq_lens=np.asarray([21, 22, 23], dtype=np.int32),
    )

    split = ScheduleBatch._split_spec_info_per_rank(state, [1, 2])

    assert all(isinstance(part, DSparkDraftInput) for part in split)
    np.testing.assert_array_equal(split[0].verified_id, [11])
    np.testing.assert_array_equal(split[1].verified_id, [12, 13])


def test_dspark_cross_round_state_filters_and_merges_minimal_fields() -> None:
    state = DSparkDraftInput(
        verified_id=np.asarray([11, 12, 13], dtype=np.int32),
        allocate_lens=np.asarray([20, 21, 22], dtype=np.int32),
    )
    state.filter_batch(np.asarray([2, 0], dtype=np.int32))
    state.merge_batch(
        DSparkDraftInput(
            verified_id=np.asarray([14], dtype=np.int32),
            allocate_lens=np.asarray([23], dtype=np.int32),
        )
    )

    np.testing.assert_array_equal(state.verified_id, [13, 11, 14])
    np.testing.assert_array_equal(state.allocate_lens, [22, 20, 23])


def test_dspark_cache_map_keeps_page_aligned_request_sections() -> None:
    req_to_token = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [11, 12, 13, 14, 15, 16],
            [21, 22, 23, 24, 25, 26],
        ],
        dtype=np.int32,
    )

    cache_loc = build_dspark_cache_loc(
        req_to_token,
        np.asarray([1, 0, 2], dtype=np.int32),
        np.asarray([3, 0, 2], dtype=np.int32),
        np.asarray([5, 0, 3], dtype=np.int32),
        page_size=4,
        capacity=16,
    )

    np.testing.assert_array_equal(cache_loc[:5], [11, 12, 13, 14, 15])
    np.testing.assert_array_equal(cache_loc[5:8], 0)
    np.testing.assert_array_equal(cache_loc[8:11], [21, 22, 23])
    np.testing.assert_array_equal(cache_loc[11:], 0)


def test_dspark_next_anchor_is_last_verified_token_per_real_request() -> None:
    verified = np.asarray(
        [
            [11, 12, 0, 0],
            [21, 22, 23, 0],
            [31, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    compact = compact_dspark_verified_ids(
        verified,
        np.asarray([2, 3, 1], dtype=np.int32),
        np.asarray([0, 2], dtype=np.int32),
        batch_size=3,
        verify_tokens=4,
    )

    np.testing.assert_array_equal(compact, [12, 31])


def test_variable_chain_output_uses_current_verify_width() -> None:
    result = SimpleNamespace(
        next_token_ids=np.asarray([10, 11, 20, 21], dtype=np.int32),
        accept_lens=np.asarray([2, 1], dtype=np.int32),
    )
    batch = SimpleNamespace(
        per_dp_bs_size=2,
        dp_size=1,
        reqs_info=[SimpleNamespace(reqs=[object(), object()])],
    )

    token_ids, accept_lens = resolve_spec_decode_token_ids(
        result,
        batch,
        draft_token_num=2,
    )

    assert token_ids == [[10, 11], [20]]
    assert accept_lens == [2, 1]
