from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.configs.dspark import (
    INKLING_SMALL_NVFP4_REPO,
    INKLING_SMALL_NVFP4_REVISION,
    validate_inkling_small_dspark_target,
)
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.managers.scheduler import can_interleave_speculative_prefill
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.speculative.dspark_worker import (
    DSparkDraftInput,
    DSparkDraftWorker,
    _gather_context_rows,
    _write_context_rows,
    build_dspark_cache_loc,
    build_dspark_verify_input,
    choose_confidence_chain_length,
    compact_dspark_verified_ids,
)
from sgl_jax.srt.models.dspark import DSparkContextCache, DSparkLayerContext
from sgl_jax.srt.speculative.base_worker import (
    gather_preserving_sharding,
    prepare_target_outputs_for_verification,
)
from sgl_jax.srt.speculative.dspark_planner import (
    DSparkScheduleConfig,
    SpsCostTable,
    compute_verify_token_budget,
    confidence_to_survival,
    load_sps_cost_table,
    load_sts_temperatures,
    schedule_verify_lengths,
)
from sgl_jax.srt.speculative.dspark_verify import (
    DSparkVerifyLayout,
    gather_compact_cache_locations,
)
from sgl_jax.srt.speculative.eagle_util import verify_compact_chain_greedy
from sgl_jax.srt.speculative.overlap_utils import resolve_spec_decode_token_ids
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dspark_target_outputs_keep_their_existing_sharding() -> None:
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("tensor",))
    logits = jax.device_put(
        jnp.arange(24, dtype=jnp.bfloat16).reshape(3, 8),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tensor")),
    )
    hidden = jax.device_put(
        jnp.arange(48, dtype=jnp.bfloat16).reshape(3, 16),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tensor")),
    )

    prepared_logits, prepared_hidden = prepare_target_outputs_for_verification(
        mesh,
        SpeculativeAlgorithm.DSPARK,
        logits,
        hidden,
    )

    assert prepared_logits is logits
    assert prepared_hidden is hidden
    assert prepared_logits.sharding == logits.sharding
    assert prepared_hidden.sharding == hidden.sharding


def test_gather_preserving_sharding_keeps_layout_and_values() -> None:
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        matrix = jax.device_put(
            jnp.arange(24, dtype=jnp.bfloat16).reshape(3, 8),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tensor")),
        )
        vector = jax.device_put(
            jnp.asarray([10, 20, 30], dtype=jnp.int32),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )
        indices = jnp.asarray([2, 0, 1], dtype=jnp.int32)

        gathered_matrix = gather_preserving_sharding(matrix, indices)
        gathered_vector = gather_preserving_sharding(vector, indices)

    assert gathered_matrix.sharding == matrix.sharding
    assert gathered_vector.sharding == vector.sharding
    np.testing.assert_array_equal(gathered_matrix, np.asarray(matrix)[[2, 0, 1]])
    np.testing.assert_array_equal(gathered_vector, [30, 10, 20])


def test_dspark_context_write_and_gather_are_batched_and_exact() -> None:
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, -1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    cache_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, None, "tensor", None),
    )
    row_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, "tensor", None),
    )
    context_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("data", None, "tensor", None),
    )
    with jax.set_mesh(mesh):
        keys = tuple(
            jax.device_put(jnp.zeros((4, 8, 8, 3), dtype=jnp.float32), cache_sharding)
            for _ in range(2)
        )
        values = tuple(jnp.zeros_like(key) for key in keys)
        encoded = DSparkContextCache(
            layers=tuple(
                DSparkLayerContext(
                    key=jnp.full((2, 1, 8, 3), layer + 1, dtype=jnp.float32),
                    value=jnp.full((2, 1, 8, 3), 10 + layer, dtype=jnp.float32),
                )
                for layer in range(2)
            )
        )
        request_indices = jnp.asarray([2, 0], dtype=jnp.int32)
        ring_positions = jnp.asarray([5, 3], dtype=jnp.int32)

        updated_keys, updated_values = _write_context_rows(
            keys,
            values,
            encoded,
            request_indices,
            ring_positions,
            cache_row_sharding=row_sharding,
        )
        gathered = _gather_context_rows(
            updated_keys,
            updated_values,
            request_indices[:, None],
            jnp.asarray([[5, 0], [3, 0]], dtype=jnp.int32),
            context_sharding=context_sharding,
        )

    for layer, layer_context in enumerate(gathered.layers):
        np.testing.assert_array_equal(layer_context.key[:, 0], layer + 1)
        np.testing.assert_array_equal(layer_context.key[:, 1], 0)
        np.testing.assert_array_equal(layer_context.value[:, 0], 10 + layer)
        np.testing.assert_array_equal(layer_context.value[:, 1], 0)


def _recurrent_schedule_batch(
    start: int,
    size: int,
    *,
    recurrent_indices: np.ndarray,
    recurrent_cow_src_indices: np.ndarray | None = None,
    recurrent_track_indices: np.ndarray | None = None,
    recurrent_track_mask: np.ndarray | None = None,
) -> ScheduleBatch:
    return ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[
                    SimpleNamespace(
                        rid=str(i),
                        return_logprob=False,
                        return_output_logprob_only=False,
                        stream=True,
                        grammar=None,
                    )
                    for i in range(start, start + size)
                ],
                req_pool_indices=np.arange(start, start + size, dtype=np.int32),
                seq_lens=np.full(size, 32, dtype=np.int32),
                recurrent_indices=recurrent_indices,
                recurrent_cow_src_indices=recurrent_cow_src_indices,
                recurrent_track_indices=recurrent_track_indices,
                recurrent_track_mask=recurrent_track_mask,
            )
        ],
        dp_size=1,
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=MagicMock(),
        model_config=MagicMock(),
        forward_mode=ForwardMode.DECODE,
    )


def test_schedule_batch_merge_preserves_recurrent_row_alignment() -> None:
    running = _recurrent_schedule_batch(
        0,
        16,
        recurrent_indices=np.arange(100, 116, dtype=np.int32),
        recurrent_cow_src_indices=np.arange(200, 216, dtype=np.int32),
        recurrent_track_indices=np.arange(300, 316, dtype=np.int32),
        recurrent_track_mask=np.ones(16, dtype=np.bool_),
    )
    admitted = _recurrent_schedule_batch(
        16,
        16,
        recurrent_indices=np.arange(116, 132, dtype=np.int32),
    )

    running.merge_batch(admitted)
    info = running.reqs_info[0]

    assert len(info.reqs) == 32
    np.testing.assert_array_equal(info.recurrent_indices, np.arange(100, 132, dtype=np.int32))
    np.testing.assert_array_equal(
        info.recurrent_cow_src_indices,
        np.concatenate([np.arange(200, 216, dtype=np.int32), np.zeros(16, dtype=np.int32)]),
    )
    np.testing.assert_array_equal(
        info.recurrent_track_indices,
        np.concatenate([np.arange(300, 316, dtype=np.int32), np.zeros(16, dtype=np.int32)]),
    )
    np.testing.assert_array_equal(
        info.recurrent_track_mask,
        np.concatenate([np.ones(16, dtype=np.bool_), np.zeros(16, dtype=np.bool_)]),
    )

    keep = np.asarray([0, 15, 16, 31], dtype=np.int32)
    running.filter_batch(keep_indices={0: keep})
    np.testing.assert_array_equal(
        running.reqs_info[0].recurrent_indices,
        np.asarray([100, 115, 116, 131], dtype=np.int32),
    )


def test_schedule_batch_merge_rejects_misaligned_recurrent_rows() -> None:
    running = _recurrent_schedule_batch(
        0,
        2,
        recurrent_indices=np.asarray([100], dtype=np.int32),
    )
    admitted = _recurrent_schedule_batch(
        2,
        1,
        recurrent_indices=np.asarray([102], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="RECURRENT_BATCH_FIELD_MISMATCH"):
        running.merge_batch(admitted)


def test_dspark_requires_exact_nvfp4_target_and_tokenizer() -> None:
    validate_inkling_small_dspark_target(
        INKLING_SMALL_NVFP4_REPO,
        INKLING_SMALL_NVFP4_REVISION,
        None,
    )

    with pytest.raises(ValueError, match="DSPARK_TARGET_MODEL_MISMATCH"):
        validate_inkling_small_dspark_target(
            "thinkingmachines/Inkling-Small",
            INKLING_SMALL_NVFP4_REVISION,
            None,
        )


def test_dspark_rejects_prefix_cache_until_context_state_can_be_restored() -> None:
    with pytest.raises(ValueError, match="DSPARK_PREFIX_CACHE_UNSUPPORTED"):
        ServerArgs(
            model_path=INKLING_SMALL_NVFP4_REPO,
            revision=INKLING_SMALL_NVFP4_REVISION,
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="RadixArk/Inkling-Small-DSpark",
        )
    with pytest.raises(ValueError, match="DSPARK_TARGET_REVISION_MISMATCH"):
        validate_inkling_small_dspark_target(
            INKLING_SMALL_NVFP4_REPO,
            "wrong",
            None,
        )
    with pytest.raises(ValueError, match="DSPARK_TARGET_TOKENIZER_MISMATCH"):
        validate_inkling_small_dspark_target(
            INKLING_SMALL_NVFP4_REPO,
            INKLING_SMALL_NVFP4_REVISION,
            "different-tokenizer",
        )


def test_dspark_can_admit_prefills_between_synchronous_verify_rounds() -> None:
    assert can_interleave_speculative_prefill(False, SpeculativeAlgorithm.DSPARK)
    assert not can_interleave_speculative_prefill(False, SpeculativeAlgorithm.EAGLE)
    assert can_interleave_speculative_prefill(True, SpeculativeAlgorithm.EAGLE)


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


def test_fixed_chain_uses_dspark_verify_layout_and_absolute_positions() -> None:
    verify_input = build_dspark_verify_input(
        anchor=np.asarray([10, 20, 30], dtype=np.int32),
        proposal_ids=np.asarray([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int32),
        prefix_lens=np.asarray([100, 200, 300], dtype=np.int32),
        selector=np.asarray([0, 2], dtype=np.int32),
        real_verify_lens=np.asarray([4, 4], dtype=np.int32),
        maximum_verify_tokens=4,
        seq_lens_sum=600,
    )

    np.testing.assert_array_equal(verify_input.verify_lens, [4, 0, 4])
    np.testing.assert_array_equal(
        verify_input.compact_positions,
        [100, 101, 102, 103, 300, 301, 302, 303],
    )
    np.testing.assert_array_equal(
        verify_input.compact_input_ids,
        [10, 11, 12, 13, 30, 31, 32, 33],
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


def test_dspark_lagged_confidence_rejects_stale_slot_generation() -> None:
    worker = object.__new__(DSparkDraftWorker)
    worker.gamma = 3
    worker.confidence_logits_by_slot = np.asarray(
        [[-10.0, -10.0, -10.0], [2.0, 1.0, 0.0]], dtype=np.float32
    )
    worker.confidence_generations = np.asarray([4, 7], dtype=np.int64)

    logits = worker._lagged_confidence_logits(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([5, 7], dtype=np.int64),
    )

    np.testing.assert_array_equal(logits[0], np.full(3, 20.0, dtype=np.float32))
    np.testing.assert_array_equal(logits[1], [2.0, 1.0, 0.0])


def test_dspark_planner_selects_prefix_preserving_compact_lengths() -> None:
    survival = confidence_to_survival(
        np.asarray(
            [
                [0.95, 0.90, 0.20],
                [0.80, 0.75, 0.70],
            ],
            dtype=np.float32,
        )
    )
    config = DSparkScheduleConfig(gamma=3)

    lengths = schedule_verify_lengths(survival, budget=3, config=config)

    np.testing.assert_array_equal(lengths, [3, 2])


def test_dspark_planner_uses_measured_step_rate_to_choose_budget() -> None:
    survival = np.asarray([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float64)
    table = SpsCostTable(
        sample_batch_tokens=(2, 4, 6, 8),
        sample_steps_per_sec=(100.0, 80.0, 40.0, 20.0),
        max_batch_tokens=8,
    )

    decision = compute_verify_token_budget(
        survival,
        table,
        DSparkScheduleConfig(gamma=3),
    )

    assert decision.budget == 3
    assert decision.predicted_step_seconds == pytest.approx(0.0125)


def test_dspark_planner_loads_pinned_sps_and_sts_tables(tmp_path) -> None:
    sps_path = tmp_path / "sps.json"
    sps_path.write_text(
        '{"sample_batch_tokens":[1,8],"sample_steps_per_sec":[10.0,5.0],"max_batch_tokens":8}'
    )
    sts_path = tmp_path / "sts.json"
    sts_path.write_text('{"temperatures":[1.0,2.0,3.0]}')

    table = load_sps_cost_table(sps_path)
    temperatures = load_sts_temperatures(sts_path, gamma=3)

    assert table.sample_batch_tokens == (1, 8)
    assert table.max_batch_tokens == 8
    np.testing.assert_array_equal(temperatures, [1.0, 2.0, 3.0])


def test_dspark_planner_rejects_unmeasured_token_counts() -> None:
    table = SpsCostTable(
        sample_batch_tokens=(1, 8),
        sample_steps_per_sec=(10.0, 5.0),
        max_batch_tokens=8,
    )

    with pytest.raises(ValueError, match="DSPARK_SPS_TABLE_COVERAGE_INSUFFICIENT"):
        compute_verify_token_budget(
            np.full((3, 3), 0.5, dtype=np.float64),
            table,
            DSparkScheduleConfig(gamma=3),
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


def test_dspark_state_merge_drops_one_sided_round_metadata_for_new_admission() -> None:
    running = DSparkDraftInput(
        verified_id=np.asarray([11, 12], dtype=np.int32),
        allocate_lens=np.asarray([20, 21], dtype=np.int32),
        new_seq_lens=np.asarray([21, 22], dtype=np.int32),
        accept_length_cpu=np.asarray([1, 1], dtype=np.int32),
    )
    admitted = DSparkDraftInput(
        verified_id=np.asarray([13], dtype=np.int32),
        allocate_lens=np.asarray([7], dtype=np.int32),
    )

    running.merge_batch(admitted)

    np.testing.assert_array_equal(running.verified_id, [11, 12, 13])
    np.testing.assert_array_equal(running.allocate_lens, [20, 21, 7])
    assert running.new_seq_lens is None
    assert running.accept_length_cpu is None


def test_dspark_state_merge_requires_anchor_and_allocation_for_every_request() -> None:
    running = DSparkDraftInput(
        verified_id=np.asarray([11], dtype=np.int32),
        allocate_lens=np.asarray([20], dtype=np.int32),
    )
    admitted = DSparkDraftInput(
        verified_id=np.asarray([12], dtype=np.int32),
        allocate_lens=None,
    )

    with pytest.raises(ValueError, match="DSPARK_REQUIRED_STATE_MISSING field=allocate_lens"):
        running.merge_batch(admitted)


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


def test_compact_chain_verification_handles_full_and_partial_acceptance() -> None:
    predict, verified, accept_lens, accept_index = verify_compact_chain_greedy(
        np.asarray(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 0],
            ],
            dtype=np.int32,
        ),
        np.asarray([11, 12, 13, 14, 99, 22, 23], dtype=np.int32),
        np.asarray([4, 3], dtype=np.int32),
    )

    np.testing.assert_array_equal(accept_lens, [4, 1])
    np.testing.assert_array_equal(predict.reshape(2, 4)[0], [11, 12, 13, 14])
    np.testing.assert_array_equal(predict.reshape(2, 4)[1], [99, 0, 0, 0])
    np.testing.assert_array_equal(verified, predict)
    np.testing.assert_array_equal(
        accept_index.reshape(2, 4),
        [[0, 1, 2, 3], [4, -1, -1, -1]],
    )


def test_dspark_verify_layout_maps_compact_and_fixed_stride_rows() -> None:
    layout = DSparkVerifyLayout.from_verify_lens(
        np.asarray([4, 1, 0, 3], dtype=np.int32),
        maximum_verify_tokens=4,
    )

    np.testing.assert_array_equal(layout.query_indptr, [0, 4, 5, 5, 8])
    np.testing.assert_array_equal(layout.compact_to_strided, [0, 1, 2, 3, 4, 12, 13, 14])
    np.testing.assert_array_equal(
        layout.strided_to_compact,
        [[0, 1, 2, 3], [4, -1, -1, -1], [-1, -1, -1, -1], [5, 6, 7, -1]],
    )
    assert layout.graph_num_tokens == 8


def test_compact_cache_locations_follow_request_slots_and_positions() -> None:
    req_to_token = np.asarray(
        [
            [101, 102, 103, 104, 105, 106],
            [201, 202, 203, 204, 205, 206],
            [301, 302, 303, 304, 305, 306],
            [401, 402, 403, 404, 405, 406],
        ],
        dtype=np.int32,
    )
    compact = gather_compact_cache_locations(
        req_to_token,
        np.asarray([3, 1, 2], dtype=np.int32),
        np.asarray([1, 2, 3, 4, 2, 4], dtype=np.int32),
        np.asarray([4, 0, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        compact,
        np.asarray([402, 403, 404, 405, 303, 305], dtype=np.int32),
    )


def test_compact_cache_locations_reject_position_count_mismatch() -> None:
    with pytest.raises(ValueError, match="DSPARK_POSITION_LAYOUT_SHAPE_MISMATCH"):
        gather_compact_cache_locations(
            np.arange(40, dtype=np.int32).reshape(4, 10),
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([3, 4, 5, 6], dtype=np.int32),
            np.asarray([2, 3], dtype=np.int32),
        )


def test_compact_cache_locations_are_stable_across_uneven_round_allocations() -> None:
    req_to_token = np.zeros((5, 16), dtype=np.int32)
    request_indices = np.asarray([4, 1, 3], dtype=np.int32)
    rounds = (
        (np.asarray([4, 1, 3]), np.asarray([2, 3, 4, 7, 8, 11, 12, 13])),
        (np.asarray([1, 4, 2]), np.asarray([8, 14, 9, 10, 11, 12, 15])),
        (np.asarray([3, 2, 1]), np.asarray([9, 10, 11, 13, 14, 12])),
    )
    for round_index, (verify_lens, positions) in enumerate(rounds, start=1):
        for request, position in zip(
            np.repeat(request_indices, verify_lens), positions, strict=True
        ):
            req_to_token[request, position] = round_index * 1000 + request * 100 + position

        compact = gather_compact_cache_locations(
            req_to_token,
            request_indices,
            positions,
            verify_lens,
        )

        expected = np.asarray(
            [
                req_to_token[request, position]
                for request, position in zip(
                    np.repeat(request_indices, verify_lens), positions, strict=True
                )
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(compact, expected)


def test_compact_chain_verification_zeroes_padding_rows() -> None:
    predict, verified, accept_lens, accept_index = verify_compact_chain_greedy(
        np.asarray([[10, 11], [0, 0], [20, 21]], dtype=np.int32),
        np.asarray([11, 12, 99, 22], dtype=np.int32),
        np.asarray([2, 0, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(accept_lens, [2, 0, 1])
    np.testing.assert_array_equal(predict.reshape(3, 2), [[11, 12], [0, 0], [99, 0]])
    np.testing.assert_array_equal(verified, predict)
    np.testing.assert_array_equal(accept_index.reshape(3, 2), [[0, 1], [-1, -1], [2, -1]])
