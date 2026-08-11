import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.causal_conv1d.causal_conv1d import (
    ConvConfigs,
    preprocess_metadata,
)
from native_sglang_parity import replicate_kv_head_blocks
from sgl_jax.srt.configs.inkling import InklingConfig
from sgl_jax.srt.models.inkling import get_recurrent_state_row
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from tpu_inkling_validation import (
    align_kv_heads_for_tensor_parallel,
    put_global,
    routes_match_or_tie,
    slice_bounds,
    token_rows,
)


def test_routes_match_exactly() -> None:
    routes = np.asarray([[1, 2, 3]])
    scores = np.asarray([[0.1, 0.9, 0.8, 0.7]])
    assert routes_match_or_tie(routes, routes, scores) == (True, False)


def test_route_order_is_semantically_irrelevant() -> None:
    candidate = np.asarray([[1, 2, 3]])
    reference = np.asarray([[2, 1, 3]])
    tied_scores = np.asarray([[0.1, 0.9, 0.9, 0.7]])
    distinct_scores = np.asarray([[0.1, 0.9, 0.8, 0.7]])
    assert routes_match_or_tie(candidate, reference, tied_scores) == (True, True)
    assert routes_match_or_tie(candidate, reference, distinct_scores) == (True, True)


def test_cutoff_route_can_change_only_at_a_tie() -> None:
    candidate = np.asarray([[1, 2]])
    reference = np.asarray([[1, 3]])
    tied_scores = np.asarray([[0.1, 0.9, 0.8, 0.8]])
    distinct_scores = np.asarray([[0.1, 0.9, 0.8, 0.7]])
    assert routes_match_or_tie(candidate, reference, tied_scores) == (True, True)
    assert routes_match_or_tie(candidate, reference, distinct_scores) == (False, False)


def test_cutoff_route_accepts_one_bfloat16_step() -> None:
    cutoff = np.asarray(1.6796875, dtype=ml_dtypes.bfloat16)
    adjacent = np.nextafter(
        cutoff,
        np.asarray(np.inf, dtype=ml_dtypes.bfloat16),
    )
    scores = np.asarray([[0.0, 2.0, cutoff, adjacent]], dtype=ml_dtypes.bfloat16)
    candidate = np.asarray([[1, 3]])
    reference = np.asarray([[1, 2]])
    assert routes_match_or_tie(candidate, reference, scores) == (True, True)


def test_cutoff_route_accepts_a_cpu_proven_tie() -> None:
    scores = np.asarray([[0.0, 2.0, 1.5, 1.75]], dtype=ml_dtypes.bfloat16)
    candidate = np.asarray([[1, 3]])
    reference = np.asarray([[1, 2]])
    assert routes_match_or_tie(
        candidate,
        reference,
        scores,
        proven_ties={0: frozenset({2, 3})},
    ) == (True, True)


def test_slice_bounds_rejects_strided_callbacks() -> None:
    assert slice_bounds(3, 16) == (3, 4)
    assert slice_bounds(slice(4, 8), 16) == (4, 8)
    try:
        slice_bounds(slice(0, 8, 2), 16)
    except ValueError as error:
        assert "INKLING_EXPERT_SLICE_STEP_UNSUPPORTED" in str(error)
    else:
        raise AssertionError("expected a strided slice to fail")


def test_causal_conv_metadata_uses_rank_one_smem_values() -> None:
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()),
        ("test",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    with jax.set_mesh(mesh):
        metadata = preprocess_metadata(
            cfgs=ConvConfigs(batch_size=8, dim_size=128, kernel_size=4, tile_size=8),
            query_start_loc=jnp.asarray([0, 3, 5], dtype=jnp.int32),
            state_indices=jnp.asarray([0, 1], dtype=jnp.int32),
            has_initial_state=jnp.asarray([0, 0], dtype=jnp.int32),
            num_seqs=jnp.asarray(2, dtype=jnp.int32),
        )
    assert metadata.num_tiles.shape == (1,)


def test_kv_replication_repeats_each_head_as_an_adjacent_block() -> None:
    weight = np.arange(8).reshape(2, 4)
    replicated = replicate_kv_head_blocks(
        weight,
        target_heads=4,
        head_dim=2,
        axis=1,
    )
    np.testing.assert_array_equal(
        replicated,
        np.asarray(
            [
                [0, 1, 0, 1, 2, 3, 2, 3],
                [4, 5, 4, 5, 6, 7, 6, 7],
            ]
        ),
    )


def test_kv_head_counts_align_with_tensor_mesh() -> None:
    config = InklingConfig(
        text_config={
            "num_key_value_heads": 8,
            "swa_num_key_value_heads": 16,
        }
    )
    align_kv_heads_for_tensor_parallel(config, tensor_parallel_size=16)
    assert config.text_config.num_key_value_heads == 16
    assert config.text_config.swa_num_key_value_heads == 16


def test_token_rows_keeps_explicit_data_sharding() -> None:
    mesh = create_device_mesh([1, jax.device_count()], [1, 1])
    value = put_global(
        np.arange(12).reshape(3, 4),
        mesh,
        jax.sharding.PartitionSpec("data", None),
    )
    sliced = token_rows(value, mesh, 1, 3)
    np.testing.assert_array_equal(np.asarray(sliced), np.arange(12).reshape(3, 4)[1:3])


def test_recurrent_state_row_keeps_explicit_tensor_sharding() -> None:
    mesh = create_device_mesh([1, jax.device_count()], [1, 1])
    values = np.arange(24).reshape(2, 4, 3)
    state_table = put_global(
        values,
        mesh,
        jax.sharding.PartitionSpec("data", "tensor", None),
    )
    row = get_recurrent_state_row(state_table, 0, mesh)
    np.testing.assert_array_equal(np.asarray(row), values[0])
