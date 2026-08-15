import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    prepare_relative_bias_inputs,
    ref_ragged_paged_attention,
)


class TestFusedRelativeAttention(unittest.TestCase):
    def test_reference_adds_relative_bias_before_softmax(self):
        queries = (
            jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4) / 17
        ).astype(jnp.bfloat16)
        keys = (
            jnp.arange(2 * 4 * 2 * 4, dtype=jnp.float32).reshape(2, 4, 2, 4) / 23
        ).astype(jnp.bfloat16)
        values = (
            jnp.arange(2 * 4 * 2 * 4, dtype=jnp.float32).reshape(2, 4, 2, 4) / 29
        ).astype(jnp.bfloat16)
        relative_states = (
            jnp.arange(2 * 2 * 3, dtype=jnp.float32).reshape(2, 2, 3) / 31
        ).astype(jnp.bfloat16)
        relative_projection = (
            jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4) / 37
        ).astype(jnp.bfloat16)
        lengths = jnp.asarray([3, 2], dtype=jnp.int32)

        actual = ref_ragged_paged_attention(
            queries,
            keys,
            values,
            lengths,
            jnp.asarray([[0], [1]], dtype=jnp.int32),
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
            sm_scale=0.5,
            relative_states=relative_states,
            relative_projection=relative_projection,
        )

        expected = []
        for sequence, length in enumerate((3, 2)):
            q = queries[sequence]
            k = keys[sequence, :length]
            v = values[sequence, :length]
            logits = jnp.einsum(
                "hd,khd->hk",
                q,
                k,
                preferred_element_type=jnp.float32,
            ) * 0.5
            distances = length - 1 - jnp.arange(length)
            selected_projection = relative_projection[:, distances]
            bias = jnp.einsum(
                "hd,dk->hk",
                relative_states[sequence],
                selected_projection,
                preferred_element_type=jnp.float32,
            )
            weights = jax.nn.softmax(logits + bias, axis=-1)
            expected.append(jnp.einsum("hk,khd->hd", weights, v).astype(jnp.bfloat16))

        np.testing.assert_allclose(actual, jnp.stack(expected), rtol=2e-2, atol=2e-2)

    def test_preparation_reverses_projection_and_pads_tpu_tiles(self):
        states = jnp.ones((2, 4, 3), dtype=jnp.bfloat16)
        projection = jnp.arange(15, dtype=jnp.bfloat16).reshape(3, 5)
        prepared_states, prepared_projection = prepare_relative_bias_inputs(
            states,
            projection,
            actual_num_kv_heads=2,
            actual_num_q_heads_per_kv_head=2,
            q_packing=2,
            work_len=128,
        )

        self.assertEqual(prepared_states.shape, (2, 2, 2, 128))
        self.assertEqual(prepared_projection.shape, (128, 261))
        np.testing.assert_array_equal(
            prepared_projection[:3, 128:133], projection[:, ::-1]
        )
        np.testing.assert_array_equal(prepared_projection[:3, :128], 0)
        np.testing.assert_array_equal(prepared_projection[:3, 133:], 0)
        np.testing.assert_array_equal(prepared_projection[3:], 0)

    def test_block_projection_selection_matches_relative_distances(self):
        extent = 1024
        block_size = 128
        work_len = 4096
        projection = np.pad(
            np.arange(extent, dtype=np.int32)[::-1],
            (work_len, work_len),
        )

        for query_position in (0, 127, 1023, 1536, 4095):
            for key_start in range(0, query_position + 1, block_size):
                projection_start = (
                    work_len
                    + extent
                    - 1
                    - query_position
                    + key_start
                )
                aligned_projection_start = projection_start - projection_start % 128
                projection_offset = projection_start - aligned_projection_start
                window = projection[
                    aligned_projection_start : aligned_projection_start
                    + block_size
                    + 128
                ]
                block = np.roll(
                    window,
                    block_size + 128 - projection_offset,
                )[:block_size]
                key_positions = key_start + np.arange(block_size)
                valid = (key_positions <= query_position) & (
                    key_positions > query_position - extent
                )

                np.testing.assert_array_equal(
                    block[valid],
                    query_position - key_positions[valid],
                )


if __name__ == "__main__":
    unittest.main()
