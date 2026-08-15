import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.relative_position_bias import relative_position_bias_pallas


class TestRelativePositionBiasKernel(unittest.TestCase):
    def test_matches_reference_for_packed_sequences_and_partial_tiles(self):
        empty_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[0], dtype=object), ())
        with jax.sharding.set_mesh(empty_mesh):
            query_positions = jnp.asarray([2, 1, 4], dtype=jnp.int32)
            query_batch_ids = jnp.asarray([0, 1, 2], dtype=jnp.int32)
            query_key_starts = jnp.asarray([0, 3, 5], dtype=jnp.int32)
            key_batch_ids = jnp.asarray([0] * 3 + [1] * 2 + [2] * 5, dtype=jnp.int32)
            key_positions = jnp.asarray([0, 1, 2, 0, 1, 0, 1, 2, 3, 4], dtype=jnp.int32)
            relative_states = (
                jnp.arange(3 * 9 * 16, dtype=jnp.float32).reshape(3, 9, 16) / 100
            ).astype(jnp.bfloat16)
            projection = (
                jnp.arange(16 * 7, dtype=jnp.float32).reshape(16, 7) / 50
            ).astype(jnp.bfloat16)
            actual = relative_position_bias_pallas(
                relative_states,
                projection,
                query_positions,
                query_key_starts,
                key_positions.shape[0],
                interpret=True,
                block_keys=128,
            )

        distance = query_positions[:, None] - key_positions[None, :]
        selected = jnp.take(projection, jnp.clip(distance, 0, projection.shape[1] - 1), axis=1)
        selected = selected.transpose(1, 2, 0)
        expected = jnp.einsum(
            "qhd,qkd->qhk",
            relative_states,
            selected,
            preferred_element_type=jnp.float32,
        )
        valid = (
            (query_batch_ids[:, None] == key_batch_ids[None, :])
            & (distance >= 0)
            & (distance < projection.shape[1])
        )
        expected = jnp.where(valid[:, None, :], expected, 0.0)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
