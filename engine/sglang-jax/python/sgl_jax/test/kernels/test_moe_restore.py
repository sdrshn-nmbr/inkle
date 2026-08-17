import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.moe_restore import fused_restore_weight_reduce


def test_fused_restore_weight_reduce_matches_epmoe_reference_exactly():
    token_count = 4
    top_k = 3
    hidden_size = 128
    route_count = token_count * top_k
    key = jax.random.key(7)
    intermediate = jax.random.normal(key, (route_count, hidden_size), dtype=jnp.bfloat16)
    sorted_routes = jnp.array([5, 0, 9, 3, 7, 1, 10, 4, 8, 2, 11, 6], dtype=jnp.int32)
    inverse_routes = (
        jnp.zeros(route_count, dtype=jnp.int32)
        .at[sorted_routes]
        .set(jnp.arange(route_count, dtype=jnp.int32))
    )
    weights = jax.random.uniform(jax.random.key(8), (token_count, top_k), dtype=jnp.float32)
    restored = jnp.take(intermediate, inverse_routes, axis=0).reshape(
        token_count, top_k, hidden_size
    )
    expected = jnp.einsum(
        "TKE,TK->TE",
        restored.astype(jnp.float32),
        weights,
    ).astype(jnp.bfloat16)

    actual = fused_restore_weight_reduce(
        intermediate,
        inverse_routes,
        weights,
        top_k=top_k,
        interpret=True,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
