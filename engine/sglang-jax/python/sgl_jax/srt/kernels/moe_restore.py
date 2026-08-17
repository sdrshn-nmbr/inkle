import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _restore_weight_reduce_kernel(
    intermediate_ref,
    inverse_routes_ref,
    weights_ref,
    output_ref,
    *,
    token_count: int,
    top_k: int,
    block_hidden: int,
):
    outputs = []
    for token_index in range(token_count):
        accumulator = jnp.zeros((block_hidden,), dtype=jnp.float32)
        for route_index in range(top_k):
            flat_route = token_index * top_k + route_index
            source_row = inverse_routes_ref[flat_route]
            values = intermediate_ref[source_row, :].astype(jnp.float32)
            weight = weights_ref[token_index, route_index].astype(jnp.float32)
            accumulator = accumulator + values * weight
        outputs.append(accumulator)
    output_ref[:, :] = jnp.stack(outputs).astype(output_ref.dtype)


def fused_restore_weight_reduce(
    intermediate: jax.Array,
    inverse_routes: jax.Array,
    weights: jax.Array,
    *,
    top_k: int,
    interpret: bool = False,
    block_hidden: int = 128,
    output_dtype: jnp.dtype | None = None,
) -> jax.Array:
    if intermediate.ndim != 2:
        raise ValueError("intermediate must have shape [routes, hidden_size]")
    if inverse_routes.ndim != 1:
        raise ValueError("inverse_routes must have shape [routes]")
    if weights.ndim != 2:
        raise ValueError("weights must have shape [tokens, top_k]")
    token_count, weight_top_k = weights.shape
    route_count, hidden_size = intermediate.shape
    if weight_top_k != top_k:
        raise ValueError(f"weights top_k {weight_top_k} does not match top_k {top_k}")
    if route_count != token_count * top_k or inverse_routes.shape[0] != route_count:
        raise ValueError("route dimensions do not match token_count * top_k")
    output_dtype = output_dtype or intermediate.dtype
    if hidden_size % block_hidden != 0:
        restored = jnp.take(intermediate, inverse_routes, axis=0).reshape(
            token_count, top_k, hidden_size
        )
        return jnp.einsum(
            "TKE,TK->TE",
            restored.astype(jnp.float32),
            weights.astype(jnp.float32),
        ).astype(output_dtype)

    memory_space = None if interpret else pltpu.VMEM
    kernel = functools.partial(
        _restore_weight_reduce_kernel,
        token_count=token_count,
        top_k=top_k,
        block_hidden=block_hidden,
    )
    compiler_params = None
    if not interpret:
        compiler_params = pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((token_count, hidden_size), output_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(hidden_size // block_hidden,),
            in_specs=(
                pl.BlockSpec(
                    (route_count, block_hidden),
                    lambda hidden_block: (0, hidden_block),
                    memory_space=memory_space,
                ),
                pl.BlockSpec(
                    (route_count,),
                    lambda hidden_block: (0,),
                    memory_space=memory_space,
                ),
                pl.BlockSpec(
                    (token_count, top_k),
                    lambda hidden_block: (0, 0),
                    memory_space=memory_space,
                ),
            ),
            out_specs=pl.BlockSpec(
                (token_count, block_hidden),
                lambda hidden_block: (0, hidden_block),
                memory_space=memory_space,
            ),
        ),
        compiler_params=compiler_params,
        interpret=interpret,
        name="epmoe_restore_weight_reduce",
    )(intermediate, inverse_routes, weights)
