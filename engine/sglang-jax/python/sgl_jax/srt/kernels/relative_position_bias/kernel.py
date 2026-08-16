import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P


def _relative_position_bias_kernel(
    query_positions_ref,
    query_key_starts_ref,
    relative_states_ref,
    projection_ref,
    output_ref,
    *,
    key_len: int,
    extent: int,
    work_len: int,
):
    query_index = pl.program_id(0)
    query_position = query_positions_ref[query_index]
    query_key_start = query_key_starts_ref[query_index]
    position_count = jnp.minimum(query_position + 1, extent)
    output_start = query_key_start + jnp.maximum(query_position - extent + 1, 0)
    projection_start = extent - jnp.minimum(query_position + 1, extent)
    states = relative_states_ref[:, :, :][0].astype(jnp.float32)
    bias_sequence = pl.dot(states, projection_ref[:, :])
    bias_source = jnp.pad(bias_sequence, ((0, 0), (0, work_len - extent)))
    key_indices = jnp.arange(work_len)
    local_positions = key_indices - output_start
    shift = jnp.mod(output_start - projection_start, work_len)
    bias = pltpu.roll(bias_source, shift, axis=1)
    valid = (
        (key_indices < key_len)
        & (local_positions >= 0)
        & (local_positions < position_count)
    )
    output_ref[:, :, :] = jnp.where(valid[None, None, :], bias[None, :, :], 0.0)


def _relative_position_bias_pallas_local(
    relative_states: jax.Array,
    projection: jax.Array,
    query_positions: jax.Array,
    query_key_starts: jax.Array,
    key_len: int,
    *,
    interpret: bool = False,
    block_keys: int = 128,
) -> jax.Array:
    if relative_states.ndim != 3:
        raise ValueError("relative_states must have shape [queries, heads, relative_dim]")
    if projection.ndim != 2:
        raise ValueError("projection must have shape [relative_dim, extent]")
    query_len, num_heads, relative_dim = relative_states.shape
    if projection.shape[0] != relative_dim:
        raise ValueError("projection and relative_states must have the same relative dimension")
    if query_positions.shape != (query_len,) or query_key_starts.shape != (query_len,):
        raise ValueError("query metadata must have one entry per query")
    if key_len <= 0 or block_keys <= 0:
        raise ValueError("key length and block size must be positive")

    padded_key_len = (key_len + block_keys - 1) // block_keys * block_keys
    extent = projection.shape[1]
    work_len = max(padded_key_len, extent)
    padded_num_heads = (num_heads + 7) // 8 * 8
    padded_relative_dim = (relative_dim + 127) // 128 * 128
    relative_states = jnp.pad(
        relative_states,
        (
            (0, 0),
            (0, padded_num_heads - num_heads),
            (0, padded_relative_dim - relative_dim),
        ),
    )
    projection = jnp.pad(
        projection[:, ::-1],
        ((0, padded_relative_dim - relative_dim), (0, 0)),
    )
    kernel = functools.partial(
        _relative_position_bias_kernel,
        key_len=key_len,
        extent=extent,
        work_len=work_len,
    )
    compiler_params = None
    if not interpret:
        compiler_params = pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        )
    vmem = None if interpret else pltpu.VMEM
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (query_len, padded_num_heads, work_len),
            jnp.float32,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(query_len,),
            in_specs=(
                pl.BlockSpec(
                    (1, padded_num_heads, padded_relative_dim),
                    lambda query, *_: (query, 0, 0),
                    memory_space=vmem,
                ),
                pl.BlockSpec(projection.shape, lambda query, *_: (0, 0), memory_space=vmem),
            ),
            out_specs=pl.BlockSpec(
                (1, padded_num_heads, work_len),
                lambda query, *_: (query, 0, 0),
                memory_space=vmem,
            ),
        ),
        compiler_params=compiler_params,
        interpret=interpret,
        name="inkling_relative_position_bias",
    )(
        query_positions,
        query_key_starts,
        relative_states,
        projection,
    )[:, :num_heads, :key_len]


def relative_position_bias_pallas(
    relative_states: jax.Array,
    projection: jax.Array,
    query_positions: jax.Array,
    query_key_starts: jax.Array,
    key_len: int,
    *,
    interpret: bool = False,
    block_keys: int = 128,
) -> jax.Array:
    local_impl = functools.partial(
        _relative_position_bias_pallas_local,
        key_len=key_len,
        interpret=interpret,
        block_keys=block_keys,
    )
    if interpret:
        return local_impl(relative_states, projection, query_positions, query_key_starts)

    return jax.shard_map(
        local_impl,
        in_specs=(P(None, "tensor", None), P(), P(), P()),
        out_specs=P(None, "tensor", None),
        check_vma=False,
    )(relative_states, projection, query_positions, query_key_starts)
