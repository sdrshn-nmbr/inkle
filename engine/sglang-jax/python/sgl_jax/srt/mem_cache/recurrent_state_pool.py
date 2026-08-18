"""RecurrentStatePool -- buffer pool for linear recurrent layers (KDA/GDN)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

# Module-level cache for jitted zero-allocators (recurrent/conv buffers).
_RECURRENT_ZERO_ALLOCATOR_CACHE: dict = {}


def _get_recurrent_zero_allocator(shape, dtype, sharding):
    """Return a cached jax.jit(jnp.zeros) allocator for recurrent/conv buffers."""
    key = (
        id(sharding.mesh),
        tuple(shape),
        str(jnp.dtype(dtype)),
        repr(sharding.spec),
        getattr(sharding, "memory_kind", None),
    )
    if key not in _RECURRENT_ZERO_ALLOCATOR_CACHE:
        _RECURRENT_ZERO_ALLOCATOR_CACHE[key] = jax.jit(
            partial(jnp.zeros, shape=tuple(shape), dtype=dtype),
            out_shardings=sharding,
        )
    return _RECURRENT_ZERO_ALLOCATOR_CACHE[key]


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _resolve_dtype(env_var: str, default):
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


@dataclass(frozen=True)
class RecurrentStateDType:
    conv: jnp.dtype
    temporal: jnp.dtype


@dataclass(frozen=True)
class LinearRecurrentStateParams:
    layers: list[int]
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    dtype: RecurrentStateDType
    # GDN has asymmetric K vs V projection widths (e.g.
    # Qwen3.5 GDN: num_k_heads=16/head_k_dim=128 vs num_v_heads=32/head_v_dim=128).
    # When None (KDA / Lightning / Bailing), RecurrentStatePool falls back to
    # treating K dim = V dim.
    num_k_heads: int | None = None
    head_k_dim: int | None = None
    conv_channel_sizes: list[list[int]] | None = None
    has_temporal_state: bool = True


@register_pytree_node_class
@dataclass(frozen=True)
class RecurrentConvStateTransaction:
    """Temporary convolution inputs produced by fixed-chain target verification."""

    candidate_inputs: tuple[tuple[jax.Array, ...], ...]
    recurrent_indices: jax.Array
    draft_token_num: int

    def tree_flatten(self):
        return (self.candidate_inputs, self.recurrent_indices), self.draft_token_num

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        candidate_inputs, recurrent_indices = children
        return cls(candidate_inputs, recurrent_indices, aux_data)


def commit_packed_convolution_state(
    state_table: jax.Array,
    candidate_inputs: jax.Array,
    recurrent_indices: jax.Array,
    accepted_lengths: jax.Array,
    draft_token_num: int,
) -> jax.Array:
    """Commit only each request's accepted fixed-chain candidate prefix."""
    batch_size = recurrent_indices.shape[0]
    if candidate_inputs.shape[0] != batch_size * draft_token_num:
        raise ValueError(
            "RECURRENT_CANDIDATE_LAYOUT_MISMATCH "
            f"tokens={candidate_inputs.shape[0]} batch={batch_size} "
            f"draft_token_num={draft_token_num}"
        )
    if accepted_lengths.shape != recurrent_indices.shape:
        raise ValueError(
            "RECURRENT_ACCEPT_LENGTH_SHAPE_MISMATCH "
            f"accepted={accepted_lengths.shape} indices={recurrent_indices.shape}"
        )

    cache_width = state_table.shape[-1]
    state_sharding = jax.typeof(state_table).sharding
    candidates = candidate_inputs.reshape(
        batch_size,
        draft_token_num,
        candidate_inputs.shape[-1],
    ).swapaxes(1, 2)
    candidates = jax.sharding.reshard(candidates, state_sharding)
    safe_indices = jnp.maximum(recurrent_indices, 0)
    old_rows = state_table.at[safe_indices].get(out_sharding=state_sharding)
    history = jnp.concatenate((old_rows, candidates.astype(state_table.dtype)), axis=-1)
    accepted_lengths = jnp.clip(accepted_lengths, 0, draft_token_num)
    history_indices = (
        accepted_lengths[:, None] + jnp.arange(cache_width, dtype=accepted_lengths.dtype)[None, :]
    )
    committed_rows = jnp.take_along_axis(history, history_indices[:, None, :], axis=2)
    updated = state_table.at[safe_indices].set(
        committed_rows,
        out_sharding=state_sharding,
    )
    row_zero = state_table.at[0].get(
        out_sharding=NamedSharding(state_sharding.mesh, P("tensor", None))
    )
    return updated.at[0].set(row_zero, out_sharding=state_sharding)


def commit_convolution_transaction_buffers(
    conv_buffers: tuple[tuple[jax.Array, ...], ...],
    candidate_inputs: tuple[tuple[jax.Array, ...], ...],
    recurrent_indices: jax.Array,
    accepted_lengths: jax.Array,
    draft_token_num: int,
) -> tuple[tuple[jax.Array, ...], ...]:
    return tuple(
        tuple(
            commit_packed_convolution_state(
                state_table,
                candidates,
                recurrent_indices,
                accepted_lengths,
                draft_token_num,
            )
            for candidates, state_table in zip(layer_candidates, layer_buffers, strict=True)
        )
        for layer_candidates, layer_buffers in zip(candidate_inputs, conv_buffers, strict=True)
    )


_jitted_commit_convolution_transaction_buffers = jax.jit(
    commit_convolution_transaction_buffers,
    static_argnames=("draft_token_num",),
    donate_argnames=("conv_buffers",),
)


def recurrent_state_dtype() -> RecurrentStateDType:
    return RecurrentStateDType(
        conv=_resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16),
        temporal=_resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32),
    )


@register_pytree_node_class
class RecurrentStatePool:

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        mesh: Mesh,
        dp_size: int = 1,
        recurrent_partition_axis: str = "tensor",
        conv_partition_axis: str = "tensor",
        data_partition_axis: str = "data",
        temporal_dtype=None,
        conv_dtype=None,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        conv_channel_sizes: list[list[int]] | None = None,
        has_temporal_state: bool = True,
    ):
        """`size` is the **global** number of valid slots across all DP ranks
        (mirrors MHATokenToKVPool.size semantics). Internally we partition by
        DP: each rank gets `size // dp_size` valid slots + 1 dummy slot at
        index 0, so total buffer slots = size + dp_size.
        """
        state_dtype = recurrent_state_dtype()
        if temporal_dtype is None:
            temporal_dtype = state_dtype.temporal
        if conv_dtype is None:
            conv_dtype = state_dtype.conv
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids), (
            f"linear_recurrent_layer_ids must not contain duplicates, "
            f"got {linear_recurrent_layer_ids}"
        )
        self.linear_recurrent_layer_ids: list[int] = list(linear_recurrent_layer_ids)
        self.layers_mapping: dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(self.linear_recurrent_layer_ids)
        }
        self.num_linear_recurrent_layers: int = len(self.linear_recurrent_layer_ids)

        assert (
            size % dp_size == 0
        ), f"RecurrentStatePool size ({size}) must be divisible by dp_size ({dp_size})."

        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size
        self.has_temporal_state = has_temporal_state
        self.conv_channel_sizes = (
            [list(sizes) for sizes in conv_channel_sizes]
            if conv_channel_sizes is not None
            else None
        )
        if self.conv_channel_sizes is not None and len(self.conv_channel_sizes) != len(
            self.linear_recurrent_layer_ids
        ):
            raise ValueError(
                "RECURRENT_CONV_LAYOUT_MISMATCH "
                f"layers={len(self.linear_recurrent_layer_ids)} "
                f"layouts={len(self.conv_channel_sizes)}"
            )

        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k

        # Each rank reserves slot 0 as a dummy → +1 per rank.
        self.total_slots = size + dp_size

        self.mesh = mesh
        self.recurrent_partition_axis = recurrent_partition_axis
        self.conv_partition_axis = conv_partition_axis
        self.data_partition_axis = data_partition_axis

        recurrent_axis_size = mesh.shape[recurrent_partition_axis]
        conv_axis_size = mesh.shape[conv_partition_axis]
        if has_temporal_state:
            assert num_heads % recurrent_axis_size == 0, (
                f"num_heads {num_heads} must be divisible by "
                f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
            )
            assert num_k_heads % recurrent_axis_size == 0, (
                f"num_k_heads {num_k_heads} must be divisible by "
                f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
            )
        conv_sizes = (
            [self.proj_size]
            if self.conv_channel_sizes is None
            else [size for sizes in self.conv_channel_sizes for size in sizes]
        )
        for channels in conv_sizes:
            assert channels % conv_axis_size == 0, (
                f"conv channels {channels} must be divisible by "
                f"'{conv_partition_axis}' size {conv_axis_size}"
            )

        self.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        self.conv_sharding = NamedSharding(mesh, P(data_partition_axis, conv_partition_axis, None))

        self.recurrent_buffers, self.conv_buffers = self._create_buffers()

    def _create_buffers(self) -> tuple[list, list]:
        recurrent_heads = (
            self.num_heads
            if self.has_temporal_state
            else self.mesh.shape[self.recurrent_partition_axis]
        )
        recurrent_dim = self.head_dim if self.has_temporal_state else 1
        recurrent_shape = (
            self.total_slots,
            recurrent_heads,
            recurrent_dim,
            recurrent_dim,
        )
        temporal_dtype = self.temporal_dtype
        conv_dtype = self.conv_dtype

        alloc_recurrent = _get_recurrent_zero_allocator(
            recurrent_shape, temporal_dtype, self.recurrent_sharding
        )
        with self.mesh:
            recurrent_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                recurrent_buffers.append(alloc_recurrent())

            conv_buffers = []
            for layer_index in range(self.num_linear_recurrent_layers):
                channel_sizes = (
                    [self.proj_size]
                    if self.conv_channel_sizes is None
                    else self.conv_channel_sizes[layer_index]
                )
                inner = []
                for channels in channel_sizes:
                    conv_shape = (
                        self.total_slots,
                        channels,
                        self.conv_kernel_size - 1,
                    )
                    alloc_conv = _get_recurrent_zero_allocator(
                        conv_shape, conv_dtype, self.conv_sharding
                    )
                    inner.append(alloc_conv())
                conv_buffers.append(inner)

        return recurrent_buffers, conv_buffers

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def replace_buffer(self, buffers) -> None:
        new_recurrent, new_conv = buffers

        assert len(new_recurrent) == self.num_linear_recurrent_layers
        assert len(new_conv) == self.num_linear_recurrent_layers

        # tp_size==1 sharding fix: see MHATokenToKVPool.replace_buffer
        tp_degenerate = self.mesh.shape.get("tensor", 1) == 1
        for layer in range(self.num_linear_recurrent_layers):
            buf = new_recurrent[layer]
            if tp_degenerate:
                buf = jax.device_put(buf, self.recurrent_sharding)
            self.recurrent_buffers[layer] = buf

            assert len(new_conv[layer]) == len(self.conv_buffers[layer])
            for i in range(len(new_conv[layer])):
                cbuf = new_conv[layer][i]
                if tp_degenerate:
                    cbuf = jax.device_put(cbuf, self.conv_sharding)
                self.conv_buffers[layer][i] = cbuf

    def clear(self) -> None:
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])

    def copy_slots(self, src_indices, dst_indices):
        """Clone src->dst slots across all layers; rows with src==0 keep dst.
        Indices are per-DP-rank local; returns new buffers for the donated pool."""
        mesh = self.mesh
        data_axis = self.data_partition_axis

        def _temporal(buf, src, dst):
            # Donated-buffer aliasing barriers: without them the scatter races the
            # gather under multi-host SPMD -> NaN. Value-preserving; do not remove.
            buf = jax.lax.optimization_barrier(buf)
            val = jnp.where((src == 0).reshape(-1, 1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        def _conv(buf, src, dst):
            buf = jax.lax.optimization_barrier(buf)  # see _temporal
            val = jnp.where((src == 0).reshape(-1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        copy_temporal = jax.shard_map(
            _temporal,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.recurrent_partition_axis, None, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.recurrent_partition_axis, None, None),
            check_vma=False,
        )
        copy_conv = jax.shard_map(
            _conv,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.conv_partition_axis, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.conv_partition_axis, None),
            check_vma=False,
        )

        new_recurrent = [
            copy_temporal(buf, src_indices, dst_indices) for buf in self.recurrent_buffers
        ]
        new_conv = [
            [copy_conv(cbuf, src_indices, dst_indices) for cbuf in inner]
            for inner in self.conv_buffers
        ]
        return new_recurrent, new_conv

    def commit_convolution_transaction(
        self,
        transaction: RecurrentConvStateTransaction,
        accepted_lengths: jax.Array,
    ) -> None:
        if len(transaction.candidate_inputs) != len(self.conv_buffers):
            raise ValueError(
                "RECURRENT_TRANSACTION_LAYER_MISMATCH "
                f"candidates={len(transaction.candidate_inputs)} "
                f"buffers={len(self.conv_buffers)}"
            )
        accepted_lengths = jax.device_put(
            accepted_lengths,
            NamedSharding(self.mesh, P(self.data_partition_axis)),
        )
        recurrent_indices = jax.sharding.reshard(
            transaction.recurrent_indices,
            NamedSharding(self.mesh, P(self.data_partition_axis)),
        )
        for layer_candidates, layer_buffers in zip(
            transaction.candidate_inputs, self.conv_buffers, strict=True
        ):
            if len(layer_candidates) != len(layer_buffers):
                raise ValueError(
                    "RECURRENT_TRANSACTION_CONV_MISMATCH "
                    f"candidates={len(layer_candidates)} buffers={len(layer_buffers)}"
                )
        new_conv = _jitted_commit_convolution_transaction_buffers(
            tuple(tuple(layer) for layer in self.conv_buffers),
            transaction.candidate_inputs,
            recurrent_indices,
            accepted_lengths,
            transaction.draft_token_num,
        )
        self.conv_buffers = [list(layer) for layer in new_conv]

    # --- pytree ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
            self.total_slots,
            self.num_heads,
            self.head_dim,
            self.num_k_heads,
            self.head_k_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
            tuple(
                tuple(sizes) for sizes in self.conv_channel_sizes
            )
            if self.conv_channel_sizes is not None
            else None,
            self.has_temporal_state,
            self.mesh,
            self.recurrent_partition_axis,
            self.conv_partition_axis,
            self.data_partition_axis,
            self.recurrent_sharding,
            self.conv_sharding,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            size,
            dp_size,
            total_slots,
            num_heads,
            head_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
            conv_channel_sizes,
            has_temporal_state,
            mesh,
            recurrent_partition_axis,
            conv_partition_axis,
            data_partition_axis,
            recurrent_sharding,
            conv_sharding,
        ) = aux_data
        obj = cls.__new__(cls)
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.num_k_heads = num_k_heads
        obj.head_k_dim = head_k_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        obj.conv_channel_sizes = (
            [list(sizes) for sizes in conv_channel_sizes]
            if conv_channel_sizes is not None
            else None
        )
        obj.has_temporal_state = has_temporal_state
        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        obj.proj_size = proj_v + 2 * proj_k
        obj.mesh = mesh
        obj.recurrent_partition_axis = recurrent_partition_axis
        obj.conv_partition_axis = conv_partition_axis
        obj.data_partition_axis = data_partition_axis
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj
