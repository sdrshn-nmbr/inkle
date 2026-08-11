"""Short depthwise causal conv1d used by linear-attention backends (e.g. KDA).

The convolution is intentionally implemented as a stateless function (not an
nnx Module) so backends can freely combine it with their own weight containers
and cache layouts. Two execution paths are provided:

* ``decode`` — single-token step that appends the new token to a per-sequence
  ``[B, D, K-1]`` cache, runs the conv on the resulting K-token window, and
  drops the oldest slot before writing back.
* ``extend`` — variable-length packed prefill that consumes ``cu_seqlens`` to
  build a per-token sliding window mixing prior cache and in-sequence tokens.

State convention follows vLLM: cache has width ``K-1`` and stores the
prior ``K-1`` tokens (the current token is supplied via ``x`` at call time).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Sharding

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.kernels.causal_conv1d import ragged_causal_conv1d
from sgl_jax.srt.utils.profiling_utils import named_scope

# Map of supported activation names → callable. ``None`` means identity.
_ACTIVATION_FNS: dict[str | None, Callable[[jax.Array], jax.Array] | None] = {
    None: None,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
}


def _resolve_activation(
    activation: str | Callable[[jax.Array], jax.Array] | None,
) -> Callable[[jax.Array], jax.Array] | None:
    """Resolve an activation spec to a callable (or None for identity).

    Accepts either a name from ``_ACTIVATION_FNS`` or a user-supplied callable.
    """
    if activation is None or callable(activation):
        return activation
    if activation not in _ACTIVATION_FNS:
        raise ValueError(
            f"short_convolution activation must be one of {sorted(k for k in _ACTIVATION_FNS if k is not None)} "
            f"or a callable; got {activation!r}"
        )
    return _ACTIVATION_FNS[activation]


def short_convolution(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array | None,
    forward_mode: ForwardMode,
    bias: jax.Array | None = None,
    activation: str | Callable[[jax.Array], jax.Array] | None = "silu",
    x_window_sharding: Sharding | None = None,
    cache_window_sharding: Sharding | None = None,
    backend: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d with per-sequence cache.

    Args:
        x: ``[T, D]`` for ``EXTEND`` (packed varlen) or ``[B, D]`` for ``DECODE``.
        weight: depthwise kernel ``[D, K]``.
        cache: per-sequence rolling buffer ``[B, D, K-1]`` storing the prior
            ``K-1`` tokens (zeros for fresh sequences). The current token is
            supplied via ``x`` and not written into the input cache.
        cu_seqlens: ``[N+1]`` cumulative sequence lengths; required for
            ``EXTEND``, ignored for ``DECODE``.
        forward_mode: ``ForwardMode.DECODE`` or ``ForwardMode.EXTEND``.
        bias: optional ``[D]`` channel bias added before the activation.
        activation: name (e.g. ``"silu"``, ``"gelu"``, ``"sigmoid"``), a
            user-supplied callable, or ``None`` for identity.

    Returns:
        ``(y, new_cache)`` where ``y`` matches the leading dims of ``x`` and
        ``new_cache`` has the same shape as ``cache``.
    """
    activation_fn = _resolve_activation(activation)

    weight = _normalize_weight(weight)

    if backend not in (None, "pallas"):
        raise ValueError(f"INKLING_CONV_BACKEND_UNSUPPORTED backend={backend}")
    if backend == "pallas" and jax.default_backend() == "tpu":
        if not isinstance(x_window_sharding, NamedSharding):
            raise ValueError(
                "INKLING_PALLAS_CONV_SHARDING_REQUIRED "
                f"x_window_sharding={x_window_sharding}"
            )
        y, new_cache = _pallas_conv(
            x,
            weight,
            cache,
            cu_seqlens,
            bias,
            x_window_sharding.mesh,
        )
        return _apply_activation(y, activation_fn), new_cache

    if forward_mode == ForwardMode.DECODE:
        return _decode_conv(x, weight, cache, bias, activation_fn)
    if cu_seqlens is None:
        raise ValueError("short_convolution(EXTEND) requires cu_seqlens")
    return _extend_conv(
        x,
        weight,
        cache,
        cu_seqlens,
        bias,
        activation_fn,
        x_window_sharding,
        cache_window_sharding,
    )


def selected_short_convolution_backend() -> str:
    return "pallas" if jax.default_backend() == "tpu" else "jax"


def _pallas_conv(
    x: jax.Array,
    conv_kernel: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array | None,
    bias: jax.Array | None,
    mesh: jax.sharding.Mesh,
) -> tuple[jax.Array, jax.Array]:
    batch_size = cache.shape[0]
    if cu_seqlens is None:
        cu_seqlens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    cu_seqlens = jax.sharding.reshard(cu_seqlens, NamedSharding(mesh, P(None)))

    def local_conv(
        local_x: jax.Array,
        local_weight: jax.Array,
        local_cache: jax.Array,
        local_cu_seqlens: jax.Array,
        local_bias: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        channel_count = local_x.shape[-1]
        padded_channel_count = ((channel_count + 127) // 128) * 128
        channel_padding = padded_channel_count - channel_count
        local_x = jnp.pad(local_x, ((0, 0), (0, channel_padding)))
        local_weight = jnp.pad(local_weight, ((0, channel_padding), (0, 0)))
        local_cache = jnp.pad(local_cache, ((0, 0), (0, channel_padding), (0, 0)))
        if local_bias is not None:
            local_bias = jnp.pad(local_bias, ((0, channel_padding),))
        state_indices = jnp.arange(batch_size, dtype=jnp.int32)
        distribution = jnp.asarray([0, 0, batch_size], dtype=jnp.int32)
        has_initial_state = jnp.ones((batch_size,), dtype=jnp.bool_)
        output, new_cache = ragged_causal_conv1d(
            x=jnp.copy(local_x),
            conv_state=jnp.copy(jnp.swapaxes(local_cache, 1, 2)),
            conv_weight=local_weight[:, None, :],
            conv_bias=local_bias,
            query_start_loc=local_cu_seqlens,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            kernel_size=local_weight.shape[-1],
        )
        return output[:, :channel_count], jnp.swapaxes(new_cache, 1, 2)[
            :, :channel_count, :
        ]

    if bias is None:
        return shard_map(
            lambda local_x, local_weight, local_cache, local_cu_seqlens: local_conv(
                local_x,
                local_weight,
                local_cache,
                local_cu_seqlens,
                None,
            ),
            mesh=mesh,
            in_specs=(P("data", "tensor"), P("tensor", None), P("data", "tensor", None), P(None)),
            out_specs=(P("data", "tensor"), P("data", "tensor", None)),
            check_vma=False,
        )(x, conv_kernel, cache, cu_seqlens)
    bias = jax.sharding.reshard(bias, NamedSharding(mesh, P("tensor")))
    return shard_map(
        local_conv,
        mesh=mesh,
        in_specs=(
            P("data", "tensor"),
            P("tensor", None),
            P("data", "tensor", None),
            P(None),
            P("tensor"),
        ),
        out_specs=(P("data", "tensor"), P("data", "tensor", None)),
        check_vma=False,
    )(x, conv_kernel, cache, cu_seqlens, bias)


def _normalize_weight(weight: jax.Array) -> jax.Array:
    """Reduce common conv-weight layouts to ``[D, K]``."""
    # Squeeze the depthwise singleton axis if the loader handed us [D, 1, K].
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    return weight


def _apply_activation(
    y: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> jax.Array:
    if activation_fn is None:
        return y
    return activation_fn(y)


@named_scope("short_conv_decode")
def _decode_conv(
    x: jax.Array,  # [B, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    # expand x shape from [B, D] to [B, D, 1]
    new_cache = jnp.concatenate([cache, x[..., None]], axis=-1)
    y = jnp.einsum("bck,ck->bc", new_cache, conv_kernel.astype(new_cache.dtype))
    if bias is not None:
        y = y + bias.astype(y.dtype)
    y = _apply_activation(y, activation_fn)
    # return the last K-1 conv state
    return y, new_cache[:, :, 1:]


@named_scope("short_conv_extend")
def _extend_conv(
    x: jax.Array,  # [T, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    cu_seqlens: jax.Array,
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
    x_window_sharding: Sharding | None,
    cache_window_sharding: Sharding | None,
) -> tuple[jax.Array, jax.Array]:
    T = x.shape[0]
    K = conv_kernel.shape[-1]
    W = K - 1  # cache width

    # Locate every output token within its owning sequence.
    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]

    # Build the K-tap window ending at each token: source positions
    # ``[t-(K-1), ..., t]``. Positions inside the same sequence read from x;
    # positions reaching back across the sequence boundary read from cache.
    offsets = jnp.arange(K, dtype=cu_seqlens.dtype) - (K - 1)
    source_idx = token_idx[:, None] + offsets[None, :]
    from_x = source_idx >= starts[:, None]

    safe_x_idx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
    # x[safe_x_idx]: [T, K, D] (advanced indexing puts the index axes first).
    # Swap to [T, D, K] so the einsum spec matches the decode path's "bck,ck".
    if x_window_sharding is None:
        gathered_x = x[safe_x_idx]
    else:
        gathered_x = x.at[safe_x_idx].get(out_sharding=x_window_sharding)
    x_window = jnp.swapaxes(gathered_x, 1, 2)  # [T, D, K]

    # cache holds the prior W = K-1 tokens at slots [0, W-1]. Map source
    # position p (where p < starts[seq]) to cache slot ``W + (p - starts)``.
    # cache[seq_ids] is already [T, D, W]; gather along the time axis (=2).
    cache_pos = jnp.clip(W + source_idx - starts[:, None], 0, W - 1)
    if cache_window_sharding is None:
        cache_by_token = cache[seq_ids]
    else:
        cache_by_token = cache.at[seq_ids].get(out_sharding=cache_window_sharding)
    cache_window = jnp.take_along_axis(
        cache_by_token,  # [T, D, W]
        cache_pos[:, None, :],  # [T, 1, K] -> broadcasts over D
        axis=2,
    )  # [T, D, K]
    window = jnp.where(from_x[:, None, :], x_window, cache_window)  # [T, D, K]
    y = jnp.einsum("tck,ck->tc", window, conv_kernel.astype(window.dtype))
    if bias is not None:
        y = y + bias.astype(y.dtype)
    y = _apply_activation(y, activation_fn)

    # Compute the new per-sequence cache: the last W = K-1 input tokens of
    # each sequence, falling back to the prior cache when the sequence is
    # shorter than W.
    ends = cu_seqlens[1:]
    state_offsets = jnp.arange(W, dtype=cu_seqlens.dtype)
    final_idx = ends[:, None] - W + state_offsets[None, :]
    final_from_x = final_idx >= cu_seqlens[:-1, None]
    safe_final_idx = jnp.clip(final_idx, 0, jnp.maximum(T - 1, 0))
    if x_window_sharding is None:
        gathered_final_x = x[safe_final_idx]
    else:
        gathered_final_x = x.at[safe_final_idx].get(out_sharding=x_window_sharding)
    final_x = jnp.swapaxes(gathered_final_x, 1, 2)
    final_cache_pos = jnp.clip(W + final_idx - cu_seqlens[:-1, None], 0, W - 1)
    final_cache = jnp.take_along_axis(cache, final_cache_pos[:, None, :], axis=2)
    new_cache = jnp.where(final_from_x[:, None, :], final_x, final_cache)

    return y, new_cache


__all__ = ["selected_short_convolution_backend", "short_convolution"]
