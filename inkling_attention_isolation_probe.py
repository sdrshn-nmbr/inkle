import json

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.native_backend import _apply_extend_mask
from sgl_jax.srt.layers.attention.native_backend import forward_attention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def compare_requests(output: jax.Array, sequence_length: int) -> dict[str, object]:
    host = np.asarray(jax.device_get(output), dtype=np.float32)
    difference = host[:sequence_length] - host[sequence_length : 2 * sequence_length]
    return {
        "exact": bool(np.array_equal(difference, np.zeros_like(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
    }


def compare_blocks(
    value: jax.Array,
    sequence_length: int,
    *,
    key_axis: bool = False,
) -> dict[str, object]:
    host = np.asarray(jax.device_get(value), dtype=np.float32)
    first = host[:sequence_length]
    second = host[sequence_length : 2 * sequence_length]
    if key_axis:
        first = first[..., :sequence_length]
        second = second[..., sequence_length : 2 * sequence_length]
    difference = first - second
    return {
        "exact": bool(np.array_equal(difference, np.zeros_like(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
    }


def main() -> None:
    active_batch_size = 2
    padded_batch_size = 12
    sequence_length = 5
    token_count = 256
    num_heads = 32
    num_kv_heads = 16
    head_dim = 128
    relative_dimension = 16
    relative_extent = 512

    devices = jax.devices()
    mesh = create_device_mesh(
        ici_parallelism=[1, len(devices)],
        dcn_parallelism=[1, 1],
        devices=devices,
    )
    jax.sharding.set_mesh(mesh)

    def repeated_tokens(heads: int, width: int, modulus: int) -> jax.Array:
        row = jnp.arange(sequence_length * heads * width, dtype=jnp.float32)
        row = ((row % modulus) - modulus // 2) / 64
        row = row.reshape(sequence_length, heads, width)
        active = jnp.concatenate([row] * active_batch_size)
        return jnp.pad(active, ((0, token_count - active.shape[0]), (0, 0), (0, 0)))

    q = repeated_tokens(num_heads, head_dim, 101).astype(jnp.bfloat16)
    packed_k = repeated_tokens(num_kv_heads, head_dim, 97).astype(jnp.bfloat16)
    packed_v = repeated_tokens(num_kv_heads, head_dim, 89).astype(jnp.bfloat16)
    k = jnp.zeros_like(packed_k).at[1:11].set(packed_k[:10])
    v = jnp.zeros_like(packed_v).at[1:11].set(packed_v[:10])
    relative = repeated_tokens(num_heads, relative_dimension, 83).astype(jnp.bfloat16)
    projection = (
        (
            (
                jnp.arange(relative_dimension * relative_extent, dtype=jnp.float32) % 79
                - 39
            )
            / 128
        )
        .reshape(relative_dimension, relative_extent)
        .astype(jnp.bfloat16)
    )

    sequence_lengths = np.zeros(padded_batch_size, dtype=np.int32)
    sequence_lengths[:active_batch_size] = sequence_length
    sequence_lengths = jnp.asarray(sequence_lengths)
    prefix_lengths = jnp.zeros((padded_batch_size,), dtype=jnp.int32)
    locations = jnp.concatenate(
        (
            jnp.arange(1, active_batch_size * sequence_length + 1, dtype=jnp.int32),
            jnp.zeros(
                (token_count - active_batch_size * sequence_length,), dtype=jnp.int32
            ),
        )
    )

    q = jax.device_put(q, NamedSharding(mesh, P("data", "tensor", None)))
    k = jax.device_put(k, NamedSharding(mesh, P(None, "tensor", None)))
    v = jax.device_put(v, NamedSharding(mesh, P(None, "tensor", None)))
    relative = jax.device_put(relative, NamedSharding(mesh, P("data", "tensor", None)))
    projection = jax.device_put(projection, NamedSharding(mesh, P(None, None)))

    results = {}
    for name, relative_inputs in {
        "without_relative_bias": (None, None),
        "with_relative_bias": (relative, projection),
    }.items():
        output = forward_attention(
            q,
            k,
            v,
            sequence_lengths,
            locations,
            prefix_lengths,
            sequence_lengths,
            num_heads,
            num_kv_heads,
            scale=1.0 / head_dim,
            is_causal=True,
            mode=ForwardMode.EXTEND,
            kv_sharding=NamedSharding(mesh, P(None, "tensor", None)),
            mesh=mesh,
            sliding_window_size=512,
            softmax_dtype=jnp.float32,
            relative_states=relative_inputs[0],
            relative_projection=relative_inputs[1],
        )
        output.block_until_ready()
        chex.assert_shape(output, (token_count, num_heads * head_dim))
        results[name] = compare_requests(output, sequence_length)

    kv_sharding = NamedSharding(mesh, P(None, "tensor", None))
    repeated_k = jnp.repeat(
        packed_k, num_heads // num_kv_heads, axis=1, out_sharding=kv_sharding
    )
    repeated_v = jnp.repeat(
        packed_v, num_heads // num_kv_heads, axis=1, out_sharding=kv_sharding
    )
    logits = jnp.einsum(
        "qhd,khd->qhk", q, repeated_k, preferred_element_type=jnp.float32
    ) * (1.0 / head_dim)
    masked_logits = _apply_extend_mask(
        logits,
        sequence_lengths,
        prefix_lengths,
        sequence_lengths,
        True,
        512,
        mesh,
    )
    weights = jax.nn.softmax(masked_logits, axis=-1)
    manual_output = jnp.einsum(
        "qhk,khd->qhd",
        weights.astype(repeated_v.dtype),
        repeated_v,
        preferred_element_type=jnp.float32,
    )
    diagnostics = {
        "logits": compare_blocks(logits, sequence_length, key_axis=True),
        "masked_logits": compare_blocks(masked_logits, sequence_length, key_axis=True),
        "softmax_weights": compare_blocks(weights, sequence_length, key_axis=True),
        "value_combination": compare_requests(
            manual_output.reshape(token_count, -1), sequence_length
        ),
    }

    passed = all(result["exact"] for result in results.values())
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "device_count": jax.device_count(),
                "event": "INKLING_ATTENTION_ISOLATION_PROBE",
                "passed": passed,
                "diagnostics": diagnostics,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
