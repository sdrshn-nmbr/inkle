import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.relative_position_bias import relative_position_bias_pallas


def reference(
    relative_states,
    projection,
    query_positions,
    query_batch_ids,
    key_positions,
    key_batch_ids,
):
    distance = query_positions[:, None] - key_positions[None, :]
    selected = jnp.take(projection, jnp.clip(distance, 0, projection.shape[1] - 1), axis=1)
    selected = selected.transpose(1, 2, 0)
    bias = jnp.einsum(
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
    return jnp.where(valid[:, None, :], bias, 0.0)


def measure(function, iterations):
    function().block_until_ready()
    start = time.perf_counter()
    for _ in range(iterations):
        function().block_until_ready()
    return (time.perf_counter() - start) * 1e6 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--relative-dim", type=int, default=16)
    parser.add_argument("--keys", type=int, default=2048)
    parser.add_argument("--extent", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--explicit-mesh", action="store_true")
    args = parser.parse_args()

    if args.explicit_mesh:
        devices = np.asarray(jax.devices(), dtype=object).reshape(1, -1)
        jax.sharding.set_mesh(jax.sharding.Mesh(devices, ("data", "tensor")))

    lengths = np.full(args.queries, args.keys // args.queries, dtype=np.int32)
    lengths[: args.keys % args.queries] += 1
    key_batch_ids = np.repeat(np.arange(args.queries, dtype=np.int32), lengths)
    key_positions = np.concatenate([np.arange(length, dtype=np.int32) for length in lengths])
    query_batch_ids = np.arange(args.queries, dtype=np.int32)
    query_key_starts = np.cumsum(lengths, dtype=np.int32) - lengths
    query_positions = lengths - 1
    relative_states = jnp.arange(
        args.queries * args.heads * args.relative_dim,
        dtype=jnp.float32,
    ).reshape(args.queries, args.heads, args.relative_dim)
    relative_states = (relative_states / 1000).astype(jnp.bfloat16)
    projection = jnp.arange(args.relative_dim * args.extent, dtype=jnp.float32).reshape(
        args.relative_dim, args.extent
    )
    projection = (projection / 1000).astype(jnp.bfloat16)
    reference_metadata = tuple(
        jnp.asarray(value)
        for value in (query_positions, query_batch_ids, key_positions, key_batch_ids)
    )
    kernel_metadata = tuple(jnp.asarray(value) for value in (query_positions, query_key_starts))
    interpret = jax.default_backend() == "cpu"

    pallas = jax.jit(
        lambda: relative_position_bias_pallas(
            relative_states,
            projection,
            *kernel_metadata,
            args.keys,
            interpret=interpret,
        )
    )
    baseline = jax.jit(lambda: reference(relative_states, projection, *reference_metadata))
    pallas_result = pallas()
    baseline_result = baseline()
    max_absolute_error = float(jnp.max(jnp.abs(pallas_result - baseline_result)))
    baseline_us = measure(baseline, args.iterations)
    pallas_us = measure(pallas, args.iterations)
    print(
        json.dumps(
            {
                "device": str(jax.devices()[0]),
                "shape": [args.queries, args.heads, args.keys],
                "relative_dim": args.relative_dim,
                "extent": args.extent,
                "max_absolute_error": max_absolute_error,
                "baseline_microseconds": baseline_us,
                "pallas_microseconds": pallas_us,
                "speedup": baseline_us / pallas_us,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
