import argparse
import json

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-batch-size", type=int, default=2)
    parser.add_argument("--padded-batch-size", type=int)
    parser.add_argument("--padded-token-count", type=int)
    parser.add_argument("--channels", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--sequence-length", type=int, default=5)
    return parser.parse_args()


def compare(reference: jax.Array, candidate: jax.Array) -> dict[str, object]:
    reference_host = np.asarray(reference, dtype=np.float32)
    candidate_host = np.asarray(candidate, dtype=np.float32)
    difference = candidate_host - reference_host
    return {
        "exact": bool(np.array_equal(reference_host, candidate_host)),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "reference_finite": bool(np.isfinite(reference_host).all()),
        "candidate_finite": bool(np.isfinite(candidate_host).all()),
        "within_bfloat16_tolerance": bool(
            np.allclose(reference_host, candidate_host, rtol=0.02, atol=0.01)
        ),
    }


def numpy_reference(
    inputs: np.ndarray,
    weights: np.ndarray,
    cache: np.ndarray,
    *,
    sequence_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(inputs, dtype=np.float32)
    new_cache = cache.astype(np.float32).copy()
    start = 0
    for request_index, sequence_length in enumerate(sequence_lengths):
        stop = start + int(sequence_length)
        if sequence_length == 0:
            continue
        sequence = inputs[start:stop].astype(np.float32)
        history = np.concatenate(
            (cache[request_index].astype(np.float32).T, sequence), axis=0
        )
        for token_index in range(sequence_length):
            window = history[token_index : token_index + weights.shape[1]]
            output[start + token_index] = np.sum(window.T * weights, axis=1)
        new_cache[request_index] = history[-(weights.shape[1] - 1) :].T
        start = stop
    return output, new_cache


def run_case(
    mesh: Mesh,
    *,
    active_batch_size: int,
    padded_batch_size: int,
    padded_token_count: int,
    channels: int,
    sequence_length: int,
) -> dict[str, object]:
    kernel_size = 4
    row = jnp.arange(sequence_length * channels, dtype=jnp.float32).reshape(
        sequence_length, channels
    )
    row = ((row % 97) - 48) / 32
    active_inputs = jnp.concatenate([row] * active_batch_size)
    if padded_token_count < active_inputs.shape[0]:
        raise ValueError("padded token count is smaller than the active token count")
    inputs = jnp.pad(
        active_inputs,
        ((0, padded_token_count - active_inputs.shape[0]), (0, 0)),
    ).astype(jnp.bfloat16)
    weights = jnp.stack(
        [
            jnp.full((channels,), 0.1, dtype=jnp.float32),
            jnp.full((channels,), -0.2, dtype=jnp.float32),
            jnp.full((channels,), 0.3, dtype=jnp.float32),
            jnp.full((channels,), 0.4, dtype=jnp.float32),
        ],
        axis=1,
    )
    cache = jnp.zeros(
        (padded_batch_size, channels, kernel_size - 1), dtype=jnp.bfloat16
    )
    sequence_lengths = np.zeros(padded_batch_size, dtype=np.int32)
    sequence_lengths[:active_batch_size] = sequence_length
    cumulative_lengths = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(sequence_lengths))
    )
    input_sharding = NamedSharding(mesh, P("data", "tensor"))
    cache_sharding = NamedSharding(mesh, P("data", "tensor", None))
    reference_output, reference_cache = numpy_reference(
        np.asarray(inputs),
        np.asarray(weights),
        np.asarray(cache),
        sequence_lengths=sequence_lengths,
    )
    reference_output = jnp.asarray(reference_output, dtype=jnp.bfloat16)
    reference_cache = jnp.asarray(reference_cache, dtype=jnp.bfloat16)
    inputs = jax.device_put(inputs, input_sharding)
    weights = jax.device_put(weights, NamedSharding(mesh, P("tensor", None)))
    cache = jax.device_put(cache, cache_sharding)
    candidate_output, candidate_cache = short_convolution(
        inputs,
        weights,
        cache,
        cumulative_lengths,
        ForwardMode.EXTEND,
        activation=None,
        x_window_sharding=NamedSharding(mesh, P("data", None, "tensor")),
        cache_window_sharding=cache_sharding,
        backend="pallas",
    )
    candidate_output.block_until_ready()
    candidate_cache.block_until_ready()

    candidate_output_host = np.asarray(jax.device_get(candidate_output))
    outputs_by_request = candidate_output_host[: active_batch_size * sequence_length].reshape(
        active_batch_size, sequence_length, channels
    )
    request_isolation = compare(
        outputs_by_request[0],
        outputs_by_request[1],
    )
    output_comparison = compare(reference_output, candidate_output)
    cache_comparison = compare(reference_cache, candidate_cache)
    chex.assert_shape(candidate_output, inputs.shape)
    chex.assert_shape(candidate_cache, cache.shape)
    return {
        "active_batch_size": active_batch_size,
        "cache": cache_comparison,
        "channels": channels,
        "output": output_comparison,
        "padded_batch_size": padded_batch_size,
        "padded_token_count": padded_token_count,
        "request_isolation": request_isolation,
        "sequence_length": sequence_length,
    }


def main() -> None:
    args = parse_args()
    padded_batch_size = args.padded_batch_size or args.active_batch_size
    padded_token_count = (
        args.padded_token_count or args.active_batch_size * args.sequence_length
    )
    if padded_batch_size < args.active_batch_size:
        raise ValueError("padded batch size is smaller than the active batch size")
    devices = jax.devices()
    mesh = create_device_mesh(
        ici_parallelism=[1, len(devices)],
        dcn_parallelism=[1, 1],
        devices=devices,
    )
    jax.sharding.set_mesh(mesh)
    results = [
        run_case(
            mesh,
            active_batch_size=args.active_batch_size,
            padded_batch_size=padded_batch_size,
            padded_token_count=padded_token_count,
            channels=channels,
            sequence_length=args.sequence_length,
        )
        for channels in args.channels
    ]
    passed = all(
        result["request_isolation"]["exact"]
        and result["output"]["within_bfloat16_tolerance"]
        and result["cache"]["exact"]
        for result in results
    )
    payload = {
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "event": "INKLING_REQUEST_ISOLATION_PROBE",
        "passed": passed,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
