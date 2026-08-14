import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.layers.attention.native_backend import forward_attention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


def measure(run, iterations: int) -> dict[str, object]:
    for _ in range(3):
        jax.block_until_ready(run())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        jax.block_until_ready(run())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
    }


def benchmark_gmm(iterations: int) -> list[dict[str, object]]:
    expert_count = 256
    hidden_size = 4096
    intermediate_size = 2048
    weights = jax.jit(
        lambda: jnp.ones(
            (expert_count, hidden_size, intermediate_size),
            dtype=jnp.bfloat16,
        )
    )()
    results = []
    for batch_tokens in (1, 4, 8, 16):
        routed_tokens = batch_tokens * 6
        inputs = jnp.ones((routed_tokens, hidden_size), dtype=jnp.bfloat16)
        group_sizes = np.zeros((expert_count,), dtype=np.int32)
        group_sizes[:routed_tokens] = 1
        group_sizes = jnp.asarray(group_sizes)

        @jax.jit
        def run(lhs, rhs, groups):
            return gmm(
                lhs,
                rhs,
                groups,
                preferred_element_type=jnp.bfloat16,
                tiling=(min(512, routed_tokens), 1024, 1024),
            )

        result = measure(lambda: run(inputs, weights, group_sizes), iterations)
        result.update(
            {
                "batch_tokens": batch_tokens,
                "routed_tokens": routed_tokens,
                "scope": "one TPU device, one Inkling routed-expert projection",
            }
        )
        results.append(result)
    return results


def benchmark_convolution(
    mesh: jax.sharding.Mesh,
    iterations: int,
) -> list[dict[str, object]]:
    results = []
    tensor_devices = mesh.shape["tensor"]
    for local_channels in (128, 512):
        channels = local_channels * tensor_devices
        x_sharding = NamedSharding(mesh, P("data", "tensor"))
        cache_sharding = NamedSharding(mesh, P("data", "tensor", None))
        weight_sharding = NamedSharding(mesh, P("tensor", None))
        x = jax.device_put(np.ones((1, channels), np.float32), x_sharding)
        cache = jax.device_put(np.zeros((1, channels, 3), np.float32), cache_sharding)
        weight = jax.device_put(np.ones((channels, 4), np.float32), weight_sharding)

        @jax.jit
        def run(x_value, weight_value, cache_value):
            return short_convolution(
                x_value,
                weight_value,
                cache_value,
                None,
                ForwardMode.DECODE,
                activation=None,
                x_window_sharding=x_sharding,
                cache_window_sharding=cache_sharding,
                backend="pallas",
            )

        result = measure(lambda: run(x, weight, cache), iterations)
        result.update(
            {
                "global_channels": channels,
                "local_channels": local_channels,
                "scope": "all TPU devices, one Inkling decode convolution",
            }
        )
        results.append(result)
    return results


def benchmark_attention(
    mesh: jax.sharding.Mesh,
    iterations: int,
) -> list[dict[str, object]]:
    results = []
    q_sharding = NamedSharding(mesh, P(None, "tensor", None))
    kv_sharding = NamedSharding(mesh, P(None, "tensor", None))
    metadata_sharding = NamedSharding(mesh, P())
    projection_sharding = NamedSharding(mesh, P(None, None))

    for context in (128, 512, 1024):
        q = jax.device_put(np.ones((1, 32, 128), np.float32), q_sharding)
        k = jax.device_put(np.ones((context, 8, 128), np.float32), kv_sharding)
        v = jax.device_put(np.ones((context, 8, 128), np.float32), kv_sharding)
        seq_lens = jax.device_put(np.asarray([context], np.int32), metadata_sharding)
        loc = jax.device_put(np.arange(1, context + 1, dtype=np.int32), metadata_sharding)
        prefix_lens = jax.device_put(np.asarray([context - 1], np.int32), metadata_sharding)
        extend_lens = jax.device_put(np.asarray([1], np.int32), metadata_sharding)
        relative_states = jax.device_put(
            np.ones((1, 32, 128), np.float32), q_sharding
        )
        projection = jax.device_put(
            np.ones((128, 512), np.float32), projection_sharding
        )

        @jax.jit
        def run(q_value, k_value, v_value, relative_value):
            return forward_attention(
                q_value,
                k_value,
                v_value,
                seq_lens,
                loc,
                prefix_lens,
                extend_lens,
                32,
                8,
                mode=ForwardMode.DECODE,
                kv_sharding=kv_sharding,
                mesh=mesh,
                sliding_window_size=512,
                relative_states=relative_value,
                relative_projection=projection,
            )

        result = measure(lambda: run(q, k, v, relative_states), iterations)
        result.update(
            {
                "context_tokens": context,
                "scope": "all TPU devices, one Inkling relative-bias attention call",
            }
        )
        results.append(result)
    return results


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"INKLING_COMPONENT_BENCHMARK_REQUIRES_TPU backend={jax.default_backend()}"
        )
    gmm_results = benchmark_gmm(args.iterations)
    mesh = create_device_mesh(
        ici_parallelism=[1, len(jax.devices())],
        dcn_parallelism=[1, 1],
    )
    jax.sharding.set_mesh(mesh)
    result = {
        "attention": benchmark_attention(mesh, args.iterations),
        "convolution": benchmark_convolution(mesh, args.iterations),
        "device_count": len(jax.devices()),
        "device_kind": jax.devices()[0].device_kind,
        "gmm": gmm_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
