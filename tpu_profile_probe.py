import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--minimum-profile-seconds", type=float, default=2.0)
    parser.add_argument("--trace-mode")
    parser.add_argument("--profile-chip-count", type=int)
    parser.add_argument("--perf-counters", action="store_true")
    parser.add_argument("--enable-hlo-proto", action="store_true")
    return parser.parse_args()


def profiler_options(args: argparse.Namespace) -> jax.profiler.ProfileOptions:
    options = jax.profiler.ProfileOptions()
    options.raise_error_on_start_failure = True
    options.enable_hlo_proto = args.enable_hlo_proto
    advanced_configuration = {}
    if args.trace_mode:
        advanced_configuration["tpu_trace_mode"] = args.trace_mode
    if args.profile_chip_count is not None:
        advanced_configuration["tpu_num_chips_to_profile_per_task"] = args.profile_chip_count
    if args.perf_counters:
        advanced_configuration["tpu_perf_counters"] = True
    if advanced_configuration:
        options.advanced_configuration = advanced_configuration
    return options


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"TPU_PROFILE_PROBE_REQUIRES_TPU backend={jax.default_backend()}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    trace_directory = args.output_directory / "profile"
    size = args.matrix_size
    left = jnp.arange(size * size, dtype=jnp.bfloat16).reshape(size, size) % 17
    right = jnp.arange(size * size, dtype=jnp.bfloat16).reshape(size, size) % 13

    operation = jax.jit(lambda x, y: jnp.tanh(x @ y))
    lowered = operation.lower(left, right)
    (args.output_directory / "stablehlo.txt").write_text(
        str(lowered.compiler_ir(dialect="stablehlo"))
    )
    executable = lowered.compile()
    (args.output_directory / "compiled-hlo.txt").write_text(executable.as_text())

    for _ in range(3):
        jax.block_until_ready(executable(left, right))

    samples_ms = []
    jax.profiler.start_trace(
        str(trace_directory),
        profiler_options=profiler_options(args),
    )
    try:
        time.sleep(0.5)
        step = 0
        profile_started = time.monotonic()
        while step < args.iterations or (
            time.monotonic() - profile_started < args.minimum_profile_seconds
        ):
            start = time.perf_counter_ns()
            with jax.profiler.StepTraceAnnotation("tpu_matmul", step_num=step):
                output = executable(left, right)
                jax.block_until_ready(output)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)
            step += 1
        time.sleep(0.5)
    finally:
        jax.profiler.stop_trace()

    result = {
        "backend": jax.default_backend(),
        "device_count": len(jax.devices()),
        "device_kind": jax.devices()[0].device_kind,
        "iterations": args.iterations,
        "matrix_size": size,
        "median_ms": float(np.median(samples_ms)),
        "samples_ms": samples_ms,
        "trace_directory": str(trace_directory),
    }
    (args.output_directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
