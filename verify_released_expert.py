# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "huggingface-hub==1.3.3",
#   "jax==0.10.2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
# ]
# ///

import argparse
import hashlib
import json
import time

import numpy as np
from checkpoint_io import (
    BF16_REPOSITORY,
    BF16_REVISION,
    NVFP4_REPOSITORY,
    NVFP4_REVISION,
    HuggingFaceSafetensorsRepository,
)
from inkling_layout import decode_nvfp4_numpy, deinterleave_gate_up_numpy

HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 3072


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)
    difference = candidate - reference
    return {
        "cosine": float(
            np.vdot(reference.ravel(), candidate.ravel()) / (np.linalg.norm(reference) * np.linalg.norm(candidate))
        ),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(np.square(difference))) / np.sqrt(np.mean(np.square(reference)))
        ),
    }


def expert_forward(x: np.ndarray, w13: np.ndarray, w2: np.ndarray) -> np.ndarray:
    gate_up = x.astype(np.float32) @ w13.astype(np.float32).T
    gate, up = np.split(gate_up, 2, axis=-1)
    activated = gate / (1.0 + np.exp(-gate)) * up
    return activated @ w2.astype(np.float32).T


def read_expert(
    repository: HuggingFaceSafetensorsRepository,
    layer: int,
    expert: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    w13 = deinterleave_gate_up_numpy(repository.read_first_axis(f"{prefix}.w13_weight", expert))
    w2 = repository.read_first_axis(f"{prefix}.w2_weight", expert)
    return w13, w2


def read_nvfp4_expert(
    repository: HuggingFaceSafetensorsRepository,
    layer: int,
    expert: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    packed_w13 = deinterleave_gate_up_numpy(repository.read_first_axis(f"{prefix}.w13_weight", expert))
    w13_scale = deinterleave_gate_up_numpy(repository.read_first_axis(f"{prefix}.w13_weight.scale", expert))
    w13_scale2 = repository.read_first_axis(f"{prefix}.w13_weight.scale2", expert)
    packed_w2 = repository.read_first_axis(f"{prefix}.w2_weight", expert)
    w2_scale = repository.read_first_axis(f"{prefix}.w2_weight.scale", expert)
    w2_scale2 = repository.read_first_axis(f"{prefix}.w2_weight.scale2", expert)
    return (
        decode_nvfp4_numpy(packed_w13, w13_scale, w13_scale2),
        decode_nvfp4_numpy(packed_w2, w2_scale, w2_scale2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.layer < 3:
        raise ValueError("INKLING_NVFP4_EXPERT_REQUIRED layer must be at least 3")

    started = time.perf_counter()
    with HuggingFaceSafetensorsRepository(BF16_REPOSITORY, BF16_REVISION) as bf16_repository:
        bf16_w13, bf16_w2 = read_expert(bf16_repository, args.layer, args.expert)
    bf16_read_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with HuggingFaceSafetensorsRepository(NVFP4_REPOSITORY, NVFP4_REVISION) as nvfp4_repository:
        nvfp4_w13, nvfp4_w2 = read_nvfp4_expert(nvfp4_repository, args.layer, args.expert)
    nvfp4_read_seconds = time.perf_counter() - started

    rng = np.random.default_rng(args.seed)
    hidden_states = rng.standard_normal((args.tokens, HIDDEN_SIZE), dtype=np.float32)
    bf16_output = expert_forward(hidden_states, bf16_w13, bf16_w2)
    nvfp4_output = expert_forward(hidden_states, nvfp4_w13, nvfp4_w2)

    result = {
        "bf16_read_seconds": bf16_read_seconds,
        "expert": args.expert,
        "layer": args.layer,
        "nvfp4_read_seconds": nvfp4_read_seconds,
        "output_comparison": compare(bf16_output, nvfp4_output),
        "output_sha256": {
            "bf16": hashlib.sha256(bf16_output.tobytes()).hexdigest(),
            "nvfp4": hashlib.sha256(nvfp4_output.tobytes()).hexdigest(),
        },
        "tokens": args.tokens,
        "weight_comparison": {
            "w13": compare(bf16_w13, nvfp4_w13),
            "w2": compare(bf16_w2, nvfp4_w2),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
