# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "huggingface-hub==1.3.3",
#   "jax==0.10.2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
#   "tokenizers==0.22.2",
# ]
# ///

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from streaming_tpu_inference import (
    HIDDEN_SIZE,
    InklingCheckpoint,
    attention_residual,
    dense_residual,
    rms_norm,
    run_sparse_layer,
    tokenize_prompt,
)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32).ravel()
    candidate = candidate.astype(np.float32).ravel()
    difference = candidate - reference
    return {
        "cosine": float(np.vdot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(np.square(difference))) / np.sqrt(np.mean(np.square(reference)))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer > 0 and args.input is None:
        raise ValueError("INKLING_LAYER_INPUT_REQUIRED for layers after layer 0")
    device = jax.devices()[0]
    checkpoint = InklingCheckpoint(4 * 1024**3, Path("/tmp/inkling-expert-cache"))
    params = checkpoint.load_layer(args.layer, device)
    if args.input is None:
        input_ids = tokenize_prompt(args.prompt)
        embedding_rows = np.stack(
            [
                checkpoint.repository.read_first_axis("model.llm.embed.weight", int(token_id))
                for token_id in input_ids[0]
            ]
        )
        embed_norm = jax.device_put(checkpoint.read("model.llm.embed_norm.weight"), device)
        hidden_states = rms_norm(jax.device_put(embedding_rows[None, ...], device), embed_norm)
    else:
        hidden_states = jax.device_put(np.load(args.input), device)
    if hidden_states.shape[-1] != HIDDEN_SIZE:
        raise ValueError(f"INKLING_INVALID_HIDDEN_SIZE actual={hidden_states.shape[-1]}")

    hidden_states = attention_residual(params, hidden_states)
    routes = None
    if args.layer < 2:
        hidden_states = dense_residual(params, hidden_states)
    else:
        hidden_states, routes = run_sparse_layer(checkpoint, params, hidden_states, args.layer, device)
    candidate = np.asarray(hidden_states, dtype=np.float32)
    result: dict[str, object] = {
        "backend": jax.default_backend(),
        "layer": args.layer,
        "routes": None if routes is None else routes.tolist(),
    }
    if args.reference is not None:
        result["comparison"] = compare(np.load(args.reference), candidate)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, candidate)
    print(json.dumps(result, indent=2, sort_keys=True))
    checkpoint.close()


if __name__ == "__main__":
    main()
