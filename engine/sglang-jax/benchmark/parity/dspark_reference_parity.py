from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from dspark import DSparkDraftModel as TorchDSparkDraftModel
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.dspark import DSparkDraftInputs, DSparkDraftModel
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference_f32 = reference.astype(np.float32).reshape(-1)
    actual_f32 = actual.astype(np.float32).reshape(-1)
    denominator = np.linalg.norm(reference_f32) * np.linalg.norm(actual_f32)
    cosine = float(np.dot(reference_f32, actual_f32) / denominator)
    relative_l2 = float(
        np.linalg.norm(reference_f32 - actual_f32) / np.linalg.norm(reference_f32)
    )
    return {
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(reference_f32 - actual_f32))),
        "relative_l2": relative_l2,
    }


def _torch_outputs(
    checkpoint: Path,
    *,
    noise_embeddings: np.ndarray,
    target_hidden_states: np.ndarray,
    position_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], TorchDSparkDraftModel]:
    model = TorchDSparkDraftModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()
    outputs: dict[str, np.ndarray] = {}

    def save(name: str):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
        ):
            outputs[name] = output.detach().float().cpu().numpy()

        return hook

    handles = [model.hidden_norm.register_forward_hook(save("projected_target"))]
    handles.extend(
        layer.register_forward_hook(save(f"layer_{layer_id}"))
        for layer_id, layer in enumerate(model.layers)
    )
    handles.append(model.norm.register_forward_hook(save("final")))
    with torch.inference_mode():
        model(
            position_ids=torch.from_numpy(position_ids),
            noise_embedding=torch.from_numpy(noise_embeddings),
            target_hidden=torch.from_numpy(target_hidden_states),
        )
    for handle in handles:
        handle.remove()
    return outputs, model


def _jax_outputs(
    checkpoint: Path,
    *,
    noise_embeddings: np.ndarray,
    target_hidden_states: np.ndarray,
    position_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], DSparkDraftModel]:
    mesh = create_device_mesh(
        ici_parallelism=[1, 1],
        dcn_parallelism=[1, 1],
        devices=[jax.devices()[0]],
    )
    jax.sharding.set_mesh(mesh)
    model_config = ModelConfig(str(checkpoint), dtype="float32")
    model = DSparkDraftModel(model_config.hf_config, mesh, dtype=jnp.float32)
    model.load_weights(model_config)
    intermediates = model.forward_with_intermediates(
        DSparkDraftInputs(
            noise_embeddings=jnp.asarray(noise_embeddings),
            target_hidden_states=jnp.asarray(target_hidden_states),
            position_ids=jnp.asarray(position_ids),
        )
    )
    outputs = {
        "projected_target": np.asarray(intermediates.projected_target_hidden_states),
        **{
            f"layer_{layer_id}": np.asarray(hidden_states)
            for layer_id, hidden_states in enumerate(intermediates.layer_hidden_states)
        },
        "final": np.asarray(intermediates.final_hidden_states),
    }
    return outputs, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the pinned Torch and JAX Inkling-Small DSpark models."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--context-length", type=int, default=1)
    parser.add_argument("--draft-length", type=int, default=2)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    config = ModelConfig(str(args.checkpoint), dtype="float32").hf_config
    rng = np.random.default_rng(args.seed)
    noise_embeddings = rng.standard_normal(
        (1, args.draft_length, config.hidden_size), dtype=np.float32
    )
    target_hidden_states = rng.standard_normal(
        (
            1,
            args.context_length,
            len(config.dflash_config["target_layer_ids"]) * config.hidden_size,
        ),
        dtype=np.float32,
    )
    position_ids = np.arange(args.context_length + args.draft_length, dtype=np.int64)[
        None, :
    ]

    torch_outputs, torch_model = _torch_outputs(
        args.checkpoint,
        noise_embeddings=noise_embeddings,
        target_hidden_states=target_hidden_states,
        position_ids=position_ids,
    )
    jax_outputs, jax_model = _jax_outputs(
        args.checkpoint,
        noise_embeddings=noise_embeddings,
        target_hidden_states=target_hidden_states,
        position_ids=position_ids,
    )
    results = {
        name: _metrics(torch_outputs[name], jax_outputs[name]) for name in torch_outputs
    }

    anchor_ids = np.asarray([7], dtype=np.int32)
    torch_anchor = torch.from_numpy(anchor_ids)
    with torch.inference_mode():
        torch_markov = torch_model.markov_head.get_prev_embeddings(torch_anchor)
        torch_bias = torch_model.markov_head.project_bias(torch_markov)
        torch_confidence = torch_model.confidence_head(
            torch.cat(
                (
                    torch.from_numpy(torch_outputs["final"][:, 0]),
                    torch_markov,
                ),
                dim=-1,
            )
        )
    jax_markov = jax_model.markov_head.embedding(jnp.asarray(anchor_ids))
    jax_bias = jax_model.markov_head.bias(jax_markov)
    jax_confidence = jax_model.confidence_head(
        jax.device_put(
            jnp.asarray(jax_outputs["final"][:, 0]),
            NamedSharding(jax_model.mesh, P("data", None)),
        ),
        jax_markov,
    )
    results["markov_embedding"] = _metrics(torch_markov.numpy(), np.asarray(jax_markov))
    results["markov_bias"] = _metrics(torch_bias.numpy(), np.asarray(jax_bias))
    results["confidence"] = _metrics(
        torch_confidence.numpy(), np.asarray(jax_confidence)
    )

    print(json.dumps(results, indent=2, sort_keys=True))
    failures = {
        name: metrics
        for name, metrics in results.items()
        if metrics["cosine"] < args.minimum_cosine
    }
    if failures:
        raise SystemExit(f"DSpark parity failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
