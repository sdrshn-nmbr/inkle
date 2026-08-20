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
from transformers import DynamicCache

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


def _torch_proposal(
    model: TorchDSparkDraftModel,
    hidden_states: torch.Tensor,
    base_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    previous = anchor_token_ids
    token_rows = []
    logit_rows = []
    confidence_rows = []
    for step in range(hidden_states.shape[1]):
        previous_embedding = model.markov_head.get_prev_embeddings(previous)
        corrected = base_logits[:, step] + model.markov_head.project_bias(
            previous_embedding
        )
        token_ids = torch.argmax(corrected, dim=-1)
        confidence = model.confidence_head(
            torch.cat((hidden_states[:, step], previous_embedding), dim=-1)
        )
        token_rows.append(token_ids)
        logit_rows.append(corrected)
        confidence_rows.append(confidence)
        previous = token_ids
    return (
        torch.stack(token_rows, dim=1),
        torch.stack(logit_rows, dim=1),
        torch.stack(confidence_rows, dim=1),
    )


def _multi_round_results(
    torch_model: TorchDSparkDraftModel,
    jax_model: DSparkDraftModel,
    *,
    rng: np.random.Generator,
    context_width: int,
    accepted_lengths: list[int],
) -> dict[str, dict[str, float] | dict[str, bool]]:
    gamma = int(jax_model.config.block_size)
    hidden_size = int(jax_model.config.hidden_size)
    vocab_size = int(jax_model.config.vocab_size)
    torch_cache = DynamicCache()
    jax_history = np.zeros((1, 0, context_width), dtype=np.float32)
    results: dict[str, dict[str, float] | dict[str, bool]] = {}
    for round_index, accepted_length in enumerate(accepted_lengths):
        new_context_length = 2 if round_index == 0 else accepted_lengths[round_index - 1]
        new_target = rng.standard_normal(
            (1, new_context_length, context_width), dtype=np.float32
        )
        noise = rng.standard_normal((1, gamma, hidden_size), dtype=np.float32)
        previous_context_length = jax_history.shape[1]
        total_context_length = previous_context_length + new_context_length
        torch_positions = np.arange(
            previous_context_length,
            total_context_length + gamma,
            dtype=np.int64,
        )[None, :]
        with torch.inference_mode():
            torch_hidden = torch_model(
                position_ids=torch.from_numpy(torch_positions),
                noise_embedding=torch.from_numpy(noise),
                target_hidden=torch.from_numpy(new_target),
                past_key_values=torch_cache,
                use_cache=True,
                is_causal=False,
            )
        torch_cache.crop(total_context_length)

        jax_history = np.concatenate((jax_history, new_target), axis=1)
        context = jax_model.encode_context(
            jnp.asarray(jax_history),
            jnp.arange(total_context_length, dtype=jnp.int32)[None, :],
        )
        jax_hidden = jax_model.forward_cached(
            jnp.asarray(noise),
            context,
            jnp.arange(
                total_context_length,
                total_context_length + gamma,
                dtype=jnp.int32,
            )[None, :],
            jnp.ones((1, gamma, total_context_length + gamma), dtype=jnp.bool_),
        )
        hidden_metrics = _metrics(
            torch_hidden.detach().cpu().numpy(), np.asarray(jax_hidden)
        )

        base_logits = np.zeros((1, gamma, vocab_size), dtype=np.float32)
        anchor = np.asarray([17 + round_index], dtype=np.int32)
        with torch.inference_mode():
            torch_tokens, torch_logits, torch_confidence = _torch_proposal(
                torch_model,
                torch_hidden,
                torch.from_numpy(base_logits),
                torch.from_numpy(anchor),
            )
        jax_proposal = jax_model.greedy_propose(
            jnp.asarray(base_logits),
            jax_hidden,
            jnp.asarray(anchor),
        )
        results[f"round_{round_index}_hidden"] = hidden_metrics
        results[f"round_{round_index}_corrected_logits"] = _metrics(
            torch_logits.detach().cpu().numpy(),
            np.asarray(jax_proposal.corrected_logits),
        )
        results[f"round_{round_index}_confidence"] = _metrics(
            torch_confidence.detach().cpu().numpy(),
            np.asarray(jax_proposal.confidence_logits),
        )
        results[f"round_{round_index}_tokens"] = {
            "exact": bool(
                np.array_equal(
                    torch_tokens.detach().cpu().numpy(),
                    np.asarray(jax_proposal.token_ids),
                )
            )
        }
        if not 1 <= accepted_length <= gamma + 1:
            raise ValueError(
                f"accepted length must be in [1, {gamma + 1}], got {accepted_length}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the pinned Torch and JAX Inkling-Small DSpark models."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--context-length", type=int, default=1)
    parser.add_argument("--draft-length", type=int, default=2)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument(
        "--round-accept-lengths",
        type=int,
        nargs="+",
        default=[8, 1, 4],
        help="Accepted target-input prefixes used to exercise full and partial cache crops.",
    )
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
    results.update(
        _multi_round_results(
            torch_model,
            jax_model,
            rng=rng,
            context_width=len(config.dflash_config["target_layer_ids"])
            * config.hidden_size,
            accepted_lengths=args.round_accept_lengths,
        )
    )

    print(json.dumps(results, indent=2, sort_keys=True))
    failures = {
        name: metrics
        for name, metrics in results.items()
        if (
            "cosine" in metrics
            and metrics["cosine"] < args.minimum_cosine
        )
        or ("exact" in metrics and not metrics["exact"])
    }
    if failures:
        raise SystemExit(f"DSpark parity failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
