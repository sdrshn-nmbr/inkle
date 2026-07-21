# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.0,<2",
#   "huggingface-hub>=1.0,<2",
#   "scipy>=1.10,<2",
#   "torch==2.12.0",
#   "transformers==5.14.0",
# ]
# ///

import collections
import json
import tempfile
from pathlib import Path

from accelerate import init_empty_weights
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForMultimodalLM

MODEL_ID = "thinkingmachines/Inkling"
MODEL_REVISION = "32b58a6494356948441ceedc192f1194f06f2e23"
METADATA_FILES = ("config.json", "model.safetensors.index.json")


def parameter_category(name: str) -> str:
    if name.startswith("model.vision_tower"):
        return "vision"
    if name.startswith("model.audio_tower"):
        return "audio"
    if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
        return "token_embeddings_and_output"
    if ".mlp.experts." in name:
        return "routed_expert_weights"
    if ".mlp.shared_experts." in name:
        return "shared_expert_weights"
    if ".mlp.gate." in name:
        return "expert_router"
    if any(part in name for part in (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.", ".mlp.global_scale")):
        return "dense_mlp"
    if ".self_attn." in name:
        return "attention"
    if ".attn_sconv." in name or ".mlp_sconv." in name:
        return "residual_short_convolutions"
    if "norm" in name:
        return "remaining_norms"
    raise ValueError(f"Unclassified Inkling parameter: {name}")


def inspect_model() -> dict[str, object]:
    api = HfApi()
    model_info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)

    with tempfile.TemporaryDirectory(prefix="inkling-metadata-") as temporary_directory:
        local_directory = Path(temporary_directory)
        for filename in METADATA_FILES:
            api.hf_hub_download(
                MODEL_ID,
                filename,
                revision=MODEL_REVISION,
                local_dir=local_directory,
            )

        config = AutoConfig.from_pretrained(local_directory, local_files_only=True)
        with init_empty_weights(include_buffers=True):
            model = AutoModelForMultimodalLM.from_config(config)

        checkpoint_index = json.loads((local_directory / "model.safetensors.index.json").read_text())

    parameters_by_category: collections.Counter[str] = collections.Counter()
    tensors_by_category: collections.Counter[str] = collections.Counter()
    for name, parameter in model.named_parameters():
        category = parameter_category(name)
        parameters_by_category[category] += parameter.numel()
        tensors_by_category[category] += 1

    logical_parameters = sum(parameters_by_category.values())
    routed_parameters = parameters_by_category["routed_expert_weights"]
    active_parameters = (
        logical_parameters
        - routed_parameters
        + (routed_parameters * config.text_config.num_experts_per_tok // config.text_config.n_routed_experts)
    )
    mtp_sibling = next(sibling for sibling in model_info.siblings if sibling.rfilename == "mtp.safetensors")
    checkpoint_parameters = model_info.safetensors.total

    return {
        "source": {
            "model_id": MODEL_ID,
            "revision": model_info.sha,
            "checkpoint_parameters": checkpoint_parameters,
            "checkpoint_tensor_bytes": checkpoint_index["metadata"]["total_size"],
            "checkpoint_tensors": len(checkpoint_index["weight_map"]),
            "checkpoint_files": len(set(checkpoint_index["weight_map"].values())),
            "mtp_tensors": sum(name.startswith("model.mtp.") for name in checkpoint_index["weight_map"]),
            "mtp_file_bytes": mtp_sibling.size,
        },
        "transformers_reference": {
            "model_class": type(model).__name__,
            "logical_parameters_without_mtp": logical_parameters,
            "bf16_bytes_without_mtp": logical_parameters * 2,
            "checkpoint_minus_reference_parameters": checkpoint_parameters - logical_parameters,
            "estimated_active_parameters_per_token": active_parameters,
            "routed_expert_fraction": routed_parameters / logical_parameters,
            "parameter_categories": {
                category: {
                    "parameters": count,
                    "bf16_bytes": count * 2,
                    "tensors": tensors_by_category[category],
                }
                for category, count in parameters_by_category.most_common()
            },
        },
        "architecture": {
            "layers": config.text_config.num_hidden_layers,
            "sliding_attention_layers": len(config.text_config.local_layer_ids),
            "full_attention_layers": config.text_config.num_hidden_layers - len(config.text_config.local_layer_ids),
            "dense_layers": sum(layer_type == "dense" for layer_type in config.text_config.mlp_layer_types),
            "sparse_layers": sum(layer_type == "sparse" for layer_type in config.text_config.mlp_layer_types),
            "routed_experts_per_sparse_layer": config.text_config.n_routed_experts,
            "selected_routed_experts_per_token": config.text_config.num_experts_per_tok,
            "shared_experts_per_sparse_layer": config.text_config.n_shared_experts,
        },
    }


def main() -> None:
    print(json.dumps(inspect_model(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
