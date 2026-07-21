# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate==1.14.0",
#   "huggingface-hub==1.24.0",
#   "scipy==1.17.0",
#   "torch==2.12.0",
#   "transformers==5.14.0",
# ]
# ///

import collections
import json

import transformers
from accelerate import init_empty_weights
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForMultimodalLM

REPOSITORY = "thinkingmachines/Inkling"
REVISION = "32b58a6494356948441ceedc192f1194f06f2e23"


def parameter_category(name: str) -> str:
    if name.startswith("model.vision_tower."):
        return "vision"
    if name.startswith("model.audio_tower."):
        return "audio"
    if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
        return "embedding_and_output"
    if ".self_attn." in name:
        return "attention"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_experts." in name:
        return "shared_experts"
    if ".mlp.gate." in name:
        return "expert_router"
    if any(part in name for part in (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.")):
        return "dense_mlp"
    if ".attn_sconv." in name or ".mlp_sconv." in name:
        return "residual_short_convolution"
    if "norm" in name:
        return "normalization"
    return "other"


def main() -> None:
    api = HfApi()
    info = api.model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    index_path = hf_hub_download(REPOSITORY, "model.safetensors.index.json", revision=REVISION)
    with open(index_path) as index_file:
        index = json.load(index_file)

    config = AutoConfig.from_pretrained(REPOSITORY, revision=REVISION)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForMultimodalLM.from_config(config)

    category_counts = collections.Counter()
    layer_counts = collections.Counter()
    parameters = list(model.named_parameters())
    for name, parameter in parameters:
        category_counts[parameter_category(name)] += parameter.numel()
        if name.startswith("model.language_model.layers."):
            layer_counts[int(name.split(".")[3])] += parameter.numel()

    checkpoint_parameter_count = info.safetensors.total
    base_parameter_count = sum(parameter.numel() for _, parameter in parameters)
    mtp_tensor_count = sum(name.startswith("model.mtp.") for name in index["weight_map"])
    mtp_file = next(sibling for sibling in info.siblings if sibling.rfilename == "mtp.safetensors")
    text_config = config.text_config
    local_layer_ids = [index for index, kind in enumerate(text_config.layer_types) if kind == "hybrid_sliding"]
    global_layer_ids = [index for index, kind in enumerate(text_config.layer_types) if kind == "hybrid"]
    dense_layer_ids = [index for index, kind in enumerate(text_config.mlp_layer_types) if kind == "dense"]
    sparse_layer_ids = [index for index, kind in enumerate(text_config.mlp_layer_types) if kind == "sparse"]

    result = {
        "checkpoint": {
            "repository": REPOSITORY,
            "revision": info.sha,
            "parameter_count": checkpoint_parameter_count,
            "parameters_by_dtype": info.safetensors.parameters,
            "stored_tensor_bytes": index["metadata"]["total_size"],
            "tensor_count": len(index["weight_map"]),
            "file_count": len(set(index["weight_map"].values())),
            "mtp_tensor_count": mtp_tensor_count,
            "mtp_file_bytes": mtp_file.size,
        },
        "transformers_reference": {
            "version": transformers.__version__,
            "base_parameter_count": base_parameter_count,
            "base_parameter_bytes_if_bf16": base_parameter_count * 2,
            "checkpoint_parameters_not_instantiated": checkpoint_parameter_count - base_parameter_count,
            "named_parameter_tensor_count": len(parameters),
            "parameters_by_category": dict(category_counts.most_common()),
            "parameter_percentage_by_category": {
                name: round(100 * count / base_parameter_count, 6) for name, count in category_counts.most_common()
            },
            "parameters_by_layer": {str(layer_id): layer_counts[layer_id] for layer_id in sorted(layer_counts)},
        },
        "architecture": {
            "hidden_size": text_config.hidden_size,
            "layer_count": text_config.num_hidden_layers,
            "local_layer_ids": local_layer_ids,
            "global_layer_ids": global_layer_ids,
            "dense_layer_ids": dense_layer_ids,
            "sparse_layer_ids": sparse_layer_ids,
            "routed_expert_count": text_config.n_routed_experts,
            "routed_experts_used_per_token": text_config.num_experts_per_tok,
            "shared_expert_count": text_config.n_shared_experts,
            "sliding_window": text_config.sliding_window_size,
            "short_convolution_kernel": text_config.conv_kernel_size,
            "maximum_context": text_config.max_position_embeddings,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
