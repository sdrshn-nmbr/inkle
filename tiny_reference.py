# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "safetensors>=0.6,<1",
#   "torch==2.12.0",
#   "transformers==5.14.0",
# ]
# ///

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
import transformers
from safetensors.torch import save_file
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingExperts, InklingForCausalLM, InklingMLP, InklingMoE

SEED = 17
INPUT_IDS = torch.tensor([[1, 7, 11, 19, 23, 29, 31]], dtype=torch.long)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
EXPERT_IMPLEMENTATIONS = ("eager", "selected")


class SelectedInklingExperts(InklingExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        selected_gate_up = self.gate_up_proj[top_k_index]
        gate_up = torch.einsum("tkih,th->tki", selected_gate_up, hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up

        selected_down = self.down_proj[top_k_index]
        expert_outputs = torch.einsum("tkhi,tki->tkh", selected_down, activated)
        return (expert_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1).to(hidden_states.dtype)


def build_config() -> InklingTextConfig:
    config = InklingTextConfig(
        vocab_size=128,
        unpadded_vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=8,
        local_layer_ids=[0],
        max_position_embeddings=32,
        conv_kernel_size=4,
        mlp_layer_types=["dense", "sparse"],
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        log_scaling_n_floor=16,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def build_model(
    dtype: torch.dtype, expert_implementation: Literal["eager", "selected"] = "eager"
) -> InklingForCausalLM:
    torch.manual_seed(SEED)
    model = InklingForCausalLM(build_config()).eval()
    if expert_implementation == "selected":
        original_experts = model.model.layers[1].mlp.experts
        selected_experts = SelectedInklingExperts(model.config)
        selected_experts.load_state_dict(original_experts.state_dict())
        model.model.layers[1].mlp.experts = selected_experts
    model = model.to(dtype=dtype)
    model.config._experts_implementation = "eager"
    assert isinstance(model.model.layers[0].mlp, InklingMLP)
    assert isinstance(model.model.layers[1].mlp, InklingMoE)
    return model


def tensor_digest(tensor: torch.Tensor) -> str:
    bytes_view = tensor.detach().float().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(bytes_view).hexdigest()


def run_reference(
    dtype: torch.dtype,
    generated_tokens: int,
    expert_implementation: Literal["eager", "selected"],
) -> tuple[InklingForCausalLM, dict[str, object]]:
    model = build_model(dtype, expert_implementation)
    attention_mask = torch.ones_like(INPUT_IDS)

    with torch.inference_mode():
        first_output = model(
            input_ids=INPUT_IDS,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        repeated_output = model(
            input_ids=INPUT_IDS,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        generated = model.generate(
            input_ids=INPUT_IDS,
            attention_mask=attention_mask,
            max_new_tokens=generated_tokens,
            do_sample=False,
        )

    assert torch.equal(first_output, repeated_output)
    assert torch.isfinite(first_output).all()

    result = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": SEED,
        "dtype": str(dtype).removeprefix("torch."),
        "expert_implementation": expert_implementation,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "attention_types": model.config.layer_types,
        "feed_forward_types": [type(layer.mlp).__name__ for layer in model.model.layers],
        "input_ids": INPUT_IDS.tolist(),
        "logits_shape": list(first_output.shape),
        "logits_sha256_float32": tensor_digest(first_output),
        "logits_sum_float32": first_output.float().sum().item(),
        "last_token_first_8_float32": first_output[0, -1, :8].float().tolist(),
        "generated_ids": generated.tolist(),
    }
    return model, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--experts", choices=EXPERT_IMPLEMENTATIONS, default="eager")
    parser.add_argument("--generated-tokens", type=int, default=3)
    parser.add_argument("--save-weights", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, result = run_reference(DTYPES[args.dtype], args.generated_tokens, args.experts)
    if args.save_weights is not None:
        args.save_weights.parent.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), args.save_weights)
        result["weights_path"] = str(args.save_weights.resolve())
        result["weights_bytes"] = args.save_weights.stat().st_size
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
