import argparse
import json

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.debug.metrics as met
from torch_xla.experimental.xla_quantized_matmul import quantized_matmul_xla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=128)
    parser.add_argument("--skip-int4", action="store_true")
    parser.add_argument("--inkling-expert", action="store_true")
    return parser.parse_args()


def symmetric_quantize(weight: torch.Tensor, maximum_integer: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=1).clamp_min(1e-8) / maximum_integer
    quantized = torch.round(weight.float() / scale[:, None]).clamp(-maximum_integer, maximum_integer).to(torch.int8)
    return quantized, scale.to(weight.dtype)


def blockwise_symmetric_quantize(
    weight: torch.Tensor,
    maximum_integer: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_features = weight.shape[1]
    blocks = weight.float().T.contiguous().reshape(input_features // block_size, block_size, weight.shape[0])
    scale = blocks.abs().amax(dim=1).clamp_min(1e-8) / maximum_integer
    quantized = torch.round(blocks / scale[:, None, :]).clamp(-maximum_integer, maximum_integer).to(torch.int8)
    return quantized, scale.to(weight.dtype)


def error_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference - candidate).abs().float()
    reference_float = reference.float()
    candidate_float = candidate.float()
    root_mean_square_error = difference.square().mean().sqrt()
    reference_root_mean_square = reference_float.square().mean().sqrt()
    return {
        "maximum_absolute_difference": difference.max().item(),
        "mean_absolute_difference": difference.mean().item(),
        "normalized_root_mean_square_error": (root_mean_square_error / reference_root_mean_square).item(),
        "cosine_similarity": F.cosine_similarity(reference_float.flatten(), candidate_float.flatten(), dim=0).item(),
    }


def quantized_linear(
    activations: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    quantize_activation: bool,
) -> torch.Tensor:
    return quantized_matmul_xla(
        activations,
        quantized_weight,
        scale,
        None,
        block_size=-1,
        int4_weight=False,
        quantize_activation=quantize_activation,
    )


def run_inkling_expert_probe(tokens: int) -> dict[str, object]:
    hidden_size = 6144
    intermediate_size = 3072
    activations = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(2 * intermediate_size, hidden_size, dtype=torch.bfloat16) / hidden_size**0.5
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16) / intermediate_size**0.5

    gate, up = F.linear(activations, gate_up_weight).chunk(2, dim=-1)
    reference = F.linear(F.silu(gate) * up, down_weight).float()
    quantized_gate_up_weight, gate_up_scale = symmetric_quantize(gate_up_weight, maximum_integer=127)
    quantized_down_weight, down_scale = symmetric_quantize(down_weight, maximum_integer=127)

    device = torch_xla.device()
    xla_activations = activations.to(device)
    xla_gate_up_weight = quantized_gate_up_weight.to(device)
    xla_gate_up_scale = gate_up_scale.to(device)
    xla_down_weight = quantized_down_weight.to(device)
    xla_down_scale = down_scale.to(device)

    met.clear_all()
    weight_only_gate, weight_only_up = quantized_linear(
        xla_activations,
        xla_gate_up_weight,
        xla_gate_up_scale,
        quantize_activation=False,
    ).chunk(2, dim=-1)
    weight_only_output = quantized_linear(
        F.silu(weight_only_gate) * weight_only_up,
        xla_down_weight,
        xla_down_scale,
        quantize_activation=False,
    )

    w8a8_gate, w8a8_up = quantized_linear(
        xla_activations,
        xla_gate_up_weight,
        xla_gate_up_scale,
        quantize_activation=True,
    ).chunk(2, dim=-1)
    w8a8_output = quantized_linear(
        F.silu(w8a8_gate) * w8a8_up,
        xla_down_weight,
        xla_down_scale,
        quantize_activation=True,
    )

    result = {
        "device": str(device),
        "shape": {
            "tokens": tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "weight_only_int8": error_summary(reference, weight_only_output.cpu().float()),
        "weight_and_activation_int8": error_summary(reference, w8a8_output.cpu().float()),
    }
    torch_xla.sync(wait=True)
    result["uncached_compilations"] = met.counter_value("UncachedCompile")
    result["cached_compilations"] = met.counter_value("CachedCompile")
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(17)

    if args.inkling_expert:
        print(json.dumps(run_inkling_expert_probe(args.tokens), indent=2, sort_keys=True))
        return

    activations = torch.randn(args.tokens, args.in_features, dtype=torch.bfloat16)
    weight = torch.randn(args.out_features, args.in_features, dtype=torch.bfloat16) / args.in_features**0.5
    int8_weight, int8_scale = symmetric_quantize(weight, maximum_integer=127)
    int4_weight, int4_scale = symmetric_quantize(weight, maximum_integer=7)
    int4_block_size = 32
    int4_block_weight, int4_block_scale = blockwise_symmetric_quantize(
        weight,
        maximum_integer=7,
        block_size=int4_block_size,
    )

    reference = F.linear(activations, weight).float()
    device = torch_xla.device()
    xla_activations = activations.to(device)
    xla_int8_weight = int8_weight.to(device)
    xla_int8_scale = int8_scale.to(device)

    met.clear_all()
    weight_only_int8 = quantized_matmul_xla(
        xla_activations,
        xla_int8_weight,
        xla_int8_scale,
        None,
        block_size=-1,
        int4_weight=False,
        quantize_activation=False,
    )
    weight_and_activation_int8 = quantized_matmul_xla(
        xla_activations,
        xla_int8_weight,
        xla_int8_scale,
        None,
        block_size=-1,
        int4_weight=False,
        quantize_activation=True,
    )
    results = {
        "device": str(device),
        "shape": {
            "tokens": args.tokens,
            "in_features": args.in_features,
            "out_features": args.out_features,
        },
        "weight_only_int8": error_summary(reference, weight_only_int8.cpu().float()),
        "weight_and_activation_int8": error_summary(reference, weight_and_activation_int8.cpu().float()),
    }

    if not args.skip_int4:
        weight_only_int4 = quantized_matmul_xla(
            xla_activations,
            int4_weight.to(device),
            int4_scale.to(device),
            None,
            block_size=-1,
            int4_weight=True,
            quantize_activation=False,
        )
        results["weight_only_int4"] = error_summary(reference, weight_only_int4.cpu().float())
        weight_only_int4_blockwise = quantized_matmul_xla(
            xla_activations,
            int4_block_weight.to(device),
            int4_block_scale.to(device),
            None,
            block_size=int4_block_size,
            int4_weight=True,
            quantize_activation=False,
        )
        results["weight_only_int4_block32"] = error_summary(reference, weight_only_int4_blockwise.cpu().float())

    torch_xla.sync(wait=True)
    results["uncached_compilations"] = met.counter_value("UncachedCompile")
    results["cached_compilations"] = met.counter_value("CachedCompile")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
