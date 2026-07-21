import argparse
import json
from pathlib import Path

import torch
import torch_xla
import torch_xla.debug.metrics as met
from safetensors.torch import load_file
from tiny_reference import INPUT_IDS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    return parser.parse_args()


def load_model(weights_path: Path) -> torch.nn.Module:
    model = build_model(torch.bfloat16, expert_implementation="selected")
    model.load_state_dict(load_file(weights_path))
    return model


def greedy_generate(model: torch.nn.Module, input_ids: torch.Tensor, generated_tokens: int) -> torch.Tensor:
    generated = input_ids
    for _ in range(generated_tokens):
        attention_mask = torch.ones_like(generated)
        logits = model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if generated.device.type == "xla":
            torch_xla.step()
    return generated


def main() -> None:
    args = parse_args()
    attention_mask = torch.ones_like(INPUT_IDS)

    cpu_model = load_model(args.weights)
    with torch.no_grad():
        cpu_logits = cpu_model(
            input_ids=INPUT_IDS,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()

    device = torch_xla.device()
    tpu_model = load_model(args.weights).to(device)
    tpu_input_ids = INPUT_IDS.to(device)
    tpu_attention_mask = attention_mask.to(device)

    met.clear_all()
    with torch.no_grad():
        first_tpu_logits = tpu_model(
            input_ids=tpu_input_ids,
            attention_mask=tpu_attention_mask,
            use_cache=False,
        ).logits
        torch_xla.step()
        first_tpu_logits = first_tpu_logits.cpu().float()

        second_tpu_logits = tpu_model(
            input_ids=tpu_input_ids,
            attention_mask=tpu_attention_mask,
            use_cache=False,
        ).logits
        torch_xla.step()
        second_tpu_logits = second_tpu_logits.cpu().float()

        cpu_generated = greedy_generate(cpu_model, INPUT_IDS, generated_tokens=3)
        tpu_generated = greedy_generate(tpu_model, tpu_input_ids, generated_tokens=3).cpu()

    absolute_difference = (cpu_logits - first_tpu_logits).abs()
    assert torch.isfinite(first_tpu_logits).all()
    assert torch.equal(first_tpu_logits, second_tpu_logits)
    assert torch.equal(cpu_logits.argmax(dim=-1), first_tpu_logits.argmax(dim=-1))
    assert torch.equal(cpu_generated, tpu_generated)
    torch.testing.assert_close(first_tpu_logits, cpu_logits, rtol=0.2, atol=0.004)

    result = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in tpu_model.parameters()),
        "logits_shape": list(first_tpu_logits.shape),
        "cpu_logits_sum": cpu_logits.sum().item(),
        "tpu_logits_sum": first_tpu_logits.sum().item(),
        "maximum_absolute_difference": absolute_difference.max().item(),
        "mean_absolute_difference": absolute_difference.mean().item(),
        "last_token_cpu_top_id": cpu_logits[0, -1].argmax().item(),
        "last_token_tpu_top_id": first_tpu_logits[0, -1].argmax().item(),
        "cpu_generated_ids": cpu_generated.tolist(),
        "tpu_generated_ids": tpu_generated.tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(met.metrics_report())


if __name__ == "__main__":
    main()
