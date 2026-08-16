import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import uuid
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--request-mode",
        choices=("concurrent", "batch"),
        default="concurrent",
    )
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--return-routed-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def stream_request(
    url: str,
    prompt: str,
    output_tokens: int,
    start_barrier: threading.Barrier,
    ignore_eos: bool,
    return_routed_experts: bool,
) -> dict[str, object]:
    request_body = {
        "rid": uuid.uuid4().hex,
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
        },
        "stream": True,
        "return_routed_experts": return_routed_experts,
    }
    start_barrier.wait()
    start_ns = time.perf_counter_ns()
    chunks = []
    with requests.post(
        f"{url}/generate",
        json=request_body,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            chunks.append(
                {
                    "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
                    "response": json.loads(payload),
                }
            )
    end_ns = time.perf_counter_ns()
    first_ms = chunks[0]["elapsed_ms"] if chunks else None
    arrival_ms = [chunk["elapsed_ms"] for chunk in chunks]
    inter_token_ms = [
        current - previous
        for previous, current in zip(arrival_ms, arrival_ms[1:], strict=False)
    ]
    return {
        "chunks": chunks,
        "e2e_ms": (end_ns - start_ns) / 1e6,
        "inter_token_ms": inter_token_ms,
        "request": request_body,
        "time_to_first_token_ms": first_ms,
    }


def run_group(
    url: str,
    prompt: str,
    output_tokens: int,
    concurrency: int,
    request_mode: str,
    input_ids: list[int] | None,
    ignore_eos: bool,
    return_routed_experts: bool,
) -> dict[str, object]:
    if request_mode == "batch":
        return run_batch_group(
            url,
            prompt,
            output_tokens,
            concurrency,
            input_ids,
            ignore_eos,
            return_routed_experts,
        )

    start_barrier = threading.Barrier(concurrency + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_request,
                url,
                prompt,
                output_tokens,
                start_barrier,
                ignore_eos,
                return_routed_experts,
            )
            for _ in range(concurrency)
        ]
        start_barrier.wait()
        group_start_ns = time.perf_counter_ns()
        requests_out = [future.result() for future in futures]
    group_ms = (time.perf_counter_ns() - group_start_ns) / 1e6
    total_tokens = sum(
        request["chunks"][-1]["response"]["meta_info"]["completion_tokens"]
        for request in requests_out
        if request["chunks"]
    )
    return {
        "concurrency": concurrency,
        "group_ms": group_ms,
        "requests": requests_out,
        "throughput_tokens_per_second": total_tokens / (group_ms / 1000),
        "total_completion_tokens": total_tokens,
    }


def run_batch_group(
    url: str,
    prompt: str,
    output_tokens: int,
    concurrency: int,
    input_ids: list[int] | None,
    ignore_eos: bool,
    return_routed_experts: bool,
) -> dict[str, object]:
    request_body = {
        "rid": [uuid.uuid4().hex for _ in range(concurrency)],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
        },
        "stream": True,
        "return_routed_experts": return_routed_experts,
    }
    if input_ids is None:
        request_body["text"] = [prompt] * concurrency
    else:
        request_body["input_ids"] = [input_ids] * concurrency
    start_ns = time.perf_counter_ns()
    chunks_by_request = [[] for _ in range(concurrency)]
    with requests.post(
        f"{url}/generate",
        json=request_body,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            output = json.loads(payload)
            chunks_by_request[output.pop("index")].append(
                {"elapsed_ms": elapsed_ms, "response": output}
            )
    end_ns = time.perf_counter_ns()
    requests_out = []
    for chunks in chunks_by_request:
        arrival_ms = [chunk["elapsed_ms"] for chunk in chunks]
        requests_out.append(
            {
                "chunks": chunks,
                "e2e_ms": (end_ns - start_ns) / 1e6,
                "inter_token_ms": [
                    current - previous
                    for previous, current in zip(arrival_ms, arrival_ms[1:], strict=False)
                ],
                "request": request_body,
                "time_to_first_token_ms": arrival_ms[0] if arrival_ms else None,
            }
        )
    group_ms = (end_ns - start_ns) / 1e6
    total_tokens = sum(
        chunks[-1]["response"]["meta_info"]["completion_tokens"]
        for chunks in chunks_by_request
        if chunks
    )
    return {
        "concurrency": concurrency,
        "group_ms": group_ms,
        "request_mode": "batch",
        "requests": requests_out,
        "throughput_tokens_per_second": total_tokens / (group_ms / 1000),
        "total_completion_tokens": total_tokens,
    }


def summarize(groups: list[dict[str, object]]) -> dict[str, object]:
    by_concurrency = {}
    for concurrency in sorted({group["concurrency"] for group in groups}):
        selected = [group for group in groups if group["concurrency"] == concurrency]
        e2e = [request["e2e_ms"] for group in selected for request in group["requests"]]
        ttft = [
            request["time_to_first_token_ms"]
            for group in selected
            for request in group["requests"]
            if request["time_to_first_token_ms"] is not None
        ]
        itl = [
            value
            for group in selected
            for request in group["requests"]
            for value in request["inter_token_ms"]
        ]
        full_batch_metrics = [fully_active_metrics(group) for group in selected]
        output_signatures = [output_multiset_signature(group) for group in selected]
        by_concurrency[str(concurrency)] = {
            "e2e_median_ms": statistics.median(e2e),
            "e2e_p95_ms": sorted(e2e)[max(0, int(len(e2e) * 0.95) - 1)],
            "inter_token_median_ms": statistics.median(itl) if itl else None,
            "throughput_median_tokens_per_second": statistics.median(
                group["throughput_tokens_per_second"] for group in selected
            ),
            "time_to_first_token_median_ms": statistics.median(ttft) if ttft else None,
            "fully_active_model_step_median_ms": statistics.median(
                metrics["model_step_median_ms"] for metrics in full_batch_metrics
            ),
            "fully_active_throughput_median_tokens_per_second": statistics.median(
                metrics["throughput_tokens_per_second"] for metrics in full_batch_metrics
            ),
            "output_multiset_stable": len(set(output_signatures)) == 1,
        }
    return by_concurrency


def fully_active_metrics(group: dict[str, object]) -> dict[str, float]:
    requests_out = [request for request in group["requests"] if request["chunks"]]
    start_ms = max(request["chunks"][0]["elapsed_ms"] for request in requests_out)
    end_ms = min(request["chunks"][-1]["elapsed_ms"] for request in requests_out)
    if end_ms <= start_ms:
        raise ValueError("No interval exists with every request decoding")

    token_count = sum(
        start_ms <= chunk["elapsed_ms"] < end_ms
        for request in requests_out
        for chunk in request["chunks"]
    )
    model_steps = [
        request["inter_token_ms"][index]
        for request in requests_out
        for index in range(len(request["inter_token_ms"]))
        if start_ms <= request["chunks"][index]["elapsed_ms"]
        and request["chunks"][index + 1]["elapsed_ms"] <= end_ms
    ]
    return {
        "model_step_median_ms": statistics.median(model_steps),
        "throughput_tokens_per_second": token_count * 1000 / (end_ms - start_ms),
    }


def output_multiset_signature(group: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            hashlib.sha256(
                request["chunks"][-1]["response"]["text"].encode()
            ).hexdigest()
            for request in group["requests"]
            if request["chunks"]
        )
    )


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    groups = []
    for concurrency in args.concurrency:
        for repetition in range(args.repetitions):
            group = run_group(
                args.url,
                args.prompt,
                args.output_tokens,
                concurrency,
                args.request_mode,
                args.input_ids,
                args.ignore_eos,
                args.return_routed_experts,
            )
            group["repetition"] = repetition
            groups.append(group)
            print(
                json.dumps(
                    {
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "throughput_tokens_per_second": group[
                            "throughput_tokens_per_second"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    result = {
        "groups": groups,
        "summary": summarize(groups),
    }
    (args.output_directory / "serving-benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
