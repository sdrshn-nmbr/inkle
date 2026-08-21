import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests
import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum wall time for each streaming generation request.",
    )
    parser.add_argument(
        "--request-mode",
        choices=("concurrent", "batch"),
        default="concurrent",
    )
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument(
        "--prompt-cases",
        type=Path,
        help="JSON prompt corpus with stable IDs and explicit input token IDs.",
    )
    parser.add_argument(
        "--workload",
        choices=("custom", "aa-1k", "aa-10k", "aa-100k", "aa-parallel-1k"),
        default="custom",
    )
    parser.add_argument(
        "--standard-tokenizer",
        default="o200k_base",
        help="Tiktoken encoding used for provider-comparable output counts.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--atomic-admission-delay-seconds",
        type=float,
        default=0.0,
        help="Pause generation while the complete request group enters the scheduler.",
    )
    parser.add_argument(
        "--stop-after-first-completion",
        action="store_true",
        help="Abort the remaining requests after the first request completes.",
    )
    parser.add_argument(
        "--return-routed-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-fully-active",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a contiguous interval where every submitted request is "
            "simultaneously decoding. Disable for provider-style latency runs."
        ),
    )
    return parser.parse_args()


WORKLOAD_CONTRACTS = {
    "aa-1k": {"input_tokens": 1_000, "minimum_output_tokens": 1_000},
    "aa-10k": {"input_tokens": 10_000, "minimum_output_tokens": 1_500},
    "aa-100k": {"input_tokens": 100_000, "minimum_output_tokens": 2_000},
    "aa-parallel-1k": {
        "input_tokens": 1_000,
        "minimum_output_tokens": 1_000,
        "required_concurrency": 10,
    },
}


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def compact_request_body(request_body: dict[str, object]) -> dict[str, object]:
    compact = dict(request_body)
    input_ids = compact.pop("input_ids", None)
    if input_ids is not None:
        compact["input_ids_count"] = len(input_ids)
        compact["input_ids_sha256"] = sha256_json(input_ids)
    return compact


def load_prompt_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prompt cases must be a non-empty JSON list")
    cases = []
    for index, case in enumerate(payload):
        if not isinstance(case, dict):
            raise ValueError(f"Prompt case {index} must be an object")
        case_id = case.get("id")
        input_ids = case.get("input_ids")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Prompt case {index} needs a non-empty string ID")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(not isinstance(token_id, int) for token_id in input_ids)
        ):
            raise ValueError(f"Prompt case {case_id} needs integer input_ids")
        cases.append(dict(case))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Prompt case IDs must be unique")
    return cases


def validate_workload(
    workload: str,
    prompt_cases: list[dict[str, object]] | None,
    output_tokens: int,
    concurrency: list[int],
    repetitions: int,
) -> None:
    required_prompt_cases = repetitions * sum(concurrency)
    if prompt_cases is not None and len(prompt_cases) < required_prompt_cases:
        raise ValueError(
            "Prompt corpus cannot provide a unique prompt for every request "
            f"required={required_prompt_cases} available={len(prompt_cases)}"
        )
    if workload == "custom":
        return
    if prompt_cases is None:
        raise ValueError(f"{workload} requires --prompt-cases")
    contract = WORKLOAD_CONTRACTS[workload]
    expected_tokens = int(contract["input_tokens"])
    wrong_lengths = {
        str(case["id"]): len(case["input_ids"])
        for case in prompt_cases
        if len(case["input_ids"]) != expected_tokens
    }
    if wrong_lengths:
        raise ValueError(
            f"{workload} requires exactly {expected_tokens} input tokens: "
            f"{wrong_lengths}"
        )
    minimum_output_tokens = int(contract["minimum_output_tokens"])
    if output_tokens < minimum_output_tokens:
        raise ValueError(
            f"{workload} requires at least {minimum_output_tokens} output tokens"
        )
    required_concurrency = contract.get("required_concurrency")
    if required_concurrency is not None and concurrency != [required_concurrency]:
        raise ValueError(
            f"{workload} requires --concurrency {required_concurrency}"
        )


def select_prompt_cases(
    prompt_cases: list[dict[str, object]], concurrency: int, offset: int
) -> list[dict[str, object]]:
    return prompt_cases[offset : offset + concurrency]


def run_text(command: list[str]) -> str | None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_provenance(
    args: argparse.Namespace,
    prompt_cases: list[dict[str, object]] | None,
) -> dict[str, object]:
    source_path = Path(__file__).resolve()
    server_info_response = requests.get(f"{args.url}/get_server_info", timeout=30)
    server_info_response.raise_for_status()
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "benchmark_source": str(source_path),
        "benchmark_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": run_text(["git", "status", "--short"]),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_environment": {
            name: os.environ[name]
            for name in ("JAX_PLATFORM_NAME", "LIBTPU_INIT_ARGS", "XLA_FLAGS")
            if name in os.environ
        },
        "server_info": server_info_response.json(),
        "workload": args.workload,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "repetitions": args.repetitions,
        "request_timeout_seconds": args.request_timeout_seconds,
        "request_mode": args.request_mode,
        "ignore_eos": args.ignore_eos,
        "require_fully_active": args.require_fully_active,
        "standard_tokenizer": args.standard_tokenizer,
        "prompt_cases_path": (
            str(args.prompt_cases.resolve()) if args.prompt_cases else None
        ),
        "prompt_cases_sha256": sha256_json(prompt_cases),
        "prompt_case_count": len(prompt_cases) if prompt_cases is not None else None,
        "native_input_token_counts": (
            sorted({len(case["input_ids"]) for case in prompt_cases})
            if prompt_cases is not None
            else None
        ),
        "standard_input_token_counts": (
            sorted(
                {
                    int(case["standard_input_tokens"])
                    for case in prompt_cases
                    if case.get("standard_input_tokens") is not None
                }
            )
            if prompt_cases is not None
            else None
        ),
    }


def stream_request(
    url: str,
    prompt: str,
    input_ids: list[int] | None,
    output_tokens: int,
    start_barrier: threading.Barrier,
    ignore_eos: bool,
    return_routed_experts: bool,
    stop_after_first_completion: bool = False,
    stop_event: threading.Event | None = None,
    request_timeout_seconds: float = 900.0,
) -> dict[str, object]:
    request_body = {
        "rid": uuid.uuid4().hex,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": ignore_eos,
        },
        "stream": True,
        "return_routed_experts": return_routed_experts,
    }
    if input_ids is None:
        request_body["text"] = prompt
    else:
        request_body["input_ids"] = input_ids
    start_barrier.wait()
    start_ns = time.perf_counter_ns()
    chunks = []
    with requests.post(
        f"{url}/generate",
        json=request_body,
        stream=True,
        timeout=request_timeout_seconds,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            output = json.loads(payload)
            compact_previous_chunk(chunks)
            chunks.append(
                {
                    "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
                    "response": output,
                }
            )
            if (
                stop_after_first_completion
                and output.get("meta_info", {}).get("finish_reason") is not None
                and stop_event is not None
                and not stop_event.is_set()
            ):
                stop_event.set()
                abort_response = requests.post(
                    f"{url}/abort_request",
                    json={"rid": None, "abort_all": True},
                    timeout=30,
                )
                abort_response.raise_for_status()
            if (
                stop_after_first_completion
                and stop_event is not None
                and stop_event.is_set()
            ):
                break
    end_ns = time.perf_counter_ns()
    first_ms = chunks[0]["elapsed_ms"] if chunks else None
    arrival_ms = [chunk["elapsed_ms"] for chunk in chunks]
    inter_token_ms = [
        current - previous
        for previous, current in zip(arrival_ms, arrival_ms[1:], strict=False)
    ]
    return {
        "completion_token_deltas": completion_token_deltas(chunks),
        "chunks": chunks,
        "e2e_ms": (end_ns - start_ns) / 1e6,
        "inter_token_ms": inter_token_ms,
        "request": compact_request_body(request_body),
        "time_to_first_token_ms": first_ms,
    }


def wait_for_server_idle(
    url: str,
    *,
    timeout_seconds: float = 300.0,
    poll_seconds: float = 0.25,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{url}/get_server_info", timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            last_state = {"poll_error": repr(exc)}
            time.sleep(poll_seconds)
            continue
        states = response.json()["internal_states"]
        last_state = states
        if all(
            int(state["running_batch_size"]) == 0
            and int(state["waiting_queue_size"]) == 0
            and int(state["req_to_token_pool_used"]) == 0
            for state in states
        ):
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"SERVER_IDLE_TIMEOUT states={last_state}")


def run_group(
    url: str,
    prompt: str,
    output_tokens: int,
    concurrency: int,
    request_mode: str,
    input_ids: list[int] | None,
    ignore_eos: bool,
    return_routed_experts: bool,
    atomic_admission_delay_seconds: float = 0.0,
    stop_after_first_completion: bool = False,
    prompt_cases: list[dict[str, object]] | None = None,
    request_timeout_seconds: float = 900.0,
) -> dict[str, object]:
    if prompt_cases is not None and len(prompt_cases) != concurrency:
        raise ValueError(
            "The selected prompt case count must equal concurrency "
            f"cases={len(prompt_cases)} concurrency={concurrency}"
        )
    if atomic_admission_delay_seconds > 0:
        pause_response = requests.post(
            f"{url}/pause_generation",
            json={"mode": "in_place"},
            timeout=30,
        )
        pause_response.raise_for_status()
        resumed = False
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_group,
                    url=url,
                    prompt=prompt,
                    output_tokens=output_tokens,
                    concurrency=concurrency,
                    request_mode=request_mode,
                    input_ids=input_ids,
                    ignore_eos=ignore_eos,
                    return_routed_experts=return_routed_experts,
                    atomic_admission_delay_seconds=0.0,
                    stop_after_first_completion=stop_after_first_completion,
                    prompt_cases=prompt_cases,
                    request_timeout_seconds=request_timeout_seconds,
                )
                time.sleep(atomic_admission_delay_seconds)
                continue_response = requests.post(
                    f"{url}/continue_generation",
                    json={},
                    timeout=30,
                )
                continue_response.raise_for_status()
                resumed = True
                return future.result()
        finally:
            if not resumed:
                requests.post(
                    f"{url}/continue_generation",
                    json={},
                    timeout=30,
                ).raise_for_status()
    if request_mode == "batch":
        if stop_after_first_completion:
            raise ValueError(
                "First-completion stopping requires concurrent request mode"
            )
        return run_batch_group(
            url,
            prompt,
            output_tokens,
            concurrency,
            input_ids,
            ignore_eos,
            return_routed_experts,
            prompt_cases,
            request_timeout_seconds,
        )

    start_barrier = threading.Barrier(concurrency + 1)
    stop_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for index in range(concurrency):
            case = prompt_cases[index] if prompt_cases is not None else None
            request_input_ids = case["input_ids"] if case is not None else input_ids
            futures.append(
                executor.submit(
                    stream_request,
                    url,
                    prompt,
                    request_input_ids,
                    output_tokens,
                    start_barrier,
                    ignore_eos,
                    return_routed_experts,
                    stop_after_first_completion,
                    stop_event,
                    request_timeout_seconds,
                )
            )
        start_barrier.wait()
        group_start_ns = time.perf_counter_ns()
        requests_out = [future.result() for future in futures]
    if prompt_cases is not None:
        for request, case in zip(requests_out, prompt_cases, strict=True):
            request["prompt_case_id"] = case["id"]
            request["native_input_tokens"] = len(case["input_ids"])
            request["standard_input_tokens"] = case.get("standard_input_tokens")
            request["decoded_prompt_sha256"] = case.get("decoded_text_sha256")
    if stop_after_first_completion:
        wait_for_server_idle(url)
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
        "stopped_after_first_completion": stop_after_first_completion,
    }


def run_batch_group(
    url: str,
    prompt: str,
    output_tokens: int,
    concurrency: int,
    input_ids: list[int] | None,
    ignore_eos: bool,
    return_routed_experts: bool,
    prompt_cases: list[dict[str, object]] | None = None,
    request_timeout_seconds: float = 900.0,
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
    if prompt_cases is not None:
        request_body["input_ids"] = [case["input_ids"] for case in prompt_cases]
    elif input_ids is None:
        request_body["text"] = [prompt] * concurrency
    else:
        request_body["input_ids"] = [input_ids] * concurrency
    start_ns = time.perf_counter_ns()
    chunks_by_request = [[] for _ in range(concurrency)]
    with requests.post(
        f"{url}/generate",
        json=request_body,
        stream=True,
        timeout=request_timeout_seconds,
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
            request_index = output.pop("index")
            request_chunks = chunks_by_request[request_index]
            compact_previous_chunk(request_chunks)
            request_chunks.append(
                {"elapsed_ms": elapsed_ms, "response": output}
            )
    end_ns = time.perf_counter_ns()
    requests_out = []
    for index, chunks in enumerate(chunks_by_request):
        arrival_ms = [chunk["elapsed_ms"] for chunk in chunks]
        requests_out.append(
            {
                "completion_token_deltas": completion_token_deltas(chunks),
                "chunks": chunks,
                "e2e_ms": (end_ns - start_ns) / 1e6,
                "inter_token_ms": [
                    current - previous
                    for previous, current in zip(
                        arrival_ms, arrival_ms[1:], strict=False
                    )
                ],
                "request": compact_request_body(
                    {
                        **request_body,
                        "rid": request_body["rid"][index],
                        **(
                            {"input_ids": request_body["input_ids"][index]}
                            if "input_ids" in request_body
                            else {"text": request_body["text"][index]}
                        ),
                    }
                ),
                "prompt_case_id": (
                    prompt_cases[index]["id"] if prompt_cases is not None else None
                ),
                "native_input_tokens": (
                    len(prompt_cases[index]["input_ids"])
                    if prompt_cases is not None
                    else None
                ),
                "standard_input_tokens": (
                    prompt_cases[index].get("standard_input_tokens")
                    if prompt_cases is not None
                    else None
                ),
                "decoded_prompt_sha256": (
                    prompt_cases[index].get("decoded_text_sha256")
                    if prompt_cases is not None
                    else None
                ),
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


def summarize(
    groups: list[dict[str, object]],
    *,
    require_fully_active: bool = True,
    comparable_outputs: bool = True,
) -> dict[str, object]:
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
        full_batch_metrics = []
        for group in selected:
            try:
                full_batch_metrics.append(fully_active_metrics(group))
            except ValueError as error:
                if not require_fully_active:
                    continue
                repetition = group.get("repetition", "unknown")
                raise ValueError(
                    f"concurrency {concurrency} never became fully active in "
                    f"repetition {repetition}"
                ) from error
        accepted_per_step = [
            delta
            for group in selected
            for request in group["requests"]
            for delta in request["completion_token_deltas"]
        ]
        native_output_speeds = [
            speed
            for group in selected
            for request in group["requests"]
            if (speed := output_speed_after_first_token(request)) is not None
        ]
        standard_output_speeds = [
            float(request["standard_output_speed_after_first_token"])
            for group in selected
            for request in group["requests"]
            if request.get("standard_output_speed_after_first_token") is not None
        ]
        output_signatures = [output_multiset_signature(group) for group in selected]
        slot_output_signatures = [slot_output_signature(group) for group in selected]
        observed_batch_sizes = [
            int(chunk["response"]["meta_info"]["server_batch_size"])
            for group in selected
            for request in group["requests"]
            for chunk in request["chunks"]
        ]
        by_concurrency[str(concurrency)] = {
            "e2e_median_ms": statistics.median(e2e),
            "e2e_p95_ms": percentile(e2e, 0.95),
            "inter_token_median_ms": statistics.median(itl) if itl else None,
            "accepted_tokens_per_stream_step_mean": (
                statistics.mean(accepted_per_step) if accepted_per_step else None
            ),
            "throughput_median_tokens_per_second": statistics.median(
                group["throughput_tokens_per_second"] for group in selected
            ),
            "standard_throughput_median_tokens_per_second": statistics.median(
                group["standard_throughput_tokens_per_second"]
                for group in selected
            ),
            "time_to_first_token_median_ms": statistics.median(ttft) if ttft else None,
            "per_request_output_speed_median_tokens_per_second": (
                statistics.median(native_output_speeds)
                if native_output_speeds
                else None
            ),
            "per_request_output_speed_p10_tokens_per_second": (
                percentile(native_output_speeds, 0.10)
                if native_output_speeds
                else None
            ),
            "per_request_output_speed_p90_tokens_per_second": (
                percentile(native_output_speeds, 0.90)
                if native_output_speeds
                else None
            ),
            "standard_per_request_output_speed_median_tokens_per_second": (
                statistics.median(standard_output_speeds)
                if standard_output_speeds
                else None
            ),
            "fully_active_model_step_median_ms": (
                statistics.median(
                    metrics["model_step_median_ms"] for metrics in full_batch_metrics
                )
                if full_batch_metrics
                else None
            ),
            "fully_active_throughput_median_tokens_per_second": (
                statistics.median(
                    metrics["throughput_tokens_per_second"]
                    for metrics in full_batch_metrics
                )
                if full_batch_metrics
                else None
            ),
            "server_full_batch_observed": len(full_batch_metrics) == len(selected),
            "fully_active_repetitions": len(full_batch_metrics),
            "max_server_batch_size": (
                max(observed_batch_sizes) if observed_batch_sizes else 0
            ),
            "output_multiset_stable": (
                len(set(output_signatures)) == 1 if comparable_outputs else None
            ),
            "slot_output_mapping_stable": (
                len(set(slot_output_signatures)) == 1
                if comparable_outputs
                else None
            ),
            "request_state_slots": sorted(
                {
                    request_state_slot(request)
                    for group in selected
                    for request in group["requests"]
                    if request["chunks"]
                }
            ),
            "recurrent_state_slots": sorted(
                {
                    request_recurrent_state_slot(request)
                    for group in selected
                    for request in group["requests"]
                    if request["chunks"]
                }
            ),
        }
    return by_concurrency


def output_speed_after_first_token(request: dict[str, object]) -> float | None:
    chunks = request["chunks"]
    if len(chunks) < 2:
        return None
    first = chunks[0]
    last = chunks[-1]
    elapsed_ms = float(last["elapsed_ms"]) - float(first["elapsed_ms"])
    last_tokens = int(last["response"]["meta_info"]["completion_tokens"])
    measured_tokens = last_tokens - 1
    if elapsed_ms <= 0 or measured_tokens <= 0:
        return None
    return measured_tokens * 1000 / elapsed_ms


def annotate_standard_token_counts(
    group: dict[str, object], encoding: tiktoken.Encoding
) -> None:
    for request in group["requests"]:
        chunks = request["chunks"]
        if not chunks:
            continue
        request["output_speed_after_first_token"] = output_speed_after_first_token(
            request
        )
        final_text = chunks[-1]["response"].get("text", "")
        standard_tokens = len(encoding.encode(final_text))
        request["standard_completion_tokens"] = standard_tokens
        elapsed_ms = float(chunks[-1]["elapsed_ms"]) - float(
            chunks[0]["elapsed_ms"]
        )
        measured_tokens = standard_tokens - 1
        request["standard_output_speed_after_first_token"] = (
            measured_tokens * 1000 / elapsed_ms
            if elapsed_ms > 0 and measured_tokens > 0
            else None
        )
    group["standard_total_completion_tokens"] = sum(
        int(request.get("standard_completion_tokens", 0))
        for request in group["requests"]
    )
    group["standard_throughput_tokens_per_second"] = (
        group["standard_total_completion_tokens"] * 1000 / float(group["group_ms"])
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    if not 0 <= fraction <= 1:
        raise ValueError(f"Percentile fraction must be in [0, 1], got {fraction}")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fully_active_metrics(group: dict[str, object]) -> dict[str, float]:
    requests_out = [request for request in group["requests"] if request["chunks"]]
    expected_batch_size = int(group["concurrency"])
    events = sorted(
        (
            float(chunk["elapsed_ms"]),
            request_index,
            chunk_index,
            int(chunk["response"]["meta_info"]["server_batch_size"]),
            int(request["completion_token_deltas"][chunk_index]),
        )
        for request_index, request in enumerate(requests_out)
        for chunk_index, chunk in enumerate(request["chunks"])
    )
    observed_batch_sizes = [event[3] for event in events]
    if expected_batch_size not in observed_batch_sizes:
        raise ValueError(
            "Server never processed the requested full batch "
            f"expected={expected_batch_size} observed={sorted(set(observed_batch_sizes))}"
        )

    full_segments: list[list[tuple[float, int, int, int, int]]] = []
    current_segment: list[tuple[float, int, int, int, int]] = []
    for event in events:
        if event[3] == expected_batch_size:
            current_segment.append(event)
        elif current_segment:
            full_segments.append(current_segment)
            current_segment = []
    if current_segment:
        full_segments.append(current_segment)

    eligible_segments = [
        segment
        for segment in full_segments
        if segment[-1][0] > segment[0][0]
        and len({event[1] for event in segment}) == expected_batch_size
    ]
    if not eligible_segments:
        raise ValueError(
            "No contiguous full-server-batch interval contains every request "
            f"expected={expected_batch_size}"
        )
    segment = max(eligible_segments, key=lambda item: item[-1][0] - item[0][0])
    start_ms = segment[0][0]
    end_ms = segment[-1][0]

    token_count = sum(event[4] for event in segment if start_ms < event[0] <= end_ms)
    model_steps = [
        request["inter_token_ms"][index]
        for request in requests_out
        for index in range(len(request["inter_token_ms"]))
        if start_ms <= request["chunks"][index]["elapsed_ms"]
        and request["chunks"][index + 1]["elapsed_ms"] <= end_ms
        and int(request["chunks"][index]["response"]["meta_info"]["server_batch_size"])
        == expected_batch_size
        and int(
            request["chunks"][index + 1]["response"]["meta_info"]["server_batch_size"]
        )
        == expected_batch_size
    ]
    if token_count <= 0 or not model_steps:
        raise ValueError(
            "Full-server-batch interval contains no measurable model steps"
        )
    return {
        "model_step_median_ms": statistics.median(model_steps),
        "throughput_tokens_per_second": token_count * 1000 / (end_ms - start_ms),
        "measurement_start_ms": start_ms,
        "measurement_end_ms": end_ms,
        "measurement_tokens": token_count,
        "server_full_batch_observed": True,
        "max_server_batch_size": max(observed_batch_sizes),
    }


def completion_token_deltas(chunks: list[dict[str, object]]) -> list[int]:
    cumulative = [
        int(chunk["response"]["meta_info"]["completion_tokens"]) for chunk in chunks
    ]
    return [
        current - previous
        for previous, current in zip([0, *cumulative[:-1]], cumulative, strict=True)
    ]


def compact_previous_chunk(chunks: list[dict[str, object]]) -> None:
    if not chunks:
        return
    response = chunks[-1]["response"]
    response.pop("text", None)
    response.pop("output_ids", None)


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


def request_state_slot(request: dict[str, object]) -> int:
    chunks = request["chunks"]
    slots = {chunk["response"]["meta_info"]["request_state_slot"] for chunk in chunks}
    if len(slots) != 1:
        raise ValueError(f"Request changed state slots while decoding: {sorted(slots)}")
    return slots.pop()


def request_recurrent_state_slot(request: dict[str, object]) -> int:
    chunks = request["chunks"]
    slots = {chunk["response"]["meta_info"]["recurrent_state_slot"] for chunk in chunks}
    if len(slots) != 1:
        raise ValueError(f"Request changed recurrent-state slots: {sorted(slots)}")
    slot = slots.pop()
    if slot is None:
        raise ValueError("Request has no recurrent-state slot")
    return slot


def slot_output_signature(group: dict[str, object]) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        sorted(
            (
                request_state_slot(request),
                request_recurrent_state_slot(request),
                hashlib.sha256(
                    request["chunks"][-1]["response"]["text"].encode()
                ).hexdigest(),
            )
            for request in group["requests"]
            if request["chunks"]
        )
    )


def main() -> None:
    args = parse_args()
    if args.prompt_cases is not None and args.input_ids is not None:
        raise ValueError("Use either --prompt-cases or --input-ids, not both")
    prompt_cases = load_prompt_cases(args.prompt_cases) if args.prompt_cases else None
    validate_workload(
        args.workload,
        prompt_cases,
        args.output_tokens,
        args.concurrency,
        args.repetitions,
    )
    standard_encoding = tiktoken.get_encoding(args.standard_tokenizer)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(args, prompt_cases)
    groups = []
    prompt_case_offset = 0
    for concurrency in args.concurrency:
        for repetition in range(args.repetitions):
            selected_cases = (
                select_prompt_cases(prompt_cases, concurrency, prompt_case_offset)
                if prompt_cases is not None
                else None
            )
            prompt_case_offset += concurrency
            group = run_group(
                url=args.url,
                prompt=args.prompt,
                output_tokens=args.output_tokens,
                concurrency=concurrency,
                request_mode=args.request_mode,
                input_ids=args.input_ids,
                ignore_eos=args.ignore_eos,
                return_routed_experts=args.return_routed_experts,
                atomic_admission_delay_seconds=args.atomic_admission_delay_seconds,
                stop_after_first_completion=args.stop_after_first_completion,
                prompt_cases=selected_cases,
                request_timeout_seconds=args.request_timeout_seconds,
            )
            annotate_standard_token_counts(group, standard_encoding)
            group["repetition"] = repetition
            groups.append(group)
            checkpoint = {
                "groups": groups,
                "provenance": provenance,
                "summary": summarize(
                    groups,
                    require_fully_active=args.require_fully_active,
                    comparable_outputs=prompt_cases is None,
                ),
            }
            (args.output_directory / "serving-benchmark.partial.json").write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True)
            )
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
        "provenance": provenance,
        "summary": summarize(
            groups,
            require_fully_active=args.require_fully_active,
            comparable_outputs=prompt_cases is None,
        ),
    }
    (args.output_directory / "serving-benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
