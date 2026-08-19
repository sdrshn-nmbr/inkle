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
    return parser.parse_args()


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
        timeout=900,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            output = json.loads(payload)
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
        "request": request_body,
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
        response = requests.get(f"{url}/get_server_info", timeout=30)
        response.raise_for_status()
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
) -> dict[str, object]:
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
                    url,
                    prompt,
                    output_tokens,
                    concurrency,
                    request_mode,
                    input_ids,
                    ignore_eos,
                    return_routed_experts,
                    0.0,
                    stop_after_first_completion,
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
            raise ValueError("First-completion stopping requires concurrent request mode")
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
    stop_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_request,
                url,
                prompt,
                input_ids,
                output_tokens,
                start_barrier,
                ignore_eos,
                return_routed_experts,
                stop_after_first_completion,
                stop_event,
            )
            for _ in range(concurrency)
        ]
        start_barrier.wait()
        group_start_ns = time.perf_counter_ns()
        requests_out = [future.result() for future in futures]
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
        timeout=900,
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
                "completion_token_deltas": completion_token_deltas(chunks),
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
        full_batch_metrics = []
        for group in selected:
            try:
                full_batch_metrics.append(fully_active_metrics(group))
            except ValueError as error:
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
        output_signatures = [output_multiset_signature(group) for group in selected]
        slot_output_signatures = [slot_output_signature(group) for group in selected]
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
            "time_to_first_token_median_ms": statistics.median(ttft) if ttft else None,
            "fully_active_model_step_median_ms": (
                statistics.median(
                    metrics["model_step_median_ms"] for metrics in full_batch_metrics
                )
                if full_batch_metrics
                else None
            ),
            "fully_active_throughput_median_tokens_per_second": (
                statistics.median(
                    metrics["throughput_tokens_per_second"] for metrics in full_batch_metrics
                )
                if full_batch_metrics
                else None
            ),
            "server_full_batch_observed": all(
                metrics["server_full_batch_observed"] for metrics in full_batch_metrics
            ),
            "max_server_batch_size": max(
                metrics["max_server_batch_size"] for metrics in full_batch_metrics
            ),
            "output_multiset_stable": len(set(output_signatures)) == 1,
            "slot_output_mapping_stable": len(set(slot_output_signatures)) == 1,
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

    token_count = sum(
        event[4] for event in segment if start_ms < event[0] <= end_ms
    )
    model_steps = [
        request["inter_token_ms"][index]
        for request in requests_out
        for index in range(len(request["inter_token_ms"]))
        if start_ms <= request["chunks"][index]["elapsed_ms"]
        and request["chunks"][index + 1]["elapsed_ms"] <= end_ms
        and int(
            request["chunks"][index]["response"]["meta_info"]["server_batch_size"]
        )
        == expected_batch_size
        and int(
            request["chunks"][index + 1]["response"]["meta_info"][
                "server_batch_size"
            ]
        )
        == expected_batch_size
    ]
    if token_count <= 0 or not model_steps:
        raise ValueError("Full-server-batch interval contains no measurable model steps")
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
        int(chunk["response"]["meta_info"]["completion_tokens"])
        for chunk in chunks
    ]
    return [
        current - previous
        for previous, current in zip([0, *cumulative[:-1]], cumulative, strict=True)
    ]


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
    slots = {
        chunk["response"]["meta_info"]["request_state_slot"] for chunk in chunks
    }
    if len(slots) != 1:
        raise ValueError(f"Request changed state slots while decoding: {sorted(slots)}")
    return slots.pop()


def request_recurrent_state_slot(request: dict[str, object]) -> int:
    chunks = request["chunks"]
    slots = {
        chunk["response"]["meta_info"]["recurrent_state_slot"] for chunk in chunks
    }
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
                args.atomic_admission_delay_seconds,
                args.stop_after_first_completion,
            )
            group["repetition"] = repetition
            groups.append(group)
            checkpoint = {
                "groups": groups,
                "summary": summarize(groups),
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
        "summary": summarize(groups),
    }
    (args.output_directory / "serving-benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
