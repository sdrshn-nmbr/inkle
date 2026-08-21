import sys
import threading
from pathlib import Path

import pytest
import requests

from inkling_serving_benchmark import (
    compact_request_body,
    fully_active_metrics,
    load_prompt_cases,
    output_speed_after_first_token,
    parse_args,
    percentile,
    request_recurrent_state_slot,
    request_state_slot,
    slot_output_signature,
    stream_request,
    summarize,
    validate_workload,
    wait_for_server_idle,
)


def test_output_speed_starts_after_first_streamed_token() -> None:
    request = {
        "chunks": [
            {
                "elapsed_ms": 100.0,
                "response": {"meta_info": {"completion_tokens": 2}},
            },
            {
                "elapsed_ms": 300.0,
                "response": {"meta_info": {"completion_tokens": 10}},
            },
        ]
    }

    assert output_speed_after_first_token(request) == pytest.approx(45.0)


def test_output_speed_requires_tokens_after_first_token() -> None:
    request = {
        "chunks": [
            {
                "elapsed_ms": 100.0,
                "response": {"meta_info": {"completion_tokens": 2}},
            }
        ]
    }

    assert output_speed_after_first_token(request) is None


def test_compact_request_body_hashes_input_ids() -> None:
    compact = compact_request_body(
        {
            "rid": "request-id",
            "input_ids": [11, 22, 33],
            "sampling_params": {"temperature": 0},
        }
    )

    assert compact == {
        "rid": "request-id",
        "input_ids_count": 3,
        "input_ids_sha256": (
            "3cf46ba3daf30bf336dbbc80c5f3fd4185bf9cc3747b7ce49d58335723d3a72c"
        ),
        "sampling_params": {"temperature": 0},
    }


def test_load_prompt_cases_accepts_explicit_ids(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '[{"id":"alpha","input_ids":[1,2]},'
        '{"id":"beta","input_ids":[3,4]}]'
    )

    assert load_prompt_cases(path) == [
        {"id": "alpha", "input_ids": [1, 2]},
        {"id": "beta", "input_ids": [3, 4]},
    ]


def test_load_prompt_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '[{"id":"same","input_ids":[1]},'
        '{"id":"same","input_ids":[2]}]'
    )

    with pytest.raises(ValueError, match="Prompt case IDs must be unique"):
        load_prompt_cases(path)


def test_aa_workload_enforces_prompt_and_output_lengths() -> None:
    prompt_cases = [{"id": "prompt", "input_ids": list(range(1_000))}]

    validate_workload("aa-1k", prompt_cases, 1_000, [1], 1)

    with pytest.raises(ValueError, match="at least 1000 output tokens"):
        validate_workload("aa-1k", prompt_cases, 999, [1], 1)


def test_parallel_workload_requires_ten_concurrent_prompts() -> None:
    prompt_cases = [
        {"id": f"prompt-{index}", "input_ids": list(range(1_000))}
        for index in range(11)
    ]

    with pytest.raises(ValueError, match="requires --concurrency 10"):
        validate_workload("aa-parallel-1k", prompt_cases, 1_000, [1, 10], 1)


def test_benchmark_does_not_request_unconfigured_expert_capture_by_default(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["inkling_serving_benchmark.py", "--output-directory", str(tmp_path)],
    )

    assert not parse_args().return_routed_experts


def test_benchmark_accepts_long_streaming_request_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inkling_serving_benchmark.py",
            "--output-directory",
            str(tmp_path),
            "--request-timeout-seconds",
            "3600",
        ],
    )

    assert parse_args().request_timeout_seconds == 3600


def test_concurrent_request_uses_pinned_input_ids(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            return [
                b'data: {"text":"","meta_info":{"completion_tokens":0}}',
                b"data: [DONE]",
            ]

    def fake_post(url, *, json, stream, timeout):
        captured.update(url=url, json=json, stream=stream, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("inkling_serving_benchmark.requests.post", fake_post)
    stream_request(
        "http://server",
        "ignored text",
        [11, 22, 33],
        8,
        threading.Barrier(1),
        True,
        False,
        request_timeout_seconds=3600,
    )

    assert captured["json"]["input_ids"] == [11, 22, 33]
    assert "text" not in captured["json"]
    assert captured["timeout"] == 3600


def test_concurrent_request_compacts_cumulative_intermediate_outputs(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            return [
                b'data: {"text":"one","output_ids":[1],'
                b'"meta_info":{"completion_tokens":1}}',
                b'data: {"text":"one two","output_ids":[1,2],'
                b'"meta_info":{"completion_tokens":2}}',
                b"data: [DONE]",
            ]

    monkeypatch.setattr(
        "inkling_serving_benchmark.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = stream_request(
        "http://server",
        "prompt",
        None,
        2,
        threading.Barrier(1),
        True,
        False,
    )

    assert result["chunks"][0]["response"] == {
        "meta_info": {"completion_tokens": 1}
    }
    assert result["chunks"][1]["response"]["text"] == "one two"
    assert result["chunks"][1]["response"]["output_ids"] == [1, 2]


def test_concurrent_request_aborts_group_after_first_completion(monkeypatch) -> None:
    posts = []

    class FakeResponse:
        def __init__(self, lines=()):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            return self.lines

    def fake_post(url, *, json, stream=False, timeout):
        posts.append((url, json))
        if url.endswith("/generate"):
            return FakeResponse(
                [
                    b'data: {"text":"done","meta_info":{"completion_tokens":8,'
                    b'"finish_reason":{"type":"length"}}}',
                    b"data: [DONE]",
                ]
            )
        return FakeResponse()

    monkeypatch.setattr("inkling_serving_benchmark.requests.post", fake_post)
    stop_event = threading.Event()

    stream_request(
        "http://server",
        "prompt",
        None,
        8,
        threading.Barrier(1),
        True,
        False,
        True,
        stop_event,
    )

    assert stop_event.is_set()
    assert posts[-1] == (
        "http://server/abort_request",
        {"rid": None, "abort_all": True},
    )


def test_concurrent_request_stops_reading_after_peer_finishes(monkeypatch) -> None:
    stop_event = threading.Event()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            stop_event.set()
            yield b'data: {"text":"one","meta_info":{"completion_tokens":1}}'
            yield b'data: {"text":"two","meta_info":{"completion_tokens":2}}'

    monkeypatch.setattr(
        "inkling_serving_benchmark.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = stream_request(
        "http://server",
        "prompt",
        None,
        8,
        threading.Barrier(1),
        True,
        False,
        True,
        stop_event,
    )

    assert len(result["chunks"]) == 1


def test_wait_for_server_idle_waits_for_scheduler_and_pool_cleanup(monkeypatch) -> None:
    states = iter(
        [
            {
                "internal_states": [
                    {
                        "running_batch_size": 2,
                        "waiting_queue_size": 0,
                        "req_to_token_pool_used": 2,
                    }
                ]
            },
            {
                "internal_states": [
                    {
                        "running_batch_size": 0,
                        "waiting_queue_size": 0,
                        "req_to_token_pool_used": 0,
                    }
                ]
            },
        ]
    )

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr(
        "inkling_serving_benchmark.requests.get",
        lambda url, timeout: FakeResponse(next(states)),
    )
    monkeypatch.setattr("inkling_serving_benchmark.time.sleep", lambda seconds: None)

    wait_for_server_idle("http://server", poll_seconds=0.0)


def test_wait_for_server_idle_retries_transient_poll_failure(monkeypatch) -> None:
    responses = iter(
        [
            requests.Timeout("temporary"),
            {
                "internal_states": [
                    {
                        "running_batch_size": 0,
                        "waiting_queue_size": 0,
                        "req_to_token_pool_used": 0,
                    }
                ]
            },
        ]
    )

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_get(url, timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)

    monkeypatch.setattr("inkling_serving_benchmark.requests.get", fake_get)
    monkeypatch.setattr("inkling_serving_benchmark.time.sleep", lambda seconds: None)

    wait_for_server_idle("http://server", poll_seconds=0.0)


def make_request(slot: int, text: str) -> dict[str, object]:
    return {
        "chunks": [
            {
                "response": {
                    "text": text,
                    "meta_info": {
                        "request_state_slot": slot,
                        "recurrent_state_slot": slot + 100,
                        "server_batch_size": 1,
                    },
                }
            }
        ]
    }


def test_request_state_slot_is_stable_across_chunks() -> None:
    request = make_request(7, "first")
    request["chunks"].append(
        {
            "response": {
                "text": "second",
                "meta_info": {
                    "request_state_slot": 7,
                    "recurrent_state_slot": 107,
                },
            }
        }
    )

    assert request_state_slot(request) == 7


def test_request_state_slot_rejects_slot_changes() -> None:
    request = make_request(7, "first")
    request["chunks"].append(
        {
            "response": {
                "text": "second",
                "meta_info": {
                    "request_state_slot": 8,
                    "recurrent_state_slot": 107,
                },
            }
        }
    )

    with pytest.raises(ValueError, match="changed state slots"):
        request_state_slot(request)


def test_recurrent_state_slot_is_reported_separately() -> None:
    assert request_recurrent_state_slot(make_request(7, "first")) == 107


def test_slot_output_signature_follows_internal_slot_not_submission_order() -> None:
    first = make_request(4, "alpha")
    second = make_request(2, "beta")

    assert slot_output_signature(
        {"requests": [first, second]}
    ) == slot_output_signature({"requests": [second, first]})


def test_summary_rejects_group_without_fully_active_interval() -> None:
    requests = []
    for slot, elapsed_ms in enumerate((10.0, 20.0)):
        request = make_request(slot, f"output-{slot}")
        request.update(
            e2e_ms=elapsed_ms,
            time_to_first_token_ms=elapsed_ms,
            inter_token_ms=[],
            completion_token_deltas=[1],
        )
        request["chunks"][0]["elapsed_ms"] = elapsed_ms
        requests.append(request)
    group = {
        "concurrency": 2,
        "requests": requests,
        "throughput_tokens_per_second": 100.0,
    }

    with pytest.raises(ValueError, match="concurrency 2 never became fully active"):
        summarize([group])


def test_summary_can_report_provider_latency_without_fully_active_interval() -> None:
    requests = []
    for slot, elapsed_ms in enumerate((10.0, 20.0)):
        request = make_request(slot, f"output-{slot}")
        request.update(
            e2e_ms=elapsed_ms,
            time_to_first_token_ms=elapsed_ms,
            inter_token_ms=[],
            completion_token_deltas=[1],
            standard_completion_tokens=1,
            standard_output_speed_after_first_token=None,
        )
        request["chunks"][0]["elapsed_ms"] = elapsed_ms
        request["chunks"][0]["response"]["meta_info"]["server_batch_size"] = 1
        requests.append(request)
    group = {
        "concurrency": 2,
        "requests": requests,
        "throughput_tokens_per_second": 100.0,
        "standard_throughput_tokens_per_second": 100.0,
    }

    result = summarize([group], require_fully_active=False)["2"]

    assert result["server_full_batch_observed"] is False
    assert result["fully_active_repetitions"] == 0
    assert result["max_server_batch_size"] == 1
    assert result["fully_active_model_step_median_ms"] is None
    assert result["fully_active_throughput_median_tokens_per_second"] is None


def test_summary_omits_output_stability_for_distinct_prompt_cases() -> None:
    request = make_request(0, "unique-output")
    request.update(
        e2e_ms=20.0,
        time_to_first_token_ms=10.0,
        inter_token_ms=[],
        completion_token_deltas=[1],
        standard_completion_tokens=1,
        standard_output_speed_after_first_token=None,
    )
    request["chunks"][0]["elapsed_ms"] = 10.0
    group = {
        "concurrency": 1,
        "requests": [request],
        "throughput_tokens_per_second": 50.0,
        "standard_throughput_tokens_per_second": 50.0,
    }

    result = summarize(
        [group], require_fully_active=False, comparable_outputs=False
    )["1"]

    assert result["output_multiset_stable"] is None
    assert result["slot_output_mapping_stable"] is None


def test_summary_rejects_client_overlap_without_full_server_batch() -> None:
    requests = []
    for slot, times in enumerate(((10.0, 30.0), (20.0, 40.0))):
        request = make_request(slot, f"output-{slot}")
        request["chunks"].append(
            {
                "elapsed_ms": times[1],
                "response": {
                    "text": f"output-{slot}-continued",
                    "meta_info": {
                        "request_state_slot": slot,
                        "recurrent_state_slot": slot + 100,
                        "server_batch_size": 1,
                    },
                },
            }
        )
        request["chunks"][0]["elapsed_ms"] = times[0]
        request.update(
            e2e_ms=times[1],
            time_to_first_token_ms=times[0],
            inter_token_ms=[times[1] - times[0]],
            completion_token_deltas=[1, 1],
        )
        requests.append(request)

    with pytest.raises(ValueError, match="concurrency 2 never became fully active"):
        summarize(
            [
                {
                    "concurrency": 2,
                    "requests": requests,
                    "throughput_tokens_per_second": 100.0,
                }
            ]
        )


def test_fully_active_metrics_uses_only_contiguous_full_server_batch_window() -> None:
    requests = []
    event_rows = (
        ((10.0, 1, 1), (20.0, 2, 2), (30.0, 2, 3), (40.0, 1, 4)),
        ((12.0, 1, 1), (22.0, 2, 2), (32.0, 2, 4), (42.0, 1, 5)),
    )
    for slot, rows in enumerate(event_rows):
        chunks = [
            {
                "elapsed_ms": elapsed_ms,
                "response": {
                    "text": f"request-{slot}-{completion_tokens}",
                    "meta_info": {
                        "request_state_slot": slot,
                        "recurrent_state_slot": slot + 100,
                        "server_batch_size": server_batch_size,
                        "completion_tokens": completion_tokens,
                    },
                },
            }
            for elapsed_ms, server_batch_size, completion_tokens in rows
        ]
        requests.append(
            {
                "chunks": chunks,
                "completion_token_deltas": [1, 1, 1, 1 if slot == 0 else 0],
                "inter_token_ms": [10.0, 10.0, 10.0],
                "e2e_ms": rows[-1][0],
                "time_to_first_token_ms": rows[0][0],
            }
        )

    metrics = fully_active_metrics({"concurrency": 2, "requests": requests})

    assert metrics["measurement_start_ms"] == 20.0
    assert metrics["measurement_end_ms"] == 32.0
    assert metrics["measurement_tokens"] == 3
    assert metrics["throughput_tokens_per_second"] == pytest.approx(250.0)


def test_fully_active_metrics_rejects_disconnected_full_batch_events() -> None:
    requests = []
    for slot, rows in enumerate(
        (
            ((10.0, 2, 1), (30.0, 1, 2)),
            ((20.0, 1, 1), (40.0, 2, 2)),
        )
    ):
        requests.append(
            {
                "chunks": [
                    {
                        "elapsed_ms": elapsed_ms,
                        "response": {
                            "text": "x",
                            "meta_info": {
                                "request_state_slot": slot,
                                "recurrent_state_slot": slot + 100,
                                "server_batch_size": server_batch_size,
                                "completion_tokens": completion_tokens,
                            },
                        },
                    }
                    for elapsed_ms, server_batch_size, completion_tokens in rows
                ],
                "completion_token_deltas": [1, 1],
                "inter_token_ms": [20.0],
                "e2e_ms": rows[-1][0],
                "time_to_first_token_ms": rows[0][0],
            }
        )

    with pytest.raises(ValueError, match="No contiguous full-server-batch interval"):
        fully_active_metrics({"concurrency": 2, "requests": requests})


def test_percentile_is_not_below_median_for_two_samples() -> None:
    values = [8.2, 37.2]

    assert percentile(values, 0.95) >= 22.7
    assert percentile(values, 0.95) <= 37.2
