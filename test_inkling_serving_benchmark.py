import pytest

from inkling_serving_benchmark import (
    request_recurrent_state_slot,
    request_state_slot,
    slot_output_signature,
    summarize,
)


def make_request(slot: int, text: str) -> dict[str, object]:
    return {
        "chunks": [
            {
                "response": {
                    "text": text,
                    "meta_info": {
                        "request_state_slot": slot,
                        "recurrent_state_slot": slot + 100,
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

    assert slot_output_signature({"requests": [first, second]}) == slot_output_signature(
        {"requests": [second, first]}
    )


def test_summary_keeps_group_without_fully_active_interval() -> None:
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

    result = summarize([group])["2"]

    assert result["fully_active_model_step_median_ms"] is None
    assert result["fully_active_throughput_median_tokens_per_second"] is None
