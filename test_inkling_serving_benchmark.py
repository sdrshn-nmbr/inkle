import pytest

from inkling_serving_benchmark import (
    request_recurrent_state_slot,
    request_state_slot,
    slot_output_signature,
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
