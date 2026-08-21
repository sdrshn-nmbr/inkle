from types import SimpleNamespace
from unittest import mock

import pytest

from inkling_steady_decode_profile import (
    load_profile_prompt_cases,
    parse_args,
    post_profile,
    run_profile,
)


def test_profile_requires_tokens_after_capture_window() -> None:
    args = SimpleNamespace(captured_steps=64, output_tokens=64)

    with pytest.raises(ValueError, match="smaller than output tokens"):
        run_profile(args)


def test_profile_endpoint_accepts_plain_text_responses() -> None:
    response = mock.Mock(
        content=b"Profiling started",
        headers={"content-type": "text/plain; charset=utf-8"},
        status_code=200,
        text="Profiling started",
    )

    with mock.patch("inkling_steady_decode_profile.requests.post", return_value=response):
        result = post_profile("http://server", "start_profile", {})

    assert result == {"status_code": 200, "body": "Profiling started"}
    response.raise_for_status.assert_called_once_with()


def test_profile_tracer_levels_default_to_device_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "inkling_steady_decode_profile.py",
            "--output-directory",
            "/tmp/profile-output",
            "--profile-directory",
            "/tmp/profile",
        ],
    )

    args = parse_args()

    assert args.host_tracer_level == 0
    assert args.python_tracer_level == 0


def test_profile_prompt_cases_are_exact_and_unique(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '[{"id":"first","input_ids":[1,2]},'
        '{"id":"second","input_ids":[3,4]}]'
    )

    assert load_profile_prompt_cases(path, 2) == [
        {"id": "first", "input_ids": [1, 2]},
        {"id": "second", "input_ids": [3, 4]},
    ]


def test_profile_prompt_cases_reject_duplicate_selected_ids(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '[{"id":"same","input_ids":[1]},'
        '{"id":"same","input_ids":[2]}]'
    )

    with pytest.raises(ValueError, match="PROFILE_PROMPT_CASE_IDS_NOT_UNIQUE"):
        load_profile_prompt_cases(path, 2)
