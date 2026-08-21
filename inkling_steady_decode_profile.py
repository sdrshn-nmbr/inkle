import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profile-directory", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--captured-steps", type=int, default=64)
    parser.add_argument("--host-tracer-level", type=int, default=0)
    parser.add_argument("--python-tracer-level", type=int, default=0)
    parser.add_argument("--prompt", default="Write an essay on vmap in JAX.")
    parser.add_argument(
        "--prompt-cases",
        type=Path,
        help="JSON prompt corpus with stable IDs and exact input token IDs.",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_profile_prompt_cases(
    path: Path, concurrency: int
) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) < concurrency:
        raise ValueError(
            "PROFILE_PROMPT_CORPUS_TOO_SMALL "
            f"required={concurrency} available={len(payload) if isinstance(payload, list) else 0}"
        )
    selected = payload[:concurrency]
    for index, case in enumerate(selected):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not isinstance(case.get("input_ids"), list)
            or not case["input_ids"]
            or any(not isinstance(token_id, int) for token_id in case["input_ids"])
        ):
            raise ValueError(f"PROFILE_PROMPT_CASE_INVALID index={index}")
    if len({case["id"] for case in selected}) != concurrency:
        raise ValueError("PROFILE_PROMPT_CASE_IDS_NOT_UNIQUE")
    return selected


def run_text(command: list[str]) -> str | None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def post_profile(url: str, endpoint: str, body: dict | None = None) -> dict:
    response = requests.post(f"{url}/{endpoint}", json=body, timeout=300)
    response.raise_for_status()
    if not response.content:
        return {"status_code": response.status_code}
    if response.headers.get("content-type", "").startswith("application/json"):
        return {"status_code": response.status_code, "body": response.json()}
    return {"status_code": response.status_code, "body": response.text}


def run_profile(args: argparse.Namespace) -> dict[str, object]:
    if args.captured_steps >= args.output_tokens:
        raise ValueError("captured steps must be smaller than output tokens")

    prompt_cases = (
        load_profile_prompt_cases(args.prompt_cases, args.concurrency)
        if args.prompt_cases is not None
        else None
    )
    request_body = {
        "rid": [uuid.uuid4().hex for _ in range(args.concurrency)],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_routed_experts": False,
    }
    if prompt_cases is None:
        request_body["text"] = [args.prompt] * args.concurrency
    else:
        request_body["input_ids"] = [case["input_ids"] for case in prompt_cases]
    first_seen = set()
    chunks_by_request = [[] for _ in range(args.concurrency)]
    profile_started = False
    profile_stopped = False
    profile_start_response = None
    profile_stop_response = None
    start_ns = time.perf_counter_ns()

    try:
        with requests.post(
            f"{args.url}/generate",
            json=request_body,
            stream=True,
            timeout=3600,
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
                index = output.pop("index")
                chunks_by_request[index].append(
                    {"elapsed_ms": elapsed_ms, "response": output}
                )
                first_seen.add(index)

                if not profile_started and len(first_seen) == args.concurrency:
                    profile_start_response = post_profile(
                        args.url,
                        "start_profile",
                        {
                            "output_dir": args.profile_directory,
                            "host_tracer_level": args.host_tracer_level,
                            "python_tracer_level": args.python_tracer_level,
                        },
                    )
                    profile_started = True

                if (
                    profile_started
                    and not profile_stopped
                    and min(map(len, chunks_by_request)) >= args.captured_steps + 1
                ):
                    profile_stop_response = post_profile(args.url, "stop_profile")
                    profile_stopped = True
    finally:
        if profile_started and not profile_stopped:
            with contextlib.suppress(requests.RequestException):
                post_profile(args.url, "stop_profile")

    if not profile_started:
        raise RuntimeError("PROFILE_NEVER_STARTED not every request emitted a token")
    if not profile_stopped:
        post_profile(args.url, "stop_profile")
        raise RuntimeError("PROFILE_NEVER_STOPPED request ended before capture completed")

    source_path = Path(__file__).resolve()
    server_info_response = requests.get(f"{args.url}/get_server_info", timeout=30)
    server_info_response.raise_for_status()
    request_receipt = {
        key: value for key, value in request_body.items() if key != "input_ids"
    }
    if prompt_cases is not None:
        request_receipt["prompt_cases"] = [
            {
                "id": case["id"],
                "native_input_tokens": len(case["input_ids"]),
                "input_ids_sha256": sha256_bytes(
                    json.dumps(case["input_ids"], separators=(",", ":")).encode()
                ),
                "standard_input_tokens": case.get("standard_input_tokens"),
                "decoded_prompt_sha256": case.get("decoded_text_sha256"),
            }
            for case in prompt_cases
        ]
    return {
        "request": request_receipt,
        "provenance": {
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "benchmark_source": str(source_path),
            "benchmark_source_sha256": sha256_bytes(source_path.read_bytes()),
            "git_commit": run_text(["git", "rev-parse", "HEAD"]),
            "git_status_porcelain": run_text(["git", "status", "--short"]),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "server_info": server_info_response.json(),
            "prompt_cases_path": (
                str(args.prompt_cases.resolve()) if args.prompt_cases else None
            ),
            "prompt_cases_file_sha256": (
                sha256_bytes(args.prompt_cases.read_bytes())
                if args.prompt_cases
                else None
            ),
        },
        "profile_start_response": profile_start_response,
        "profile_stop_response": profile_stop_response,
        "profile_start_condition": "every request emitted at least one token",
        "admissions_during_capture": 0,
        "captured_steps_after_first_token": args.captured_steps,
        "host_tracer_level": args.host_tracer_level,
        "python_tracer_level": args.python_tracer_level,
        "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
        "chunks_by_request": chunks_by_request,
    }


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    result = run_profile(args)
    output = args.output_directory / "profile-request.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(result["chunks_by_request"]),
                "captured_steps": result["captured_steps_after_first_token"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
