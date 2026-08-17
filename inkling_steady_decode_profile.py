import argparse
import contextlib
import json
import time
import uuid
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
    parser.add_argument("--prompt", default="Write an essay on vmap in JAX.")
    return parser.parse_args()


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

    request_body = {
        "rid": [uuid.uuid4().hex for _ in range(args.concurrency)],
        "text": [args.prompt] * args.concurrency,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_routed_experts": False,
    }
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
            timeout=600,
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
                            "host_tracer_level": 0,
                            "python_tracer_level": 0,
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

    return {
        "request": request_body,
        "profile_start_response": profile_start_response,
        "profile_stop_response": profile_stop_response,
        "profile_start_condition": "every request emitted at least one token",
        "admissions_during_capture": 0,
        "captured_steps_after_first_token": args.captured_steps,
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
