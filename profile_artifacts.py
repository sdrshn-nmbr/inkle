import argparse
import json
from pathlib import Path

from xprof.convert import raw_to_tool_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-device-events", action="store_true")
    parser.add_argument("--require-compiled-program", action="store_true")
    return parser.parse_args()


def inspect_profile(profile_directory: Path) -> dict[str, object]:
    profile_files = sorted(profile_directory.rglob("*.xplane.pb"))
    files = []
    device_event_count = 0
    compiled_event_count = 0
    compiled_terms = ("xla", "hlo", "fusion", "custom", "dot", "matmul", "module")

    for profile_file in profile_files:
        profile_paths = [str(profile_file)]
        tool_names = raw_to_tool_data.xspace_to_tool_names(profile_paths)
        trace_data, _ = raw_to_tool_data.xspace_to_tool_data(
            profile_paths,
            "trace_viewer",
            {"use_saved_result": False},
        )
        trace = json.loads(trace_data)
        process_names = {
            event["pid"]: event.get("args", {}).get("name", "")
            for event in trace.get("traceEvents", [])
            if event.get("ph") == "M" and event.get("name") == "process_name"
        }
        device_pids = {
            pid
            for pid, name in process_names.items()
            if "/device:" in name.lower()
            and ("tpu" in name.lower() or "megascale" in name.lower())
        }
        events = [
            event
            for event in trace.get("traceEvents", [])
            if event.get("pid") in device_pids and event.get("ph") != "M"
        ]
        device_event_count += len(events)
        compiled_event_count += sum(
            any(term in event.get("name", "").lower() for term in compiled_terms)
            for event in events
        )
        files.append(
            {
                "device_event_count": len(events),
                "device_processes": {
                    str(pid): process_names[pid] for pid in sorted(device_pids)
                },
                "file": str(profile_file),
                "tool_names": tool_names,
            }
        )

    return {
        "compiled_event_count": compiled_event_count,
        "device_event_count": device_event_count,
        "files": files,
        "profile_file_count": len(profile_files),
    }


def main() -> None:
    args = parse_args()
    result = inspect_profile(args.profile_directory)
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)
    if args.require_device_events and result["device_event_count"] == 0:
        raise SystemExit("TPU_PROFILE_REJECTED reason=no_device_events")
    if args.require_compiled_program and result["compiled_event_count"] == 0:
        raise SystemExit("TPU_PROFILE_REJECTED reason=no_compiled_program_events")


if __name__ == "__main__":
    main()
