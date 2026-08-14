import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures = []
    verified_bytes = 0
    for entry in manifest["files"]:
        path = args.snapshot / entry["path"]
        expected_hash = (entry["lfs"] or {}).get("sha256", entry["blob_id"])
        if not path.exists():
            failures.append({"path": entry["path"], "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = path.resolve().name
        if actual_size != entry["size"] or actual_hash != expected_hash:
            failures.append(
                {
                    "actual_hash": actual_hash,
                    "actual_size": actual_size,
                    "expected_hash": expected_hash,
                    "expected_size": entry["size"],
                    "path": entry["path"],
                    "reason": "metadata_mismatch",
                }
            )
            continue
        verified_bytes += actual_size

    result = {
        "failure_count": len(failures),
        "failures": failures,
        "file_count": len(manifest["files"]),
        "revision": manifest["revision"],
        "verified_bytes": verified_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
