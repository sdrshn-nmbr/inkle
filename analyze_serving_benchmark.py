import argparse
import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    output_sequences = Counter()
    expert_counts = Counter()
    sequences_by_concurrency: dict[int, Counter] = {}
    request_count = 0
    for group in benchmark["groups"]:
        group_sequences = sequences_by_concurrency.setdefault(
            group["concurrency"], Counter()
        )
        for request in group["requests"]:
            if not request["chunks"]:
                continue
            request_count += 1
            final = request["chunks"][-1]["response"]
            completion_tokens = final["meta_info"]["completion_tokens"]
            output_ids = tuple(final["output_ids"][:completion_tokens])
            output_sequences[output_ids] += 1
            group_sequences[output_ids] += 1
            encoded_routes = final["meta_info"].get("routed_experts")
            if encoded_routes:
                routes = np.frombuffer(base64.b64decode(encoded_routes), dtype=np.int32)
                expert_counts.update(int(route) for route in routes if route >= 0)

    result = {
        "expert_selection_count": sum(expert_counts.values()),
        "most_selected_experts": expert_counts.most_common(20),
        "request_count": request_count,
        "unique_outputs_by_concurrency": {
            str(concurrency): {
                "request_count": sum(sequences.values()),
                "unique_output_sequences": len(sequences),
                "most_common_frequency": sequences.most_common(1)[0][1],
            }
            for concurrency, sequences in sorted(sequences_by_concurrency.items())
        },
        "unique_output_sequences": len(output_sequences),
        "output_sequence_frequencies": [
            {"count": count, "output_ids": list(tokens)}
            for tokens, count in output_sequences.most_common()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
