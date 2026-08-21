# Inkle

This repository is our working implementation and benchmark suite for running
Inkling-Small on TPU7x with SGLang-JAX and DSpark speculative decoding.

## What works

- Inkling-Small target inference on eight TPU7x devices.
- A six-layer DSpark draft model that proposes seven tokens at a time. The target
  checks the current token plus those seven proposals.
- Sharded target verification, batched context reads and writes, and exact
  request, cache, recurrent-state, and slot checks.
- Strict steady-decode benchmarks that exclude compilation, prompt processing,
  admissions, and partially filled batches.
- Provider-shaped prompt sets with exact token IDs, native and standardized token
  counts, and full source, model, request, and runtime receipts.
- Raw TPU timing traces with separate counter captures.

## Measured baseline

These numbers use native Inkling tokens. They measure different workloads and
should not be compared as if they were the same test.

| Workload | Result | Notes |
|---|---:|---|
| 1K prompt, one request | 37.66 tokens/s | 34.69s to first token; 66.56ms decode step |
| 1K prompts, ten active requests | 479.12 total tokens/s | 47.9 tokens/s per active request; 77.80ms decode step |
| Best accepted historical 48-request decode | about 1,422 total tokens/s | Pinned revisions on one four-chip TPU7x host; acceptance was about 2.85 tokens per round |
| Repetitive late 48-request decode | about 2,952 total tokens/s | Ceiling probe only; acceptance rose to about 6.11 |

We did not call a hosted provider while building this baseline. Published
provider numbers are useful reference points, but they do not disclose replica
count, hardware allocation, or sustained capacity at matching concurrency.

## What the traces showed

The TPU does more model work as concurrency increases, but the fixed time between
operations barely changes:

| Concurrency | Round | TPU active | Waiting between work | Launches per round |
|---:|---:|---:|---:|---:|
| 1 | 66.87ms | 17.36ms | 49.47ms | 69 |
| 10 | 76.63ms | 29.00ms | 47.06ms | 66 |
| 48, late ceiling trace | 102.48ms | 49.58ms | 52.90ms | about 66 |

The 1,422 result predates the current `main`; it is the best accepted matched
historical run, not a fresh performance claim for the current commit.

This is the main result. Single-request performance is dominated by fixed
orchestration and many small operations. At high concurrency, target model work
and draft acceptance matter as well.

The existing hardware-counter captures contain one cumulative sample per
counter. They cannot support trustworthy utilization or bandwidth rates.

## Ideas we tested and rejected

- **Packed NVFP4 with software decoding:** much slower than BF16.
- **The available native float4 grouped matmul:** about 2.24 times slower end to
  end than a fair BF16 comparison. It saved memory but not time.
- **Moving proposal metadata to the TPU:** removed transfers but added reshards
  and launches. It did not shorten the round enough.
- **Broad device-side finalization:** exact, but roughly 19% slower.
- **A fused post-target tail:** saved 2.15ms in isolation, only about 2.3% of a
  complete round.
- **Combining all six context writes and reads:** exact, but 4.90% slower than
  the two existing operations.
- **Different BF16 grouped-matmul tiles:** the best c48 tile was 2.25% faster
  inside that kernel, worth less than 0.5% of a complete round.
- **Two c24 speculative lanes:** even a perfect two-lane estimate lost to the
  existing c48 path.
- **Context window 512:** slower than the current 256-token DSpark context.
- **Proposal widths below eight:** K8 remained best.

## What would be worth doing next

There are two credible performance projects left:

1. Train or obtain a stronger Inkling draft model. The current checkpoint is
   built for seven proposals, so K16 or tree verification is not a command-line
   setting. It needs a new checkpoint and a new verification contract.
2. Redesign DSpark so independent requests overlap safely. This requires clear
   ownership of cache, recurrent, convolution, context, request-slot, and
   streaming state. Another small helper operation will not be enough.

Long-context prompt processing, admission behavior, API reliability, and KV
movement are important serving work, but they are separate from steady-decode
throughput.
