# Memory Prefetcher Evaluation

Reproducible ChampSim campaigns and workload-level analysis for cache prefetchers.

## Contents

- `experiments/` contains the campaign runner, result normalization, access-stream analysis, miss attribution, report generators, and the feedback-directed IP-stride module.
- `tests/` contains unit tests for campaign configuration and result handling.
- ChampSim DPC4 and the Berti artifact are pinned as Git submodules.

## Setup

```sh
git clone --recurse-submodules https://github.com/durgesh-k-sharma/memory-prefetcher-evaluation.git
cd memory-prefetcher-evaluation
python3 -m pip install -r requirements.txt
```

Running a campaign requires a ChampSim-supported C++ build toolchain and an exact trace file. Trace files and raw simulation outputs are not included; use trace data available to you under its applicable terms. The runner requires the trace path explicitly and records its size and SHA-256 in the result.

For example:

```sh
python3 experiments/run_campaign.py \
  --workload 603.bwaves_s-891B \
  --trace /path/to/603.bwaves_s-891B.champsimtrace.xz \
  --prefetcher ip_stride \
  --output results/raw/bwaves-ip-stride.json
```

Campaign options, supported workloads, telemetry, and normalization/report workflows are documented in [`experiments/README.md`](experiments/README.md).
