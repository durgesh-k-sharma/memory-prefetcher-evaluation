# Reproducible ChampSim campaigns

`run_campaign.py` runs one named workload against one cache-level prefetcher configuration. Pass the exact trace path explicitly; the runner does not download or guess trace files.

```sh
python3 experiments/run_campaign.py \
  --workload 603.bwaves_s-891B \
  --trace /path/to/603.bwaves_s-891B.champsimtrace.xz \
  --prefetcher ip_stride \
  --output results/603.bwaves_s-891B-ip-stride.json
```

Supported workload IDs are the six SPEC traces (`603.bwaves_s-891B`, `619.lbm_s-4268B`, `649.fotonik3d_s-1176B`, `605.mcf_s-1554B`, `620.omnetpp_s-141B`, `623.xalancbmk_s-165B`), `ligra-bfs-cc`, `ligra-pagerankdelta`, `ligra-triangle`, and `llama2`.

The runner uses the tracked `1C.fullBW.baseline` single-core profile. That profile has every prefetcher set to `no`. Each generated configuration keeps only the requested prefetcher active: IP-stride, its `ip_stride_fdp` feedback-directed variant, SPP, AMPM, or Pythia at L2C, and Berti at L1D. No-prefetch runs require an explicit level so the baseline is paired with the intended scenario:

```sh
python3 experiments/run_campaign.py \
  --workload ligra-bfs-cc \
  --trace /path/to/ligra_BFSCC.champsimtrace.xz \
  --prefetcher none --level L1D \
  --output results/ligra-bfs-cc-berti-no-pref.json
```

The runner uses a config-specific ChampSim build by default. It copies the ChampSim source tree under the output directory's `.campaign-builds` folder, configures that copy, and builds the selected module set there. ChampSim's dependencies must already be available to its normal build. To run a binary that was built from the emitted configuration, pass `--simulator`; the effective configuration is saved next to the result as `<output-stem>.config.json`.

Each successful run writes the ChampSim JSON ROI output, stdout/stderr logs, a normalized result JSON, an L2 load sample CSV, and a telemetry JSON sidecar. The result records the exact trace path, byte size, and SHA-256; the source profile and effective configuration SHA-256; simulator binary identity; workload and prefetcher cache level; warmup and measured instruction bounds; raw ChampSim ROI statistics; and normalized instruction count, cycles, IPC, cache statistics, and DRAM traffic. ChampSim may retire up to `retire_width - 1` instructions beyond the requested ROI boundary; the result records the actual retired instruction count alongside the requested 100 million.

The L2 sample contains at most 10,000 demand `LOAD` tag checks selected with deterministic reservoir sampling across the measured region. Each CSV row has ROI access order, load PC, a physical cache-line address aligned to the configured block size, and hit/miss status. The sidecar records the sample cap, total L2 demand loads, retained row count, and the same run identity and ROI as the normalized result.

DRAM read/write transactions are counted when their data-bus transfer starts; transferred bytes are the configured channel width times the DRAM burst length. ChampSim retains the request type that reaches the DRAM controller, so the sidecar reports per-type transaction and byte counts and groups `LOAD`/`RFO` as demand reads and `PREFETCH` as prefetch reads. These labels describe the retained request type at DRAM service; they do not reconstruct original traffic sources when a demand and prefetch merged earlier in the hierarchy. Other request types remain separately visible in the per-type counters.

The `ip_stride_fdp` module keeps the original per-PC stride table, page-boundary checks, and MSHR-aware request behavior. It changes only the lookahead degree, clamped to 1–3. Accepted prefetches are tracked by cache-line address and tagged with the latest degree submitting that still-pending address; duplicate submissions to one pending address count once. A demand use, eviction, or demand fill resolves that prefetch address episode. After 4,096 resolved prefetch address episodes for one degree, the controller reduces degree below 30% usefulness, increases it at or above 45% when that degree is still active, and otherwise holds. Pending addresses at end of run are excluded from the usefulness fraction. The conceptual controller adds a degree register and per-degree outcome counters. ChampSim uses a pending-address map to associate delayed events with the issuing degree; that map is feedback instrumentation, not an area estimate. These module counters span warmup and ROI, while normalized performance and traffic metrics cover the ROI.

## Normalize and generate reports

After each campaign finishes, normalize its ten-workload results. `l2-required` includes the required L2 configurations; Pythia and Berti remain separate scenarios because they use their own campaign directories and Berti observes at L1D.

```sh
python3 experiments/normalize_campaign.py --scenario l2-required \
  --candidate-dir results/raw/ticket03 --output-dir results/normalized/l2-required
python3 experiments/normalize_campaign.py --scenario pythia-l2 \
  --candidate-dir results/raw/ticket05 --baseline-dir results/raw/ticket03 \
  --output-dir results/normalized/pythia-l2
python3 experiments/normalize_campaign.py --scenario berti-l1d \
  --candidate-dir results/raw/ticket06 --baseline-dir results/raw/ticket06 \
  --output-dir results/normalized/berti-l1d
python3 experiments/normalize_campaign.py --scenario fdp-ip-stride \
  --candidate-dir results/raw/ticket08-final --baseline-dir results/raw/ticket03 \
  --output-dir results/normalized/fdp-ip-stride
```

Then regenerate miss attribution and the endsem report:

```sh
python3 experiments/attribute_misses.py \
  --l2-dir results/raw/ticket03 --pythia-dir results/raw/ticket05 \
  --berti-dir results/raw/ticket06 --fdp-dir results/raw/ticket08-final \
  --output results/normalized/endsem/miss-attribution.json
python3 experiments/generate_endsem_report.py
```

The report generator reads the normalized CSVs, access characterization, run identities, and attribution summaries; it creates LaTeX, figures, and a compiled PDF under `reports/endsem/`.
