#!/usr/bin/env python3
"""Generate the midsem LaTeX report and plots from normalized L2 campaign data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stridewise-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFETCHERS = ("none", "ip_stride", "spp_dev", "va_ampm_lite")
LABELS = {"none": "No prefetch", "ip_stride": "IP-stride", "spp_dev": "SPP", "va_ampm_lite": "VA-AMPM"}
COLORS = {"none": "#777777", "ip_stride": "#2878b5", "spp_dev": "#e1812c", "va_ampm_lite": "#3a923a"}
WORKLOAD_LABELS = {
    "603.bwaves_s-891B": "bwaves",
    "619.lbm_s-4268B": "lbm",
    "649.fotonik3d_s-1176B": "fotonik3d",
    "605.mcf_s-1554B": "mcf",
    "620.omnetpp_s-141B": "omnetpp",
    "623.xalancbmk_s-165B": "xalanc",
    "ligra-bfs-cc": "BFS",
    "ligra-pagerankdelta": "PageRank",
    "ligra-triangle": "Triangle",
    "llama2": "Llama2",
}


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def read_campaign(data_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    with (data_dir / "metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    runs = json.loads((data_dir / "runs.json").read_text(encoding="utf-8"))
    characterization = json.loads((data_dir / "access-characterization.json").read_text(encoding="utf-8"))
    if len(rows) != 40 or len(characterization.get("workloads", [])) != 10:
        raise ValueError("midsem report requires 40 normalized runs and ten workload characterizations")
    return rows, runs, characterization


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        result[row["workload"]][row["prefetcher"]] = row
    return result


def plot_speedups(rows: list[dict[str, str]], output: Path) -> None:
    data = grouped(rows)
    workloads = list(data)
    x = np.arange(len(workloads))
    width = 0.19
    figure, axis = plt.subplots(figsize=(14, 6.2), constrained_layout=True)
    for index, prefetcher in enumerate(PREFETCHERS):
        values = [number(data[workload][prefetcher], "ipc_speedup") for workload in workloads]
        axis.bar(x + (index - 1.5) * width, values, width, label=LABELS[prefetcher], color=COLORS[prefetcher])
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axis.set_ylabel("IPC speedup over matched no-prefetch L2 run")
    axis.set_xticks(
        x,
        [WORKLOAD_LABELS[workload] for workload in workloads],
        rotation=28,
        ha="right",
        fontsize=9,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_coverage_speedup(rows: list[dict[str, str]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 5.1), constrained_layout=True)
    for prefetcher in PREFETCHERS[1:]:
        selected = [row for row in rows if row["prefetcher"] == prefetcher]
        axis.scatter(
            [100 * number(row, "coverage") for row in selected],
            [number(row, "ipc_speedup") for row in selected],
            label=LABELS[prefetcher],
            color=COLORS[prefetcher],
            s=42,
            alpha=0.85,
        )
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axis.axvline(0.0, color="#999999", linewidth=0.6)
    axis.set_xlabel("L2 coverage from matched demand misses (%)")
    axis.set_ylabel("IPC speedup over no-prefetch L2")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_traffic(rows: list[dict[str, str]], output: Path) -> None:
    data = grouped(rows)
    workloads = list(data)
    x = np.arange(len(workloads))
    width = 0.24
    figure, axis = plt.subplots(figsize=(14, 6.2), constrained_layout=True)
    for index, prefetcher in enumerate(PREFETCHERS[1:]):
        relative = []
        for workload in workloads:
            candidate = number(data[workload][prefetcher], "dram_bytes")
            baseline = number(data[workload]["none"], "dram_bytes")
            relative.append(candidate / baseline if baseline else 0)
        axis.bar(x + (index - 1) * width, relative, width, label=LABELS[prefetcher], color=COLORS[prefetcher])
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--", label="No-prefetch baseline")
    axis.set_ylabel("Total DRAM bytes / matched no-prefetch bytes")
    axis.set_xticks(
        x,
        [WORKLOAD_LABELS[workload] for workload in workloads],
        rotation=28,
        ha="right",
        fontsize=9,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def metric_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        workload = tex_escape(row["workload"])
        prefetcher = tex_escape(LABELS[row["prefetcher"]])
        baseline_mpki = number(row, "baseline_demand_mpki")
        coverage = number(row, "coverage") * 100
        accuracy = row["accuracy"]
        accuracy_text = "n/a" if accuracy == "" else f"{float(accuracy):.2f}"
        lines.append(
            f"{workload} & {prefetcher} & {baseline_mpki:.3f} & {int(row['demand_misses']):,} & "
            f"{coverage:.1f}\\% & {accuracy_text} & {number(row, 'ipc_speedup'):.3f} \\\\"
        )
    return "\n".join(lines)


def traffic_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{tex_escape(row['workload'])} & {tex_escape(LABELS[row['prefetcher']])} & "
            f"{int(row['read_transactions']):,} & {int(row['write_transactions']):,} & "
            f"{float(row['demand_read_bytes']) / (1024**2):.1f} & "
            f"{float(row['prefetch_read_bytes']) / (1024**2):.1f} & "
            f"{float(row['dram_bytes']) / (1024**2):.1f} \\\\"
        )
    return "\n".join(lines)


def top_pc_line(workload: dict[str, Any]) -> str:
    pc = workload["important_pcs"][0]
    exact = pc["dominant_exact_contiguous_delta_bytes"]
    if (
        exact
        and pc["sampled_pc_local_transitions_with_no_intervening_l2_loads"] >= 5
        and exact[0]["fraction_of_exact_contiguous_pc_local_transitions"] >= 0.5
    ):
        dominant = exact[0]
        delta = f"${dominant['delta_bytes']:+,}$ B in {dominant['sampled_transition_count']}/" \
            f"{pc['sampled_pc_local_transitions_with_no_intervening_l2_loads']} exact contiguous transitions"
    else:
        sampled = pc["dominant_sampled_transition_deltas"]
        if sampled:
            dominant = sampled[0]
            delta = f"${dominant['delta_bytes']:+,}$ B in {dominant['sampled_transition_count']} sampled transitions"
        else:
            delta = "no sampled PC-local transition"

    sequences = pc["recurring_ordered_contiguous_delta_sequences"]
    if sequences:
        sequence = sequences[0]
        sequence_text = ", ".join(f"{value:+,}" for value in sequence["delta_bytes_sequence"]) + " B"
        sequence_summary = f"recurring contiguous sequence ${sequence_text}$ (n={sequence['sampled_sequence_count']})"
    else:
        sequence_summary = "no repeated three-delta contiguous sequence observed"

    offsets = workload["page_local_footprints"]["top_recurring_line_offsets"][:3]
    offset_summary = ", ".join(
        f"{item['offset_bytes_within_4k_page']} B ({item['distinct_pages']} pages)" for item in offsets
    )
    sample = workload["sample"]
    return (
        f"\\paragraph{{{tex_escape(workload['workload'])}.}} The highest sampled L2 miss PC is "
        f"\\texttt{{{pc['pc']}}}: {pc['sampled_misses']:,} sampled misses, with an expanded estimate of "
        f"{pc['estimated_misses_in_full_stream']:,.0f} misses across {pc['sampled_loads']:,} sampled loads. "
        f"The leading address-delta evidence is {delta}; {sequence_summary}. The most recurrent sampled "
        f"4~KiB-page line offsets are {offset_summary}. The reservoir retained "
        f"{sample['sampled_accesses']:,}/{sample['total_accesses']:,} L2 loads "
        f"({100 * sample['sampling_fraction']:.2f}\\%).\\par\n"
    )


def workload_interpretations(rows: list[dict[str, str]], characterization: dict[str, Any]) -> str:
    data = grouped(rows)
    access_by_workload = {item["workload"]: item for item in characterization["workloads"]}
    paragraphs = []
    for workload, results in data.items():
        candidates = [results[prefetcher] for prefetcher in PREFETCHERS[1:]]
        best_speedup = max(candidates, key=lambda row: number(row, "ipc_speedup"))
        best_coverage = max(candidates, key=lambda row: number(row, "coverage"))
        exact_pc = access_by_workload[workload]["important_pcs"][0]
        delta = exact_pc["dominant_exact_contiguous_delta_bytes"]
        if (
            delta
            and exact_pc["sampled_pc_local_transitions_with_no_intervening_l2_loads"] >= 5
            and delta[0]["fraction_of_exact_contiguous_pc_local_transitions"] >= 0.5
        ):
            stream_evidence = (
                f"PC \\texttt{{{exact_pc['pc']}}} shows a leading exact contiguous sampled delta of "
                f"{delta[0]['delta_bytes']:+,}~B across {exact_pc['sampled_pc_local_transitions_with_no_intervening_l2_loads']} pairs"
            )
        elif exact_pc["sampled_pc_local_transitions_with_no_intervening_l2_loads"] >= 5:
            stream_evidence = (
                f"PC \\texttt{{{exact_pc['pc']}}} leads sampled L2 misses, but its exact contiguous pairs "
                f"({exact_pc['sampled_pc_local_transitions_with_no_intervening_l2_loads']}) show no dominant delta"
            )
        else:
            stream_evidence = (
                f"PC \\texttt{{{exact_pc['pc']}}} leads sampled L2 misses, but the bounded sample contains "
                "too few exact contiguous pairs to establish its stride"
            )
        paragraphs.append(
            f"\\paragraph{{{tex_escape(workload)}.}} {stream_evidence}. "
            f"{tex_escape(LABELS[best_coverage['prefetcher']])} has the largest measured L2 coverage "
            f"({100 * number(best_coverage, 'coverage'):.1f}\\%), while "
            f"{tex_escape(LABELS[best_speedup['prefetcher']])} has the largest IPC speedup "
            f"({number(best_speedup, 'ipc_speedup'):.3f}$\\times$). This separates miss reduction from "
            "performance: cache coverage does not encode prefetch timeliness, interference, or DRAM service cost.\\par\n"
        )
    return "\n".join(paragraphs)


def trace_table(identities: dict[str, Any]) -> str:
    baseline_by_workload = {
        identity["workload"]: identity
        for identity in identities.values()
        if identity["prefetcher"]["name"] == "none"
    }
    lines = []
    for workload, identity in sorted(baseline_by_workload.items()):
        trace = identity["trace"]
        lines.append(
            f"{tex_escape(workload)} & \\url{{{trace['filename']}}} & "
            f"\\texttt{{{trace['sha256'][:16]}}} \\\\"
        )
    return "\n".join(lines)


def make_tex(rows: list[dict[str, str]], run_document: dict[str, Any], characterization: dict[str, Any], figures: Path) -> str:
    identities = run_document["run_identities"]
    revisions = sorted({identity["simulator"]["source_revision"] for identity in identities.values()})
    profile_hashes = sorted({identity["profile"]["sha256"] for identity in identities.values()})
    roi = next(iter(identities.values()))["roi"]
    actual = [identity["roi"]["actual_simulation_instructions"] for identity in identities.values()]
    figure_prefix = figures.name
    return rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.7cm]{{geometry}}
\usepackage{{booktabs,longtable,graphicx,float,hyperref,parskip}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{Midsem: L2 Prefetcher Characterization}}
\author{{Stridewise project simulation results}}
\date{{}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This report compares no prefetching, IP-stride, SPP, and VA-AMPM at L2 on the fixed ten-workload trace set. It reports matched L2 demand misses, project-defined coverage and accuracy, IPC speedup, DRAM traffic, and bounded access-stream characterization. These are project-specific ChampSim trace simulations, not official SPEC CPU2017 results.
\end{{abstract}}

\section{{Experiment setup}}
Each run uses the same single-core profile (SHA-256 prefix \texttt{{{profile_hashes[0][:16]}}}), warms up for 50 million instructions, then requests a 100 million-instruction ROI from instruction {roi['measured_instruction_start']:,} through {roi['measured_instruction_end']:,}. ChampSim records {min(actual):,}--{max(actual):,} retired instructions across the reported ROIs because it completes the final retire group; the runner accepts only the requested count plus fewer than one retire width. Cache line size is 64~B. All prefetchers outside L2C are disabled, and the measured L2 configuration is the only active prefetcher \cite{{champsim}}. The pinned simulator source is {', '.join(tex_escape(revision) for revision in revisions)}; full profile and trace hashes are in the normalized run identities.

\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{@{{}}p{{3.2cm}}p{{9.4cm}}p{{3.5cm}}@{{}}}}
\caption{{Workload trace identities (SHA-256 prefixes)}}\\
\toprule Workload & Trace & SHA-256 prefix \\
\midrule
\endfirsthead
\toprule Workload & Trace & SHA-256 prefix \\
\midrule
\endhead
{trace_table(identities)}
\bottomrule
\end{{longtable}}
\normalsize

\section{{Metric definitions}}
Baseline L2 MPKI is baseline L2 LOAD misses divided by baseline ROI instructions/1000. L2 coverage is (matched no-prefetch L2 LOAD misses $-$ configured L2 LOAD misses) / matched no-prefetch L2 LOAD misses. Accuracy follows the project formula, useful prefetch events / L2 prefetch fills \cite{{project}}. IPC speedup is configured ROI IPC / matched no-prefetch ROI IPC, where IPC is ROI instructions / ROI cycles. DRAM traffic reports read and write transaction counts and bytes separately; demand and prefetch read counters use the request type retained when a transaction is serviced.

Some accuracy ratios exceed 1. ChampSim increments its useful-prefetch event counter when demand accesses merge onto an outstanding prefetch, so this project-defined ratio is not a probability. The normalized CSV retains each numerator and denominator, and the evidence archive preserves the counters.

\section{{Four-way L2 results}}
\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{@{{}}p{{3.0cm}}p{{1.5cm}}rrrrr@{{}}}}
\caption{{L2 miss, coverage, accuracy, and IPC results for every workload}}\\
\toprule Workload & Configuration & Baseline L2 MPKI & Config. L2 misses & Coverage & Accuracy & IPC speedup \\
\midrule
\endfirsthead
\toprule Workload & Configuration & Baseline L2 MPKI & Config. L2 misses & Coverage & Accuracy & IPC speedup \\
\midrule
\endhead
{metric_table(rows)}
\bottomrule
\end{{longtable}}
\normalsize

The full normalized data in \url{{results/normalized/l2-required/metrics.csv}} includes the coverage, accuracy, MPKI, and IPC numerator/denominator fields, configuration hashes, trace hashes, and per-run identifiers.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{figure_prefix}/ipc-speedup.png}}
\caption{{Per-workload IPC speedup at L2; the dashed line is the matched no-prefetch baseline.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{{figure_prefix}/coverage-vs-speedup.png}}
\caption{{L2 demand-miss coverage and IPC speedup are related but distinct outcomes.}}
\end{{figure}}

\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{@{{}}p{{3.0cm}}p{{1.5cm}}rrrrr@{{}}}}
\caption{{DRAM traffic during each measured ROI}}\\
\toprule Workload & Configuration & Read tx. & Write tx. & Demand read MiB & Prefetch read MiB & Total MiB \\
\midrule
\endfirsthead
\toprule Workload & Configuration & Read tx. & Write tx. & Demand read MiB & Prefetch read MiB & Total MiB \\
\midrule
\endhead
{traffic_table(rows)}
\bottomrule
\end{{longtable}}
\normalsize

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{figure_prefix}/relative-dram-bytes.png}}
\caption{{Total DRAM bytes relative to the matched no-prefetch run for each workload.}}
\end{{figure}}

\section{{Access-stream characterization}}
The L2 demand sample is a deterministic, bounded reservoir sample of at most 10,000 accesses per run. PC-local full-stream miss counts are expanded estimates from the sampling fraction; raw sample counts and access-order gaps remain available in \texttt{{access-characterization.json}}. Sampled transitions are not treated as consecutive real accesses unless their global L2 access-order gap is exactly one. This avoids turning a gap between reservoir observations into a claimed hardware stride. The table below highlights each workload's highest sampled-miss PC, strongest contiguous-delta evidence, observed ordered sequence evidence, and recurring 4~KiB line offsets.

{''.join(top_pc_line(workload) for workload in characterization['workloads'])}

Only \texttt{{ligra-pagerankdelta}} has a repeated ordered three-delta sequence in the sampled-observation histories of its top miss PCs; no repeated three-delta sequence appears in the strict contiguous subset. This is a limit of the bounded evidence, not proof that other workloads lack correlation. Page-local footprints provide complementary, order-independent evidence for AMPM-like spatial prediction.

\section{{Interpretation}}
IP-stride tracks addresses by load PC and benefits from stable PC-local deltas. SPP learns recurring spatial patterns and correlations from recent accesses \cite{{spp}}. VA-AMPM-lite records page-local access-map state and can exploit recurring footprints without depending on one PC's stride \cite{{ampm}}. These predictor classes therefore use different evidence: PC-local order for stride, recurring correlated patterns for SPP, and page-local spatial footprint for AMPM.

{workload_interpretations(rows, characterization)}

Across workloads, geometric-mean IPC speedup is {math.exp(sum(math.log(number(row, 'ipc_speedup')) for row in rows if row['prefetcher'] == 'ip_stride') / 10):.3f} for IP-stride, {math.exp(sum(math.log(number(row, 'ipc_speedup')) for row in rows if row['prefetcher'] == 'spp_dev') / 10):.3f} for SPP, and {math.exp(sum(math.log(number(row, 'ipc_speedup')) for row in rows if row['prefetcher'] == 'va_ampm_lite') / 10):.3f} for VA-AMPM. These geometric means are within the L2 group and use the matched L2 no-prefetch IPC for each workload.

\section{{Reproduction artifacts}}
The normalized metrics, run identities, access characterization, and archived raw evidence are in \url{{results/normalized/l2-required/}}. Each identity records the trace and configuration hashes, simulator revision, prefetcher level, warmup, requested ROI, and actual retired instructions. Rerun individual simulations with \url{{experiments/run_campaign.py}} and regenerate the summaries with \url{{experiments/normalize_campaign.py}} and \url{{experiments/analyze_access_streams.py}}.

\begin{{thebibliography}}{{9}}
\bibitem{{project}} Project 3: Prefetcher Characterization, course brief, \texttt{{docs/project/CA\_Project3\_Prefetcher\_Characterization.md}}.
\bibitem{{ampm}} T. Ishii, M. Inaba, and K. Hiraki, ``Access Map Pattern Matching for High Performance Data Cache Prefetch,'' JILP, 2011.
\bibitem{{spp}} J. Kim et al., ``Spatial Memory Streaming,'' ISCA, 2016.
\bibitem{{champsim}} ChampSim DPC4 source pinned at {tex_escape(revisions[0])}.
\end{{thebibliography}}
\end{{document}}
"""


def generate(data_dir: Path, output_dir: Path, compile_pdf: bool) -> None:
    rows, run_document, characterization = read_campaign(data_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_speedups(rows, figures / "ipc-speedup.png")
    plot_coverage_speedup(rows, figures / "coverage-vs-speedup.png")
    plot_traffic(rows, figures / "relative-dram-bytes.png")
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "midsem.tex"
    tex_path.write_text(make_tex(rows, run_document, characterization, figures), encoding="utf-8")
    if compile_pdf:
        for _ in range(2):
            compiled = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "midsem.tex"],
                cwd=output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if compiled.returncode:
                raise RuntimeError(compiled.stdout[-6000:])
    print(f"Wrote report source to {tex_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/normalized/l2-required"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/midsem"))
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    generate(args.data_dir, args.output_dir, compile_pdf=not args.no_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
