#!/usr/bin/env python3
"""Generate the endsem evaluation report from normalized campaign artifacts."""

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


WORKLOADS = (
    "603.bwaves_s-891B", "619.lbm_s-4268B", "649.fotonik3d_s-1176B", "605.mcf_s-1554B",
    "620.omnetpp_s-141B", "623.xalancbmk_s-165B", "ligra-bfs-cc", "ligra-pagerankdelta",
    "ligra-triangle", "llama2",
)
L2_PREFETCHERS = ("none", "ip_stride", "spp_dev", "va_ampm_lite", "pythia")
LABELS = {
    "none": "No prefetch", "ip_stride": "IP-stride", "spp_dev": "SPP", "va_ampm_lite": "VA-AMPM",
    "pythia": "Pythia", "berti": "Berti", "ip_stride_fdp": "FDP-IP-stride",
}
COLORS = {
    "none": "#777777", "ip_stride": "#2878b5", "spp_dev": "#e1812c", "va_ampm_lite": "#3a923a",
    "pythia": "#8968ac", "berti": "#b25454", "ip_stride_fdp": "#17a2a4",
}
WORKLOAD_LABELS = {
    "603.bwaves_s-891B": "bwaves", "619.lbm_s-4268B": "lbm", "649.fotonik3d_s-1176B": "fotonik3d",
    "605.mcf_s-1554B": "mcf", "620.omnetpp_s-141B": "omnetpp", "623.xalancbmk_s-165B": "xalanc",
    "ligra-bfs-cc": "BFS", "ligra-pagerankdelta": "PageRank", "ligra-triangle": "Triangle", "llama2": "Llama2",
}
MECHANISMS = {
    "ip_stride": "per-PC last-stride state with fixed-degree lookahead",
    "spp_dev": "signature and pattern-table prediction of ordered spatial correlations",
    "va_ampm_lite": "page-local access-map prediction of unordered footprints",
    "pythia": "feature-conditioned action selection updated by delayed reinforcement rewards",
}


class ReportError(Exception):
    """Raised when the campaign artifacts are incomplete or inconsistent."""


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def tex_break_hash(value: str) -> str:
    return r"\allowbreak{}".join(value[index : index + 8] for index in range(0, len(value), 8))


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error) as error:
        raise ReportError(f"cannot read {path}: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"expected a JSON object in {path}")
    return value


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else math.nan


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        result[row["workload"]][row["prefetcher"]] = row
    return result


def geomean(rows: list[dict[str, str]], field: str = "ipc_speedup") -> float:
    values = [number(row, field) for row in rows]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ReportError(f"cannot calculate a geometric mean from {field}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def plot_l2(rows: list[dict[str, str]], output: Path) -> None:
    data = grouped(rows)
    x = np.arange(len(WORKLOADS))
    width = 0.15
    fig, axis = plt.subplots(figsize=(14, 6.4), constrained_layout=True)
    for index, prefetcher in enumerate(L2_PREFETCHERS):
        values = [number(data[workload][prefetcher], "ipc_speedup") for workload in WORKLOADS]
        axis.bar(x + (index - 2) * width, values, width, label=LABELS[prefetcher], color=COLORS[prefetcher])
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axis.set_ylabel("L2 IPC speedup over matched no-prefetch L2 run")
    axis.set_xticks(x, [WORKLOAD_LABELS[item] for item in WORKLOADS], rotation=28, ha="right", fontsize=9)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_geomeans(l2_rows: list[dict[str, str]], berti_rows: list[dict[str, str]], fdp_rows: list[dict[str, str]], output: Path) -> None:
    l2 = grouped(l2_rows)
    l2_names = ("ip_stride", "spp_dev", "va_ampm_lite", "pythia")
    means = [geomean([l2[workload][name] for workload in WORKLOADS]) for name in l2_names]
    fdp_mean = geomean(fdp_rows)
    berti_mean = geomean(berti_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), constrained_layout=True)
    axes[0].bar([LABELS[name] for name in l2_names], means, color=[COLORS[name] for name in l2_names])
    axes[0].axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axes[0].set_title("L2 vs no-prefetch L2")
    axes[0].set_ylabel("Geometric-mean IPC speedup")
    axes[0].tick_params(axis="x", rotation=24)
    axes[1].bar(["Berti@L1D"], [berti_mean], color=COLORS["berti"])
    axes[1].axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axes[1].set_title("Separate L1D scenario")
    axes[1].set_ylim(bottom=0)
    axes[2].bar(["FDP-IP-stride"], [fdp_mean], color=COLORS["ip_stride_fdp"])
    axes[2].axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axes[2].set_title("Variant vs original IP-stride")
    axes[2].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.bar_label(axis.containers[0], fmt="%.3f", padding=3, fontsize=9)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_l2_traffic(rows: list[dict[str, str]], output: Path) -> None:
    data = grouped(rows)
    x = np.arange(len(WORKLOADS))
    width = 0.15
    fig, axis = plt.subplots(figsize=(14, 6.4), constrained_layout=True)
    for index, prefetcher in enumerate(L2_PREFETCHERS):
        values = [number(data[workload][prefetcher], "dram_bytes") / number(data[workload]["none"], "dram_bytes") for workload in WORKLOADS]
        axis.bar(x + (index - 2) * width, values, width, label=LABELS[prefetcher], color=COLORS[prefetcher])
    axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axis.set_ylabel("Total DRAM bytes / no-prefetch L2 bytes")
    axis.set_xticks(x, [WORKLOAD_LABELS[item] for item in WORKLOADS], rotation=28, ha="right", fontsize=9)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_fdp(fdp_rows: list[dict[str, str]], output: Path) -> None:
    x = np.arange(len(WORKLOADS))
    ipc = [number(row, "ipc_speedup") for row in fdp_rows]
    traffic = [number(row, "dram_bytes") / number(row, "baseline_dram_bytes") for row in fdp_rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, constrained_layout=True)
    axes[0].bar(x, ipc, color=COLORS["ip_stride_fdp"])
    axes[0].axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axes[0].set_ylabel("IPC speedup vs IP-stride")
    axes[1].bar(x, traffic, color="#4e79a7")
    axes[1].axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    axes[1].set_ylabel("DRAM bytes / IP-stride bytes")
    axes[1].set_xticks(x, [WORKLOAD_LABELS[item] for item in WORKLOADS], rotation=25, ha="right", fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def pct(row: dict[str, str], key: str) -> str:
    value = number(row, key)
    return "n/a" if not math.isfinite(value) else f"{100 * value:.1f}\\%"


def ratio(row: dict[str, str], key: str) -> str:
    value = number(row, key)
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def mib(row: dict[str, str], key: str = "dram_bytes") -> str:
    value = number(row, key)
    return "n/a" if not math.isfinite(value) else f"{value / (1024 * 1024):.1f}"


def l2_table(rows: list[dict[str, str]]) -> str:
    data = grouped(rows)
    lines = []
    for workload in WORKLOADS:
        for prefetcher in L2_PREFETCHERS:
            row = data[workload][prefetcher]
            lines.append(
                f"{WORKLOAD_LABELS[workload]} & {LABELS[prefetcher]} & {number(row, 'baseline_demand_mpki'):.2f} & "
                f"{pct(row, 'coverage')} & {ratio(row, 'accuracy')} & {number(row, 'ipc_speedup'):.3f} & "
                f"{int(number(row, 'read_transactions')):,} & {int(number(row, 'write_transactions')):,} & {mib(row)} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    return "\n".join(lines)


def berti_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in sorted(rows, key=lambda item: WORKLOADS.index(item["workload"])):
        lines.append(
            f"{WORKLOAD_LABELS[row['workload']]} & {number(row, 'baseline_demand_mpki'):.2f} & "
            f"{number(row, 'demand_misses'):,.0f} & {pct(row, 'coverage')} & {ratio(row, 'accuracy')} & "
            f"{number(row, 'ipc_speedup'):.3f} & {int(number(row, 'read_transactions')):,} & "
            f"{int(number(row, 'write_transactions')):,} & {mib(row)} \\\\"
        )
    return "\n".join(lines)


def fdp_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in sorted(rows, key=lambda item: WORKLOADS.index(item["workload"])):
        lines.append(
            f"{WORKLOAD_LABELS[row['workload']]} & {pct(row, 'coverage')} & {ratio(row, 'accuracy')} & "
            f"{ratio(row, 'feedback_useful_per_resolved_address_episode')} & {number(row, 'ipc_speedup'):.3f} & "
            f"{number(row, 'dram_bytes') / number(row, 'baseline_dram_bytes'):.3f} & "
            f"{row['feedback_final_degree']} & {row['feedback_degree_increases']}/{row['feedback_degree_decreases']} \\\\"
        )
    return "\n".join(lines)


def access_evidence(workload: dict[str, Any]) -> str:
    pcs = workload.get("important_pcs", [])
    if not pcs:
        return "The bounded reservoir did not retain a load PC with sampled misses."
    top = pcs[0]
    exact = top.get("dominant_exact_contiguous_delta_bytes", [])
    if exact and exact[0].get("fraction_of_exact_contiguous_pc_local_transitions", 0) >= 0.5:
        delta = exact[0]
        sequence_text = (
            f"At the top sampled-miss PC {top['pc']}, the leading exact adjacent delta is {delta['delta_bytes']:+} B "
            f"({delta['sampled_transition_count']} pairs; {100 * delta['fraction_of_exact_contiguous_pc_local_transitions']:.1f}\\% of exact adjacent pairs)."
        )
    else:
        sequence_text = (
            f"At the top sampled-miss PC {top['pc']}, the bounded sample has no dominant exact adjacent delta "
            f"({top.get('sampled_pc_local_transitions_with_no_intervening_l2_loads', 0)} gap-free pairs)."
        )

    history = top.get("recurring_ordered_sample_transition_sequences", [])
    if history:
        item = history[0]
        deltas = ", ".join(f"{delta:+} B" for delta in item.get("delta_bytes_sequence", []))
        gaps = ", ".join(str(gap) for gap in item.get("access_order_gap_sequence", []))
        sequence_text += (
            f" A recurring sampled-observation history ({item.get('sampled_sequence_count', 0)} occurrences) has deltas "
            f"{deltas} and global access-order gaps [{gaps}]; it is not treated as a contiguous stride."
        )
    else:
        sequence_text += " No recurring ordered three-delta history was retained for this PC."

    footprint = workload.get("page_local_footprints", {})
    offsets = footprint.get("top_recurring_line_offsets", []) if isinstance(footprint, dict) else []
    if offsets:
        common = ", ".join(
            f"{item['offset_bytes_within_4k_page']} B on {item['distinct_pages']} pages" for item in offsets[:3]
        )
        sequence_text += f" Most common sampled 4 KiB page offsets are {common}."
    sample = workload.get("sample", {})
    sequence_text += (
        f" The access sample retains {sample.get('sampled_accesses', 0):,}/{sample.get('total_accesses', 0):,} "
        f"L2 loads ({100 * sample.get('sampling_fraction', 0):.2f}\\%)."
    )
    return sequence_text


def pattern_interpretation(workload: dict[str, Any], winner_name: str) -> str:
    pcs = workload.get("important_pcs", [])
    if not pcs:
        return "No sampled-miss PC was retained, so the winner's workload-level advantage cannot be tied to a specific address pattern."

    top = pcs[0]
    exact = top.get("dominant_exact_contiguous_delta_bytes", [])
    delta = exact[0] if exact else {}
    repeated_stride = (
        delta.get("sampled_transition_count", 0) >= 20
        and delta.get("fraction_of_exact_contiguous_pc_local_transitions", 0) >= 0.5
    )
    offsets = workload.get("page_local_footprints", {}).get("top_recurring_line_offsets", [])
    repeated_footprint = bool(offsets and offsets[0].get("distinct_pages", 0) >= 100)

    if repeated_stride:
        stride_text = (
            f"The leading sampled-miss PC {top['pc']} repeats a {delta['delta_bytes']:+} B exact adjacent delta "
            f"in {delta['sampled_transition_count']} pairs ({100 * delta['fraction_of_exact_contiguous_pc_local_transitions']:.1f}\\% of exact pairs)."
        )
        if winner_name == "ip_stride":
            winner_text = "That is direct stream evidence for the per-PC stride pattern IP-stride models."
        elif winner_name == "spp_dev":
            winner_text = "That repeated same-PC transition is compatible with SPP's ordered spatial model, though no signature lookup or confidence state was recorded."
        elif winner_name == "va_ampm_lite":
            winner_text = "The repeated transition is a per-PC sequence; the available evidence does not show which page-footprint entries VA-AMPM used."
        else:
            winner_text = "Pythia wins on measured IPC; the repeated per-PC sequence is workload evidence, but its feature/action state was not exported, so the policy's use of this or other patterns cannot be established."
    else:
        if delta:
            count = delta.get("sampled_transition_count", 0)
            stride_text = (
                f"At the leading sampled-miss PC {top['pc']}, the strongest exact adjacent delta appears in only {count} pairs; "
                "the bounded sample does not establish a repeated per-PC stride."
            )
        else:
            stride_text = f"At the leading sampled-miss PC {top['pc']}, the bounded sample does not establish a repeated exact per-PC stride."
        if winner_name == "va_ampm_lite" and repeated_footprint:
            winner_text = "The recurring page offsets are compatible with VA-AMPM's page-local footprint model, but access-map state was not recorded."
        elif winner_name == "spp_dev":
            winner_text = "No recurring ordered multi-delta history was retained to explain SPP's result; its internal signature state was not recorded."
        elif winner_name == "ip_stride":
            winner_text = "The bounded sample does not independently support the per-PC stride hypothesis behind IP-stride's result."
        else:
            if repeated_footprint:
                winner_text = "Pythia wins on measured IPC, and repeated page offsets show recurring spatial structure; its feature/action state was not exported, so this pattern cannot be linked to a policy decision."
            else:
                winner_text = "Pythia wins on measured IPC, but the bounded sample shows neither a dominant stride nor a recurring page footprint; its feature/action state was not exported, so the winning policy remains unexplained."

    footprint_text = ""
    if repeated_footprint:
        most_common = offsets[0]
        footprint_text = (
            f" The most common sampled page offset is {most_common['offset_bytes_within_4k_page']} B across "
            f"{most_common['distinct_pages']} pages."
        )
    return f"{stride_text} {winner_text}{footprint_text}"


def competitor_pattern_interpretation(workload: dict[str, Any], prefetcher: str) -> str:
    pcs = workload.get("important_pcs", [])
    if not pcs:
        return "No sampled-miss PC was retained, so the stream sample cannot show whether this mechanism had a usable pattern."

    top = pcs[0]
    exact = top.get("dominant_exact_contiguous_delta_bytes", [])
    delta = exact[0] if exact else {}
    repeatable_stride = (
        delta.get("sampled_transition_count", 0) >= 20
        and delta.get("fraction_of_exact_contiguous_pc_local_transitions", 0) >= 0.5
    )
    histories = top.get("recurring_ordered_sample_transition_sequences", [])
    offsets = workload.get("page_local_footprints", {}).get("top_recurring_line_offsets", [])
    recurring_offsets = [item for item in offsets[:3] if item.get("distinct_pages", 0) >= 100]

    if prefetcher == "ip_stride":
        if repeatable_stride:
            evidence = (
                f"The sample shows {delta['sampled_transition_count']} exact {delta['delta_bytes']:+} B pairs at {top['pc']}, "
                "so a per-PC stride opportunity is visible."
            )
        else:
            count = delta.get("sampled_transition_count", 0)
            evidence = (
                f"Only {count} exact pairs at {top['pc']} support the leading sampled delta, so the bounded sample offers little "
                "evidence for a repeatable per-PC stride."
            )
        limit = "Tracker hits, confidence, and lookahead decisions were not exported, so the measured gap cannot be assigned to training or timeliness."
    elif prefetcher == "spp_dev":
        if histories:
            sequence = histories[0]
            deltas = ", ".join(f"{item:+} B" for item in sequence.get("delta_bytes_sequence", []))
            evidence = (
                f"The bounded sample retains an ordered history at {top['pc']} ({deltas}, "
                f"{sequence.get('sampled_sequence_count', 0)} observations), which is compatible with signature training."
            )
        else:
            evidence = f"No recurring ordered history was retained at the leading sampled-miss PC {top['pc']}, leaving little sampled evidence for signature training."
        limit = "The sample is bounded and SPP signature/confidence state was not exported, so it cannot establish whether a signature was learned or matched."
    elif prefetcher == "va_ampm_lite":
        if recurring_offsets:
            summary = ", ".join(
                f"{item['offset_bytes_within_4k_page']} B on {item['distinct_pages']} pages" for item in recurring_offsets
            )
            evidence = f"The sample shows recurring page-local offsets ({summary}), which are opportunities for footprint prediction."
        else:
            evidence = "No sampled page-local offset repeats across 100 pages, so the retained footprint evidence is weak."
        limit = "Access-map replacement and issued-map match state were not exported, so the coverage and traffic outcome cannot be tied to a specific footprint decision."
    else:
        if repeatable_stride or recurring_offsets:
            structures = []
            if repeatable_stride:
                structures.append(f"a {delta['delta_bytes']:+} B per-PC delta at {top['pc']}")
            if recurring_offsets:
                structures.append(f"page offsets recurring on {recurring_offsets[0]['distinct_pages']} pages")
            evidence = "The sample contains " + " and ".join(structures) + ", but does not expose Pythia's feature/action values."
        else:
            evidence = "The bounded sample contains no strong stride or page-footprint signal, and Pythia's feature/action values were not exported."
        limit = "The result therefore shows the policy outcome but cannot identify which feature or delayed reward explains it."

    return f"{evidence} {limit} Per-request timeliness was not recorded."


def stats_for(row: dict[str, str]) -> dict[str, int]:
    path = Path(row["candidate_result_path"])
    result = read_json(path)
    cache = result.get("normalized", {}).get(f"cpu0_{row['cache_level']}", {})
    return {
        key: cache.get(source, 0) if type(cache.get(source, 0)) is int else 0
        for key, source in (
            ("issued", "prefetch issued"), ("requested", "prefetch requested"), ("fills", "prefetch fill"),
            ("useful", "useful prefetch"), ("useless", "useless prefetch"),
        )
    }


def dram_transactions_for(row: dict[str, str]) -> tuple[int, int, int, int]:
    candidate = read_json(Path(row["candidate_result_path"]))
    baseline = read_json(Path(row["baseline_result_path"]))
    candidate_traffic = candidate.get("normalized", {}).get("dram_traffic", {})
    baseline_traffic = baseline.get("normalized", {}).get("dram_traffic", {})
    keys = ("read_transactions", "write_transactions")
    values = tuple(candidate_traffic.get(key) for key in keys) + tuple(baseline_traffic.get(key) for key in keys)
    if any(type(value) is not int for value in values):
        raise ReportError(f"missing DRAM transaction counts for {row['workload']}")
    return values


def attribution_summary(records: list[dict[str, Any]], workload: str) -> tuple[int, int, int]:
    selected = [record for record in records if record["workload"] == workload]
    return (
        sum(record["classified_misses"] for record in selected),
        sum(record["unresolved_demand_LOAD_misses"] for record in selected),
        sum(
            cause["status"] == "not_applicable"
            for record in selected
            for cause in record["causes"].values()
        ),
    )


def predictor_diagnosis(
    workload: str,
    l2_data: dict[str, dict[str, str]],
    access: dict[str, Any],
    pythia_row: dict[str, str],
    berti_row: dict[str, str],
    fdp_row: dict[str, str],
    attribution: list[dict[str, Any]],
) -> str:
    candidates = [l2_data[workload][name] for name in L2_PREFETCHERS]
    winner = max(candidates, key=lambda row: number(row, "ipc_speedup"))
    winner_name = winner["prefetcher"]
    best_coverage = max(candidates[1:], key=lambda row: number(row, "coverage"))
    winner_stats = stats_for(winner) if winner_name != "none" else {"issued": 0, "fills": 0, "useful": 0, "useless": 0}
    trace_summary = access_evidence(access)

    lines = [
        f"\\subsection{{{tex_escape(WORKLOAD_LABELS[workload])}}}",
        f"\\textbf{{L2 winner.}} {LABELS[winner_name]} has the highest measured L2 IPC, "
        f"{number(winner, 'ipc_speedup'):.3f}$\\times$ the matched no-prefetch run, with "
        f"{pct(winner, 'coverage')} coverage, accuracy {ratio(winner, 'accuracy')}, and "
        f"{mib(winner)} MiB of total DRAM traffic. Its ROI counters show "
        f"{winner_stats['issued']:,} accepted requests, {winner_stats['fills']:,} fills, "
        f"{winner_stats['useful']:,} useful events, and {winner_stats['useless']:,} useless evictions. "
        f"The best coverage instead belongs to {LABELS[best_coverage['prefetcher']]} "
        f"({pct(best_coverage, 'coverage')}, {number(best_coverage, 'ipc_speedup'):.3f}$\\times$ IPC), "
        "showing whether miss reduction and speedup selected the same configuration.\\par",
        f"\\textbf{{Memory-access evidence.}} {trace_summary}\\par",
        f"\\textbf{{Pattern interpretation.}} {pattern_interpretation(access, winner_name)}\\par",
    ]

    failures = []
    for prefetcher in L2_PREFETCHERS[1:]:
        row = l2_data[workload][prefetcher]
        if prefetcher == winner_name:
            continue
        explanation = competitor_pattern_interpretation(access, prefetcher)
        event_stats = stats_for(row)
        coverage_gap = 100 * (number(row, "coverage") - number(winner, "coverage"))
        traffic_gap = (number(row, "dram_bytes") - number(winner, "dram_bytes")) / (1024 * 1024)
        ipc_gap = number(winner, "ipc_speedup") - number(row, "ipc_speedup")
        if coverage_gap > 0.5:
            traffic_direction = "more" if traffic_gap >= 0 else "less"
            outcome = (
                f"Its miss coverage is {coverage_gap:.1f} percentage points above the winner's, yet its IPC trails by "
                f"{ipc_gap:.3f} IPC ratio points and transfers {abs(traffic_gap):.1f} MiB {traffic_direction} DRAM data; higher coverage did not translate into higher IPC here."
            )
        elif number(row, "ipc_speedup") < 1.0:
            outcome = "It falls below the matched no-prefetch IPC despite issuing prefetches, so its net result is a regression on this workload."
        elif coverage_gap < -0.5:
            outcome = (
                f"Its miss coverage is {-coverage_gap:.1f} percentage points below the winner's and its IPC trails by "
                f"{ipc_gap:.3f} IPC ratio points, while its DRAM traffic differs by {traffic_gap:+.1f} MiB."
            )
        else:
            outcome = (
                f"Its coverage is within 0.5 percentage points of the winner but it trails by "
                f"{ipc_gap:.3f} IPC ratio points; the measured traffic difference is {traffic_gap:+.1f} MiB."
            )
        failures.append(
            f"{LABELS[prefetcher]}: {number(row, 'ipc_speedup'):.3f}$\\times$ IPC, {pct(row, 'coverage')} coverage, "
            f"accuracy {ratio(row, 'accuracy')}, {mib(row)} MiB DRAM, and {event_stats['issued']:,} accepted requests; "
            f"{outcome} {explanation}"
        )
    lines.append("\\textbf{Other L2 predictors.} " + " ".join(failures) + "\\par")

    fdp_ratio = number(fdp_row, "dram_bytes") / number(fdp_row, "baseline_dram_bytes")
    fdp_read_tx, fdp_write_tx, baseline_read_tx, baseline_write_tx = dram_transactions_for(fdp_row)
    feedback_scope = "Counters span warmup and ROI."
    lines.append(
        f"\\textbf{{Feedback-directed IP-stride.}} Relative to original IP-stride, the variant records "
        f"{number(fdp_row, 'ipc_speedup'):.3f}$\\times$ IPC, {fdp_ratio:.3f} DRAM-byte ratio, and "
        f"read/write DRAM transactions {fdp_read_tx:,}/{fdp_write_tx:,} vs original IP-stride "
        f"{baseline_read_tx:,}/{baseline_write_tx:,}; "
        f"a useful/resolved prefetch address episode ratio of {ratio(fdp_row, 'feedback_useful_per_resolved_address_episode')}, "
        f"with final degree {fdp_row['feedback_final_degree']} and "
        f"{fdp_row['feedback_degree_increases']} degree increases / {fdp_row['feedback_degree_decreases']} decreases. "
        f"{feedback_scope}\\par"
    )
    lines.append(
        f"\\textbf{{Separate L1D Berti result.}} Berti has {number(berti_row, 'ipc_speedup'):.3f}$\\times$ "
        f"IPC vs its matched L1D no-prefetch run, {pct(berti_row, 'coverage')} L1D coverage, "
        f"accuracy {ratio(berti_row, 'accuracy')}, and {mib(berti_row)} MiB DRAM. "
        "These values are a separate L1D scenario and are not ranked against the L2 winner.\\par"
    )
    classified, unresolved, not_applicable = attribution_summary(attribution, workload)
    lines.append(
        f"\\textbf{{Miss attribution.}} Across the measured prefetcher runs, {classified:,} L2/L1D LOAD misses "
        f"were causally classified and {unresolved:,} remain unresolved; {not_applicable} cause entries are marked not applicable. "
        "No persistent address-linked event timeline or predictor-state snapshot was exported.\\par"
    )
    return "\n".join(lines)


def make_tex(
    l2_rows: list[dict[str, str]], pythia_rows: list[dict[str, str]], berti_rows: list[dict[str, str]], fdp_rows: list[dict[str, str]],
    access_doc: dict[str, Any], attribution_doc: dict[str, Any], output_dir: Path,
) -> str:
    l2 = grouped(l2_rows)
    pythia = {row["workload"]: row for row in pythia_rows}
    berti = {row["workload"]: row for row in berti_rows}
    fdp = {row["workload"]: row for row in fdp_rows}
    access = {row["workload"]: row for row in access_doc["workloads"]}
    attribution = attribution_doc["records"]

    ip_mean = geomean([l2[workload]["ip_stride"] for workload in WORKLOADS])
    spp_mean = geomean([l2[workload]["spp_dev"] for workload in WORKLOADS])
    ampm_mean = geomean([l2[workload]["va_ampm_lite"] for workload in WORKLOADS])
    pythia_mean = geomean([pythia[workload] for workload in WORKLOADS])
    berti_mean = geomean(berti_rows)
    fdp_mean = geomean(fdp_rows)
    actual_roi_counts = [int(row["roi_instructions"]) for row in l2_rows + berti_rows + fdp_rows]
    actual_roi_min, actual_roi_max = min(actual_roi_counts), max(actual_roi_counts)
    fdp_improved_ipc = sum(number(row, "ipc_speedup") > 1.0 for row in fdp_rows)
    fdp_lower_traffic_no_loss = sum(
        number(row, "dram_bytes") < number(row, "baseline_dram_bytes") and number(row, "ipc_speedup") >= 1.0
        for row in fdp_rows
    )
    fdp_regressions = sum(number(row, "ipc_speedup") < 1.0 for row in fdp_rows)
    baseline_feedback = []
    for workload in WORKLOADS:
        counters = stats_for(l2[workload]["ip_stride"])
        if counters["issued"]:
            baseline_feedback.append((counters["useful"] / counters["issued"], workload))
    lowest_feedback, lowest_workload = min(baseline_feedback)
    highest_feedback, highest_workload = max(baseline_feedback)
    omnet_baseline = l2["620.omnetpp_s-141B"]["ip_stride"]
    mcf_baseline = l2["605.mcf_s-1554B"]["ip_stride"]
    l2_wins = defaultdict(int)
    for workload in WORKLOADS:
        winner = max((l2[workload][name] for name in L2_PREFETCHERS), key=lambda row: number(row, "ipc_speedup"))
        l2_wins[winner["prefetcher"]] += 1
    source_revisions = sorted(
        {read_json(Path(row["candidate_result_path"]))["identity"]["simulator"]["source_revision"] for row in l2_rows if row["prefetcher"] != "none"}
    )

    winner_counts = ", ".join(f"{LABELS[name]} {count}" for name, count in l2_wins.items() if count)
    diagnosis = "\n\n".join(
        predictor_diagnosis(workload, l2, access[workload], pythia[workload], berti[workload], fdp[workload], attribution)
        for workload in WORKLOADS
    )
    return rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,flushend,graphicx,makecell,url}}
\usepackage[hidelinks]{{hyperref}}
\hypersetup{{pdftitle={{Workload-Level Prefetcher Diagnosis}}}}
\author{{}}
\title{{Workload-Level Prefetcher Diagnosis}}
\setlength{{\emergencystretch}}{{2em}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This evaluation adds Pythia at L2, Berti in a separate L1D scenario, conservative mechanism-aware miss attribution, and one feedback-directed IP-stride variant to the required L2 comparison. Every conclusion is tied to matched trace runs, normalized counters, and bounded access-stream evidence. These are project-specific ChampSim trace simulations, not official SPEC CPU2017 results.
\end{{abstract}}

\begin{{IEEEkeywords}}
hardware prefetching, memory hierarchy, ChampSim, performance evaluation
\end{{IEEEkeywords}}

\section{{Introduction}}
Within the same-level L2 group, geometric-mean IPC speedups are {ip_mean:.3f} for IP-stride, {spp_mean:.3f} for SPP, {ampm_mean:.3f} for VA-AMPM, and {pythia_mean:.3f} for Pythia, each against a matched no-prefetch L2 run. The highest per-workload L2 IPC results are distributed as follows: {tex_escape(winner_counts)}. Berti's separate L1D scenario has a geometric-mean speedup of {berti_mean:.3f} against matched L1D no-prefetch. It is not ranked against L2 prefetchers.

The feedback-directed IP-stride variant has a geometric-mean IPC ratio of {fdp_mean:.3f} against original IP-stride. It gains IPC on {fdp_improved_ipc}/10 workloads, reduces DRAM bytes without an IPC loss on {fdp_lower_traffic_no_loss}/10, and regresses IPC on {fdp_regressions}/10. The per-workload table reports both outcomes separately; no aggregate score combines them.

\section{{Campaign setup and comparison groups}}
Each campaign uses the pinned ChampSim DPC4 source revision \texttt{{{tex_break_hash(source_revisions[0] if source_revisions else "unavailable")}}} \cite{{champsim}}, the tracked one-core full-bandwidth profile (SHA-256 prefix \texttt{{3c4306007d9c1d39}}), 50 million warmup instructions, and a requested 100 million-instruction ROI. Across the compared runs, actual retired ROI instructions range from {actual_roi_min:,} to {actual_roi_max:,}. The runner records each exact count, trace hash, effective-config hash, binary hash, and run ID. All prefetchers except the selected target are disabled.

The L2 comparison contains no prefetching, IP-stride, SPP \cite{{spp}}, VA-AMPM-lite \cite{{ampm}}, and Pythia \cite{{pythia}}, all at L2C against their workload-matched no-prefetch L2 controls. The Berti campaign runs only at L1D and uses its own matched L1D no-prefetch controls \cite{{berti}}. The feedback-directed variant is compared with original IP-stride on identical traces, profile, and ROI \cite{{fdp}}. Its two source files are fingerprinted as a project overlay on the same pinned ChampSim revision.

\section{{Metrics and evidence limits}}
L2 coverage is (matched no-prefetch L2 LOAD misses minus configured L2 LOAD misses) / matched no-prefetch L2 LOAD misses. Berti coverage uses the analogous L1D misses. Accuracy follows the project formula: useful-prefetch events / configured-level prefetch fills. The data preserve the numerator and denominator for each row. ChampSim increments useful events for demand merges onto outstanding prefetches, so this project-defined ratio can exceed one and is not a probability. IPC speedup is candidate ROI IPC / matched baseline ROI IPC. DRAM bytes and transactions are counted at the memory controller; request-type attribution reflects the type retained at service and does not reverse earlier demand/prefetch merges.

The bounded L2 sample retains at most 10,000 demand loads per run. PC-local counts expanded from the reservoir are estimates. A transition is called contiguous only when its global access-order gap is exactly one. The feedback controller's pending-address map only matches delayed use/eviction outcomes to issue-degree cohorts; it records no event timestamps or reusable request history. The campaign does not retain a per-request issue-to-fill-to-demand timeline or full predictor-state snapshot. Therefore late prefetching, predictor-table evictions, and policy-learning failures are discussed as unresolved where the measured evidence cannot separate them. Useful and useless counters are aggregate diagnostics, not address-level timeliness labels.

\section{{Same-level L2 results}}
Figs.~\ref{{fig:l2-ipc}}--\ref{{fig:l2-traffic}} summarize workload-level IPC and traffic outcomes; Table~\ref{{tab:l2-results}} gives the corresponding counters and ratios.

\begin{{figure*}}[!t]\centering
\includegraphics[width=0.95\textwidth]{{figures/l2-ipc-speedup.png}}
\caption{{Per-workload IPC speedup for the five L2 configurations against matched no-prefetch L2.}}\label{{fig:l2-ipc}}\end{{figure*}}

\begin{{figure*}}[!t]\centering
\includegraphics[width=0.95\textwidth]{{figures/l2-geomeans.png}}
\caption{{Geometric means within their comparison groups. Berti is evaluated at L1D; FDP is compared with original IP-stride.}}\label{{fig:l2-geomeans}}\end{{figure*}}

\begin{{figure*}}[!t]\centering
\includegraphics[width=0.95\textwidth]{{figures/l2-dram-bytes.png}}
\caption{{Total DRAM bytes relative to matched no-prefetch L2.}}\label{{fig:l2-traffic}}\end{{figure*}}

\begin{{table*}}[!t]
\caption{{Per-workload L2 results, DRAM transactions, and bytes}}\label{{tab:l2-results}}
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}p{{1.2cm}}p{{1.45cm}}rrrrrrr@{{}}}}
\toprule Workload & Config. & Base MPKI & Coverage & Accuracy & IPC speedup & Read tx & Write tx & DRAM MiB \\
\midrule
{l2_table(l2_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\section{{Separate L1D Berti scenario}}
Berti is evaluated at the cache level where its implementation observes demand accesses. Its coverage, accuracy, and speedup use L1D counters and an L1D no-prefetch baseline; those rows are not part of the L2 ranking. Table~\ref{{tab:berti}} summarizes this separate comparison.

\begin{{table*}}[!t]
\caption{{Berti at L1D against matched L1D no-prefetch}}\label{{tab:berti}}
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}p{{1.7cm}}rrrrrrrr@{{}}}}
\toprule Workload & Base MPKI & LOAD misses & Coverage & Accuracy & IPC speedup & Read tx & Write tx & DRAM MiB \\
\midrule
{berti_table(berti_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\section{{Feedback-directed IP-stride variant}}
The variant retains the original 256-set, four-way per-PC last-address/last-stride table, page-boundary check, MSHR-aware request choice, and three-line maximum lookahead. Its one controlled parameter is lookahead degree, clamped to 1--3. Accepted prefetches are tracked by cache-line address and tagged with the latest degree that submitted that still-pending address; duplicate submissions to one pending address count once. A demand use, eviction, or demand fill resolves that prefetch address episode. Every 4,096 resolved prefetch address episodes for one degree, the controller computes that cohort's useful fraction, reduces degree below 30\%, and raises it at or above 45\% only when that degree remains active. Counters span warmup and ROI; pending addresses at end of run are excluded. This is an FDP-inspired single-control response, not a reproduction of FDP's sampled lateness/pollution controller.

The signal choice follows the observed failure modes. IP-stride provides only {pct(omnet_baseline, 'coverage')} L2 coverage and {number(omnet_baseline, 'ipc_speedup'):.3f}$\times$ IPC on omnetpp, and {pct(mcf_baseline, 'coverage')} coverage with {number(mcf_baseline, 'ipc_speedup'):.3f}$\times$ IPC on mcf. Across the ten baseline runs, the aggregate useful-event/accepted-request diagnostic ranges from {lowest_feedback:.3f} ({WORKLOAD_LABELS[lowest_workload]}) to {highest_feedback:.3f} ({WORKLOAD_LABELS[highest_workload]}), supporting usefulness as a candidate feedback signal. The baseline did not record prefetch address episodes, so this aggregate ratio does not calibrate the 30\%/45\% thresholds. Those thresholds are initial low/high settings for evaluating the single degree control; they should not be read as empirically tuned cutoffs. The controller uses resolved prefetch address episodes grouped by the latest degree associated with each address. Its conceptual state is one degree register and per-degree outcome counters. The simulator uses a pending-address map to associate delayed use and eviction callbacks with the degree; this is measurement instrumentation, not a hardware design or area estimate. Cumulative totals are diagnostic.

The workload-level tradeoffs are shown in Fig.~\ref{{fig:fdp}} and Table~\ref{{tab:fdp}}.

\begin{{figure*}}[!t]\centering
\includegraphics[width=0.95\textwidth]{{figures/fdp-ipc-and-traffic.png}}
\caption{{Per-workload FDP-IP-stride IPC and traffic ratios against original IP-stride.}}\label{{fig:fdp}}\end{{figure*}}

\begin{{table*}}[!t]
\caption{{FDP-IP-stride relative to original IP-stride}}\label{{tab:fdp}}
\centering\scriptsize\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{@{{}}p{{1.35cm}}rrrrrrr@{{}}}}
\toprule Workload & \makecell{{Incremental\\coverage}} & Accuracy & \makecell{{Useful / resolved\\episodes}} & \makecell{{IPC\\ratio}} & \makecell{{DRAM byte\\ratio}} & \makecell{{Final\\degree}} & Up/down \\
\midrule
{fdp_table(fdp_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}

An outcome counts as an IPC improvement when its ratio exceeds 1.0. A traffic improvement counts only when DRAM bytes decrease and IPC does not fall. The table leaves higher-traffic IPC gains and lower-traffic IPC regressions visible as separate outcomes. Accuracy uses the project fill denominator. Useful/resolved prefetch address episodes is the fraction of resolved prefetch address episodes that were later used, counted once per pending address even when multiple accepted submissions target it; unresolved addresses are omitted. No late-prefetch ratio is inferred. Degree changes and counters are retained in the normalized campaign and raw run logs.

\section{{Mechanism-aware miss attribution}}
The analysis applies the PRISM-inspired cause names only where predictor mechanism and evidence can support them \cite{{prism}}. IP-stride, VA-AMPM, Pythia, Berti, and the FDP variant do not expose the prior-address correlation state required to distinguish unseen correlation from other limitations; those entries are marked not applicable. SPP can represent spatial signatures, so unseen-correlation is potentially applicable, but no per-miss table lookup was recorded. Compulsory misses need complete address history, late prefetches need address-linked timing, and metadata evictions need predictor-state history. Since none of those evidence streams was exported, all measured LOAD misses remain unresolved and zero are assigned a causal label. The per-workload file reports counts for every tested predictor and cache level in \url{{results/normalized/endsem/miss-attribution.json}}.

\section{{Workload-level diagnosis}}
For each workload the L2 winner is selected by measured IPC within the L2 group. Each explanation reports its coverage, accuracy, traffic, and prefetch counters; contrasts the remaining L2 configurations; relates those outcomes to the bounded stream evidence and predictor class; and preserves uncertainty where internal predictor state or timeliness was unavailable. Berti is reported separately at L1D.

{diagnosis}

\section{{Reproducibility artifacts}}
Normalized result tables, run identities, feedback summaries, and raw evidence archives are in \url{{results/normalized/}}. The L2, Pythia, Berti, and feedback-directed campaigns preserve trace names and SHA-256 values, effective-config identities, simulator revision/source fingerprints, actual ROI instruction counts, telemetry, and bounded L2 samples. The miss-attribution JSON records all considered causes and unresolved counts. LaTeX and compiled PDF are generated by \url{{experiments/generate_endsem_report.py}} from the normalized campaign directories.

\begin{{thebibliography}}{{9}}
\bibitem{{fdp}} S. Srinath, O. Mutlu, H. Kim, and Y. N. Patt, \emph{{Feedback Directed Prefetching: Improving the Performance and Bandwidth-Efficiency of Hardware Prefetchers}}, in Proceedings of the 13th International Symposium on High-Performance Computer Architecture (HPCA), pp. 63--74, 2007. doi: \url{{https://doi.org/10.1109/HPCA.2007.346185}}.
\bibitem{{ampm}} Y. Ishii, M. Inaba, and K. Hiraki, \emph{{Access Map Pattern Matching for High Performance Data Cache Prefetch}}, Journal of Instruction-Level Parallelism, vol. 13, pp. 1--24, 2011. \url{{https://jilp.org/vol13/v13paper3.pdf}}.
\bibitem{{spp}} S. Somogyi, T. F. Wenisch, A. Ailamaki, B. Falsafi, and A. Moshovos, \emph{{Spatial Memory Streaming}}, in Proceedings of the 33rd Annual International Symposium on Computer Architecture (ISCA), pp. 252--263, 2006. doi: \url{{https://doi.org/10.1109/ISCA.2006.38}}.
\bibitem{{berti}} A. Navarro-Torres, B. Panda, J. Alastruey-Benedé, P. Ibáñez, V. Viñals-Yúfera, and A. Ros, \emph{{Berti: An Accurate Local-Delta Data Prefetcher}}, in Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 975--991, 2022. doi: \url{{https://doi.org/10.1109/MICRO56248.2022.00072}}.
\bibitem{{pythia}} R. Bera, K. Kanellopoulos, A. V. Nori, T. Shahroodi, S. Subramoney, and O. Mutlu, \emph{{Pythia: A Customizable Hardware Prefetching Framework Using Online Reinforcement Learning}}, in Proceedings of the 54th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2021. doi: \url{{https://doi.org/10.1145/3466752.3480114}}.
\bibitem{{prism}} Y. Lin, J. Liu, Y. Chen, W. Lin, Z. Wang, and W. Chen, \emph{{PRISM: A Miss-Attribution-Guided Temporal Prefetcher Beyond Metadata Management}}, in Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2026. \url{{https://www.microarch.org/micro59/program/}}.
\bibitem{{champsim}} N. Gober, G. Chacon, L. Wang, P. V. Gratz, D. A. Jimenez, E. Teran, S. Pugsley, and J. Kim, \emph{{The Championship Simulator: Architectural Simulation for Education and Competition}}, arXiv:2210.14324, 2022. doi: \url{{https://doi.org/10.48550/arXiv.2210.14324}}. The DPC4 source revision used here is \texttt{{{tex_break_hash(source_revisions[0] if source_revisions else "unavailable")}}}: \url{{https://github.com/CMU-SAFARI/ChampSim/tree/{tex_escape(source_revisions[0] if source_revisions else "unavailable")}}}.
\end{{thebibliography}}
\end{{document}}
"""


def generate(data_root: Path, output_dir: Path, compile_pdf: bool) -> None:
    l2_rows = read_csv(data_root / "l2-required/metrics.csv")
    pythia_rows = read_csv(data_root / "pythia-l2/metrics.csv")
    l2_rows = l2_rows + pythia_rows
    berti_rows = read_csv(data_root / "berti-l1d/metrics.csv")
    fdp_rows = read_csv(data_root / "fdp-ip-stride/metrics.csv")
    access_doc = read_json(data_root / "l2-required/access-characterization.json")
    attribution_doc = read_json(data_root / "endsem/miss-attribution.json")
    if len(l2_rows) != 50 or len(pythia_rows) != 10 or len(berti_rows) != 10 or len(fdp_rows) != 10:
        raise ReportError("endsem report needs 50 L2, 10 Pythia, 10 Berti, and 10 FDP-IP-stride rows")
    if len(access_doc.get("workloads", [])) != 10 or len(attribution_doc.get("records", [])) != 60:
        raise ReportError("endsem report needs ten workload characterizations and 60 attribution records")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_l2(l2_rows, figures / "l2-ipc-speedup.png")
    plot_geomeans(l2_rows, berti_rows, fdp_rows, figures / "l2-geomeans.png")
    plot_l2_traffic(l2_rows, figures / "l2-dram-bytes.png")
    plot_fdp(fdp_rows, figures / "fdp-ipc-and-traffic.png")
    tex_path = output_dir / "endsem.tex"
    tex_path.write_text(make_tex(l2_rows, pythia_rows, berti_rows, fdp_rows, access_doc, attribution_doc, output_dir), encoding="utf-8")
    if compile_pdf:
        for _ in range(2):
            compiled = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "endsem.tex"],
                cwd=output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if compiled.returncode:
                raise ReportError(compiled.stdout[-6000:])
    print(f"Wrote endsem report source to {tex_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("results/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/endsem"))
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    try:
        generate(args.data_root, args.output_dir, compile_pdf=not args.no_pdf)
    except (ReportError, OSError, KeyError, ValueError, TypeError) as error:
        parser.exit(1, f"endsem report generation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
