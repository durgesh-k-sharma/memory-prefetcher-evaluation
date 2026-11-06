#!/usr/bin/env python3
"""Summarize bounded, ordered L2 load samples for workload characterization."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKLOADS = (
    "603.bwaves_s-891B",
    "619.lbm_s-4268B",
    "649.fotonik3d_s-1176B",
    "605.mcf_s-1554B",
    "620.omnetpp_s-141B",
    "623.xalancbmk_s-165B",
    "ligra-bfs-cc",
    "ligra-pagerankdelta",
    "ligra-triangle",
    "llama2",
)

PAGE_SIZE = 4096
LINE_SIZE = 64
TOP_PCS = 20
TOP_DELTAS = 8
TOP_SEQUENCES = 5


class AccessAnalysisError(Exception):
    """Raised when the expected bounded L2 access evidence is unavailable."""


def load_baseline_runs(raw_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith((".champsim.json", ".config.json", ".telemetry.json")):
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AccessAnalysisError(f"cannot read campaign result {path}: {error}") from error
        identity = result.get("identity", {})
        prefetcher = identity.get("prefetcher", {})
        if prefetcher.get("name") != "none" or prefetcher.get("level") != "L2C":
            continue
        workload = identity.get("workload")
        if workload in selected:
            raise AccessAnalysisError(f"duplicate no-prefetch L2 result for {workload}")
        selected[workload] = (path, result)
    missing = sorted(set(WORKLOADS) - set(selected))
    if missing:
        raise AccessAnalysisError(f"missing no-prefetch L2 samples for: {', '.join(missing)}")
    return selected


def read_sample(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise AccessAnalysisError(f"cannot read access sample {path}: {error}") from error
    if not rows:
        raise AccessAnalysisError(f"access sample is empty: {path}")

    parsed = []
    for row in rows:
        try:
            access_order = int(row["access_order"])
            pc = int(row["pc"], 16)
            address = int(row["cache_line_address"], 16)
        except (KeyError, ValueError) as error:
            raise AccessAnalysisError(f"invalid row in {path}: {row}") from error
        hit_miss = row.get("hit_miss")
        if access_order < 1 or pc < 0 or address < 0 or address % LINE_SIZE or hit_miss not in ("hit", "miss"):
            raise AccessAnalysisError(f"invalid L2 sample row in {path}: {row}")
        parsed.append({"access_order": access_order, "pc": pc, "address": address, "miss": hit_miss == "miss"})
    parsed.sort(key=lambda item: item["access_order"])
    return parsed


def pc_summary(rows: list[dict[str, Any]], total_accesses: int) -> list[dict[str, Any]]:
    by_pc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pc[row["pc"]].append(row)

    summaries = []
    for pc, pc_rows in by_pc.items():
        misses = sum(row["miss"] for row in pc_rows)
        pairs = []
        deltas = []
        for previous, current in zip(pc_rows, pc_rows[1:]):
            delta = current["address"] - previous["address"]
            gap = current["access_order"] - previous["access_order"]
            pairs.append({"delta_bytes": delta, "access_order_gap": gap})
            deltas.append((delta, gap))

        delta_counts = Counter(delta for delta, _gap in deltas)
        gap_counts = Counter(gap for _delta, gap in deltas)
        sequences = Counter(tuple(deltas[index : index + 3]) for index in range(max(0, len(deltas) - 2)))
        contiguous_deltas = [delta for delta, gap in deltas if gap == 1]
        contiguous_delta_counts = Counter(contiguous_deltas)
        contiguous_runs: list[list[int]] = []
        current_run: list[int] = []
        previous_was_contiguous = False
        for delta, gap in deltas:
            if gap == 1:
                if not previous_was_contiguous and current_run:
                    contiguous_runs.append(current_run)
                    current_run = []
                current_run.append(delta)
                previous_was_contiguous = True
            else:
                if current_run:
                    contiguous_runs.append(current_run)
                    current_run = []
                previous_was_contiguous = False
        if current_run:
            contiguous_runs.append(current_run)
        contiguous_sequences = Counter(
            tuple(run[index : index + 3])
            for run in contiguous_runs
            for index in range(max(0, len(run) - 2))
        )
        summaries.append(
            {
                "pc": f"0x{pc:x}",
                "sampled_loads": len(pc_rows),
                "sampled_misses": misses,
                "sampled_miss_fraction": misses / len(pc_rows),
                "estimated_loads_in_full_stream": len(pc_rows) * total_accesses / len(rows),
                "estimated_misses_in_full_stream": misses * total_accesses / len(rows),
                "sampled_pc_local_transitions": len(pairs),
                "sampled_pc_local_transitions_with_no_intervening_l2_loads": sum(pair["access_order_gap"] == 1 for pair in pairs),
                "sampled_transition_access_order_gap_histogram": {
                    str(gap): count for gap, count in sorted(gap_counts.items())
                },
                "dominant_sampled_transition_deltas": [
                    {
                        "delta_bytes": delta,
                        "sampled_transition_count": count,
                        "fraction_of_sampled_pc_local_transitions": count / len(pairs) if pairs else None,
                        "transitions_with_no_intervening_l2_loads": sum(
                            pair["delta_bytes"] == delta and pair["access_order_gap"] == 1 for pair in pairs
                        ),
                    }
                    for delta, count in delta_counts.most_common(TOP_DELTAS)
                ],
                "dominant_exact_contiguous_delta_bytes": [
                    {
                        "delta_bytes": delta,
                        "sampled_transition_count": count,
                        "fraction_of_exact_contiguous_pc_local_transitions": count / len(contiguous_deltas),
                    }
                    for delta, count in contiguous_delta_counts.most_common(TOP_DELTAS)
                ],
                "recurring_ordered_sample_transition_sequences": [
                    {
                        "delta_bytes_sequence": [delta for delta, _gap in sequence],
                        "access_order_gap_sequence": [gap for _delta, gap in sequence],
                        "sampled_sequence_count": count,
                    }
                    for sequence, count in sequences.most_common(TOP_SEQUENCES)
                    if count > 1
                ],
                "recurring_ordered_contiguous_delta_sequences": [
                    {"delta_bytes_sequence": list(sequence), "sampled_sequence_count": count}
                    for sequence, count in contiguous_sequences.most_common(TOP_SEQUENCES)
                    if count > 1
                ],
            }
        )
    return sorted(summaries, key=lambda item: (-item["sampled_misses"], -item["sampled_loads"], item["pc"]))


def page_footprints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_page: dict[tuple[int, int], set[int]] = defaultdict(set)
    offset_pages: dict[int, set[int]] = defaultdict(set)
    offset_samples: Counter[int] = Counter()
    for row in rows:
        page = row["address"] // PAGE_SIZE
        offset = row["address"] % PAGE_SIZE
        line_offset = offset // LINE_SIZE
        per_page[(row["pc"], page)].add(line_offset)
        offset_pages[line_offset].add(page)
        offset_samples[line_offset] += 1

    per_pc: dict[int, list[set[int]]] = defaultdict(list)
    for (pc, _page), lines in per_page.items():
        per_pc[pc].append(lines)

    return {
        "unique_sampled_pages": len({page for _pc, page in per_page}),
        "mean_distinct_cache_lines_per_pc_page": (
            sum(len(lines) for lines in per_page.values()) / len(per_page) if per_page else 0
        ),
        "top_recurring_line_offsets": [
            {
                "offset_bytes_within_4k_page": line_offset * LINE_SIZE,
                "sampled_accesses": offset_samples[line_offset],
                "distinct_pages": len(offset_pages[line_offset]),
            }
            for line_offset, _count in sorted(offset_samples.items(), key=lambda item: (-len(offset_pages[item[0]]), -item[1], item[0]))[:32]
        ],
        "top_pc_page_footprints": [
            {
                "pc": f"0x{pc:x}",
                "sampled_pages": len(page_sets),
                "mean_distinct_cache_lines_per_page": sum(map(len, page_sets)) / len(page_sets),
                "max_distinct_cache_lines_on_a_sampled_page": max(map(len, page_sets)),
            }
            for pc, page_sets in sorted(per_pc.items(), key=lambda item: (-len(item[1]), -sum(map(len, item[1]))))[:TOP_PCS]
        ],
    }


def characterize(workload: str, result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    sample_path = result_path.with_name(f"{result_path.stem}.l2-load-sample.csv")
    rows = read_sample(sample_path)
    sample_meta = result.get("normalized", {}).get("l2_load_sample", {})
    total_accesses = sample_meta.get("total_accesses")
    sample_limit = sample_meta.get("sample_limit")
    if type(total_accesses) is not int or total_accesses < len(rows) or type(sample_limit) is not int:
        raise AccessAnalysisError(f"invalid sample metadata in {result_path}")

    pcs = pc_summary(rows, total_accesses)
    return {
        "workload": workload,
        "trace": result["identity"]["trace"],
        "sample": {
            "cache_level": "L2C",
            "sample_method": "deterministic reservoir sample",
            "sample_limit": sample_limit,
            "sampled_accesses": len(rows),
            "total_accesses": total_accesses,
            "sampling_fraction": len(rows) / total_accesses if total_accesses else None,
            "note": "PC-local deltas and ordered histories join consecutive sampled observations; access_order_gap records how many L2 demand loads separate observations. Full-stream counts are estimates expanded by the reservoir sampling fraction.",
        },
        "sampled_loads": len(rows),
        "sampled_misses": sum(row["miss"] for row in rows),
        "estimated_total_load_misses": sum(row["miss"] for row in rows) * total_accesses / len(rows),
        "important_pcs": pcs[:TOP_PCS],
        "page_local_footprints": page_footprints(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        selected = load_baseline_runs(args.raw_dir)
        workloads = [characterize(name, *selected[name]) for name in WORKLOADS]
    except (AccessAnalysisError, OSError, KeyError, TypeError) as error:
        print(f"access-stream analysis failed: {error}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "cache_level": "L2C",
        "line_size_bytes": LINE_SIZE,
        "page_size_bytes": PAGE_SIZE,
        "workloads": workloads,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote access characterization for {len(workloads)} workloads to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
