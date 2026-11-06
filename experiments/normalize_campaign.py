#!/usr/bin/env python3
"""Normalize completed ChampSim campaigns and preserve their run evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable

from run_campaign import champsim_source_diff


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

SCENARIOS = {
    "l2-required": {"level": "L2C", "prefetchers": ("none", "ip_stride", "spp_dev", "va_ampm_lite")},
    "pythia-l2": {"level": "L2C", "prefetchers": ("pythia",)},
    "berti-l1d": {"level": "L1D", "prefetchers": ("berti",)},
    "fdp-ip-stride": {"level": "L2C", "prefetchers": ("ip_stride_fdp",)},
}

METRIC_DEFINITIONS = {
    "baseline_demand_mpki": "matched comparison-run LOAD misses / (matched ROI instructions / 1000)",
    "coverage": "(matched comparison-run LOAD misses - configured LOAD misses) / matched comparison-run LOAD misses",
    "accuracy": "configured-level useful prefetch events / configured-level prefetch fills",
    "ipc_speedup": "configured IPC / matched comparison-run IPC; IPC is ROI instructions / ROI cycles",
    "dram_transactions": "DRAM read transactions + DRAM write transactions during the ROI",
    "dram_bytes": "DRAM read bytes + DRAM write bytes during the ROI",
}


class CampaignDataError(Exception):
    """Raised when campaign outputs cannot support a comparable result."""


def result_files(directories: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            raise CampaignDataError(f"results directory does not exist: {directory}")
        for path in directory.glob("*.json"):
            if path.name.endswith((".champsim.json", ".config.json", ".telemetry.json")):
                continue
            paths.add(path.resolve())
    return sorted(paths)


def load_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignDataError(f"cannot read completed run {path}: {error}") from error
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise CampaignDataError(f"unsupported or incomplete result file: {path}")
    identity = result.get("identity")
    normalized = result.get("normalized")
    if not isinstance(identity, dict) or not isinstance(normalized, dict):
        raise CampaignDataError(f"result is missing its identity or normalized ROI: {path}")
    if not identity.get("roi", {}).get("actual_simulation_instructions"):
        raise CampaignDataError(f"result has no measured ROI: {path}")
    return result


def collect_runs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    runs = []
    for path in paths:
        result = load_result(path)
        run = {
            "path": path,
            "result": result,
            "workload": result["identity"]["workload"],
            "prefetcher": result["identity"]["prefetcher"]["name"],
            "level": result["identity"]["prefetcher"]["level"],
        }
        runs.append(run)
    return runs


def one_per_workload(runs: Iterable[dict[str, Any]], prefetcher: str, level: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run["prefetcher"] != prefetcher or run["level"] != level:
            continue
        workload = run["workload"]
        if workload in selected:
            raise CampaignDataError(f"duplicate {prefetcher}@{level} result for {workload}")
        selected[workload] = run
    missing = sorted(set(WORKLOADS) - set(selected))
    unexpected = sorted(set(selected) - set(WORKLOADS))
    if missing or unexpected:
        raise CampaignDataError(f"{prefetcher}@{level} workload mismatch; missing={missing}, unexpected={unexpected}")
    return selected


def level_stats(run: dict[str, Any], level: str) -> dict[str, Any]:
    cache_name = f"cpu0_{level}"
    stats = run["result"]["normalized"].get(cache_name)
    if not isinstance(stats, dict):
        raise CampaignDataError(f"{run['path']} has no {cache_name} statistics")
    load = stats.get("LOAD")
    misses = load.get("miss") if isinstance(load, dict) else None
    if not isinstance(misses, list) or len(misses) != 1 or type(misses[0]) is not int:
        raise CampaignDataError(f"{run['path']} has invalid {cache_name} LOAD miss counts")
    return stats


def ensure_matched(candidate: dict[str, Any], baseline: dict[str, Any], *, allow_source_extension: bool = False) -> None:
    candidate_id = candidate["result"]["identity"]
    baseline_id = baseline["result"]["identity"]
    if candidate_id["trace"].get("sha256") != baseline_id["trace"].get("sha256"):
        raise CampaignDataError(f"{candidate['workload']} candidate and baseline use different traces")
    if candidate_id["profile"].get("sha256") != baseline_id["profile"].get("sha256"):
        raise CampaignDataError(f"{candidate['workload']} candidate and baseline use different profiles")
    if allow_source_extension:
        validate_fdp_source_extension(candidate_id, baseline_id, candidate["workload"])
    else:
        for field in ("source_revision", "source_diff_sha256"):
            if candidate_id["simulator"].get(field) != baseline_id["simulator"].get(field):
                raise CampaignDataError(f"{candidate['workload']} candidate and baseline use different simulator sources")
    roi_fields = ("warmup_instructions", "simulation_instructions", "measured_instruction_start", "measured_instruction_end")
    if any(candidate_id["roi"].get(key) != baseline_id["roi"].get(key) for key in roi_fields):
        raise CampaignDataError(f"{candidate['workload']} candidate and baseline use different ROIs")
    if candidate["workload"] != baseline["workload"]:
        raise CampaignDataError("candidate and baseline workloads do not match")


def validate_fdp_source_extension(candidate_id: dict[str, Any], baseline_id: dict[str, Any], workload: str) -> None:
    source_dir = Path(__file__).resolve().parents[1] / "third_party" / "champsim-dpc4"
    candidate_revision = candidate_id["simulator"].get("source_revision")
    baseline_revision = baseline_id["simulator"].get("source_revision")
    candidate_diff = candidate_id["simulator"].get("source_diff_sha256")
    baseline_diff = baseline_id["simulator"].get("source_diff_sha256")
    extension_dir = Path(__file__).resolve().parent / "champsim_modules" / "ip_stride_fdp"
    expected_files = {"ip_stride_fdp.cc", "ip_stride_fdp.h"}
    try:
        current = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        tracked_changes = subprocess.run(
            ["git", "-C", str(source_dir), "diff", "--quiet", "HEAD"], check=False,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CampaignDataError(f"cannot verify the FDP source extension: {error}") from error
    try:
        current_diff = champsim_source_diff(source_dir, "ip_stride_fdp")
    except OSError as error:
        raise CampaignDataError(f"cannot fingerprint the FDP source extension: {error}") from error
    if (
        not candidate_revision
        or candidate_revision != baseline_revision
        or current != candidate_revision
        or tracked_changes.returncode != 0
        or candidate_diff != current_diff
        or baseline_diff != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        or not extension_dir.is_dir()
        or {path.name for path in extension_dir.iterdir() if path.is_file()} != expected_files
    ):
        raise CampaignDataError(f"{workload} FDP source must be the reproducible two-file project extension of original IP-stride")


def normalized_row(
    candidate: dict[str, Any], baseline: dict[str, Any], level: str, *, allow_source_extension: bool = False
) -> dict[str, Any]:
    ensure_matched(candidate, baseline, allow_source_extension=allow_source_extension)
    candidate_roi = candidate["result"]["normalized"]
    baseline_roi = baseline["result"]["normalized"]
    candidate_cache = level_stats(candidate, level)
    baseline_cache = level_stats(baseline, level)
    demand_misses = candidate_cache["LOAD"]["miss"][0]
    baseline_misses = baseline_cache["LOAD"]["miss"][0]
    fills = candidate_cache.get("prefetch fill")
    useful = candidate_cache.get("useful prefetch")
    if type(fills) is not int or type(useful) is not int:
        raise CampaignDataError(f"{candidate['path']} has invalid {level} prefetch counters")

    instructions = candidate_roi.get("instructions")
    baseline_instructions = baseline_roi.get("instructions")
    ipc = candidate_roi.get("ipc")
    baseline_ipc = baseline_roi.get("ipc")
    if type(instructions) is not int or type(baseline_instructions) is not int:
        raise CampaignDataError(f"{candidate['path']} has invalid ROI instruction counts")
    if not isinstance(ipc, (int, float)) or not isinstance(baseline_ipc, (int, float)) or ipc <= 0 or baseline_ipc <= 0:
        raise CampaignDataError(f"{candidate['path']} has invalid IPC values")

    dram = candidate_roi.get("dram_traffic")
    baseline_dram = baseline_roi.get("dram_traffic")
    if not isinstance(dram, dict) or not isinstance(baseline_dram, dict):
        raise CampaignDataError(f"{candidate['path']} has no normalized DRAM traffic")
    read_transactions = dram["read_transactions"]
    write_transactions = dram["write_transactions"]
    read_bytes = dram["read_bytes"]
    write_bytes = dram["write_bytes"]
    coverage_numerator = baseline_misses - demand_misses
    coverage = coverage_numerator / baseline_misses if baseline_misses else None
    accuracy = useful / fills if fills else None
    baseline_mpki_denominator = baseline_instructions / 1000

    row = {
        "scenario": None,
        "workload": candidate["workload"],
        "prefetcher": candidate["prefetcher"],
        "cache_level": level,
        "baseline_prefetcher": baseline["prefetcher"],
        "baseline_demand_mpki": baseline_misses / baseline_mpki_denominator,
        "baseline_demand_misses": baseline_misses,
        "baseline_mpki_denominator_instructions_div_1000": baseline_mpki_denominator,
        "demand_misses": demand_misses,
        "coverage": coverage,
        "coverage_numerator_baseline_misses_minus_configured_misses": coverage_numerator,
        "coverage_denominator_baseline_demand_misses": baseline_misses,
        "useful_prefetch_events": useful,
        "prefetch_fills": fills,
        "accuracy": accuracy,
        "accuracy_numerator_useful_prefetch_events": useful,
        "accuracy_denominator_prefetch_fills": fills,
        "ipc": ipc,
        "baseline_ipc": baseline_ipc,
        "ipc_speedup": ipc / baseline_ipc,
        "ipc_speedup_numerator_candidate_ipc": ipc,
        "ipc_speedup_denominator_baseline_ipc": baseline_ipc,
        "roi_instructions": instructions,
        "roi_cycles": candidate_roi["cycles"],
        "read_transactions": read_transactions,
        "write_transactions": write_transactions,
        "dram_transactions": read_transactions + write_transactions,
        "demand_read_transactions": dram.get("demand_read_transactions"),
        "prefetch_read_transactions": dram.get("prefetch_read_transactions"),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "dram_bytes": read_bytes + write_bytes,
        "baseline_read_bytes": baseline_dram["read_bytes"],
        "baseline_write_bytes": baseline_dram["write_bytes"],
        "baseline_dram_bytes": baseline_dram["read_bytes"] + baseline_dram["write_bytes"],
        "demand_read_bytes": dram.get("demand_read_bytes"),
        "prefetch_read_bytes": dram.get("prefetch_read_bytes"),
        "dram_attribution_basis": dram.get("attribution_basis"),
        "trace_sha256": candidate["result"]["identity"]["trace"]["sha256"],
        "configuration_sha256": candidate["result"]["identity"]["configuration"]["sha256"],
        "simulator_sha256": candidate["result"]["identity"]["simulator"]["sha256"],
        "candidate_run_id": candidate["result"]["run_id"],
        "baseline_run_id": baseline["result"]["run_id"],
        "candidate_result_path": str(candidate["path"]),
        "baseline_result_path": str(baseline["path"]),
    }
    feedback = candidate["result"]["normalized"].get("prefetcher_feedback", {})
    row.update(
        {
            "feedback_signal": feedback.get("signal"),
            "feedback_control": feedback.get("primary_control"),
            "feedback_final_degree": feedback.get("final_degree"),
            "feedback_degree_increases": feedback.get("degree_increases"),
            "feedback_degree_decreases": feedback.get("degree_decreases"),
            "feedback_windows": feedback.get("feedback_windows"),
            "feedback_total_useful_events": feedback.get("total_useful_events"),
            "feedback_useful_address_episodes": feedback.get("useful_address_episodes"),
            "feedback_accepted_prefetch_requests": feedback.get("accepted_prefetch_requests"),
            "feedback_tracked_address_episodes": feedback.get("tracked_address_episodes"),
            "feedback_resolved_address_episodes": feedback.get("resolved_address_episodes"),
            "feedback_batch_address_episodes": feedback.get("feedback_batch_address_episodes"),
            "feedback_useful_per_resolved_address_episode": (
                feedback.get("useful_address_episodes") / feedback.get("resolved_address_episodes")
                if feedback.get("resolved_address_episodes")
                else None
            ),
        }
    )
    return row


CSV_FIELDS = (
    "scenario", "workload", "prefetcher", "cache_level", "baseline_prefetcher",
    "baseline_demand_mpki", "baseline_demand_misses", "baseline_mpki_denominator_instructions_div_1000",
    "demand_misses", "coverage", "coverage_numerator_baseline_misses_minus_configured_misses",
    "coverage_denominator_baseline_demand_misses", "useful_prefetch_events", "prefetch_fills", "accuracy",
    "accuracy_numerator_useful_prefetch_events", "accuracy_denominator_prefetch_fills", "ipc", "baseline_ipc",
    "ipc_speedup", "ipc_speedup_numerator_candidate_ipc", "ipc_speedup_denominator_baseline_ipc",
    "roi_instructions", "roi_cycles", "read_transactions", "write_transactions", "dram_transactions",
    "demand_read_transactions", "prefetch_read_transactions", "read_bytes", "write_bytes", "dram_bytes",
    "baseline_read_bytes", "baseline_write_bytes", "baseline_dram_bytes",
    "demand_read_bytes", "prefetch_read_bytes", "dram_attribution_basis", "trace_sha256",
    "configuration_sha256", "simulator_sha256", "candidate_run_id", "baseline_run_id",
    "candidate_result_path", "baseline_result_path",
    "feedback_signal", "feedback_control", "feedback_final_degree", "feedback_degree_increases",
    "feedback_degree_decreases", "feedback_windows", "feedback_total_useful_events", "feedback_useful_address_episodes",
    "feedback_accepted_prefetch_requests", "feedback_tracked_address_episodes", "feedback_resolved_address_episodes",
    "feedback_batch_address_episodes", "feedback_useful_per_resolved_address_episode",
)


def add_evidence(archive: tarfile.TarFile, run: dict[str, Any], prefix: str) -> bool:
    path: Path = run["path"]
    stem = path.stem
    required_files = (
        path,
        path.with_name(f"{stem}.telemetry.json"),
        path.with_name(f"{stem}.config.json"),
        path.with_name(f"{stem}.l2-load-sample.csv"),
    )
    for evidence in required_files:
        if not evidence.is_file():
            raise CampaignDataError(f"completed run is missing evidence file: {evidence}")
        archive.add(evidence, arcname=f"{prefix}/{run['workload']}/{run['prefetcher']}/{evidence.name}")
    raw_champsim = path.with_name(f"{stem}.champsim.json")
    if raw_champsim.is_file():
        archive.add(raw_champsim, arcname=f"{prefix}/{run['workload']}/{run['prefetcher']}/{raw_champsim.name}")
        return True
    return False


def run_normalization(args: argparse.Namespace) -> None:
    scenario = args.scenario
    level = SCENARIOS[scenario]["level"]
    expected_prefetchers = SCENARIOS[scenario]["prefetchers"]
    candidates = collect_runs(result_files(args.candidate_dir))
    baselines = collect_runs(result_files(args.baseline_dir))

    if scenario == "l2-required":
        baselines = candidates
    elif scenario == "fdp-ip-stride":
        pass
    elif scenario == "berti-l1d":
        all_available = candidates + [run for run in baselines if run["path"] not in {c["path"] for c in candidates}]
        baselines = all_available

    rows = []
    selected_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for prefetcher in expected_prefetchers:
        selected_candidates[prefetcher] = one_per_workload(candidates, prefetcher, level)

    baseline_prefetcher = "ip_stride" if scenario == "fdp-ip-stride" else "none"
    selected_baseline = one_per_workload(baselines, baseline_prefetcher, level)
    for prefetcher in expected_prefetchers:
        for workload in WORKLOADS:
            if scenario == "fdp-ip-stride" and not selected_candidates[prefetcher][workload]["result"]["normalized"].get("prefetcher_feedback"):
                raise CampaignDataError(f"{prefetcher} result is missing its measured feedback summary")
            row = normalized_row(
                selected_candidates[prefetcher][workload],
                selected_baseline[workload],
                level,
                allow_source_extension=scenario == "fdp-ip-stride",
            )
            row["scenario"] = scenario
            rows.append(row)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    runs_document = {
        "schema_version": 1,
        "scenario": scenario,
        "comparison_cache_level": level,
        "workloads": list(WORKLOADS),
        "prefetchers": list(expected_prefetchers),
        "metric_definitions": METRIC_DEFINITIONS,
        "runs": rows,
    }
    included: dict[Path, dict[str, Any]] = {}
    for run_map in selected_candidates.values():
        included.update((run["path"], run) for run in run_map.values())
    included.update((run["path"], run) for run in selected_baseline.values())

    identities = {
        run["result"]["run_id"]: run["result"]["identity"]
        for run in included.values()
    }
    missing_raw_phase_json = []
    with tarfile.open(output_dir / "raw-evidence.tar.gz", "w:gz") as archive:
        for run in included.values():
            if not add_evidence(archive, run, "runs"):
                missing_raw_phase_json.append(run["result"]["run_id"])

    runs_document["run_identities"] = identities
    runs_document["raw_evidence_archive"] = "raw-evidence.tar.gz"
    runs_document["missing_full_champsim_phase_json_run_ids"] = missing_raw_phase_json
    (output_dir / "runs.json").write_text(json.dumps(runs_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = (
        f"# {scenario} normalized campaign\n\n"
        f"This directory contains {len(rows)} comparable rows for the ten fixed workloads.\n\n"
        "- `metrics.csv` and `runs.json` contain normalized metrics and explicit numerator/denominator fields.\n"
        "- `raw-evidence.tar.gz` preserves each included run's result, telemetry, effective configuration, and bounded L2 load sample; full ChampSim phase JSON is included when available.\n"
        f"- Comparisons use {level} demand LOAD misses and a matched {baseline_prefetcher} run with the same trace, profile, warmup, and requested ROI.\n\n"
        "Accuracy follows the project formula (useful prefetch events / prefetch fills). ChampSim counts demand merges onto outstanding prefetches as useful events, so this ratio can exceed 1; interpret it with the preserved raw counters rather than as a probability.\n"
    )
    if scenario == "fdp-ip-stride":
        readme += (
            "\nThe candidate uses a two-file project source extension to the same pinned ChampSim revision as original IP-stride. The extension content is included in its source fingerprint. The selected primary control is lookahead degree; it reacts to the recent useful-event-to-accepted-request ratio. Feedback summary counters span warmup and ROI, while performance and cache metrics are ROI-only.\n"
        )
    if missing_raw_phase_json:
        readme += (
            f"\nFull ChampSim phase JSON was absent for {len(missing_raw_phase_json)} run(s). Their measured normalized ROI, identity, telemetry, and sample remain in the archive; see `missing_full_champsim_phase_json_run_ids` in `runs.json`.\n"
        )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(rows)} normalized rows to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--baseline-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_normalization(args)
    except (CampaignDataError, OSError, tarfile.TarError) as error:
        print(f"campaign normalization failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
