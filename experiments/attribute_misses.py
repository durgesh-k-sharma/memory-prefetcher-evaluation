#!/usr/bin/env python3
"""Emit conservative, mechanism-aware miss-attribution evidence for the endsem report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREDICTOR_MECHANISMS = {
    "ip_stride": {
        "level": "L2C",
        "mechanism": "Per-PC last-address and last-stride tracker with an LRU replacement table and degree-three lookahead.",
        "prism_correlation_model": "not_applicable",
        "reason": "It predicts a constant stride from per-PC state rather than consulting a learned prior-address-to-current-address correlation table.",
    },
    "ip_stride_fdp": {
        "level": "L2C",
        "mechanism": "Per-PC last-address and last-stride tracker with LRU replacement; measured usefulness feedback adjusts lookahead degree from one to three.",
        "prism_correlation_model": "not_applicable",
        "reason": "Degree feedback changes stride aggressiveness but does not add prior-address-to-current-address correlation state.",
    },
    "spp_dev": {
        "level": "L2C",
        "mechanism": "Signature path history and pattern tables predict spatially correlated cache-line offsets.",
        "prism_correlation_model": "potentially_applicable",
        "reason": "Its signature and pattern state can represent learned spatial correlations, but the campaign did not export per-miss table lookups or evictions.",
    },
    "va_ampm_lite": {
        "level": "L2C",
        "mechanism": "Page-local access maps match spatial footprints without indexing the prediction by load PC or ordered address history.",
        "prism_correlation_model": "not_applicable",
        "reason": "Its predictor state represents unordered page footprints rather than prior-address correlations.",
    },
    "pythia": {
        "level": "L2C",
        "mechanism": "A feature/action policy chooses page-relative prefetch offsets and updates reinforcement-learning state from delayed rewards.",
        "prism_correlation_model": "not_applicable",
        "reason": "Its feature-conditioned action/Q-value state is not a PRISM-style prior-address correlation table.",
    },
    "berti": {
        "level": "L1D",
        "mechanism": "A local-delta predictor tracks ordered address-difference patterns at L1D.",
        "prism_correlation_model": "not_applicable",
        "reason": "It observes local-delta histories at L1D and is not the L2 prior-address correlation mechanism used by the PRISM taxonomy.",
    },
}

CAUSE_DEFINITIONS = {
    "compulsory_miss": "The missed address is being accessed for the first time.",
    "unseen_correlation": "The predictor has not learned the needed prior-address-to-current-address correlation.",
    "late_prefetch": "A correct prefetch exists but has not arrived before the demand.",
    "metadata_eviction": "The needed correlation was learned and later evicted from predictor metadata.",
}

CAUSE_SUPPORT = {
    "compulsory_miss": {
        "status": "potentially_applicable",
        "evidence_required": "Complete address history for the measured ROI.",
    },
    "unseen_correlation": {
        "status": "predictor_specific",
        "evidence_required": "A per-miss lookup and training record for the predictor's learned state.",
    },
    "late_prefetch": {
        "status": "potentially_applicable",
        "evidence_required": "Address-linked prefetch issue, fill, and demand-arrival times.",
    },
    "metadata_eviction": {
        "status": "predictor_specific",
        "evidence_required": "The relevant predictor-state entry and its eviction/replacement history.",
    },
}


class AttributionError(Exception):
    """Raised when completed run evidence is unavailable or inconsistent."""


def load_runs(directories: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for directory in directories:
        if not directory.is_dir():
            raise AttributionError(f"results directory does not exist: {directory}")
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith((".champsim.json", ".config.json", ".telemetry.json")):
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise AttributionError(f"cannot read result {path}: {error}") from error
            identity = result.get("identity", {})
            prefetcher = identity.get("prefetcher", {})
            if prefetcher.get("name") == "none":
                continue
            if not result.get("normalized") or not identity.get("roi", {}).get("actual_simulation_instructions"):
                raise AttributionError(f"run is incomplete: {path}")
            runs.append({"path": path, "result": result})
    return runs


def classify_evidence(run: dict[str, Any]) -> dict[str, Any]:
    result = run["result"]
    identity = result["identity"]
    prefetcher = identity["prefetcher"]["name"]
    level = identity["prefetcher"]["level"]
    stats = result["normalized"].get(f"cpu0_{level}")
    if not isinstance(stats, dict) or not isinstance(stats.get("LOAD"), dict):
        raise AttributionError(f"missing {level} demand counters in {run['path']}")
    misses = stats["LOAD"].get("miss")
    if not isinstance(misses, list) or len(misses) != 1 or type(misses[0]) is not int:
        raise AttributionError(f"invalid {level} LOAD miss counter in {run['path']}")

    causes = {}
    for cause, definition in CAUSE_DEFINITIONS.items():
        correlation_model = PREDICTOR_MECHANISMS[prefetcher]["prism_correlation_model"]
        if cause in ("unseen_correlation", "metadata_eviction") and correlation_model == "not_applicable":
            status = "not_applicable"
            reason = (
                "The exact PRISM label concerns learned prior-address correlation state. This predictor uses a different state model; "
                "its predictor-specific state loss is not renamed as correlation eviction."
            )
        elif cause in ("unseen_correlation", "metadata_eviction"):
            status = "unresolved"
            reason = (
                "The mechanism can represent learned spatial correlations, but per-miss lookup, training, and replacement events were not exported."
            )
        else:
            status = "unresolved"
            reason = {
                "compulsory_miss": "No complete ROI-wide address-history set was recorded; the bounded reservoir sample cannot establish first touch.",
                "unseen_correlation": "Per-miss predictor lookup state was not exported, so absent training cannot be distinguished from a different predictor limit.",
                "late_prefetch": "Aggregate issue/fill/use counters are available, but no address-linked request-to-fill-to-demand timeline was recorded.",
                "metadata_eviction": "Predictor table state and eviction history were not exported for the measured ROI.",
            }[cause]
        causes[cause] = {
            "status": status,
            "definition": definition,
            "reason": reason,
            "mechanism_applicability": "not_applicable_to_this_predictor_state_model"
            if status == "not_applicable"
            else "potentially_applicable_to_this_correlation_state"
            if cause in ("unseen_correlation", "metadata_eviction")
            else "potentially_applicable_with_required_instrumentation",
            "evidence_required": CAUSE_SUPPORT[cause]["evidence_required"],
        }

    return {
        "workload": identity["workload"],
        "prefetcher": prefetcher,
        "cache_level": level,
        "run_id": result["run_id"],
        "trace_sha256": identity["trace"]["sha256"],
        "configuration_sha256": identity["configuration"]["sha256"],
        "demand_LOAD_misses": misses[0],
        "classified_misses": 0,
        "unresolved_demand_LOAD_misses": misses[0],
        "unresolved_fraction": 1.0 if misses[0] else 0.0,
        "aggregate_prefetch_counters": {
            "requested": stats.get("prefetch requested"),
            "issued": stats.get("prefetch issued"),
            "fills": stats.get("prefetch fill"),
            "useful_events": stats.get("useful prefetch"),
            "useless_evictions": stats.get("useless prefetch"),
            "address_linkage_available": False,
            "predictor_state_snapshot_available": False,
        },
        "causes": causes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--pythia-dir", type=Path, required=True)
    parser.add_argument("--berti-dir", type=Path, required=True)
    parser.add_argument("--fdp-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        directories = [args.l2_dir, args.pythia_dir, args.berti_dir]
        if args.fdp_dir is not None:
            directories.append(args.fdp_dir)
        runs = load_runs(directories)
        records = [classify_evidence(run) for run in runs]
    except (AttributionError, KeyError, TypeError) as error:
        print(f"miss attribution failed: {error}")
        return 1

    document = {
        "schema_version": 1,
        "taxonomy": "PRISM-inspired, mechanism-aware; labels are emitted only when supported by event and state evidence.",
        "cause_definitions": CAUSE_DEFINITIONS,
        "cause_support_requirements": CAUSE_SUPPORT,
        "predictor_mechanisms": PREDICTOR_MECHANISMS,
        "evidence_limits": [
            "The bounded access reservoir is not a complete address-history set.",
            "The measured counters aggregate prefetch events and cannot link a prefetch to a particular demand miss.",
            "Predictor training state, lookup outcomes, and metadata evictions were not exported by these campaigns.",
            "No PRISM-specific causal label is assigned from access-sample or coverage data alone.",
        ],
        "records": sorted(records, key=lambda item: (item["cache_level"], item["workload"], item["prefetcher"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote conservative attribution summaries for {len(records)} measured configurations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
