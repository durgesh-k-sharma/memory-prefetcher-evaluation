#!/usr/bin/env python3
"""Run one reproducible, single-workload ChampSim experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


WARMUP_INSTRUCTIONS = 50_000_000
SIMULATION_INSTRUCTIONS = 100_000_000
L2_LOAD_SAMPLE_LIMIT = 10_000
CACHE_LEVELS = ("L1I", "L1D", "L2C", "LLC")
DRAM_REQUEST_TYPES = ("LOAD", "RFO", "PREFETCH", "WRITE", "TRANSLATION")
PREFETCHER_LEVELS = {
    "ip_stride": "L2C",
    "ip_stride_fdp": "L2C",
    "spp_dev": "L2C",
    "va_ampm_lite": "L2C",
    "pythia": "L2C",
    "berti": "L1D",
}
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


class CampaignError(Exception):
    """An invalid campaign input or failed ChampSim run."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"expected a JSON object in {path}")
    return value


def selected_level(prefetcher: str, requested_level: str | None) -> str:
    if prefetcher == "none":
        if requested_level is None:
            raise CampaignError("--level is required when --prefetcher is none")
        return requested_level

    expected_level = PREFETCHER_LEVELS[prefetcher]
    if requested_level is not None and requested_level != expected_level:
        raise CampaignError(f"{prefetcher} is configured at {expected_level}, not {requested_level}")
    return expected_level


def make_effective_config(profile: dict[str, Any], prefetcher: str, level: str, profile_sha256: str) -> dict[str, Any]:
    cores = profile.get("ooo_cpu")
    if profile.get("num_cores") != 1 or not isinstance(cores, list) or len(cores) != 1:
        raise CampaignError("the campaign profile must describe exactly one core")

    config = json.loads(json.dumps(profile))
    for cache_name in CACHE_LEVELS:
        cache = config.get(cache_name)
        if not isinstance(cache, dict) or "prefetcher" not in cache:
            raise CampaignError(f"the campaign profile must define a prefetcher for {cache_name}")
        cache["prefetcher"] = "no"

    if prefetcher != "none":
        config[level]["prefetcher"] = prefetcher

    signature = digest_json(
        {
            "profile_sha256": profile_sha256,
            "prefetcher": prefetcher,
            "level": level,
        }
    )[:12]
    safe_prefetcher = prefetcher.replace("-", "_")
    config["executable_name"] = f"stridewise_{safe_prefetcher}_{signature}"
    return config


def champsim_revision(source_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def champsim_source_diff(source_dir: Path, source_extension: str | None = None) -> str | None:
    try:
        diff = subprocess.run(
            ["git", "-C", str(source_dir), "diff", "HEAD", "--binary"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if source_extension is not None:
        extension_dir = Path(__file__).resolve().parent / "champsim_modules" / source_extension
        if not extension_dir.is_dir():
            raise CampaignError(f"ChampSim source extension is unavailable: {extension_dir}")
        parts = [diff]
        for path in sorted(item for item in extension_dir.rglob("*") if item.is_file()):
            parts.extend((b"\0", path.relative_to(extension_dir).as_posix().encode("utf-8"), b"\0", path.read_bytes()))
        diff = b"".join(parts)
    return hashlib.sha256(diff).hexdigest()


def build_simulator(
    source_dir: Path, config_path: Path, config_sha256: str, output_dir: Path, source_extension: str | None = None
) -> tuple[Path, str | None, str | None]:
    source_dir = source_dir.resolve()
    if not (source_dir / "config.sh").is_file() or not (source_dir / "Makefile").is_file():
        raise CampaignError(
            f"ChampSim source is unavailable at {source_dir}; initialize the champsim-dpc4 submodule or pass --simulator"
        )

    revision = champsim_revision(source_dir)
    source_diff = champsim_source_diff(source_dir, source_extension)
    source_key = revision or sha256_file(source_dir / "README.md")
    build_key = digest_json({"configuration": config_sha256, "source": source_key, "source_diff": source_diff})[:20]
    build_root = output_dir / ".campaign-builds" / build_key
    copied_source = build_root / "champsim"
    binary_path = copied_source / "bin" / read_json(config_path)["executable_name"]

    if not binary_path.is_file():
        if not copied_source.exists():
            build_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_dir,
                copied_source,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "bin",
                    ".csconfig",
                    "vcpkg_installed",
                    "__pycache__",
                    ".cache",
                    "*.o",
                    "*.d",
                ),
            )
            installed_vcpkg = source_dir / "vcpkg_installed"
            if installed_vcpkg.is_dir():
                (copied_source / "vcpkg_installed").symlink_to(installed_vcpkg.resolve(), target_is_directory=True)

        if source_extension is not None:
            extension_source = Path(__file__).resolve().parent / "champsim_modules" / source_extension
            extension_target = copied_source / "prefetcher" / source_extension
            shutil.copytree(extension_source, extension_target, dirs_exist_ok=True)

        configure = subprocess.run(
            [sys.executable, str(copied_source / "config.sh"), "--prefix", str(copied_source), str(config_path)],
            cwd=copied_source,
            capture_output=True,
            text=True,
        )
        if configure.returncode != 0:
            raise CampaignError(f"ChampSim configuration failed:\n{configure.stderr.strip() or configure.stdout.strip()}")

        build = subprocess.run(["make", "-j1"], cwd=copied_source, capture_output=True, text=True)
        if build.returncode != 0:
            raise CampaignError(f"ChampSim build failed:\n{build.stderr.strip() or build.stdout.strip()}")

    if not binary_path.is_file():
        raise CampaignError(f"ChampSim build did not produce {binary_path}")
    return binary_path, revision, source_diff


def simulation_roi(champsim_output: Any, retire_width: int) -> dict[str, Any]:
    if not isinstance(champsim_output, list):
        raise CampaignError("ChampSim JSON output must be an array of phases")
    simulation_phase = next(
        (phase for phase in champsim_output if isinstance(phase, dict) and phase.get("name") == "Simulation"),
        None,
    )
    if simulation_phase is None or not isinstance(simulation_phase.get("roi"), dict):
        raise CampaignError("ChampSim JSON output is missing the Simulation ROI statistics")

    roi = simulation_phase["roi"]
    cores = roi.get("cores")
    if not isinstance(cores, list) or len(cores) != 1 or not isinstance(cores[0], dict):
        raise CampaignError("ChampSim ROI output must contain statistics for exactly one core")
    actual = cores[0].get("instructions")
    if type(actual) is not int or actual < SIMULATION_INSTRUCTIONS or actual >= SIMULATION_INSTRUCTIONS + retire_width:
        raise CampaignError(
            f"ChampSim measured {actual} instructions; expected {SIMULATION_INSTRUCTIONS} with at most "
            f"{retire_width - 1} retire-boundary instructions"
        )
    if not isinstance(cores[0].get("cycles"), (int, float)) or cores[0]["cycles"] <= 0:
        raise CampaignError("ChampSim ROI output is missing a positive core cycle count")
    return roi


def normalized_roi(roi: dict[str, Any]) -> dict[str, Any]:
    core = roi["cores"][0]
    normalized: dict[str, Any] = {
        "instructions": core["instructions"],
        "cycles": core["cycles"],
        "ipc": core["instructions"] / core["cycles"],
    }
    for name, value in roi.items():
        if name != "cores":
            normalized[name] = value
    return normalized


def l2_load_sample(roi: dict[str, Any], cache_line_size: int) -> tuple[str, int, int, list[dict[str, Any]]]:
    l2_caches = [(name, stats) for name, stats in roi.items() if (name == "L2C" or name.endswith("_L2C")) and isinstance(stats, dict)]
    if len(l2_caches) != 1:
        raise CampaignError("ChampSim ROI output is missing L2C access-sample statistics")
    cache_name, l2_stats = l2_caches[0]
    sample_limit = l2_stats.get("load_access_sample_limit")
    total_accesses = l2_stats.get("load_access_sample_total")
    sample = l2_stats.get("load_access_sample")
    if type(sample_limit) is not int or sample_limit <= 0:
        raise CampaignError("ChampSim ROI output has an invalid L2 load sample limit")
    if type(total_accesses) is not int or total_accesses < 0:
        raise CampaignError("ChampSim ROI output has an invalid L2 load access count")
    if not isinstance(sample, list) or len(sample) > sample_limit:
        raise CampaignError("ChampSim ROI output has an invalid or unbounded L2 load sample")

    rows: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    for item in sample:
        if not isinstance(item, dict):
            raise CampaignError("ChampSim L2 load sample contains a non-object row")
        access_order = item.get("access_order")
        pc = item.get("pc")
        line_address = item.get("cache_line_address")
        hit = item.get("hit")
        if type(access_order) is not int or access_order < 1 or access_order > total_accesses:
            raise CampaignError("ChampSim L2 load sample contains an invalid access order")
        if type(pc) is not int or pc < 0 or type(line_address) is not int or line_address < 0 or type(hit) is not bool:
            raise CampaignError("ChampSim L2 load sample contains an invalid PC, address, or hit status")
        if line_address % cache_line_size != 0:
            raise CampaignError("ChampSim L2 load sample address is not aligned to the configured cache-line size")
        if access_order in seen_orders:
            raise CampaignError("ChampSim L2 load sample contains duplicate access orders")
        seen_orders.add(access_order)
        rows.append({"access_order": access_order, "pc": pc, "cache_line_address": line_address, "hit": hit})

    rows.sort(key=lambda row: row["access_order"])
    return cache_name, sample_limit, total_accesses, rows


def dram_traffic(roi: dict[str, Any]) -> dict[str, Any]:
    channels = roi.get("DRAM")
    if not isinstance(channels, list) or not channels or any(not isinstance(channel, dict) for channel in channels):
        raise CampaignError("ChampSim ROI output is missing per-channel DRAM telemetry")

    counter_names = ("read_transactions", "read_bytes", "write_transactions", "write_bytes")
    by_type_names = (
        "read_transactions_by_type",
        "read_bytes_by_type",
        "write_transactions_by_type",
        "write_bytes_by_type",
    )
    totals: dict[str, Any] = {name: 0 for name in counter_names}
    for name in by_type_names:
        totals[name] = {request_type: 0 for request_type in DRAM_REQUEST_TYPES}

    for channel in channels:
        for name in counter_names:
            value = channel.get(name)
            if type(value) is not int or value < 0:
                raise CampaignError(f"ChampSim DRAM telemetry has an invalid {name} counter")
            totals[name] += value
        for name in by_type_names:
            values = channel.get(name)
            if not isinstance(values, dict) or set(values) != set(DRAM_REQUEST_TYPES):
                raise CampaignError(f"ChampSim DRAM telemetry has invalid request-type attribution in {name}")
            for request_type in DRAM_REQUEST_TYPES:
                value = values[request_type]
                if type(value) is not int or value < 0:
                    raise CampaignError(f"ChampSim DRAM telemetry has an invalid {name}.{request_type} counter")
                totals[name][request_type] += value

    for total_name, type_name in (
        ("read_transactions", "read_transactions_by_type"),
        ("read_bytes", "read_bytes_by_type"),
        ("write_transactions", "write_transactions_by_type"),
        ("write_bytes", "write_bytes_by_type"),
    ):
        if totals[total_name] != sum(totals[type_name].values()):
            raise CampaignError(f"ChampSim DRAM {total_name} does not match its request-type counters")

    # These categories reflect the request type retained by ChampSim when each
    # DRAM transaction is serviced. They do not reconstruct original traffic
    # sources after requests have merged.
    totals["demand_read_transactions"] = totals["read_transactions_by_type"]["LOAD"] + totals["read_transactions_by_type"]["RFO"]
    totals["demand_read_bytes"] = totals["read_bytes_by_type"]["LOAD"] + totals["read_bytes_by_type"]["RFO"]
    totals["prefetch_read_transactions"] = totals["read_transactions_by_type"]["PREFETCH"]
    totals["prefetch_read_bytes"] = totals["read_bytes_by_type"]["PREFETCH"]
    totals["attribution_basis"] = "request type retained at DRAM transaction service"
    return totals


def prefetcher_feedback(stdout_path: Path) -> dict[str, Any]:
    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in output.splitlines() if line.startswith("FDP_IP_STRIDE_STATS ")]
    if len(lines) != 1:
        raise CampaignError(f"expected one FDP IP-stride summary in {stdout_path}, found {len(lines)}")

    fields: dict[str, int] = {}
    for item in lines[0].split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or not value.isdecimal():
            raise CampaignError(f"invalid FDP IP-stride summary field: {item}")
        fields[key] = int(value)

    expected = {
        "final_degree", "degree_increases", "degree_decreases", "feedback_windows",
        "total_useful_events", "useful_address_episodes", "accepted_prefetch_requests", "tracked_address_episodes",
        "resolved_address_episodes", "feedback_batch_address_episodes",
        "low_usefulness_percent", "high_usefulness_percent",
    }
    if (
        set(fields) != expected
        or fields["final_degree"] not in (1, 2, 3)
        or fields["useful_address_episodes"] > fields["resolved_address_episodes"]
        or fields["resolved_address_episodes"] > fields["tracked_address_episodes"]
        or fields["tracked_address_episodes"] > fields["accepted_prefetch_requests"]
    ):
        raise CampaignError(f"incomplete or invalid FDP IP-stride summary in {stdout_path}")
    return {
        "signal": "useful resolved prefetch-address episodes / all resolved prefetch-address episodes in 4,096-episode batches, tagged by the latest accepted issue degree; duplicate submissions to one pending address count once",
        "primary_control": "stride lookahead degree, clamped to 1–3",
        "counter_scope": "full simulator execution, including warmup; these counters are diagnostic, not ROI-only",
        **fields,
    }


def write_access_sample(path: Path, sample: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("access_order", "pc", "cache_line_address", "hit_miss"))
        for row in sample:
            writer.writerow(
                (
                    row["access_order"],
                    f"0x{row['pc']:x}",
                    f"0x{row['cache_line_address']:x}",
                    "hit" if row["hit"] else "miss",
                )
            )


def run_campaign(args: argparse.Namespace) -> Path:
    level = selected_level(args.prefetcher, args.level)
    trace_path = args.trace.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not trace_path.is_file():
        raise CampaignError(f"trace file does not exist: {trace_path}")
    if not profile_path.is_file():
        raise CampaignError(f"ChampSim profile does not exist: {profile_path}")

    profile = read_json(profile_path)
    cache_line_size = profile.get("block_size")
    if type(cache_line_size) is not int or cache_line_size <= 0:
        raise CampaignError("the campaign profile must define a positive cache-line block_size")
    profile_sha256 = sha256_file(profile_path)
    trace_sha256 = sha256_file(trace_path)
    trace_size_bytes = trace_path.stat().st_size
    effective_config = make_effective_config(profile, args.prefetcher, level, profile_sha256)
    effective_config_bytes = (json.dumps(effective_config, indent=2, sort_keys=True) + "\n").encode("utf-8")
    effective_config_sha256 = hashlib.sha256(effective_config_bytes).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    config_path = output_path.with_name(f"{stem}.config.json")
    raw_output_path = output_path.with_name(f"{stem}.champsim.json")
    sample_path = output_path.with_name(f"{stem}.l2-load-sample.csv")
    telemetry_path = output_path.with_name(f"{stem}.telemetry.json")
    stdout_path = output_path.with_name(f"{stem}.stdout.log")
    stderr_path = output_path.with_name(f"{stem}.stderr.log")
    config_path.write_bytes(effective_config_bytes)

    if args.simulator is not None:
        simulator_path = args.simulator.expanduser().resolve()
        revision = champsim_revision(args.champsim_source.resolve())
        source_extension = "ip_stride_fdp" if args.prefetcher == "ip_stride_fdp" else None
        source_diff = champsim_source_diff(args.champsim_source.resolve(), source_extension)
        if not simulator_path.is_file() or not os.access(simulator_path, os.X_OK):
            raise CampaignError(f"simulator must be an executable file: {simulator_path}")
    else:
        simulator_path, revision, source_diff = build_simulator(
            args.champsim_source,
            config_path,
            effective_config_sha256,
            output_path.parent,
            "ip_stride_fdp" if args.prefetcher == "ip_stride_fdp" else None,
        )

    simulator_sha256 = sha256_file(simulator_path)
    command = [
        str(simulator_path),
        "--warmup-instructions",
        str(WARMUP_INSTRUCTIONS),
        "--simulation-instructions",
        str(SIMULATION_INSTRUCTIONS),
        "--hide-heartbeat",
        "--json",
        str(raw_output_path),
        str(trace_path),
    ]
    raw_output_path.unlink(missing_ok=True)
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            completed = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                env={**os.environ, "CHAMPSIM_L2_LOAD_SAMPLE_LIMIT": str(L2_LOAD_SAMPLE_LIMIT)},
            )
    except OSError as error:
        raise CampaignError(f"unable to execute ChampSim binary {simulator_path}: {error}") from error
    if completed.returncode != 0:
        details = stderr_path.read_text(errors="replace").strip()
        raise CampaignError(f"ChampSim exited with status {completed.returncode}: {details}")
    if not raw_output_path.is_file():
        raise CampaignError("ChampSim completed without writing its JSON output")
    try:
        champsim_output = json.loads(raw_output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read ChampSim JSON output {raw_output_path}: {error}") from error

    retire_width = profile["ooo_cpu"][0].get("retire_width")
    if type(retire_width) is not int or retire_width <= 0:
        raise CampaignError("the campaign profile must define a positive core retire_width")
    roi = simulation_roi(champsim_output, retire_width)
    simulator_cache_name, sample_limit, total_load_accesses, sample = l2_load_sample(roi, cache_line_size)
    traffic = dram_traffic(roi)
    identity = {
        "workload": args.workload,
        "trace": {
            "path": str(trace_path),
            "filename": trace_path.name,
            "size_bytes": trace_size_bytes,
            "sha256": trace_sha256,
        },
        "prefetcher": {"name": args.prefetcher, "level": level},
        "profile": {"path": str(profile_path), "sha256": profile_sha256},
        "configuration": {
            "path": str(config_path),
            "sha256": effective_config_sha256,
            "executable_name": effective_config["executable_name"],
        },
        "simulator": {
            "path": str(simulator_path),
            "sha256": simulator_sha256,
            "source_revision": revision,
            "source_diff_sha256": source_diff,
        },
        "roi": {
            "warmup_instructions": WARMUP_INSTRUCTIONS,
            "simulation_instructions": SIMULATION_INSTRUCTIONS,
            "actual_simulation_instructions": roi["cores"][0]["instructions"],
            "measured_instruction_start": WARMUP_INSTRUCTIONS,
            "measured_instruction_end": WARMUP_INSTRUCTIONS + SIMULATION_INSTRUCTIONS,
        },
    }
    normalized = normalized_roi(roi)
    normalized["l2_load_sample"] = {
        "cache_level": "L2C",
        "simulator_cache_name": simulator_cache_name,
        "sample_limit": sample_limit,
        "total_accesses": total_load_accesses,
        "sampled_accesses": len(sample),
    }
    normalized["dram_traffic"] = traffic
    if args.prefetcher == "ip_stride_fdp":
        normalized["prefetcher_feedback"] = prefetcher_feedback(stdout_path)
    result = {
        "schema_version": 1,
        "run_id": digest_json(identity),
        "identity": identity,
        "champsim_roi": roi,
        "normalized": normalized,
        "artifacts": {
            "effective_config": str(config_path),
            "champsim_json": str(raw_output_path),
            "l2_load_sample": str(sample_path),
            "telemetry": str(telemetry_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
    }
    telemetry = {
        "schema_version": 1,
        "run_id": result["run_id"],
        "identity": identity,
        "cache_level": "L2C",
        "roi": identity["roi"],
        "l2_load_sample": {
            "path": str(sample_path),
            "simulator_cache_name": simulator_cache_name,
            "total_accesses": total_load_accesses,
            "sampled_accesses": len(sample),
            "sample_limit": sample_limit,
            "rows": sample,
        },
        "dram": traffic,
    }
    temporary_sample_path = sample_path.with_name(f".{sample_path.name}.tmp")
    write_access_sample(temporary_sample_path, sample)
    temporary_sample_path.replace(sample_path)
    temporary_telemetry_path = telemetry_path.with_name(f".{telemetry_path.name}.tmp")
    temporary_telemetry_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_telemetry_path.replace(telemetry_path)
    temporary_result_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_result_path.replace(output_path)
    return output_path


def argument_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=WORKLOADS, help="one project workload name")
    parser.add_argument("--trace", required=True, type=Path, help="the exact trace file for this workload")
    parser.add_argument("--prefetcher", required=True, choices=("none", *PREFETCHER_LEVELS), help="one active prefetcher, or none")
    parser.add_argument("--level", choices=("L1D", "L2C"), help="required for none; otherwise checked against the prefetcher level")
    parser.add_argument(
        "--profile",
        type=Path,
        default=repository / "experiments" / "config" / "1C.fullBW.baseline.json",
        help="official ChampSim single-core profile (defaults to the tracked project profile)",
    )
    parser.add_argument("--output", required=True, type=Path, help="normalized result JSON output path")
    parser.add_argument("--simulator", type=Path, help="prebuilt ChampSim binary compiled from the emitted effective configuration")
    parser.add_argument(
        "--champsim-source",
        type=Path,
        default=repository / "third_party" / "champsim-dpc4",
        help="ChampSim source tree used when the runner builds a config-specific binary",
    )
    return parser


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    try:
        result_path = run_campaign(args)
    except CampaignError as error:
        parser.exit(2, f"error: {error}\n")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
