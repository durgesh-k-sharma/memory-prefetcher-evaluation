import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "experiments" / "run_campaign.py"


class CampaignRunnerTests(unittest.TestCase):
    def test_result_writes_bounded_sample_and_roi_dram_telemetry_with_matching_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            profile_path = REPOSITORY / "experiments" / "config" / "1C.fullBW.baseline.json"
            trace_path = workspace / "603.bwaves_s-891B.champsimtrace.xz"
            trace_path.write_bytes(b"deterministic trace fixture")
            output_path = workspace / "result.json"

            fake_simulator = workspace / "fake-champsim"
            fake_simulator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "out = pathlib.Path(args[args.index('--json') + 1])\n"
                "out.write_text(json.dumps([{'name':'Simulation','traces':[args[-1]],'roi':{\n"
                " 'cores':[{'instructions':100000003,'cycles':50000000}],\n"
                " 'cpu0_L2C':{'load_access_sample_limit':4,'load_access_sample_total':12,"
                "'load_access_sample':[\n"
                "  {'access_order':2,'pc':4096,'cache_line_address':8192,'hit':True},\n"
                "  {'access_order':9,'pc':8192,'cache_line_address':12288,'hit':False},\n"
                "  {'access_order':12,'pc':4096,'cache_line_address':8256,'hit':True}],"
                "'prefetch fill':6},\n"
                " 'DRAM':[{'read_transactions':7,'read_bytes':448,'write_transactions':2,'write_bytes':128,"
                "'read_transactions_by_type':{'LOAD':5,'RFO':0,'PREFETCH':2,'WRITE':0,'TRANSLATION':0},"
                "'read_bytes_by_type':{'LOAD':320,'RFO':0,'PREFETCH':128,'WRITE':0,'TRANSLATION':0},"
                "'write_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':2,'TRANSLATION':0},"
                "'write_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':128,'TRANSLATION':0}}]}}]))\n"
            )
            fake_simulator.chmod(0o755)

            process = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--workload",
                    "603.bwaves_s-891B",
                    "--trace",
                    str(trace_path),
                    "--prefetcher",
                    "ip_stride",
                    "--profile",
                    str(profile_path),
                    "--simulator",
                    str(fake_simulator),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output_path.read_text())
            sample_path = workspace / "result.l2-load-sample.csv"
            telemetry_path = workspace / "result.telemetry.json"
            self.assertEqual(result["artifacts"]["l2_load_sample"], str(sample_path))
            self.assertEqual(result["artifacts"]["telemetry"], str(telemetry_path))
            self.assertEqual(
                sample_path.read_text().splitlines(),
                [
                    "access_order,pc,cache_line_address,hit_miss",
                    "2,0x1000,0x2000,hit",
                    "9,0x2000,0x3000,miss",
                    "12,0x1000,0x2040,hit",
                ],
            )
            telemetry = json.loads(telemetry_path.read_text())
            self.assertEqual(telemetry["run_id"], result["run_id"])
            self.assertEqual(telemetry["identity"], result["identity"])
            self.assertEqual(telemetry["cache_level"], "L2C")
            self.assertEqual(telemetry["l2_load_sample"]["simulator_cache_name"], "cpu0_L2C")
            self.assertEqual(telemetry["roi"], result["identity"]["roi"])
            self.assertEqual(telemetry["l2_load_sample"]["total_accesses"], 12)
            self.assertEqual(telemetry["l2_load_sample"]["sampled_accesses"], 3)
            self.assertEqual(telemetry["dram"]["read_transactions"], 7)
            self.assertEqual(telemetry["dram"]["read_bytes"], 448)
            self.assertEqual(telemetry["dram"]["write_transactions"], 2)
            self.assertEqual(telemetry["dram"]["write_bytes"], 128)
            self.assertEqual(telemetry["dram"]["read_transactions_by_type"]["PREFETCH"], 2)
            self.assertEqual(result["champsim_roi"]["cpu0_L2C"]["prefetch fill"], 6)
            self.assertEqual(result["identity"]["roi"]["actual_simulation_instructions"], 100000003)

    def test_l2_run_records_roi_and_isolated_configuration_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            profile_path = REPOSITORY / "experiments" / "config" / "1C.fullBW.baseline.json"
            trace_path = workspace / "603.bwaves_s-891B.champsimtrace.xz"
            trace_path.write_bytes(b"deterministic trace fixture")
            output_path = workspace / "result.json"

            fake_simulator = workspace / "fake-champsim"
            fake_simulator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "out = pathlib.Path(args[args.index('--json') + 1])\n"
                "out.write_text(json.dumps([\n"
                " {'name':'Warmup','roi':{}},\n"
                " {'name':'Simulation','traces':[args[-1]],'roi':{\n"
                "  'cores':[{'instructions':100000000,'cycles':50000000}],\n"
                "  'L2C':{'Load':{'hit':[20],'miss':[5]},'Prefetch':{'hit':[0],'miss':[3]},'prefetch issued':3,'useful prefetch':2,"
                "'load_access_sample_limit':10000,'load_access_sample_total':2,'load_access_sample':["
                "{'access_order':1,'pc':4096,'cache_line_address':8192,'hit':True},"
                "{'access_order':2,'pc':8192,'cache_line_address':12288,'hit':False}]},\n"
                "  'DRAM':[{'read_transactions':0,'read_bytes':0,'write_transactions':0,'write_bytes':0,"
                "'read_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'read_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0}}]}}\n"
                "]))\n"
            )
            fake_simulator.chmod(0o755)

            process = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--workload",
                    "603.bwaves_s-891B",
                    "--trace",
                    str(trace_path),
                    "--prefetcher",
                    "ip_stride",
                    "--profile",
                    str(profile_path),
                    "--simulator",
                    str(fake_simulator),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output_path.read_text())
            config = json.loads((workspace / "result.config.json").read_text())

            self.assertEqual(config["L2C"]["prefetcher"], "ip_stride")
            self.assertEqual(
                {name: config[name]["prefetcher"] for name in ("L1I", "L1D", "L2C", "LLC")},
                {"L1I": "no", "L1D": "no", "L2C": "ip_stride", "LLC": "no"},
            )
            self.assertEqual(result["identity"]["workload"], "603.bwaves_s-891B")
            self.assertEqual(result["identity"]["prefetcher"], {"name": "ip_stride", "level": "L2C"})
            self.assertEqual(
                result["identity"]["roi"],
                {
                    "warmup_instructions": 50000000,
                    "simulation_instructions": 100000000,
                    "actual_simulation_instructions": 100000000,
                    "measured_instruction_start": 50000000,
                    "measured_instruction_end": 150000000,
                },
            )
            self.assertEqual(result["identity"]["trace"]["sha256"], hashlib.sha256(trace_path.read_bytes()).hexdigest())
            self.assertEqual(result["champsim_roi"]["cores"][0]["instructions"], 100000000)
            self.assertEqual(result["normalized"]["ipc"], 2.0)
            self.assertEqual(result["normalized"]["L2C"]["Load"]["miss"], [5])
            self.assertEqual(
                {name: json.loads(profile_path.read_text())[name]["prefetcher"] for name in ("L1I", "L1D", "L2C", "LLC")},
                {"L1I": "no", "L1D": "no", "L2C": "no", "LLC": "no"},
            )
            official_profile = json.loads(profile_path.read_text())
            self.assertEqual((official_profile["num_cores"], official_profile["ooo_cpu"][0]["frequency"]), (1, 4000))
            self.assertEqual((official_profile["L1D"]["sets"], official_profile["L1D"]["ways"], official_profile["L1D"]["latency"]), (64, 12, 5))
            self.assertEqual((official_profile["L2C"]["sets"], official_profile["L2C"]["ways"], official_profile["L2C"]["latency"]), (2048, 16, 10))
            self.assertEqual((official_profile["LLC"]["sets"], official_profile["LLC"]["ways"], official_profile["LLC"]["latency"]), (4096, 12, 35))
            self.assertEqual(
                tuple(official_profile["physical_memory"][key] for key in ("data_rate", "tCAS", "tRCD", "tRP", "tRAS")),
                (4800, 36, 36, 36, 78),
            )

            baseline_path = workspace / "603.bwaves-no-pref.json"
            baseline_process = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--workload",
                    "603.bwaves_s-891B",
                    "--trace",
                    str(trace_path),
                    "--prefetcher",
                    "none",
                    "--level",
                    "L2C",
                    "--profile",
                    str(profile_path),
                    "--simulator",
                    str(fake_simulator),
                    "--output",
                    str(baseline_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(baseline_process.returncode, 0, baseline_process.stderr)
            baseline = json.loads(baseline_path.read_text())
            baseline_config = json.loads(workspace.joinpath("603.bwaves-no-pref.config.json").read_text())
            self.assertEqual(baseline["identity"]["prefetcher"], {"name": "none", "level": "L2C"})
            self.assertEqual(
                {name: baseline_config[name]["prefetcher"] for name in ("L1I", "L1D", "L2C", "LLC")},
                {"L1I": "no", "L1D": "no", "L2C": "no", "LLC": "no"},
            )
            self.assertEqual(result["identity"]["trace"], baseline["identity"]["trace"])
            self.assertEqual(result["identity"]["roi"], baseline["identity"]["roi"])
            self.assertNotEqual(result["run_id"], baseline["run_id"])

    def test_berti_l1d_run_has_a_matching_l1d_no_pref_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            profile_path = REPOSITORY / "experiments" / "config" / "1C.fullBW.baseline.json"
            trace_path = workspace / "ligra_BFSCC.fixture.trace"
            trace_path.write_bytes(b"same fixture trace for both runs")
            fake_simulator = workspace / "fake-champsim"
            fake_simulator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "out = pathlib.Path(args[args.index('--json') + 1])\n"
                "out.write_text(json.dumps([{'name':'Simulation','roi':{\n"
                " 'cores':[{'instructions':100000000,'cycles':50000000}],\n"
                " 'L1D':{'Load':{'hit':[20],'miss':[5]},'Prefetch':{'hit':[0],'miss':[3]}},\n"
                " 'L2C':{'load_access_sample_limit':10000,'load_access_sample_total':1,'load_access_sample':["
                "{'access_order':1,'pc':4096,'cache_line_address':8192,'hit':True}]},\n"
                " 'DRAM':[{'read_transactions':0,'read_bytes':0,'write_transactions':0,'write_bytes':0,"
                "'read_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'read_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0}}]}}]))\n"
            )
            fake_simulator.chmod(0o755)

            results = {}
            for label, prefetcher, level_args in (
                ("berti", "berti", []),
                ("no-pref", "none", ["--level", "L1D"]),
            ):
                output_path = workspace / f"{label}.json"
                process = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--workload",
                        "ligra-bfs-cc",
                        "--trace",
                        str(trace_path),
                        "--prefetcher",
                        prefetcher,
                        *level_args,
                        "--profile",
                        str(profile_path),
                        "--simulator",
                        str(fake_simulator),
                        "--output",
                        str(output_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads(output_path.read_text())
                config = json.loads(output_path.with_name(f"{output_path.stem}.config.json").read_text())
                results[label] = result

                self.assertEqual(
                    {name: config[name]["prefetcher"] for name in ("L1I", "L1D", "L2C", "LLC")},
                    {"L1I": "no", "L1D": "berti" if prefetcher == "berti" else "no", "L2C": "no", "LLC": "no"},
                )
                self.assertEqual(result["identity"]["prefetcher"]["level"], "L1D")
                self.assertEqual(result["champsim_roi"]["cores"][0]["instructions"], 100000000)

            self.assertEqual(results["berti"]["identity"]["trace"], results["no-pref"]["identity"]["trace"])
            self.assertEqual(results["berti"]["identity"]["roi"], results["no-pref"]["identity"]["roi"])
            self.assertNotEqual(results["berti"]["run_id"], results["no-pref"]["run_id"])

    def test_runner_builds_a_configuration_specific_binary_in_its_output_area(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            profile_path = REPOSITORY / "experiments" / "config" / "1C.fullBW.baseline.json"
            trace_path = workspace / "fixture.trace"
            trace_path.write_bytes(b"trace fixture")

            fake_simulator = workspace / "fake-champsim"
            fake_simulator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "out = pathlib.Path(args[args.index('--json') + 1])\n"
                "out.write_text(json.dumps([{'name':'Simulation','roi':{\n"
                " 'cores':[{'instructions':100000000,'cycles':50000000}],\n"
                " 'L2C':{'load_access_sample_limit':10000,'load_access_sample_total':0,'load_access_sample':[]},\n"
                " 'DRAM':[{'read_transactions':0,'read_bytes':0,'write_transactions':0,'write_bytes':0,"
                "'read_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'read_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_transactions_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0},"
                "'write_bytes_by_type':{'LOAD':0,'RFO':0,'PREFETCH':0,'WRITE':0,'TRANSLATION':0}}]}}]))\n"
            )
            fake_simulator.chmod(0o755)

            fake_source = workspace / "fake-champsim-source"
            fake_source.mkdir()
            (fake_source / "README.md").write_text("fake source revision\n")
            (fake_source / "Makefile").write_text("all:\n\t@true\n")
            (fake_source / "config.sh").write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, shutil, sys\n"
                "source = pathlib.Path(__file__).parent\n"
                "config = json.loads(pathlib.Path(sys.argv[-1]).read_text())\n"
                "binary = source / 'bin' / config['executable_name']\n"
                "binary.parent.mkdir(parents=True, exist_ok=True)\n"
                f"shutil.copyfile({str(fake_simulator)!r}, binary)\n"
                "binary.chmod(0o755)\n"
                "(source / 'Makefile').write_text('all:\\n\\t@true\\n')\n"
            )

            output_path = workspace / "built.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--workload",
                    "llama2",
                    "--trace",
                    str(trace_path),
                    "--prefetcher",
                    "spp_dev",
                    "--profile",
                    str(profile_path),
                    "--champsim-source",
                    str(fake_source),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output_path.read_text())
            self.assertTrue(Path(result["identity"]["simulator"]["path"]).is_file())
            self.assertIn(str(workspace / ".campaign-builds"), result["identity"]["simulator"]["path"])
            config = json.loads(output_path.with_name("built.config.json").read_text())
            self.assertEqual(config["L2C"]["prefetcher"], "spp_dev")


if __name__ == "__main__":
    unittest.main()
