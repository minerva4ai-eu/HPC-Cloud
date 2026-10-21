import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmark_compare as benchmark
import cloud_hpc


PROFILE = {
    "requested_cpu": "4",
    "requested_memory": "4Gi",
    "workload_script": "workloads/tier-a/scripts/train_cpu_ai_v1.py",
}


def catalog():
    return json.loads((ROOT / "config" / "workload-profiles.json").read_text())


def k8s_raw():
    return {
        "job": {"metadata": {"creationTimestamp": "2026-08-28T10:00:00Z"}},
        "pod": {
            "spec": {
                "nodeName": "worker-1",
                "containers": [{"resources": {
                    "requests": {"cpu": "4", "memory": "4Gi"},
                    "limits": {"cpu": "4", "memory": "4Gi"},
                }}],
            },
            "status": {
                "phase": "Succeeded",
                "startTime": "2026-08-28T10:00:02Z",
                "containerStatuses": [{"state": {"terminated": {
                    "exitCode": 0, "finishedAt": "2026-08-28T10:00:12Z"
                }}}],
            },
        },
        "result": {
            "experiment": "synthetic", "seed": 42,
            "train_samples": 120000, "test_samples": 20000,
            "features": 32, "classes": 4, "epochs": 3, "batch_size": 64,
            "cpu_threads": 4, "test_accuracy": 0.91, "test_loss": 0.2,
            "inference_samples_per_second": 5000.0, "max_rss_mb": 123.0,
        },
        "success": True,
        "output_sha256": "def",
    }


def mn5_raw():
    return {
        "job_id": "TEST-JOB",
        "accounting": {
            "JobIDRaw": "TEST-JOB", "State": "COMPLETED", "ExitCode": "0:0",
            "Submit": "2026-08-28T10:01:00+00:00",
            "Start": "2026-08-28T10:01:05+00:00",
            "End": "2026-08-28T10:01:25+00:00", "AllocCPUS": "8",
            "ReqMem": "8G", "MaxRSS": "512M", "NodeList": "test-node",
        },
        "result": {
            "experiment": "synthetic", "seed": 42,
            "train_samples": 120000, "test_samples": 20000,
            "features": 32, "classes": 4, "epochs": 3, "batch_size": 64,
            "cpu_threads": 4, "test_accuracy": 0.91, "test_loss": 0.2,
            "inference_samples_per_second": 4000.0, "max_rss_mb": 512.0,
        },
        "sha256": {"result.json": "abc"},
        "accounting_timezone": "UTC",
    }


class NormalizationTests(unittest.TestCase):
    def test_timing_calculations_are_distinct(self):
        value = benchmark.timing(
            "2026-08-28T10:00:00Z", "2026-08-28T10:00:03Z", "2026-08-28T10:00:13Z"
        )
        self.assertEqual(value["queue_wait_s"], 3.0)
        self.assertEqual(value["execution_time_s"], 10.0)
        self.assertEqual(value["end_to_end_s"], 13.0)

    def test_kubernetes_normalization(self):
        value = benchmark.normalize_k8s(k8s_raw(), "CMP-X", "RUN-K", "synthetic-light-v1", PROFILE, "hash")
        self.assertEqual(value["identity"]["node_host"], "worker-1")
        self.assertEqual(value["timing"]["execution_time_s"], 10.0)
        self.assertEqual(value["resources"]["peak_memory_mb"], 123.0)
        self.assertTrue(value["status"]["success"])

    def test_mn5_normalization(self):
        value = benchmark.normalize_mn5(mn5_raw(), "CMP-X", "RUN-M", "synthetic-light-v1", PROFILE, "hash")
        self.assertEqual(value["identity"]["execution_run_id"], "TEST-JOB")
        self.assertEqual(value["identity"]["node_host"], "test-node")
        self.assertEqual(value["timing"]["queue_wait_s"], 5.0)
        self.assertEqual(value["resources"]["accounting_alloc_cpus"], "8")
        self.assertEqual(value["resources"]["scheduler_allocated_cpu"], "8")
        self.assertIsNone(value["resources"]["allocated_cpu"])
        self.assertEqual(value["resources"]["workload_cpu_threads"], 4)
        self.assertEqual(value["resources"]["scheduler_requested_memory"], "8G")
        self.assertEqual(value["resources"]["memory_policy_source"], "configured HPC memory policy")
        self.assertEqual(value["resources"]["memory_per_requested_core_gb"], 2)
        self.assertEqual(value["resources"]["expected_memory_from_cpu"], "8000M")
        self.assertEqual(value["resources"]["expected_memory_from_cpu_gb"], 8)
        self.assertTrue(value["resources"]["memory_policy_match"])

    def test_mn5_memory_policy_uses_requested_cpu_not_alloc_cpus(self):
        for requested_cpu, req_mem, expected_gb in (
            ("4", "8000M", 8), ("8", "16000M", 16), ("16", "32000M", 32),
        ):
            with self.subTest(requested_cpu=requested_cpu):
                raw = mn5_raw()
                raw["accounting"]["ReqMem"] = req_mem
                raw["accounting"]["AllocCPUS"] = "999"
                profile = dict(PROFILE, requested_cpu=requested_cpu)
                value = benchmark.normalize_mn5(
                    raw, "CMP-X", "RUN-M", "w", profile, "hash"
                )
                resources = value["resources"]
                self.assertEqual(resources["scheduler_requested_memory"], req_mem)
                self.assertEqual(resources["requested_memory"], req_mem)
                self.assertEqual(resources["expected_memory_from_cpu_gb"], expected_gb)
                self.assertEqual(resources["expected_memory_from_cpu"], f"{expected_gb * 1000}M")
                self.assertTrue(resources["memory_policy_match"])
                self.assertEqual(resources["accounting_alloc_cpus"], "999")

    def test_timezone_less_scheduler_times_require_declared_timezone(self):
        raw = mn5_raw()
        raw["accounting"]["Submit"] = "2026-08-28T18:50:39"
        raw["accounting"]["Start"] = "2026-08-28T18:50:47"
        raw["accounting"]["End"] = "2026-08-28T18:51:05"
        value = benchmark.normalize_mn5(raw, "CMP-X", "RUN-M", "w", PROFILE, "hash")
        self.assertEqual(value["timing"]["start_time"], "2026-08-28T18:50:47+00:00")
        self.assertEqual(value["timing"]["raw_scheduler_timestamps"]["start_time"], "2026-08-28T18:50:47")
        self.assertEqual(value["timing"]["execution_time_s"], 18.0)

    def test_missing_optional_metrics_are_null(self):
        value = benchmark.normalize_k8s({}, "CMP-X", "RUN-K", "w", PROFILE, "hash")
        self.assertIsNone(value["energy"])
        self.assertIsNone(value["cost"])
        self.assertIsNone(value["resources"]["runtime_cpu_utilization"])
        self.assertEqual(value["provenance"]["energy"], "unavailable")


class ComparisonTests(unittest.TestCase):
    def _comparison(self, kraw=None, mraw=None):
        k = benchmark.normalize_k8s(kraw or k8s_raw(), "CMP-X", "RUN-K", "w", PROFILE, "hash")
        m = benchmark.normalize_mn5(mraw or mn5_raw(), "CMP-X", "RUN-M", "w", PROFILE, "hash")
        return benchmark.build_comparison("CMP-X", "w", "hash", {"k8s": k, "mn5": m}, "s", "f")

    def test_different_raw_hashes_can_be_semantically_equivalent(self):
        comparison = self._comparison()
        for platform in ("k8s", "mn5"):
            integrity = comparison["executions"][platform]["integrity"]
            self.assertFalse(integrity["raw_output_sha256_match"])
            self.assertTrue(integrity["semantic_output_equivalent"])
            self.assertTrue(integrity["output_equivalent"])
            self.assertIsNotNone(integrity["raw_output_sha256"])
        self.assertEqual(comparison["executions"]["k8s"]["integrity"]["raw_output_sha256"], "def")
        self.assertEqual(comparison["executions"]["mn5"]["integrity"]["raw_output_sha256"], "abc")

    def test_correctness_difference_is_not_semantically_equivalent(self):
        raw = mn5_raw()
        raw["result"]["test_accuracy"] = 0.81
        integrity = self._comparison(mraw=raw)["executions"]["mn5"]["integrity"]
        self.assertFalse(integrity["semantic_output_equivalent"])
        self.assertEqual(integrity["semantic_mismatches"][0]["field"], "test_accuracy")

    def test_performance_only_differences_do_not_change_semantics(self):
        raw = mn5_raw()
        raw["result"].update({"train_seconds": 99, "data_seconds": 88, "max_rss_mb": 999})
        integrity = self._comparison(mraw=raw)["executions"]["mn5"]["integrity"]
        self.assertTrue(integrity["semantic_output_equivalent"])
        self.assertNotIn("train_seconds", integrity["semantic_fields_checked"])

    def test_same_parent_and_workload_spec_for_both_children(self):
        calls = []
        def k_runner(workload, comparison_id, run_id, profile):
            calls.append((workload, comparison_id, profile.copy()))
            return k8s_raw()
        def m_runner(workload, comparison_id, run_id, profile):
            calls.append((workload, comparison_id, profile.copy()))
            return mn5_raw()
        with tempfile.TemporaryDirectory() as tmp:
            comparison, _ = benchmark.run_comparison(
                "synthetic-light-v1", PROFILE, k_runner, m_runner, tmp
            )
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(
            comparison["executions"]["k8s"]["identity"]["comparison_id"],
            comparison["executions"]["mn5"]["identity"]["comparison_id"],
        )
        self.assertEqual(comparison["execution_order"], ["KUBERNETES", "MN5"])

    def test_one_platform_failure_preserves_other_evidence(self):
        def fail(*_args):
            raise RuntimeError("k8s unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            comparison, directory = benchmark.run_comparison(
                "synthetic-light-v1", PROFILE, fail, lambda *_: mn5_raw(), tmp
            )
            self.assertFalse(comparison["executions"]["k8s"]["status"]["success"])
            self.assertTrue(comparison["executions"]["mn5"]["status"]["success"])
            self.assertTrue((directory / "comparison.json").exists())
            integrity = comparison["executions"]["mn5"]["integrity"]
            self.assertIsNone(integrity["semantic_output_equivalent"])
            self.assertIsNone(integrity["raw_output_sha256_match"])
            self.assertEqual(integrity["semantic_mismatches"], [])
            self.assertEqual(
                integrity["semantic_comparison_status"],
                "NOT_PERFORMED_MISSING_OUTPUT",
            )

    def test_static_infeasibility_skips_k8s_but_runs_mn5_and_writes_artifact(self):
        k8s_runner = mock.Mock(side_effect=AssertionError("must not submit"))
        mn5_runner = mock.Mock(return_value=mn5_raw())
        infeasible = {
            "platform_feasible": False,
            "state": "INFEASIBLE",
            "failure_category": "RESOURCE_INFEASIBLE",
            "failure_reason": "No eligible Kubernetes node can satisfy requested CPU 16",
            "max_eligible_node_cpu": 12,
            "success": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            comparison, directory = benchmark.run_comparison(
                "synthetic-heavy-v1", catalog()["synthetic-heavy-v1"],
                k8s_runner, mn5_runner, tmp,
                k8s_feasibility=lambda _profile: infeasible,
            )
            self.assertTrue((directory / "comparison.json").exists())
        k8s_runner.assert_not_called()
        mn5_runner.assert_called_once()
        k8s = comparison["executions"]["k8s"]
        self.assertFalse(k8s["status"]["platform_feasible"])
        self.assertEqual(k8s["status"]["scheduler_state"], "INFEASIBLE")
        self.assertEqual(k8s["status"]["failure_category"], "RESOURCE_INFEASIBLE")
        self.assertEqual(k8s["resources"]["requested_cpu"], "16")
        self.assertEqual(k8s["resources"]["max_eligible_node_cpu"], 12)

    def test_heavy_profiles_preserve_capacity_and_common_resource_levels(self):
        profiles = catalog()
        capacity = profiles["synthetic-heavy-v1"]
        common = profiles["synthetic-heavy-common-v1"]
        self.assertEqual((capacity["requested_cpu"], capacity["cpu_threads"]), ("16", 16))
        self.assertEqual((common["requested_cpu"], common["cpu_threads"]), ("8", 8))
        self.assertEqual(capacity["workload_script"], common["workload_script"])
        script = (ROOT / capacity["workload_script"]).read_text()
        for definition in (
            "SEED = 20260813", "TRAIN_SAMPLES = 400_000", "TEST_SAMPLES = 50_000",
            "FEATURES = 256", "CLASSES = 10", "EPOCHS = 6", "BATCH_SIZE = 2048",
        ):
            self.assertIn(definition, script)
        self.assertIn('os.environ.get(\n        "CPU_THREADS",\n        "16",', script)
        manifest = (ROOT / "k8s" / "b1-10-local-heavy-common.yaml").read_text()
        slurm = (
            ROOT / "workloads" / "synthetic-heavy" / "slurm"
            / "run_heavy_common_mn5_v1.slurm"
        ).read_text()
        self.assertIn('value: "8"', manifest)
        self.assertIn('cpu: "8"', manifest)
        self.assertIn("#SBATCH --cpus-per-task=8", slurm)
        self.assertIn("export CPU_THREADS=8", slurm)

    def test_json_csv_and_summary_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            comparison, directory = benchmark.run_comparison(
                "synthetic-light-v1", PROFILE, lambda *_: k8s_raw(), lambda *_: mn5_raw(), tmp
            )
            loaded = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(loaded["comparison_id"], comparison["comparison_id"])
            with open(directory / "comparison.csv", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([row["platform"] for row in rows], ["KUBERNETES", "MN5"])
            self.assertEqual(rows[1]["scheduler_requested_memory"], "8G")
            self.assertEqual(rows[1]["expected_memory_from_cpu"], "8000M")
            self.assertEqual(rows[1]["memory_policy_match"], "True")
            summary = (directory / "summary.md").read_text()
            self.assertIn("| Queue wait (s) |", summary)
            self.assertIn("| Kubernetes | MN5 |", summary)
            self.assertIn("| Requested/reserved memory | 4Gi | 8G |", summary)
            self.assertIn("| MN5 memory policy match | unavailable | True |", summary)
            self.assertTrue((directory / "k8s" / "metrics.json").exists())
            self.assertTrue((directory / "mn5" / "metrics.json").exists())


class CliRegressionTests(unittest.TestCase):
    def test_operator_menu_preserves_existing_actions_and_adds_compare_last(self):
        output = io.StringIO()
        with mock.patch("builtins.input", return_value="C"), contextlib.redirect_stdout(output):
            self.assertEqual(cloud_hpc.human_review("LOCAL", "details"), ("compare", None))
        menu = output.getvalue()
        entries = (
            "[A] Approve recommendation", "[R] Reject request",
            "[O] Override target", "[I] More information",
            "[C] Compare Kubernetes + MN5",
        )
        positions = [menu.index(entry) for entry in entries]
        self.assertEqual(positions, sorted(positions))

    @mock.patch.object(cloud_hpc, "dispatch_mn5")
    @mock.patch.object(cloud_hpc, "dispatch_local")
    @mock.patch.object(cloud_hpc, "wait_review_controller")
    @mock.patch.object(cloud_hpc, "submit_review")
    @mock.patch.object(cloud_hpc, "delete_job_visible")
    @mock.patch.object(cloud_hpc, "human_review", return_value=("compare", None))
    @mock.patch.object(cloud_hpc, "start_decision_controller", return_value=("MN5", "test", "logs"))
    @mock.patch.object(cloud_hpc, "load_catalog", return_value={"synthetic-light-v1": PROFILE})
    @mock.patch.object(cloud_hpc, "run_compare_mode", return_value=0)
    def test_interactive_compare_reuses_cli_path_without_placement_submission(
        self, compare, _catalog, decision, _review, cleanup, submit, wait,
        dispatch_local, dispatch_mn5,
    ):
        with mock.patch.object(sys, "argv", ["cloud-hpc", "synthetic-light-v1"]):
            self.assertEqual(cloud_hpc.main(), 0)
        decision.assert_called_once_with("synthetic-light-v1")
        compare.assert_called_once_with("synthetic-light-v1", PROFILE)
        cleanup.assert_called_once_with(cloud_hpc.CONTROLLER_JOB)
        submit.assert_not_called()
        wait.assert_not_called()
        dispatch_local.assert_not_called()
        dispatch_mn5.assert_not_called()

    def test_live_assessment_reports_per_node_cpu_infeasibility(self):
        nodes = {"items": [
            {
                "metadata": {"name": name, "labels": {"kubernetes.io/hostname": name}},
                "spec": {},
                "status": {
                    "allocatable": {"cpu": "12", "memory": "32Gi"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
            for name in ("worker-a", "worker-b")
        ]}
        pods = {"items": []}
        responses = [
            mock.Mock(stdout=json.dumps(nodes)), mock.Mock(stdout=json.dumps(pods)),
        ]
        with mock.patch.object(cloud_hpc, "kubectl", side_effect=responses) as kubectl:
            result = cloud_hpc.assess_compare_kubernetes(catalog()["synthetic-heavy-v1"])
        self.assertFalse(result["platform_feasible"])
        self.assertEqual(result["failure_category"], "RESOURCE_INFEASIBLE")
        self.assertEqual(result["max_eligible_node_cpu"], 12)
        self.assertIn("requested CPU 16", result["failure_reason"])
        self.assertEqual(kubectl.call_count, 2)

    @mock.patch.object(cloud_hpc, "run_compare_mode", return_value=0)
    @mock.patch.object(cloud_hpc, "load_catalog", return_value={"synthetic-light-v1": PROFILE})
    def test_compare_and_both_are_recognized_without_external_calls(self, _catalog, compare):
        for mode in ("COMPARE", "BOTH"):
            with self.subTest(mode=mode), mock.patch.object(
                sys, "argv", ["cloud-hpc", "synthetic-light-v1", "--mode", mode]
            ):
                self.assertEqual(cloud_hpc.main(), 0)
        self.assertEqual(compare.call_count, 2)

    @mock.patch.object(cloud_hpc, "dispatch_local")
    @mock.patch.object(cloud_hpc, "wait_review_controller", return_value=(0, "APPROVED_TARGET=LOCAL\n"))
    @mock.patch.object(cloud_hpc, "submit_review")
    @mock.patch.object(cloud_hpc, "human_review", return_value=("approved", "local"))
    @mock.patch.object(cloud_hpc, "start_decision_controller", return_value=("LOCAL", "test", "logs"))
    @mock.patch.object(cloud_hpc, "load_catalog", return_value={"synthetic-light-v1": PROFILE})
    @mock.patch.object(cloud_hpc, "run_compare_mode")
    def test_default_mode_keeps_existing_placement_route(
        self, compare, _catalog, decision, _review, _submit, _wait, dispatch
    ):
        with mock.patch.object(sys, "argv", ["cloud-hpc", "synthetic-light-v1"]):
            self.assertEqual(cloud_hpc.main(), 0)
        decision.assert_called_once_with("synthetic-light-v1")
        dispatch.assert_called_once_with("synthetic-light-v1")
        compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
