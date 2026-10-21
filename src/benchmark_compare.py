#!/usr/bin/env python3

"""B1.10 paired Kubernetes/MN5 benchmark evidence and normalization."""

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PLATFORMS = ("k8s", "mn5")
SEMANTIC_RESULT_FIELDS = (
    "experiment", "seed", "train_samples", "test_samples", "features",
    "classes", "epochs", "batch_size", "cpu_threads",
    "final_train_accuracy", "test_accuracy", "test_loss",
)
SEMANTIC_FLOAT_REL_TOL = 1e-9
SEMANTIC_FLOAT_ABS_TOL = 1e-12
DEFAULT_HPC_MEMORY_PER_REQUESTED_CORE_GB = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def new_comparison_id(now=None):
    current = now or datetime.now(timezone.utc)
    return "CMP-" + current.strftime("%Y%m%dT%H%M%S%fZ")


def _parse_time(value, source_timezone=None):
    if not value or value in ("Unknown", "N/A", "None"):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        if source_timezone is None:
            return None
        parsed = parsed.replace(tzinfo=source_timezone)
    return parsed


def timing(submit_time, start_time, end_time, source_timezone=None):
    submit = _parse_time(submit_time, source_timezone)
    start = _parse_time(start_time, source_timezone)
    end = _parse_time(end_time, source_timezone)

    def seconds(later, earlier):
        if later is None or earlier is None:
            return None
        return round(max((later - earlier).total_seconds(), 0.0), 6)

    return {
        "submit_time": submit.isoformat() if submit else None,
        "start_time": start.isoformat() if start else None,
        "end_time": end.isoformat() if end else None,
        "raw_scheduler_timestamps": {
            "submit_time": submit_time or None,
            "start_time": start_time or None,
            "end_time": end_time or None,
        },
        "queue_wait_s": seconds(start, submit),
        "execution_time_s": seconds(end, start),
        "end_to_end_s": seconds(end, submit),
    }


def _quantity(value):
    return value if value not in ("", None) else None


def _memory_megabytes(value):
    if value in ("", None):
        return None
    text = str(value).strip().upper()
    try:
        if text.endswith("GB"):
            return float(text[:-2]) * 1000
        if text.endswith("G"):
            return float(text[:-1]) * 1000
        if text.endswith("MB"):
            return float(text[:-2])
        if text.endswith("M"):
            return float(text[:-1])
    except ValueError:
        return None
    return None


def _base(comparison_id, run_id, workload, platform, backend, spec_hash):
    return {
        "identity": {
            "comparison_id": comparison_id,
            "execution_run_id": run_id,
            "workload_profile": workload,
            "platform": platform,
            "backend_scheduler": backend,
            "node_host": None,
            "workload_config_sha256": spec_hash,
        },
        "timing": timing(None, None, None),
        "status": {
            "platform_feasible": None,
            "scheduler_state": None,
            "success": False,
            "exit_code": None,
            "failure_category": None,
            "failure_reason": None,
        },
        "resources": {
            "requested_cpu": None,
            "requested_memory": None,
            "scheduler_allocated_cpu": None,
            "accounting_alloc_cpus": None,
            "runtime_cpus_per_task": None,
            "runtime_cpus_on_node": None,
            "workload_cpu_threads": None,
            "max_eligible_node_cpu": None,
            "allocated_cpu": None,
            "allocated_memory": None,
            "peak_memory_mb": None,
            "runtime_cpu_utilization": None,
            "scheduler_requested_memory": None,
            "memory_policy_source": None,
            "memory_per_requested_core_gb": None,
            "expected_memory_from_cpu": None,
            "expected_memory_from_cpu_gb": None,
            "memory_policy_match": None,
        },
        "workload": {
            "input_size": None,
            "output_size": None,
            "processed_work_units": None,
            "throughput": None,
        },
        "integrity": {
            "raw_output_sha256": None,
            "raw_output_sha256_match": None,
            "semantic_output_equivalent": None,
            "semantic_fields_checked": [],
            "semantic_mismatches": [],
            "semantic_float_tolerance": {
                "relative": SEMANTIC_FLOAT_REL_TOL,
                "absolute": SEMANTIC_FLOAT_ABS_TOL,
            },
            "output_sha256": None,
            "output_equivalent": None,
            "semantic_values": {},
        },
        "energy": None,
        "cost": None,
        "evidence": {},
        "provenance": {},
    }


def normalize_k8s(raw, comparison_id, run_id, workload, profile, spec_hash):
    result = _base(
        comparison_id, run_id, workload, "KUBERNETES", "Kubernetes", spec_hash
    )
    job = raw.get("job", {})
    pod = raw.get("pod", {})
    status = pod.get("status", {})
    pod_spec = pod.get("spec", {})
    containers = status.get("containerStatuses", [])
    terminated = {}
    if containers:
        terminated = containers[0].get("state", {}).get("terminated", {})
    container_specs = pod_spec.get("containers", [])
    resources = container_specs[0].get("resources", {}) if container_specs else {}
    requests = resources.get("requests", {})
    limits = resources.get("limits", {})
    submit = job.get("metadata", {}).get("creationTimestamp") or raw.get("submit_time")
    start = status.get("startTime") or raw.get("start_time")
    end = terminated.get("finishedAt") or raw.get("end_time")
    phase = status.get("phase") or raw.get("state")
    exit_code = terminated.get("exitCode", raw.get("exit_code"))
    success = raw.get("success")
    if success is None:
        success = phase == "Succeeded" and exit_code in (None, 0)
    workload_result = raw.get("result", {}) or {}

    result["identity"]["node_host"] = pod_spec.get("nodeName")
    result["timing"] = timing(submit, start, end)
    result["status"].update({
        "platform_feasible": raw.get("platform_feasible", True),
        "scheduler_state": phase,
        "success": bool(success),
        "exit_code": exit_code,
        "failure_category": raw.get("failure_category"),
        "failure_reason": raw.get("failure_reason") or terminated.get("reason"),
    })
    result["resources"].update({
        "requested_cpu": _quantity(requests.get("cpu") or profile.get("requested_cpu")),
        "requested_memory": _quantity(requests.get("memory") or profile.get("requested_memory")),
        "allocated_cpu": _quantity(limits.get("cpu")),
        "allocated_memory": _quantity(limits.get("memory")),
        "peak_memory_mb": workload_result.get("max_rss_mb"),
        "workload_cpu_threads": workload_result.get("cpu_threads"),
        "max_eligible_node_cpu": raw.get("max_eligible_node_cpu"),
    })
    result["workload"].update({
        "input_size": workload_result.get("train_samples"),
        "processed_work_units": workload_result.get("test_samples"),
        "throughput": workload_result.get("inference_samples_per_second"),
    })
    result["integrity"]["raw_output_sha256"] = raw.get("output_sha256") or (
        sha256_json(workload_result) if workload_result else None
    )
    result["integrity"]["output_sha256"] = result["integrity"]["raw_output_sha256"]
    result["integrity"]["semantic_values"] = {
        field: workload_result[field]
        for field in SEMANTIC_RESULT_FIELDS if field in workload_result
    }
    result["evidence"] = {"logs": "k8s/logs.txt" if raw.get("logs") else None}
    if raw.get("resource_assessment") is not None:
        result["evidence"]["resource_assessment"] = raw["resource_assessment"]
    result["provenance"] = {
        "timestamps": "Kubernetes API / container status",
        "status": "Kubernetes API / container status",
        "resources": "Kubernetes Pod specification",
        "peak_memory_mb": "workload output" if workload_result.get("max_rss_mb") is not None else "unavailable",
        "runtime_cpu_utilization": "unavailable (metrics-server not required)",
        "energy": "unavailable",
        "cost": "unavailable",
    }
    return result


def normalize_mn5(raw, comparison_id, run_id, workload, profile, spec_hash):
    result = _base(comparison_id, run_id, workload, "MN5", "Slurm", spec_hash)
    accounting = raw.get("accounting", {}) or {}
    workload_result = raw.get("result", {}) or {}
    scheduler_requested_memory = _quantity(accounting.get("ReqMem"))
    try:
        workload_requested_cpu = float(profile.get("requested_cpu"))
    except (TypeError, ValueError):
        workload_requested_cpu = None
    memory_per_requested_core_gb = float(
        profile.get(
            "hpc_memory_per_requested_core_gb",
            DEFAULT_HPC_MEMORY_PER_REQUESTED_CORE_GB,
        )
    )
    expected_memory_gb = (
        workload_requested_cpu * memory_per_requested_core_gb
        if workload_requested_cpu is not None else None
    )
    expected_memory_mb = expected_memory_gb * 1000 if expected_memory_gb is not None else None
    observed_memory_mb = _memory_megabytes(scheduler_requested_memory)
    state = accounting.get("State") or raw.get("remote_terminal_state") or raw.get("state")
    exit_code = accounting.get("ExitCode") or raw.get("remote_exit_code") or raw.get("exit_code")
    success = raw.get("success")
    if success is None:
        success = str(state or "").split("+")[0] == "COMPLETED" and exit_code in (None, "0:0", 0)
    result["identity"]["execution_run_id"] = str(
        raw.get("job_id") or accounting.get("JobIDRaw") or run_id
    )
    result["identity"]["node_host"] = accounting.get("NodeList")
    result["timing"] = timing(
        accounting.get("Submit") or raw.get("submit_time"),
        accounting.get("Start") or raw.get("start_time"),
        accounting.get("End") or raw.get("end_time"),
        timezone.utc if raw.get("accounting_timezone") == "UTC" else None,
    )
    result["status"].update({
        "platform_feasible": raw.get("platform_feasible", True),
        "scheduler_state": state,
        "success": bool(success),
        "exit_code": exit_code,
        "failure_reason": raw.get("failure_reason") if not success else None,
    })
    result["resources"].update({
        "requested_cpu": _quantity(accounting.get("ReqCPUS") or profile.get("requested_cpu")),
        "requested_memory": _quantity(accounting.get("ReqMem") or profile.get("requested_memory")),
        "scheduler_requested_memory": scheduler_requested_memory,
        "memory_policy_source": "configured HPC memory policy",
        "memory_per_requested_core_gb": memory_per_requested_core_gb,
        "expected_memory_from_cpu": (
            f"{expected_memory_mb:g}M" if expected_memory_mb is not None else None
        ),
        "expected_memory_from_cpu_gb": expected_memory_gb,
        "memory_policy_match": (
            observed_memory_mb == expected_memory_mb
            if observed_memory_mb is not None and expected_memory_mb is not None
            else None
        ),
        "scheduler_allocated_cpu": _quantity(accounting.get("AllocCPUS")),
        "accounting_alloc_cpus": _quantity(accounting.get("AllocCPUS")),
        "runtime_cpus_per_task": _quantity(raw.get("runtime_cpus_per_task")),
        "runtime_cpus_on_node": _quantity(raw.get("runtime_cpus_on_node")),
        "workload_cpu_threads": workload_result.get("cpu_threads"),
        "allocated_memory": _quantity(accounting.get("AllocMem")),
        "peak_memory_mb": accounting.get("MaxRSS") or workload_result.get("max_rss_mb"),
    })
    result["workload"].update({
        "input_size": workload_result.get("train_samples"),
        "processed_work_units": workload_result.get("test_samples"),
        "throughput": workload_result.get("inference_samples_per_second"),
    })
    hashes = raw.get("sha256", {}) or {}
    result["integrity"]["raw_output_sha256"] = hashes.get("result.json") or (
        sha256_json(workload_result) if workload_result else None
    )
    result["integrity"]["output_sha256"] = result["integrity"]["raw_output_sha256"]
    result["integrity"]["semantic_values"] = {
        field: workload_result[field]
        for field in SEMANTIC_RESULT_FIELDS if field in workload_result
    }
    result["evidence"] = {"existing_job_directory": raw.get("job_dir")}
    result["provenance"] = {
        "timestamps": "Slurm sacct queried with TZ=UTC",
        "scheduler_timestamp_timezone": raw.get("accounting_timezone") or "unknown",
        "status": "Slurm sacct",
        "resources": "Slurm sacct (scheduler allocation) / workload result and profile",
        "accounting_alloc_cpus": "raw Slurm sacct AllocCPUS",
        "scheduler_requested_memory": "raw Slurm sacct ReqMem",
        "memory_policy": (
            "configured HPC policy; "
            f"{memory_per_requested_core_gb:g} GB per requested workload CPU core"
        ),
        "peak_memory_mb": "Slurm sacct or workload output" if result["resources"]["peak_memory_mb"] is not None else "unavailable",
        "runtime_cpu_utilization": "unavailable",
        "energy": "unavailable",
        "cost": "unavailable",
    }
    return result


def failure_metrics(comparison_id, run_id, workload, platform, profile, spec_hash, exc):
    backend = "Kubernetes" if platform == "KUBERNETES" else "Slurm"
    result = _base(comparison_id, run_id, workload, platform, backend, spec_hash)
    result["status"]["failure_reason"] = str(exc)
    result["resources"]["requested_cpu"] = profile.get("requested_cpu")
    result["resources"]["requested_memory"] = profile.get("requested_memory")
    result["provenance"] = {"failure": "controller exception", "optional_metrics": "unavailable"}
    return result


def _delta(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(right - left, 6)
    return None


def _semantic_comparison(left, right):
    if not left or not right:
        return [], [], None
    fields = [field for field in SEMANTIC_RESULT_FIELDS if field in left or field in right]
    mismatches = []
    for field in fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            equal = math.isclose(
                left_value, right_value,
                rel_tol=SEMANTIC_FLOAT_REL_TOL,
                abs_tol=SEMANTIC_FLOAT_ABS_TOL,
            )
        else:
            equal = left_value == right_value
        if not equal:
            mismatches.append({"field": field, "k8s": left_value, "mn5": right_value})
    return fields, mismatches, (not mismatches if fields else None)


def build_comparison(comparison_id, workload, spec_hash, executions, started, finished):
    k8s = executions["k8s"]
    mn5 = executions["mn5"]
    k8s_hash = k8s["integrity"]["raw_output_sha256"]
    mn5_hash = mn5["integrity"]["raw_output_sha256"]
    raw_match = None if not k8s_hash or not mn5_hash else k8s_hash == mn5_hash
    fields, mismatches, semantic_equivalent = _semantic_comparison(
        k8s["integrity"]["semantic_values"], mn5["integrity"]["semantic_values"]
    )
    for execution in (k8s, mn5):
        integrity = execution["integrity"]
        integrity["raw_output_sha256_match"] = raw_match
        integrity["semantic_output_equivalent"] = semantic_equivalent
        integrity["semantic_fields_checked"] = fields
        integrity["semantic_mismatches"] = mismatches
        integrity["semantic_comparison_status"] = (
            "COMPARED" if semantic_equivalent is not None
            else "NOT_PERFORMED_MISSING_OUTPUT"
        )
        # Backward-compatible alias; equivalence now means workload correctness.
        integrity["output_equivalent"] = semantic_equivalent
    return {
        "schema_version": "b1.10-v1",
        "comparison_id": comparison_id,
        "workload_profile": workload,
        "workload_config_sha256": spec_hash,
        "execution_order": ["KUBERNETES", "MN5"],
        "started_at": started,
        "finished_at": finished,
        "executions": executions,
        "derived_deltas_mn5_minus_k8s": {
            name: _delta(k8s["timing"].get(name), mn5["timing"].get(name))
            for name in ("queue_wait_s", "execution_time_s", "end_to_end_s")
        },
    }


def write_artifacts(root, workload_spec, comparison, raw_records=None):
    directory = Path(root) / comparison["comparison_id"]
    (directory / "k8s").mkdir(parents=True, exist_ok=True)
    (directory / "mn5").mkdir(parents=True, exist_ok=True)

    def write_json(path, value):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, sort_keys=True)
            fh.write("\n")

    write_json(directory / "workload.json", workload_spec)
    for platform in PLATFORMS:
        write_json(directory / platform / "metrics.json", comparison["executions"][platform])
        execution = {
            "identity": comparison["executions"][platform]["identity"],
            "timing": comparison["executions"][platform]["timing"],
            "status": comparison["executions"][platform]["status"],
        }
        write_json(directory / platform / "execution.json", execution)
    raw_records = raw_records or {}
    k8s_logs = (raw_records.get("k8s") or {}).get("logs")
    if k8s_logs:
        (directory / "k8s" / "logs.txt").write_text(k8s_logs, encoding="utf-8")
    write_json(directory / "comparison.json", comparison)

    rows = []
    for platform in PLATFORMS:
        item = comparison["executions"][platform]
        rows.append({
            "comparison_id": comparison["comparison_id"],
            "workload_profile": comparison["workload_profile"],
            "platform": item["identity"]["platform"],
            "execution_run_id": item["identity"]["execution_run_id"],
            "success": item["status"]["success"],
            "scheduler_state": item["status"]["scheduler_state"],
            "queue_wait_s": item["timing"]["queue_wait_s"],
            "execution_time_s": item["timing"]["execution_time_s"],
            "end_to_end_s": item["timing"]["end_to_end_s"],
            "throughput": item["workload"]["throughput"],
            "requested_memory": item["resources"]["requested_memory"],
            "scheduler_requested_memory": item["resources"]["scheduler_requested_memory"],
            "expected_memory_from_cpu": item["resources"]["expected_memory_from_cpu"],
            "memory_policy_match": item["resources"]["memory_policy_match"],
            "peak_memory_mb": item["resources"]["peak_memory_mb"],
            "energy": item["energy"],
            "cost": item["cost"],
            "failure_reason": item["status"]["failure_reason"],
        })
    with open(directory / "comparison.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = (
        ("Status", "status", "success"),
        ("Queue wait (s)", "timing", "queue_wait_s"),
        ("Execution runtime (s)", "timing", "execution_time_s"),
        ("End-to-end (s)", "timing", "end_to_end_s"),
        ("Throughput", "workload", "throughput"),
        ("Peak memory (MB)", "resources", "peak_memory_mb"),
        ("Requested/reserved memory", "resources", "requested_memory"),
        ("MN5 memory policy expected", "resources", "expected_memory_from_cpu"),
        ("MN5 memory policy match", "resources", "memory_policy_match"),
        ("Energy", None, "energy"),
        ("Cost", None, "cost"),
        ("Output SHA-256", "integrity", "output_sha256"),
        ("Raw SHA-256 match", "integrity", "raw_output_sha256_match"),
        ("Semantic output equivalent", "integrity", "semantic_output_equivalent"),
    )
    lines = [
        f"# B1.10 comparison {comparison['comparison_id']}", "",
        f"Workload: `{comparison['workload_profile']}`", "",
        "Execution order: Kubernetes, then MN5.", "",
        "| KPI | Kubernetes | MN5 |", "|---|---:|---:|",
    ]
    for label, section, key in labels:
        values = []
        for platform in PLATFORMS:
            item = comparison["executions"][platform]
            value = item.get(key) if section is None else item[section].get(key)
            values.append("unavailable" if value is None else str(value))
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def run_comparison(
    workload, profile, k8s_runner, mn5_runner, artifact_root,
    k8s_feasibility=None,
):
    comparison_id = new_comparison_id()
    spec_hash = sha256_json({"profile_name": workload, "profile": profile})
    started = utc_now()
    executions = {}
    raw_records = {}
    runners = (("k8s", "KUBERNETES", k8s_runner, normalize_k8s),
               ("mn5", "MN5", mn5_runner, normalize_mn5))
    for index, (key, platform, runner, normalizer) in enumerate(runners, start=1):
        run_id = f"{comparison_id}-{key.upper()}"
        try:
            raw = None
            assessment = None
            if key == "k8s" and k8s_feasibility is not None:
                assessment = k8s_feasibility(profile)
                if assessment.get("platform_feasible") is False:
                    raw = assessment
            if raw is None:
                raw = runner(workload, comparison_id, run_id, profile)
                if assessment:
                    raw = dict(raw or {})
                    raw.setdefault("platform_feasible", True)
                    raw.setdefault(
                        "resource_assessment",
                        assessment.get("resource_assessment"),
                    )
            raw_records[key] = raw or {}
            executions[key] = normalizer(raw or {}, comparison_id, run_id, workload, profile, spec_hash)
        except Exception as exc:  # preserve the other platform's evidence
            executions[key] = failure_metrics(
                comparison_id, run_id, workload, platform, profile, spec_hash, exc
            )
        executions[key]["identity"]["execution_order"] = index
    comparison = build_comparison(
        comparison_id, workload, spec_hash, executions, started, utc_now()
    )
    directory = write_artifacts(
        artifact_root,
        {"profile_name": workload, "profile": profile, "sha256": spec_hash},
        comparison,
        raw_records,
    )
    return comparison, directory
