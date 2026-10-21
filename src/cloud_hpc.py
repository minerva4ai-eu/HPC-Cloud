#!/usr/bin/env python3

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from benchmark_compare import run_comparison
from unified_controller import assess_kubernetes


BASE = Path(__file__).resolve().parents[1]

NAMESPACE = os.environ.get(
    "CLOUD_HPC_NAMESPACE",
    "cloud-hpc-workloads",
)

REQUEST_CM = "cloud-hpc-b17c-request"
CONTROLLER_JOB = "b17c-unified-controller"

CONTROLLER_MANIFEST = (
    BASE / "k8s" / "b1-7c-controller-job.yaml"
)

LOCAL_MANIFEST = (
    BASE / "k8s" / "b1-6-local-tier-a.yaml"
)

HEAVY_LOCAL_MANIFEST = (
    BASE / "k8s" / "b1-10-local-heavy.yaml"
)

COMMON_HEAVY_LOCAL_MANIFEST = (
    BASE / "k8s" / "b1-10-local-heavy-common.yaml"
)

WORKLOAD_CATALOG = (
    BASE / "config" / "workload-profiles.json"
)

MN5_SPECS = {
    "synthetic-light-v1":
        BASE / "job-specs" / "tier-a-replay.json",

    "synthetic-heavy-v1":
        BASE / "job-specs" / "synthetic-heavy-v1.json",

    "synthetic-heavy-common-v1":
        BASE / "job-specs" / "synthetic-heavy-common-v1.json",
}


def run(
    args,
    *,
    check=True,
    capture=True,
    timeout=30,
    input_text=None,
):
    try:
        proc = subprocess.run(
            list(map(str, args)),
            cwd=BASE,
            text=True,
            capture_output=capture,
            timeout=timeout,
            input=input_text,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Command timed out after "
            f"{timeout}s: "
            + " ".join(map(str, args))
        ) from exc

    if check and proc.returncode != 0:
        if capture:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(
                    proc.stderr,
                    file=sys.stderr,
                    end="",
                )

        raise RuntimeError(
            "Command failed: "
            + " ".join(map(str, args))
            + f" rc={proc.returncode}"
        )

    return proc


def kubectl(*args, **kwargs):
    return run(
        ["kubectl", *args],
        **kwargs,
    )


def apply_generated_configmap(name, file_args):
    """Create/update a ConfigMap from local files without storing generated YAML."""
    create_cmd = [
        "kubectl",
        "-n",
        NAMESPACE,
        "create",
        "configmap",
        name,
    ]

    for key, path in file_args:
        create_cmd.append(
            f"--from-file={key}={path}"
        )

    create_cmd += [
        "--dry-run=client",
        "-o",
        "yaml",
    ]

    rendered = run(
        create_cmd,
        timeout=20,
    )

    applied = run(
        ["kubectl", "-n", NAMESPACE, "apply", "-f", "-"],
        timeout=20,
        input_text=rendered.stdout,
    )

    return applied


def apply_manifest(path):
    """Apply a manifest after rendering the public namespace placeholder."""
    rendered = Path(path).read_text(encoding="utf-8").replace(
        "<KUBERNETES_NAMESPACE>",
        NAMESPACE,
    )
    return run(
        ["kubectl", "-n", NAMESPACE, "apply", "-f", "-"],
        timeout=20,
        input_text=rendered,
    )


def sync_controller_inputs():
    print("[PREPARE] Synchronizing decision-controller inputs...", flush=True)

    apply_generated_configmap(
        "cloud-hpc-workload-catalog",
        [(
            "workload-profiles.json",
            BASE / "config" / "workload-profiles.json",
        )],
    )

    apply_generated_configmap(
        "b17c-controller-code",
        [
            ("decision_engine.py", BASE / "src" / "decision_engine.py"),
            ("unified_controller.py", BASE / "src" / "unified_controller.py"),
        ],
    )

    print("[PREPARE] Workload catalog........ PASS", flush=True)
    print("[PREPARE] Controller source....... PASS", flush=True)


def latest_job_pod(job_name):
    proc = kubectl(
        "-n",
        NAMESPACE,
        "get",
        "pods",
        "-l",
        f"job-name={job_name}",
        "-o",
        "json",
        check=False,
        timeout=10,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None

    items = data.get("items", [])
    if not items:
        return None, None

    items.sort(
        key=lambda item: item.get("metadata", {}).get("creationTimestamp", "")
    )
    pod = items[-1]
    return (
        pod.get("metadata", {}).get("name"),
        pod.get("status", {}).get("phase", "UNKNOWN"),
    )


def delete_job_visible(job_name, timeout_seconds=30):
    print(f"[CLEANUP] Removing previous {job_name} if present...", flush=True)

    kubectl(
        "-n",
        NAMESPACE,
        "delete",
        "job",
        job_name,
        "--ignore-not-found",
        "--wait=false",
        check=False,
        timeout=15,
    )

    started = time.monotonic()
    last_report = -5

    while True:
        proc = kubectl(
            "-n",
            NAMESPACE,
            "get",
            "job",
            job_name,
            "-o",
            "name",
            check=False,
            timeout=10,
        )

        if proc.returncode != 0 or not proc.stdout.strip():
            print("[CLEANUP] Previous controller....... CLEARED", flush=True)
            return

        elapsed = int(time.monotonic() - started)
        if elapsed - last_report >= 5:
            print(
                f"[CLEANUP] {elapsed:>3}s waiting for old controller deletion...",
                flush=True,
            )
            last_report = elapsed

        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out deleting previous Kubernetes Job {job_name}"
            )

        time.sleep(1)


def load_catalog():
    with open(
        WORKLOAD_CATALOG,
        "r",
        encoding="utf-8",
    ) as fh:
        return json.load(fh)


def request_id():
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return f"b18-{stamp}"


def patch_request(data):
    payload = json.dumps(
        {"data": data}
    )

    kubectl(
        "-n",
        NAMESPACE,
        "patch",
        "configmap",
        REQUEST_CM,
        "--type",
        "merge",
        "-p",
        payload,
    )


def controller_logs():
    pod_name, _ = latest_job_pod(CONTROLLER_JOB)

    if not pod_name:
        return ""

    proc = kubectl(
        "-n",
        NAMESPACE,
        "logs",
        pod_name,
        check=False,
        timeout=10,
    )

    if proc.returncode != 0:
        return ""

    return proc.stdout


def controller_job_failed():
    proc = kubectl(
        "-n",
        NAMESPACE,
        "get",
        "job",
        CONTROLLER_JOB,
        "-o",
        "jsonpath={.status.failed}",
        check=False,
    )

    return (
        proc.returncode == 0
        and proc.stdout.strip()
        not in ("", "0")
    )


def start_decision_controller(
    workload,
):
    rid = request_id()

    print()
    print("=============================================")
    print("[1/6] REQUEST")
    print("=============================================")
    print(f"Request ID      : {rid}")
    print(f"Workload        : {workload}")

    sync_controller_inputs()

    print("[PREPARE] Resetting request state...", flush=True)
    patch_request({
        "requestId": rid,
        "workload": workload,
        "reviewPolicy": "required",
        "decision": "pending",
        "approved": "false",
        "approvedTarget": "",
        "rejected": "false",
    })
    print("[PREPARE] Request state............ PASS", flush=True)

    delete_job_visible(CONTROLLER_JOB)

    print()
    print("=============================================")
    print("[2/6] PLACEMENT ASSESSMENT")
    print("=============================================")
    print("[DECISION] Launching Kubernetes assessment controller...", flush=True)

    apply_manifest(CONTROLLER_MANIFEST)

    started = time.monotonic()
    last_logs = ""
    last_progress = -5
    last_phase = None

    while True:
        pod_name, phase = latest_job_pod(CONTROLLER_JOB)
        logs = controller_logs()
        elapsed = int(time.monotonic() - started)

        if phase != last_phase or elapsed - last_progress >= 5:
            print(
                f"[DECISION] {elapsed:>3}s  "
                f"pod={pod_name or 'STARTING':<35} "
                f"phase={phase or 'STARTING':<9} "
                "recommendation=PENDING",
                flush=True,
            )
            last_progress = elapsed
            last_phase = phase

        if logs:
            last_logs = logs

            match = re.search(
                r"^RECOMMENDED_TARGET=(LOCAL|MN5)$",
                logs,
                re.MULTILINE,
            )

            if match:
                recommendation = match.group(1)

                reason_match = re.search(
                    r"^DECISION_REASON=(.+)$",
                    logs,
                    re.MULTILINE,
                )

                reason = (
                    reason_match.group(1)
                    if reason_match
                    else "UNKNOWN"
                )

                print(
                    f"[DECISION] recommendation={recommendation} "
                    f"reason={reason}",
                    flush=True,
                )

                return recommendation, reason, logs

        if controller_job_failed():
            if last_logs:
                print(last_logs)
            raise RuntimeError(
                "Decision controller failed before recommendation"
            )

        if time.monotonic() - started > 90:
            if last_logs:
                print(last_logs)
            raise TimeoutError(
                "Timed out waiting for placement recommendation"
            )

        time.sleep(2)


def show_recommendation(
    workload,
    profile,
    recommendation,
    reason,
):
    print()
    print(
        "============================================="
    )
    print(
        "Cloud-HPC Placement Recommendation"
    )
    print(
        "============================================="
    )

    print(
        f"Workload       : {workload}"
    )

    print(
        "Class          : "
        + str(
            profile.get(
                "workload_class",
                "unknown",
            )
        )
    )

    print(
        "Requested CPU  : "
        + str(
            profile.get(
                "requested_cpu"
            )
        )
    )

    print(
        "Requested RAM  : "
        + str(
            profile.get(
                "requested_memory"
            )
        )
    )

    print(
        "GPU required   : "
        + str(
            profile.get(
                "gpu_required",
                False,
            )
        ).lower()
    )

    print()
    print(
        f"Recommendation : {recommendation}"
    )

    print(
        f"Reason         : {reason}"
    )


def human_review(
    recommendation,
    logs,
):
    while True:
        print()
        print(
            "[A] Approve recommendation"
        )
        print(
            "[R] Reject request"
        )
        print(
            "[O] Override target"
        )
        print(
            "[I] More information"
        )
        print(
            "[C] Compare Kubernetes + MN5"
        )

        choice = input(
            "\nChoice: "
        ).strip().upper()

        if choice == "I":
            print()
            print(
                "========== DECISION DETAILS =========="
            )
            print(logs)
            continue

        if choice == "R":
            return (
                "rejected",
                None,
            )

        if choice == "C":
            return (
                "compare",
                None,
            )

        if choice == "A":
            return (
                "approved",
                recommendation.lower(),
            )

        if choice == "O":
            print()
            print(
                "Override target:"
            )
            print(
                "  [L] LOCAL"
            )
            print(
                "  [M] MN5"
            )

            target = input(
                "Target: "
            ).strip().upper()

            if target == "L":
                return (
                    "override",
                    "local",
                )

            if target == "M":
                return (
                    "override",
                    "mn5",
                )

            print(
                "Invalid override target."
            )
            continue

        print(
            "Invalid choice."
        )


def submit_review(
    action,
    target,
):
    if action == "rejected":

        patch_request({
            "decision": "rejected",
            "approved": "false",
            "approvedTarget": "",
            "rejected": "true",
        })

        return

    patch_request({
        "decision": "approved",
        "approved": "true",
        "approvedTarget": target,
        "rejected": "false",
    })


def wait_review_controller():
    print()
    print("[REVIEW] Waiting for controller validation...", flush=True)

    started = time.monotonic()
    last_logs = ""
    last_report = -5

    while True:
        logs = controller_logs()
        if logs:
            last_logs = logs

        terminal_markers = (
            "HUMAN_REVIEW_STATE=REJECTED",
            "HUMAN_REVIEW_STATE=APPROVED",
            "OVERRIDE_VALIDATION=FAIL",
            "HUMAN_REVIEW_STATE=TIMEOUT",
            "REQUEST_MUTATION=FAIL",
        )

        if any(marker in last_logs for marker in terminal_markers):
            job = kubectl(
                "-n",
                NAMESPACE,
                "get",
                "job",
                CONTROLLER_JOB,
                "-o",
                "json",
                check=False,
                timeout=10,
            )
            rc = 0
            if job.returncode == 0 and job.stdout.strip():
                try:
                    status = json.loads(job.stdout).get("status", {})
                    if status.get("failed", 0):
                        rc = 1
                except json.JSONDecodeError:
                    pass

            print()
            print("========== REVIEW RESULT ==========")
            print(last_logs)
            return rc, last_logs

        if controller_job_failed():
            print()
            print("========== REVIEW RESULT ==========")
            print(last_logs)
            return 1, last_logs

        elapsed = int(time.monotonic() - started)
        if elapsed - last_report >= 5:
            _, phase = latest_job_pod(CONTROLLER_JOB)
            print(
                f"[REVIEW] {elapsed:>3}s controller={phase or 'UNKNOWN'} "
                "validation=PENDING",
                flush=True,
            )
            last_report = elapsed

        if elapsed >= 120:
            raise TimeoutError(
                "Timed out waiting for human-review controller validation"
            )

        time.sleep(2)


def dispatch_local(workload, capture_result=False):
    if workload not in (
        "synthetic-light-v1", "synthetic-heavy-v1",
        "synthetic-heavy-common-v1",
    ):
        raise RuntimeError(
            "No validated LOCAL backend "
            f"registered for {workload}"
        )

    print()
    print(
        "========== LOCAL DISPATCH =========="
    )
    print(
        "Backend=KUBERNETES"
    )

    job_name = "b16-local-tier-a"
    manifest = LOCAL_MANIFEST
    if workload in ("synthetic-heavy-v1", "synthetic-heavy-common-v1"):
        common = workload == "synthetic-heavy-common-v1"
        job_name = "b110-local-heavy-common" if common else "b110-local-heavy"
        manifest = COMMON_HEAVY_LOCAL_MANIFEST if common else HEAVY_LOCAL_MANIFEST
        apply_generated_configmap(
            "cloud-hpc-synthetic-heavy-v1-script",
            [("train_heavy_ai_v1.py", BASE / profile_script(workload))],
        )

    submit_time = datetime.now(timezone.utc).isoformat()

    kubectl(
        "-n",
        NAMESPACE,
        "delete",
        "job",
        job_name,
        "--ignore-not-found",
    )

    apply_manifest(manifest)

    wait_seconds = 300 if workload == "synthetic-light-v1" else 1200
    proc = kubectl(
        "-n",
        NAMESPACE,
        "wait",
        "--for=condition=complete",
        f"job/{job_name}",
        f"--timeout={wait_seconds}s",
        check=False,
        timeout=wait_seconds + 30,
    )

    logs_proc = kubectl(
        "-n",
        NAMESPACE,
        "logs",
        f"job/{job_name}",
        check=False,
    )

    logs = logs_proc.stdout

    print()
    print(
        "========== LOCAL WORKLOAD LOG =========="
    )
    print(logs)

    if proc.returncode != 0:
        pod_proc = kubectl(
            "-n", NAMESPACE, "get", "pods", "-l", f"job-name={job_name}",
            "-o", "json", check=False,
        )
        pod_items = (
            json.loads(pod_proc.stdout).get("items", [])
            if pod_proc.returncode == 0 and pod_proc.stdout.strip() else []
        )
        pod = pod_items[-1] if pod_items else {}
        conditions = pod.get("status", {}).get("conditions", [])
        scheduling = next(
            (item for item in conditions if item.get("reason") == "Unschedulable"),
            {},
        )
        reason = scheduling.get("message") or "Kubernetes Job did not complete"
        return {
            "job": {}, "pod": pod, "logs": logs, "result": {},
            "success": False,
            "platform_feasible": None,
            "state": scheduling.get("reason") or pod.get("status", {}).get("phase"),
            "failure_category": "SCHEDULING_FAILED" if scheduling else "EXECUTION_FAILED",
            "failure_reason": reason,
            "submit_time": submit_time,
        }

    required = (
        [
            "CPU_AI_BASELINE=PASS",
            "RESULT_ACCEPTANCE=PASS",
            "LOCAL_TIER_A=PASS",
            "B1_6B_LOCAL_EXECUTION=PASS",
        ]
        if workload == "synthetic-light-v1"
        else [
            "HEAVY_AI_BASELINE=PASS",
            "RESULT_ACCEPTANCE=PASS",
            "B1_10_LOCAL_HEAVY_EXECUTION=PASS",
        ]
    )

    for marker in required:
        if marker not in logs:
            raise RuntimeError(
                "LOCAL validation marker "
                f"missing: {marker}"
            )

        print(
            f"VALIDATE_LOCAL {marker}=PASS"
        )

    print(
        "LOCAL_BACKEND=PASS"
    )

    if not capture_result:
        return None

    job_proc = kubectl(
        "-n", NAMESPACE, "get", "job", job_name, "-o", "json",
        check=False,
    )
    pod_proc = kubectl(
        "-n", NAMESPACE, "get", "pods", "-l", f"job-name={job_name}",
        "-o", "json", check=False,
    )
    job = json.loads(job_proc.stdout) if job_proc.returncode == 0 else {}
    pod_items = json.loads(pod_proc.stdout).get("items", []) if pod_proc.returncode == 0 else []
    result = parse_workload_result(logs)
    return {
        "job": job,
        "pod": pod_items[-1] if pod_items else {},
        "logs": logs,
        "result": result,
        "output_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True).encode("utf-8")
        ).hexdigest() if result else None,
        "submit_time": submit_time,
        "success": True,
    }


def profile_script(workload):
    catalog = load_catalog()
    return catalog[workload]["workload_script"]


def parse_workload_result(logs):
    decoder = json.JSONDecoder()
    candidates = []
    for position, character in enumerate(logs):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(logs[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "test_accuracy" in value:
            candidates.append(value)
    return candidates[-1] if candidates else {}


def dispatch_mn5(workload, capture_result=False):
    if workload not in MN5_SPECS:
        raise RuntimeError(
            "No MN5 job specification "
            f"registered for {workload}"
        )

    spec = MN5_SPECS[workload]

    if not spec.exists():
        raise FileNotFoundError(
            f"MN5 spec missing: {spec}"
        )

    print()
    print(
        "========== MN5 DISPATCH =========="
    )

    print(
        "Backend=KUBE2HPC_LITE"
    )

    print(
        f"SPEC={spec.relative_to(BASE)}"
    )

    cmd = [
        sys.executable,
        "-u",
        str(
            BASE
            / "src"
            / "run_pipeline.py"
        ),
        str(spec),
        "--poll-seconds",
        "5",
        "--timeout-seconds",
        "900",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )

    output = []

    assert proc.stdout is not None

    try:
        for line in proc.stdout:
            print(
                line,
                end="",
                flush=True,
            )
            output.append(line)

        rc = proc.wait()

    except KeyboardInterrupt:
        print()
        print(
            "[CANCEL] Forwarding operator interrupt "
            "to MN5 lifecycle...",
            flush=True,
        )

        if proc.poll() is None:
            proc.send_signal(
                signal.SIGINT
            )

        try:
            remainder, _ = proc.communicate(
                timeout=45
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            remainder, _ = proc.communicate()
            if remainder:
                print(
                    remainder,
                    end="" if remainder.endswith("\n") else "\n",
                    flush=True,
                )
            raise RuntimeError(
                "MN5 lifecycle did not finish "
                "cancellation handling within 45s"
            )

        if remainder:
            print(
                remainder,
                end="" if remainder.endswith("\n") else "\n",
                flush=True,
            )
            output.append(remainder)

        rc = proc.returncode

        print(
            "[CANCEL] MN5 lifecycle cancellation "
            f"handler exit_code={rc}",
            flush=True,
        )

        raise

    text = "".join(output)

    if rc != 0:
        raise RuntimeError(
            "MN5 Kube2HPC pipeline "
            f"failed rc={rc}"
        )

    if (
        "KUBE2HPC_PIPELINE=PASS"
        not in text
    ):
        raise RuntimeError(
            "MN5 pipeline did not emit "
            "PASS marker"
        )

    print(
        "MN5_BACKEND=PASS"
    )

    if not capture_result:
        return None

    match = re.search(r"^SUMMARY_JSON=(.+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("MN5 pipeline summary path was not emitted")
    with open(match.group(1).strip(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_compare_mode(workload, profile):
    print("=============================================")
    print("Cloud-HPC B1.10 Paired Benchmark")
    print("=============================================")
    print(f"Workload        : {workload}")
    print("Execution order : KUBERNETES -> MN5")

    comparison, directory = run_comparison(
        workload,
        profile,
        lambda name, _comparison_id, _run_id, _profile: dispatch_local(
            name, capture_result=True
        ),
        lambda name, _comparison_id, _run_id, _profile: dispatch_mn5(
            name, capture_result=True
        ),
        BASE / "artifacts" / "benchmarks",
        k8s_feasibility=assess_compare_kubernetes,
    )
    print(f"COMPARISON_ID={comparison['comparison_id']}")
    print(f"COMPARISON_RESULT={directory}")
    print("CLOUD_HPC_COMPARE=PASS")
    return 0


def assess_compare_kubernetes(profile):
    nodes_proc = kubectl("get", "nodes", "-o", "json", timeout=20)
    pods_proc = kubectl("get", "pods", "--all-namespaces", "-o", "json", timeout=20)
    nodes = json.loads(nodes_proc.stdout).get("items", [])
    pods = json.loads(pods_proc.stdout).get("items", [])
    assessments, feasible_nodes = assess_kubernetes(profile, nodes=nodes, pods=pods)
    if feasible_nodes:
        return {"platform_feasible": True, "resource_assessment": assessments}

    requested_cpu = profile.get("requested_cpu")
    max_cpu = max((item["alloc_cpu"] for item in assessments), default=None)
    cpu_infeasible = bool(assessments) and all(
        not item["cpu_ok"] for item in assessments
    )
    if max_cpu is None:
        reason = "No eligible Kubernetes worker was reported by live resource assessment."
    elif cpu_infeasible:
        reason = (
            f"No eligible Kubernetes node can satisfy requested CPU {requested_cpu}; "
            f"eligible workers expose at most {max_cpu:g} allocatable CPUs per node."
        )
    else:
        reason = "No eligible Kubernetes node can satisfy the complete resource request."
    return {
        "platform_feasible": False,
        "state": "INFEASIBLE",
        "failure_category": "RESOURCE_INFEASIBLE",
        "failure_reason": reason,
        "requested_cpu": requested_cpu,
        "max_eligible_node_cpu": max_cpu,
        "resource_assessment": assessments,
        "success": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cloud-HPC unified "
            "placement and execution controller"
        )
    )

    parser.add_argument(
        "workload",
        help=(
            "Workload profile name, e.g. "
            "synthetic-light-v1 or "
            "synthetic-heavy-common-v1 or "
            "synthetic-heavy-v1"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str.upper,
        choices=("PLACEMENT", "COMPARE", "BOTH"),
        default="PLACEMENT",
        help="PLACEMENT (default), or paired benchmark mode COMPARE/BOTH",
    )

    args = parser.parse_args()

    catalog = load_catalog()

    if args.workload not in catalog:
        print(
            "CLOUD_HPC_ERROR="
            f"Unknown workload: {args.workload}",
            file=sys.stderr,
        )
        return 2

    profile = catalog[
        args.workload
    ]

    if args.mode in ("COMPARE", "BOTH"):
        return run_compare_mode(args.workload, profile)

    print(
        "============================================="
    )
    print(
        "Cloud-HPC Unified Controller"
    )
    print(
        "============================================="
    )

    try:

        (
            recommendation,
            reason,
            decision_logs,
        ) = start_decision_controller(
            args.workload
        )

        show_recommendation(
            args.workload,
            profile,
            recommendation,
            reason,
        )

        (
            action,
            approved_target,
        ) = human_review(
            recommendation,
            decision_logs,
        )

        if action == "compare":
            print()
            print("OPERATIONAL_INTENT=COMPARE")
            print("PLACEMENT_RECOMMENDATION=UNCHANGED")
            delete_job_visible(CONTROLLER_JOB)
            return run_compare_mode(args.workload, profile)

        submit_review(
            action,
            approved_target,
        )

        (
            review_rc,
            review_logs,
        ) = wait_review_controller()

        if action == "rejected":

            if (
                "HUMAN_REVIEW_STATE=REJECTED"
                not in review_logs
            ):
                raise RuntimeError(
                    "Controller did not confirm rejection"
                )

            print()
            print(
                "EXECUTION_DISPATCH=NONE"
            )
            print(
                "CLOUD_HPC_REQUEST=REJECTED"
            )

            return 0

        if "OVERRIDE_VALIDATION=FAIL" in review_logs:
            reason_match = re.search(
                r"^REASON=(.+)$",
                review_logs,
                re.MULTILINE,
            )

            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "OVERRIDE_REJECTED"
            )

            print()
            print(
                "========== EXECUTION =========="
            )
            print(
                "OVERRIDE_VALIDATION=FAIL"
            )
            print(
                f"REASON={reason}"
            )
            print(
                "EXECUTION_DISPATCH=NONE"
            )
            print(
                "CLOUD_HPC_REQUEST=BLOCKED"
            )

            return 2

        if review_rc != 0:
            raise RuntimeError(
                "Human-review controller "
                "rejected/failed the requested target"
            )

        match = re.search(
            r"^APPROVED_TARGET="
            r"(LOCAL|MN5)$",
            review_logs,
            re.MULTILINE,
        )

        if not match:
            raise RuntimeError(
                "Unable to determine "
                "approved execution target"
            )

        target = match.group(1)

        print()
        print(
            "========== EXECUTION =========="
        )
        print(
            f"APPROVED_TARGET={target}"
        )

        if target == "LOCAL":
            dispatch_local(
                args.workload
            )

        elif target == "MN5":
            dispatch_mn5(
                args.workload
            )

        else:
            raise RuntimeError(
                f"Unsupported target {target}"
            )

        print()
        print(
            "============================================="
        )
        print(
            "CLOUD_HPC_REQUEST=PASS"
        )
        print(
            "============================================="
        )

        return 0

    except KeyboardInterrupt:

        print()
        print(
            "[CANCEL] Request interrupted by operator.",
            flush=True,
        )

        kubectl(
            "-n",
            NAMESPACE,
            "delete",
            "job",
            CONTROLLER_JOB,
            "--ignore-not-found",
            check=False,
        )

        print(
            "[CANCEL] Decision controller cleaned up.",
            flush=True,
        )

        print(
            "CLOUD_HPC_REQUEST=CANCELLED"
        )

        return 130

    except Exception as exc:
        print(
            f"CLOUD_HPC_ERROR={exc}",
            file=sys.stderr,
        )

        print(
            "CLOUD_HPC_REQUEST=FAIL",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
