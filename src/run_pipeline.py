#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
CONTROLLER = BASE / "src" / "kube2hpc.py"


CURRENT_JOB_ID = None
CURRENT_JOB_METADATA = None


TERMINAL_FAILURES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def controller(*args, check=True):
    cmd = [
        sys.executable,
        str(CONTROLLER),
        *map(str, args),
    ]

    proc = subprocess.run(
        cmd,
        cwd=BASE,
        text=True,
        capture_output=True,
    )

    if proc.stdout:
        print(proc.stdout, end="")

    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Controller command failed: "
            f"{' '.join(cmd)} "
            f"rc={proc.returncode}"
        )

    return proc


def parse_jobid(text):
    match = re.search(
        r"^JOBID=(\d+)$",
        text,
        re.MULTILINE,
    )

    if not match:
        raise RuntimeError(
            "Unable to parse Slurm Job ID"
        )

    return match.group(1)


def read_metadata(jobid):
    path = (
        BASE
        / "results"
        / "remote-jobs"
        / f"job-{jobid}"
        / "metadata.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata missing: {path}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def query_remote_status(jobid, metadata):
    alias = metadata["target"]["ssh_alias"]

    active_cmd = (
        f"squeue -h -j {jobid} "
        "-o '%T|%M|%R' "
        "| head -n 1"
    )

    proc = subprocess.run(
        ["ssh", alias, active_cmd],
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"squeue failed rc={proc.returncode}: "
            f"{proc.stderr.strip()}"
        )

    active = proc.stdout.strip()

    if active:
        state = active.split("|", 1)[0]

        return {
            "source": "squeue",
            "state": state,
            "raw": active,
        }

    acct_cmd = (
        f"TZ=UTC sacct -j {jobid} "
        "-X -n -P "
        "--format="
        "JobIDRaw,State,ExitCode,Elapsed,"
        "Submit,Start,End,AllocCPUS,ReqMem,MaxRSS,NodeList "
        "| head -n 1"
    )

    proc = subprocess.run(
        ["ssh", alias, acct_cmd],
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"sacct failed rc={proc.returncode}: "
            f"{proc.stderr.strip()}"
        )

    raw = proc.stdout.strip()

    if not raw:
        return {
            "source": "sacct",
            "state": "UNKNOWN",
            "raw": "",
        }

    fields = raw.split("|")
    names = (
        "JobIDRaw", "State", "ExitCode", "Elapsed", "Submit", "Start",
        "End", "AllocCPUS", "ReqMem", "MaxRSS", "NodeList",
    )
    accounting = {
        name: fields[index] if index < len(fields) and fields[index] else None
        for index, name in enumerate(names)
    }

    state = fields[1] if len(fields) > 1 else "UNKNOWN"
    exit_code = fields[2] if len(fields) > 2 else ""

    return {
        "source": "sacct",
        "accounting_timezone": "UTC",
        "state": state,
        "exit_code": exit_code,
        "raw": raw,
        "accounting": accounting,
    }



def remember_remote_job(jobid, metadata):
    global CURRENT_JOB_ID
    global CURRENT_JOB_METADATA

    CURRENT_JOB_ID = str(jobid)
    CURRENT_JOB_METADATA = metadata


def write_cancellation_evidence(jobid, reason, before, after, cancel_requested):
    job_dir = (
        BASE
        / "results"
        / "remote-jobs"
        / f"job-{jobid}"
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "job_id": str(jobid),
        "reason": reason,
        "recorded_utc": utc_now(),
        "cancel_requested": bool(cancel_requested),
        "state_before": before,
        "state_after": after,
        "orphan_protection": "PASS",
    }

    path = job_dir / "cancellation.json"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            payload,
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    print(f"CANCELLATION_JSON={path}")
    print(f"CANCELLATION_SHA256={sha256_file(path)}")


def protect_remote_job(reason, confirm_timeout_seconds=30):
    jobid = CURRENT_JOB_ID
    metadata = CURRENT_JOB_METADATA

    if not jobid or metadata is None:
        print("REMOTE_JOB_SUBMITTED=NO")
        print("REMOTE_CANCEL_REQUESTED=NO")
        print("REMOTE_ORPHAN_PROTECTION=PASS")
        return

    print()
    print("========== REMOTE ORPHAN PROTECTION ==========")
    print(f"JOBID={jobid}")
    print(f"CANCEL_REASON={reason}")

    before = query_remote_status(
        jobid,
        metadata,
    )

    print(
        "REMOTE_STATE_BEFORE_CANCEL="
        f"{before['state']}"
    )

    cancel_requested = False

    if before["source"] == "squeue":
        cancel_proc = controller(
            "cancel",
            jobid,
            check=False,
        )

        if cancel_proc.returncode != 0:
            raise RuntimeError(
                "Remote Slurm cancellation request failed "
                f"for job {jobid}"
            )

        cancel_requested = True
        print("REMOTE_CANCEL_REQUESTED=YES")
    else:
        print("REMOTE_CANCEL_REQUESTED=NO")
        print("REMOTE_JOB_ALREADY_TERMINAL=YES")

    deadline = (
        time.monotonic()
        + confirm_timeout_seconds
    )

    after = before

    while cancel_requested:
        after = query_remote_status(
            jobid,
            metadata,
        )

        if after["source"] != "squeue":
            break

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Remote job still active after cancellation "
                f"request: job {jobid} state={after['state']}"
            )

        print(
            "REMOTE_CANCEL_WAIT "
            f"jobid={jobid} state={after['state']}"
        )
        time.sleep(2)

    if cancel_requested:
        print("REMOTE_CANCEL_CONFIRMED=YES")

    print(
        "REMOTE_STATE_AFTER_CANCEL="
        f"{after['state']}"
    )
    print("REMOTE_ORPHAN_PROTECTION=PASS")

    write_cancellation_evidence(
        jobid,
        reason,
        before,
        after,
        cancel_requested,
    )


def wait_for_job(
    jobid,
    metadata,
    poll_seconds,
    timeout_seconds,
):
    started = time.monotonic()

    while True:
        info = query_remote_status(
            jobid,
            metadata,
        )

        elapsed = time.monotonic() - started

        print(
            f"POLL jobid={jobid} "
            f"source={info['source']} "
            f"state={info['state']} "
            f"elapsed_wait={elapsed:.1f}s"
        )

        state = info["state"].split("+")[0]

        if state == "COMPLETED":
            if info.get("exit_code") != "0:0":
                raise RuntimeError(
                    "Slurm reports COMPLETED "
                    f"but ExitCode={info.get('exit_code')}"
                )

            return info

        if state in TERMINAL_FAILURES:
            raise RuntimeError(
                f"Remote Slurm job ended in "
                f"{state}: {info['raw']}"
            )

        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after "
                f"{timeout_seconds}s waiting "
                f"for job {jobid}"
            )

        time.sleep(poll_seconds)


def validate_outputs(jobid):
    job_dir = (
        BASE
        / "results"
        / "remote-jobs"
        / f"job-{jobid}"
    )

    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    result_path = job_dir / "result.json"
    metadata_path = job_dir / "metadata.json"

    required = [
        metadata_path,
        stdout_path,
        stderr_path,
        result_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required output missing: {path}"
            )

    stdout_text = stdout_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as fh:
        metadata = json.load(fh)

    spec = metadata.get(
        "spec",
        {},
    )

    required_markers = spec.get(
        "success_markers",
        [
            "SCRIPT_CHECKSUM=PASS",
            "SIF_CHECKSUM=PASS",
            "CPU_AI_BASELINE=PASS",
            "AI_EXIT_CODE=0",
            "MN5_TIER_A=PASS",
        ],
    )

    if not required_markers:
        raise RuntimeError(
            "No success markers defined for workload"
        )

    print(
        "VALIDATION_PROFILE="
        + spec.get(
            "name",
            "legacy",
        )
    )

    marker_results = {}

    for marker in required_markers:
        passed = marker in stdout_text
        marker_results[marker] = passed

        print(
            f"VALIDATE_MARKER "
            f"{marker}="
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            raise RuntimeError(
                f"Required success marker "
                f"missing: {marker}"
            )

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as fh:
        result_data = json.load(fh)

    if result_data.get("test_accuracy", 0) < 0.95:
        raise RuntimeError(
            "Result accuracy acceptance "
            "criterion failed"
        )

    hashes = {}

    for path in required:
        hashes[path.name] = sha256_file(path)

    return {
        "job_dir": str(job_dir),
        "markers": marker_results,
        "result": result_data,
        "sha256": hashes,
    }


def run_pipeline(
    spec,
    poll_seconds,
    timeout_seconds,
):
    pipeline_start = time.monotonic()
    started_utc = utc_now()

    print(
        "============================================="
    )
    print(
        "Kube2HPC-Lite Automated Lifecycle"
    )
    print(
        "============================================="
    )

    print(f"PIPELINE_START_UTC={started_utc}")
    print(f"SPEC={spec}")

    print()
    print("========== STAGE 1: PREFLIGHT ==========")

    preflight = controller(
        "preflight",
        spec,
        check=False,
    )

    if preflight.returncode != 0:
        raise RuntimeError(
            "Pipeline preflight failed"
        )

    if "KUBE2HPC_PREFLIGHT=PASS" not in preflight.stdout:
        raise RuntimeError(
            "Preflight did not emit PASS marker"
        )

    print()
    print("========== STAGE 2: SUBMIT ==========")

    submit = controller(
        "submit",
        spec,
        check=False,
    )

    if submit.returncode != 0:
        raise RuntimeError(
            "Controller submission failed"
        )

    jobid = parse_jobid(
        submit.stdout
    )

    print(f"PIPELINE_JOBID={jobid}")

    metadata = read_metadata(
        jobid
    )

    remember_remote_job(
        jobid,
        metadata,
    )

    print()
    print("========== STAGE 3: MONITOR ==========")

    terminal = wait_for_job(
        jobid,
        metadata,
        poll_seconds,
        timeout_seconds,
    )

    print(
        f"REMOTE_TERMINAL_STATE="
        f"{terminal['state']}"
    )

    print(
        f"REMOTE_EXIT_CODE="
        f"{terminal.get('exit_code', '')}"
    )

    print()
    print("========== STAGE 4: LOG RETRIEVAL ==========")

    logs_proc = controller(
        "logs",
        jobid,
        check=False,
    )

    if logs_proc.returncode != 0:
        raise RuntimeError(
            "Log retrieval failed"
        )

    print()
    print("========== STAGE 5: RESULT RETRIEVAL ==========")

    result_proc = controller(
        "result",
        jobid,
        check=False,
    )

    if result_proc.returncode != 0:
        raise RuntimeError(
            "Result retrieval failed"
        )

    print()
    print("========== STAGE 6: VALIDATION ==========")

    validation = validate_outputs(
        jobid
    )

    pipeline_seconds = (
        time.monotonic()
        - pipeline_start
    )

    summary = {
        "pipeline": "kube2hpc-lite-b1.4",
        "status": "PASS",
        "job_id": jobid,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "pipeline_seconds": round(
            pipeline_seconds,
            6,
        ),
        "remote_terminal_state":
            terminal["state"],
        "remote_exit_code":
            terminal.get("exit_code"),
        "accounting": terminal.get("accounting", {}),
        "accounting_timezone": terminal.get("accounting_timezone"),
        "job_dir":
            validation["job_dir"],
        "success_markers":
            validation["markers"],
        "result":
            validation["result"],
        "sha256":
            validation["sha256"],
    }

    summary_path = (
        Path(validation["job_dir"])
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            summary,
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    summary_hash = sha256_file(
        summary_path
    )

    print()
    print("========== PIPELINE SUMMARY ==========")
    print(f"JOBID={jobid}")
    print(
        f"REMOTE_STATE="
        f"{terminal['state']}"
    )
    print(
        f"REMOTE_EXIT_CODE="
        f"{terminal.get('exit_code')}"
    )
    print(
        f"PIPELINE_SECONDS="
        f"{pipeline_seconds:.6f}"
    )
    print(
        f"SUMMARY_JSON={summary_path}"
    )
    print(
        f"SUMMARY_SHA256={summary_hash}"
    )
    print(
        "KUBE2HPC_PIPELINE=PASS"
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "One-command lifecycle runner "
            "for Kube2HPC-Lite"
        )
    )

    parser.add_argument(
        "spec",
        help="Job specification JSON",
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
    )

    args = parser.parse_args()

    global CURRENT_JOB_ID
    global CURRENT_JOB_METADATA

    CURRENT_JOB_ID = None
    CURRENT_JOB_METADATA = None

    try:
        return run_pipeline(
            args.spec,
            args.poll_seconds,
            args.timeout_seconds,
        )

    except KeyboardInterrupt:
        print()
        print(
            "KUBE2HPC_PIPELINE_CANCEL_REASON=OPERATOR_INTERRUPT",
            flush=True,
        )

        try:
            protect_remote_job(
                "OPERATOR_INTERRUPT"
            )
        except Exception as cancel_exc:
            print(
                "KUBE2HPC_CANCEL_ERROR="
                f"{cancel_exc}",
                file=sys.stderr,
            )
            print(
                "KUBE2HPC_PIPELINE=CANCEL_FAILED",
                file=sys.stderr,
            )
            return 1

        print("KUBE2HPC_PIPELINE=CANCELLED")
        return 130

    except TimeoutError as exc:
        print(
            f"KUBE2HPC_PIPELINE_TIMEOUT={exc}",
            file=sys.stderr,
        )

        try:
            protect_remote_job(
                "PIPELINE_TIMEOUT"
            )
        except Exception as cancel_exc:
            print(
                "KUBE2HPC_CANCEL_ERROR="
                f"{cancel_exc}",
                file=sys.stderr,
            )
            print(
                "KUBE2HPC_PIPELINE=TIMEOUT_CANCEL_FAILED",
                file=sys.stderr,
            )
            return 1

        print("KUBE2HPC_PIPELINE=TIMEOUT")
        return 124

    except Exception as exc:
        print(
            f"KUBE2HPC_PIPELINE_ERROR={exc}",
            file=sys.stderr,
        )
        print(
            "KUBE2HPC_PIPELINE=FAIL",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
