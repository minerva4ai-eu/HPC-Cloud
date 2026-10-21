#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

TARGETS_FILE = BASE / "config" / "targets.json"

STATE_ROOT = BASE / "results" / "remote-jobs"


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_ssh(alias, remote_cmd, check=True):

    proc = subprocess.run(
        ["ssh", alias, remote_cmd],
        text=True,
        capture_output=True,
    )

    if check and proc.returncode != 0:

        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")

        raise RuntimeError(
            f"SSH command failed: alias={alias} rc={proc.returncode}"
        )

    return proc


def resolve_spec(spec_path):

    spec_path = Path(spec_path).resolve()

    spec = load_json(spec_path)

    targets = load_json(TARGETS_FILE)

    target_name = spec["target"]

    if target_name not in targets:
        raise KeyError(f"Unknown target: {target_name}")

    target = dict(targets[target_name])

    # A workload/job specification may override the
    # infrastructure target's default remote root.
    if spec.get("remote_root"):
        target["remote_root"] = spec["remote_root"]

    return spec_path, spec, target


def remote_path(root, rel):

    rel = str(rel)

    # Absolute paths are already fully resolved.
    if rel.startswith("/"):
        return rel

    return f"{root.rstrip('/')}/{rel.lstrip('/')}"


def validate_jobid(jobid):

    if not re.fullmatch(r"\d+", str(jobid)):
        raise ValueError("Job ID must contain digits only")

    return str(jobid)


def state_dir(jobid):

    return STATE_ROOT / f"job-{validate_jobid(jobid)}"


def save_metadata(jobid, spec_path, spec, target):

    directory = state_dir(jobid)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {

        "job_id": str(jobid),

        "submitted_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "spec_path": str(spec_path),

        "spec": spec,

        "target": target
    }

    with open(
        directory / "metadata.json",
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            payload,
            fh,
            indent=2,
            sort_keys=True
        )

        fh.write("\n")

    return directory


def load_metadata(jobid):

    path = (
        state_dir(jobid)
        / "metadata.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"No local metadata for job {jobid}: {path}"
        )

    return load_json(path)


# ==========================================================
# PREFLIGHT
# ==========================================================

def preflight(spec_path):

    spec_path, spec, target = resolve_spec(spec_path)

    alias = target["ssh_alias"]

    root = target["remote_root"]

    print(
        "========== KUBE2HPC PREFLIGHT =========="
    )

    print(f"target={spec['target']}")

    print(f"ssh_alias={alias}")

    print(f"remote_root={root}")

    command = (

        "echo REMOTE_HOST=$(hostname); "

        "echo REMOTE_USER=$(whoami); "

        "echo REMOTE_ARCH=$(uname -m); "

        "echo SLURM_VERSION=$(sbatch --version | awk '{print $2}'); "

        f"test -d {shlex.quote(root)} "
        "&& echo REMOTE_ROOT=VISIBLE"
    )

    proc = run_ssh(
        alias,
        command
    )

    print(
        proc.stdout,
        end=""
    )

    slurm_path = remote_path(
        root,
        spec["slurm_script"]
    )

    proc = run_ssh(
        alias,
        (
            f"test -f {shlex.quote(slurm_path)} "
            "&& echo SLURM_SCRIPT=VISIBLE"
        )
    )

    print(
        proc.stdout,
        end=""
    )

    failures = 0

    for artifact in spec.get(
        "artifacts",
        []
    ):

        path = remote_path(
            root,
            artifact["path"]
        )

        proc = run_ssh(
            alias,
            f"sha256sum {shlex.quote(path)}"
        )

        actual = (
            proc.stdout
            .strip()
            .split()[0]
        )

        expected = artifact["sha256"]

        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )

        print(
            f"ARTIFACT={artifact['path']}"
        )

        print(
            f"EXPECTED_SHA256={expected}"
        )

        print(
            f"ACTUAL_SHA256={actual}"
        )

        print(
            f"CHECKSUM={status}"
        )

        if status != "PASS":
            failures += 1

    if failures:

        print(
            "KUBE2HPC_PREFLIGHT=FAIL"
        )

        return 2

    print(
        "KUBE2HPC_PREFLIGHT=PASS"
    )

    return 0


# ==========================================================
# SUBMIT
# ==========================================================

def submit(spec_path):

    rc = preflight(spec_path)

    if rc != 0:
        return rc

    spec_path, spec, target = resolve_spec(
        spec_path
    )

    alias = target["ssh_alias"]

    root = target["remote_root"]

    script = spec["slurm_script"]

    sbatch_args = spec.get("sbatch_args", [])
    if not isinstance(sbatch_args, list) or not all(
        isinstance(arg, str) and arg.startswith("--") and "\n" not in arg
        for arg in sbatch_args
    ):
        raise ValueError("sbatch_args must be a list of long-form Slurm options")

    rendered_args = " ".join(shlex.quote(arg) for arg in sbatch_args)
    if rendered_args:
        rendered_args += " "

    command = (
        f"cd {shlex.quote(root)} && "
        f"sbatch --parsable {rendered_args}{shlex.quote(script)}"
    )

    proc = run_ssh(
        alias,
        command
    )

    match = re.search(
        r"(?:^|\n)(\d+)(?:;[^\n]+)?(?:\n|$)",
        proc.stdout
    )

    if not match:

        print(
            proc.stdout,
            file=sys.stderr
        )

        raise RuntimeError(
            "Could not parse Slurm job ID"
        )

    jobid = match.group(1)

    directory = save_metadata(
        jobid,
        spec_path,
        spec,
        target
    )

    print(
        "KUBE2HPC_SUBMIT=PASS"
    )

    print(
        f"JOBID={jobid}"
    )

    print(
        f"LOCAL_STATE={directory}"
    )

    return 0


# ==========================================================
# STATUS
# ==========================================================

def status(jobid):

    jobid = validate_jobid(
        jobid
    )

    metadata = load_metadata(
        jobid
    )

    alias = (
        metadata["target"]
        ["ssh_alias"]
    )

    active = run_ssh(
        alias,
        (
            f"squeue -h -j {jobid} "
            "-o '%T|%M|%R' "
            "| head -n 1"
        )
    ).stdout.strip()

    if active:

        print(
            "SOURCE=squeue"
        )

        print(
            f"JOBID={jobid}"
        )

        print(
            f"STATUS={active}"
        )

        return 0

    accounting = run_ssh(
        alias,
        (
            "sacct -j "
            + jobid
            + " -X -n -P "
            "--format="
            "JobIDRaw,State,"
            "ExitCode,Elapsed,"
            "Start,End "
            "| head -n 1"
        )
    ).stdout.strip()

    print(
        "SOURCE=sacct"
    )

    print(
        f"JOBID={jobid}"
    )

    print(
        f"STATUS={accounting or 'UNKNOWN'}"
    )

    return 0


# ==========================================================
# FILE RETRIEVAL
# ==========================================================

def fetch_text(
    jobid,
    field,
    local_name
):

    jobid = validate_jobid(
        jobid
    )

    metadata = load_metadata(
        jobid
    )

    spec = metadata["spec"]

    target = metadata["target"]

    alias = target["ssh_alias"]

    root = target["remote_root"]

    relative = (
        spec[field]
        .format(job_id=jobid)
    )

    path = remote_path(
        root,
        relative
    )

    proc = run_ssh(
        alias,
        f"cat {shlex.quote(path)}"
    )

    directory = state_dir(
        jobid
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        directory
        / local_name
    )

    output.write_text(
        proc.stdout,
        encoding="utf-8"
    )

    print(
        f"REMOTE_PATH={path}"
    )

    print(
        f"LOCAL_PATH={output}"
    )

    return proc.stdout


# ==========================================================
# LOGS
# ==========================================================

def logs(jobid):

    print(
        "========== STDOUT =========="
    )

    stdout = fetch_text(
        jobid,
        "stdout",
        "stdout.log"
    )

    print(
        stdout,
        end=""
        if stdout.endswith("\n")
        else "\n"
    )

    print(
        "========== STDERR =========="
    )

    stderr = fetch_text(
        jobid,
        "stderr",
        "stderr.log"
    )

    print(
        stderr,
        end=""
        if stderr.endswith("\n")
        else "\n"
    )

    return 0


# ==========================================================
# RESULT
# ==========================================================

def result(jobid):

    text = fetch_text(
        jobid,
        "result",
        "result.json"
    )

    data = json.loads(
        text
    )

    print(
        json.dumps(
            data,
            indent=2,
            sort_keys=True
        )
    )

    return 0


# ==========================================================
# CANCEL
# ==========================================================

def cancel(jobid):

    jobid = validate_jobid(
        jobid
    )

    metadata = load_metadata(
        jobid
    )

    alias = (
        metadata["target"]
        ["ssh_alias"]
    )

    run_ssh(
        alias,
        f"scancel {jobid}"
    )

    print(
        "KUBE2HPC_CANCEL=REQUESTED"
    )

    print(
        f"JOBID={jobid}"
    )

    return 0


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Kube2HPC-Lite "
            "B1 controller"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    parser_preflight = (
        sub.add_parser(
            "preflight"
        )
    )

    parser_preflight.add_argument(
        "spec"
    )

    parser_submit = (
        sub.add_parser(
            "submit"
        )
    )

    parser_submit.add_argument(
        "spec"
    )

    for name in (
        "status",
        "logs",
        "result",
        "cancel"
    ):

        parser_job = (
            sub.add_parser(
                name
            )
        )

        parser_job.add_argument(
            "jobid"
        )

    args = parser.parse_args()

    try:

        if args.command == "preflight":

            return preflight(
                args.spec
            )

        if args.command == "submit":

            return submit(
                args.spec
            )

        if args.command == "status":

            return status(
                args.jobid
            )

        if args.command == "logs":

            return logs(
                args.jobid
            )

        if args.command == "result":

            return result(
                args.jobid
            )

        if args.command == "cancel":

            return cancel(
                args.jobid
            )

    except Exception as exc:

        print(
            f"KUBE2HPC_ERROR={exc}",
            file=sys.stderr
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
