#!/usr/bin/env python3
"""Compute resource-budget metrics from generated B1.10 comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def value(item, *keys):
    for key in keys:
        item = item.get(key, {}) if isinstance(item, dict) else {}
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def memory_gib(text):
    if not isinstance(text, str) or not text.endswith("Gi"):
        return None
    try:
        return float(text[:-2])
    except ValueError:
        return None


def load_comparisons(root):
    result = []
    for path in sorted(root.rglob("comparison.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("schema_version") == "b1.10-v1":
            result.append(item)
    if not result:
        raise SystemExit(f"No B1.10 comparison.json files found under {root}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=ROOT / "artifacts" / "benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "analysis" / "cost_resource_metrics.json")
    parser.add_argument("--cost-model", type=Path, default=ROOT / "config" / "cost-model.json")
    args = parser.parse_args()
    cost_model = json.loads(args.cost_model.read_text(encoding="utf-8"))
    cpu_rate = cost_model["cloud_rates"].get("eur_per_vcpu_hour")
    memory_rate = cost_model["cloud_rates"].get("eur_per_gib_ram_hour")
    rates_available = isinstance(cpu_rate, (int, float)) and isinstance(memory_rate, (int, float))

    metrics = []
    for comparison in load_comparisons(args.evidence):
        row = {"workload_profile": comparison["workload_profile"]}
        for key in ("k8s", "mn5"):
            execution = comparison["executions"][key]
            seconds = value(execution, "timing", "execution_time_s")
            cpu = value(execution, "resources", "requested_cpu")
            memory = (
                memory_gib(execution["resources"].get("requested_memory"))
                if key == "k8s"
                else value(execution, "resources", "expected_memory_from_cpu_gb")
            )
            cpu_hours = cpu * seconds / 3600 if cpu is not None and seconds is not None else None
            memory_hours = memory * seconds / 3600 if memory is not None and seconds is not None else None
            platform = {
                "compute_resource_hours": cpu_hours,
                "memory_reservation_hours": memory_hours,
                "modelled_cost_eur": None,
            }
            if key == "k8s" and rates_available and cpu_hours is not None and memory_hours is not None:
                platform["modelled_cost_eur"] = cpu_hours * cpu_rate + memory_hours * memory_rate
            row[key] = platform
        metrics.append(row)

    output = {
        "schema_version": "b1.10-resource-analysis-v1",
        "source_directory": str(args.evidence),
        "cloud_monetary_rates_available": rates_available,
        "hpc_incremental_user_charge_eur_per_job": cost_model.get("hpc", {}).get("incremental_user_charge_eur_per_job"),
        "comparisons": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("B1.10_COST_RESOURCE_ANALYSIS=PASS")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
