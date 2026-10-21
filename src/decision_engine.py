#!/usr/bin/env python3

import json
import os
import ssl
import urllib.request
from decimal import Decimal
from pathlib import Path


SA_ROOT = Path(
    "/var/run/secrets/kubernetes.io/serviceaccount"
)

TOKEN_FILE = SA_ROOT / "token"
CA_FILE = SA_ROOT / "ca.crt"

REQUEST_ROOT = Path("/request")

API_HOST = os.environ.get("KUBERNETES_SERVICE_HOST")
API_PORT = os.environ.get(
    "KUBERNETES_SERVICE_PORT_HTTPS",
    "443",
)

API_SERVER = f"https://{API_HOST}:{API_PORT}" if API_HOST else None


def read_text(path):
    return Path(path).read_text(
        encoding="utf-8"
    ).strip()


def api_get(path):
    if API_SERVER is None:
        raise RuntimeError("KUBERNETES_SERVICE_HOST is unavailable")
    token = read_text(TOKEN_FILE)

    context = ssl.create_default_context(
        cafile=str(CA_FILE)
    )

    request = urllib.request.Request(
        API_SERVER + path,
        headers={
            "Authorization":
                f"Bearer {token}",
            "Accept":
                "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        context=context,
        timeout=20,
    ) as response:

        return json.load(response)


def cpu_to_millicpu(value):
    if not value:
        return 0

    value = str(value)

    if value.endswith("m"):
        return int(
            Decimal(value[:-1])
        )

    if value.endswith("u"):
        return int(
            Decimal(value[:-1])
            / Decimal(1000)
        )

    if value.endswith("n"):
        return int(
            Decimal(value[:-1])
            / Decimal(1000000)
        )

    return int(
        Decimal(value)
        * Decimal(1000)
    )


MEMORY_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024 ** 2,
    "Gi": 1024 ** 3,
    "Ti": 1024 ** 4,
    "Pi": 1024 ** 5,
    "Ei": 1024 ** 6,
    "K": 1000,
    "M": 1000 ** 2,
    "G": 1000 ** 3,
    "T": 1000 ** 4,
    "P": 1000 ** 5,
    "E": 1000 ** 6,
}


def memory_to_bytes(value):
    if not value:
        return 0

    value = str(value)

    for suffix, multiplier in (
        sorted(
            MEMORY_SUFFIXES.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )
    ):
        if value.endswith(suffix):
            number = value[
                : -len(suffix)
            ]

            return int(
                Decimal(number)
                * multiplier
            )

    return int(
        Decimal(value)
    )


def container_request(
    container,
    resource,
):
    resources = container.get(
        "resources",
        {},
    )

    requests = resources.get(
        "requests",
        {},
    )

    limits = resources.get(
        "limits",
        {},
    )

    if resource in requests:
        return requests[resource]

    # Conservative fallback for workloads where
    # a limit exists but an explicit request is absent.
    if resource in limits:
        return limits[resource]

    return None


def pod_requests(
    pod,
    gpu_resource,
):
    spec = pod.get(
        "spec",
        {},
    )

    containers = spec.get(
        "containers",
        [],
    )

    init_containers = spec.get(
        "initContainers",
        [],
    )

    regular_cpu = sum(
        cpu_to_millicpu(
            container_request(
                c,
                "cpu",
            )
        )
        for c in containers
    )

    regular_memory = sum(
        memory_to_bytes(
            container_request(
                c,
                "memory",
            )
        )
        for c in containers
    )

    regular_gpu = sum(
        int(
            container_request(
                c,
                gpu_resource,
            )
            or 0
        )
        for c in containers
    )

    init_cpu = max(
        [
            cpu_to_millicpu(
                container_request(
                    c,
                    "cpu",
                )
            )
            for c in init_containers
        ]
        or [0]
    )

    init_memory = max(
        [
            memory_to_bytes(
                container_request(
                    c,
                    "memory",
                )
            )
            for c in init_containers
        ]
        or [0]
    )

    init_gpu = max(
        [
            int(
                container_request(
                    c,
                    gpu_resource,
                )
                or 0
            )
            for c in init_containers
        ]
        or [0]
    )

    cpu = max(
        regular_cpu,
        init_cpu,
    )

    memory = max(
        regular_memory,
        init_memory,
    )

    gpu = max(
        regular_gpu,
        init_gpu,
    )

    # Kubernetes 1.36 may also expose Pod-level
    # CPU/memory requests.
    pod_level = (
        spec.get("resources", {})
        .get("requests", {})
    )

    if pod_level.get("cpu"):
        cpu = cpu_to_millicpu(
            pod_level["cpu"]
        )

    if pod_level.get("memory"):
        memory = memory_to_bytes(
            pod_level["memory"]
        )

    overhead = spec.get(
        "overhead",
        {},
    )

    cpu += cpu_to_millicpu(
        overhead.get("cpu")
    )

    memory += memory_to_bytes(
        overhead.get("memory")
    )

    return cpu, memory, gpu


def is_ready_worker(node):
    metadata = node.get(
        "metadata",
        {},
    )

    spec = node.get(
        "spec",
        {},
    )

    labels = metadata.get(
        "labels",
        {},
    )

    if spec.get(
        "unschedulable",
        False,
    ):
        return False

    if (
        "node-role.kubernetes.io/control-plane"
        in labels
        or
        "node-role.kubernetes.io/master"
        in labels
    ):
        return False

    conditions = (
        node.get("status", {})
        .get("conditions", [])
    )

    for condition in conditions:
        if (
            condition.get("type")
            == "Ready"
            and
            condition.get("status")
            == "True"
        ):
            return True

    return False


def gib(value):
    return (
        value
        / (1024 ** 3)
    )


def main():
    print(
        "============================================="
    )
    print(
        "Cloud-HPC B1.7A Decision Engine"
    )
    print(
        "============================================="
    )

    request_id = read_text(
        REQUEST_ROOT / "requestId"
    )

    requested_cpu_text = read_text(
        REQUEST_ROOT / "requestedCpu"
    )

    requested_memory_text = read_text(
        REQUEST_ROOT
        / "requestedMemory"
    )

    gpu_required = (
        read_text(
            REQUEST_ROOT
            / "gpuRequired"
        ).lower()
        == "true"
    )

    gpu_resource = read_text(
        REQUEST_ROOT
        / "gpuResource"
    )

    requested_gpu = int(
        read_text(
            REQUEST_ROOT
            / "requestedGpu"
        )
    )

    if gpu_required and requested_gpu < 1:
        requested_gpu = 1

    requested_cpu = (
        cpu_to_millicpu(
            requested_cpu_text
        )
    )

    requested_memory = (
        memory_to_bytes(
            requested_memory_text
        )
    )

    print(
        f"REQUEST_ID={request_id}"
    )
    print(
        f"REQUESTED_CPU={requested_cpu_text}"
    )
    print(
        f"REQUESTED_MEMORY={requested_memory_text}"
    )
    print(
        f"GPU_REQUIRED={str(gpu_required).lower()}"
    )
    print(
        f"GPU_RESOURCE={gpu_resource}"
    )
    print(
        f"REQUESTED_GPU={requested_gpu}"
    )

    print()
    print(
        "DECISION_MODE=RECOMMENDATION_ONLY"
    )
    print(
        "DECISION_SCOPE=RESOURCE_FEASIBILITY"
    )
    print(
        "HPC_SUBMISSION_CAPABILITY=DISABLED"
    )

    print()
    print(
        "========== KUBERNETES API =========="
    )

    nodes_response = api_get(
        "/api/v1/nodes"
    )

    pods_response = api_get(
        "/api/v1/pods"
    )

    print(
        "K8S_API_READ=PASS"
    )

    nodes = nodes_response.get(
        "items",
        [],
    )

    pods = pods_response.get(
        "items",
        [],
    )

    usage = {}

    for pod in pods:
        phase = (
            pod.get("status", {})
            .get("phase", "")
        )

        if phase in (
            "Succeeded",
            "Failed",
        ):
            continue

        node_name = (
            pod.get("spec", {})
            .get("nodeName")
        )

        if not node_name:
            continue

        cpu, memory, gpu = (
            pod_requests(
                pod,
                gpu_resource,
            )
        )

        item = usage.setdefault(
            node_name,
            {
                "cpu": 0,
                "memory": 0,
                "gpu": 0,
                "pods": 0,
            },
        )

        item["cpu"] += cpu
        item["memory"] += memory
        item["gpu"] += gpu
        item["pods"] += 1

    feasible_nodes = []

    print()
    print(
        "========== NODE ASSESSMENT =========="
    )

    for node in sorted(
        nodes,
        key=lambda x:
            x["metadata"]["name"],
    ):
        if not is_ready_worker(
            node
        ):
            continue

        name = node[
            "metadata"
        ]["name"]

        allocatable = (
            node.get("status", {})
            .get("allocatable", {})
        )

        alloc_cpu = (
            cpu_to_millicpu(
                allocatable.get("cpu")
            )
        )

        alloc_memory = (
            memory_to_bytes(
                allocatable.get(
                    "memory"
                )
            )
        )

        alloc_gpu = int(
            allocatable.get(
                gpu_resource,
                0,
            )
        )

        used = usage.get(
            name,
            {
                "cpu": 0,
                "memory": 0,
                "gpu": 0,
                "pods": 0,
            },
        )

        free_cpu = max(
            alloc_cpu - used["cpu"],
            0,
        )

        free_memory = max(
            alloc_memory
            - used["memory"],
            0,
        )

        free_gpu = max(
            alloc_gpu
            - used["gpu"],
            0,
        )

        cpu_ok = (
            free_cpu
            >= requested_cpu
        )

        memory_ok = (
            free_memory
            >= requested_memory
        )

        gpu_ok = (
            not gpu_required
            or
            free_gpu >= requested_gpu
        )

        feasible = (
            cpu_ok
            and memory_ok
            and gpu_ok
        )

        print(
            "NODE="
            f"{name}"
            " "
            f"ALLOC_CPU={alloc_cpu / 1000:.3f}"
            " "
            f"REQUESTED_CPU_CURRENT={used['cpu'] / 1000:.3f}"
            " "
            f"FREE_CPU_EST={free_cpu / 1000:.3f}"
        )

        print(
            "NODE="
            f"{name}"
            " "
            f"ALLOC_MEM_GIB={gib(alloc_memory):.3f}"
            " "
            f"REQUESTED_MEM_GIB_CURRENT={gib(used['memory']):.3f}"
            " "
            f"FREE_MEM_GIB_EST={gib(free_memory):.3f}"
        )

        print(
            "NODE="
            f"{name}"
            " "
            f"ALLOC_GPU={alloc_gpu}"
            " "
            f"REQUESTED_GPU_CURRENT={used['gpu']}"
            " "
            f"FREE_GPU_EST={free_gpu}"
        )

        print(
            "NODE="
            f"{name}"
            " "
            f"ACTIVE_PODS_COUNTED={used['pods']}"
            " "
            f"CPU_OK={str(cpu_ok).lower()}"
            " "
            f"MEMORY_OK={str(memory_ok).lower()}"
            " "
            f"GPU_OK={str(gpu_ok).lower()}"
            " "
            f"FEASIBLE={str(feasible).lower()}"
        )

        if feasible:
            feasible_nodes.append(
                name
            )

    print()
    print(
        "========== RECOMMENDATION =========="
    )

    if feasible_nodes:
        recommendation = "LOCAL"

        reason = (
            "K8S_NODE_CAN_SATISFY_REQUEST"
        )

        selected = feasible_nodes[0]

        print(
            f"RECOMMENDED_TARGET={recommendation}"
        )
        print(
            f"DECISION_REASON={reason}"
        )
        print(
            f"FEASIBLE_K8S_NODE={selected}"
        )

    else:
        recommendation = "MN5"

        if gpu_required:
            reason = (
                "NO_K8S_NODE_CAN_SATISFY_GPU_AND_RESOURCE_REQUEST"
            )
        else:
            reason = (
                "NO_K8S_NODE_CAN_SATISFY_RESOURCE_REQUEST"
            )

        print(
            f"RECOMMENDED_TARGET={recommendation}"
        )
        print(
            f"DECISION_REASON={reason}"
        )
        print(
            "FEASIBLE_K8S_NODE=NONE"
        )

    print(
        "HUMAN_APPROVAL_REQUIRED_FOR_MN5=true"
    )

    print()
    print(
        "DECISION_ENGINE=PASS"
    )
    print(
        "B1_7A_RECOMMENDATION=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
