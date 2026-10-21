#!/usr/bin/env python3

import json
import os
import time

from decision_engine import (
    api_get,
    cpu_to_millicpu,
    memory_to_bytes,
    pod_requests,
    is_ready_worker,
    gib,
)


NAMESPACE = os.environ.get(
    "CLOUD_HPC_NAMESPACE",
    "cloud-hpc-workloads",
)

REQUEST_CONFIGMAP = (
    "cloud-hpc-b17c-request"
)

CATALOG_CONFIGMAP = (
    "cloud-hpc-workload-catalog"
)

GPU_RESOURCE = "nvidia.com/gpu"

REVIEW_TIMEOUT_SECONDS = 600

POLL_SECONDS = 5


def bool_value(value):
    return str(value).strip().lower() == "true"


def get_configmap(name):
    return api_get(
        f"/api/v1/namespaces/"
        f"{NAMESPACE}/configmaps/{name}"
    )


def get_request():
    cm = get_configmap(
        REQUEST_CONFIGMAP
    )

    return cm.get(
        "data",
        {},
    )


def get_catalog():
    cm = get_configmap(
        CATALOG_CONFIGMAP
    )

    raw = cm.get(
        "data",
        {},
    ).get(
        "workload-profiles.json"
    )

    if not raw:
        raise RuntimeError(
            "WORKLOAD_CATALOG_EMPTY"
        )

    return json.loads(raw)


def assess_kubernetes(profile, nodes=None, pods=None):
    requested_cpu = (
        cpu_to_millicpu(
            profile["requested_cpu"]
        )
    )

    requested_memory = (
        memory_to_bytes(
            profile["requested_memory"]
        )
    )

    gpu_required = bool(
        profile.get(
            "gpu_required",
            False,
        )
    )

    requested_gpu = int(
        profile.get(
            "requested_gpu",
            0,
        )
    )

    if (
        gpu_required
        and requested_gpu < 1
    ):
        requested_gpu = 1

    if nodes is None:
        nodes = api_get(
            "/api/v1/nodes"
        ).get(
            "items",
            [],
        )

    if pods is None:
        pods = api_get(
            "/api/v1/pods"
        ).get(
            "items",
            [],
        )

    usage = {}

    for pod in pods:

        phase = (
            pod.get(
                "status",
                {},
            )
            .get(
                "phase",
                "",
            )
        )

        if phase in (
            "Succeeded",
            "Failed",
        ):
            continue

        node_name = (
            pod.get(
                "spec",
                {},
            )
            .get(
                "nodeName"
            )
        )

        if not node_name:
            continue

        cpu, memory, gpu = (
            pod_requests(
                pod,
                GPU_RESOURCE,
            )
        )

        entry = usage.setdefault(
            node_name,
            {
                "cpu": 0,
                "memory": 0,
                "gpu": 0,
                "pods": 0,
            },
        )

        entry["cpu"] += cpu
        entry["memory"] += memory
        entry["gpu"] += gpu
        entry["pods"] += 1

    assessments = []

    feasible_nodes = []

    for node in sorted(
        nodes,
        key=lambda item:
            item["metadata"]["name"],
    ):

        if not is_ready_worker(
            node
        ):
            continue

        name = (
            node["metadata"]["name"]
        )

        allocatable = (
            node.get(
                "status",
                {},
            )
            .get(
                "allocatable",
                {},
            )
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
                GPU_RESOURCE,
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
            alloc_cpu
            - used["cpu"],
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
            free_gpu
            >= requested_gpu
        )

        feasible = (
            cpu_ok
            and memory_ok
            and gpu_ok
        )

        assessment = {
            "node": name,

            "alloc_cpu":
                alloc_cpu / 1000,

            "requested_cpu_current":
                used["cpu"] / 1000,

            "free_cpu_est":
                free_cpu / 1000,

            "alloc_mem_gib":
                gib(alloc_memory),

            "requested_mem_gib_current":
                gib(used["memory"]),

            "free_mem_gib_est":
                gib(free_memory),

            "alloc_gpu":
                alloc_gpu,

            "free_gpu_est":
                free_gpu,

            "active_pods":
                used["pods"],

            "cpu_ok":
                cpu_ok,

            "memory_ok":
                memory_ok,

            "gpu_ok":
                gpu_ok,

            "feasible":
                feasible,
        }

        assessments.append(
            assessment
        )

        if feasible:
            feasible_nodes.append(
                name
            )

    return (
        assessments,
        feasible_nodes,
    )


def main():
    print(
        "============================================="
    )

    print(
        "Cloud-HPC B1.7C Unified Controller"
    )

    print(
        "============================================="
    )

    print(
        "CONTROLLER_MODE="
        "DECISION_AND_HUMAN_REVIEW"
    )

    print(
        "EXECUTION_CAPABILITY=DISABLED"
    )

    print(
        "K8S_JOB_CREATION=DISABLED"
    )

    print(
        "HPC_SUBMISSION=DISABLED"
    )

    request = get_request()

    request_id = request.get(
        "requestId",
        "",
    )

    workload = request.get(
        "workload",
        "",
    )

    review_policy = request.get(
        "reviewPolicy",
        "",
    )

    print()
    print(
        "========== REQUEST =========="
    )

    print(
        f"REQUEST_ID={request_id}"
    )

    print(
        f"WORKLOAD={workload}"
    )

    print(
        f"REVIEW_POLICY={review_policy}"
    )

    if not request_id:
        raise RuntimeError(
            "REQUEST_ID_MISSING"
        )

    if not workload:
        raise RuntimeError(
            "WORKLOAD_MISSING"
        )

    if review_policy != "required":
        raise RuntimeError(
            "REVIEW_POLICY_NOT_SUPPORTED"
        )

    catalog = get_catalog()

    if workload not in catalog:
        print(
            "WORKLOAD_LOOKUP=FAIL"
        )

        print(
            "DECISION_ENGINE=FAIL"
        )

        return 2

    profile = catalog[workload]

    print(
        "WORKLOAD_LOOKUP=PASS"
    )

    print()
    print(
        "========== WORKLOAD PROFILE =========="
    )

    print(
        "WORKLOAD_CLASS="
        + profile["workload_class"]
    )

    print(
        "REQUESTED_CPU="
        + profile["requested_cpu"]
    )

    print(
        "REQUESTED_MEMORY="
        + profile["requested_memory"]
    )

    print(
        "CPU_THREADS="
        + str(
            profile["cpu_threads"]
        )
    )

    print(
        "GPU_REQUIRED="
        + str(
            profile["gpu_required"]
        ).lower()
    )

    print()
    print(
        "========== LIVE KUBERNETES ASSESSMENT =========="
    )

    assessments, feasible_nodes = (
        assess_kubernetes(
            profile
        )
    )

    print(
        "K8S_API_READ=PASS"
    )

    for item in assessments:

        print(
            "NODE="
            f"{item['node']} "
            "ALLOC_CPU="
            f"{item['alloc_cpu']:.3f} "
            "REQUESTED_CPU_CURRENT="
            f"{item['requested_cpu_current']:.3f} "
            "FREE_CPU_EST="
            f"{item['free_cpu_est']:.3f} "
            "CPU_OK="
            f"{str(item['cpu_ok']).lower()}"
        )

        print(
            "NODE="
            f"{item['node']} "
            "ALLOC_MEM_GIB="
            f"{item['alloc_mem_gib']:.3f} "
            "REQUESTED_MEM_GIB_CURRENT="
            f"{item['requested_mem_gib_current']:.3f} "
            "FREE_MEM_GIB_EST="
            f"{item['free_mem_gib_est']:.3f} "
            "MEMORY_OK="
            f"{str(item['memory_ok']).lower()}"
        )

        print(
            "NODE="
            f"{item['node']} "
            "ALLOC_GPU="
            f"{item['alloc_gpu']} "
            "FREE_GPU_EST="
            f"{item['free_gpu_est']} "
            "GPU_OK="
            f"{str(item['gpu_ok']).lower()} "
            "FEASIBLE="
            f"{str(item['feasible']).lower()}"
        )

    print()
    print(
        "========== RECOMMENDATION =========="
    )

    if feasible_nodes:

        recommendation = "local"

        reason = (
            "K8S_NODE_CAN_SATISFY_REQUEST"
        )

    else:

        recommendation = "mn5"

        reason = (
            "NO_K8S_NODE_CAN_SATISFY_RESOURCE_REQUEST"
        )

    print(
        "RECOMMENDED_TARGET="
        + recommendation.upper()
    )

    print(
        "DECISION_REASON="
        + reason
    )

    print(
        "FEASIBLE_K8S_NODES="
        + (
            ",".join(
                feasible_nodes
            )
            if feasible_nodes
            else "NONE"
        )
    )

    print(
        "DECISION_ENGINE=PASS"
    )

    print()
    print(
        "========== HUMAN REVIEW =========="
    )

    print(
        "HUMAN_REVIEW_REQUIRED=true"
    )

    print(
        "HUMAN_REVIEW_STATE=AWAITING_REVIEW"
    )

    review_start = (
        time.monotonic()
    )

    while True:

        current = get_request()

        if (
            current.get(
                "requestId"
            )
            != request_id
        ):

            print(
                "REQUEST_MUTATION=FAIL"
            )

            print(
                "REASON=REQUEST_ID_CHANGED_DURING_REVIEW"
            )

            return 3

        if (
            current.get(
                "workload"
            )
            != workload
        ):

            print(
                "REQUEST_MUTATION=FAIL"
            )

            print(
                "REASON=WORKLOAD_CHANGED_DURING_REVIEW"
            )

            return 4

        if bool_value(
            current.get(
                "rejected",
                "false",
            )
        ):

            print(
                "HUMAN_REVIEW_STATE=REJECTED"
            )

            print(
                "EXECUTION_DISPATCH=DISABLED"
            )

            print(
                "B1_7C_REVIEW_GATE=PASS"
            )

            return 0

        approved = bool_value(
            current.get(
                "approved",
                "false",
            )
        )

        if approved:

            approved_target = (
                current.get(
                    "approvedTarget",
                    "",
                )
                .strip()
                .lower()
            )

            if approved_target not in (
                "local",
                "mn5",
            ):

                print(
                    "HUMAN_REVIEW_STATE="
                    "APPROVAL_INCOMPLETE"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            if (
                approved_target
                == "local"
                and
                not feasible_nodes
            ):

                print(
                    "OVERRIDE_VALIDATION=FAIL"
                )

                print(
                    "REASON="
                    "LOCAL_RESOURCE_INFEASIBLE"
                )

                print(
                    "EXECUTION_DISPATCH=DISABLED"
                )

                return 5

            if (
                approved_target
                == recommendation
            ):

                action = (
                    "RECOMMENDATION_APPROVED"
                )

            else:

                action = (
                    "OVERRIDE_APPROVED"
                )

            print(
                "HUMAN_REVIEW_STATE=APPROVED"
            )

            print(
                "HUMAN_REVIEW_ACTION="
                + action
            )

            print(
                "APPROVED_TARGET="
                + approved_target.upper()
            )

            print(
                "OVERRIDE_VALIDATION=PASS"
            )

            print(
                "EXECUTION_DISPATCH=DISABLED"
            )

            print(
                "B1_7C_REVIEW_GATE=PASS"
            )

            return 0

        elapsed = int(
            time.monotonic()
            - review_start
        )

        print(
            "HUMAN_REVIEW_STATE="
            "AWAITING_REVIEW "
            f"wait_seconds={elapsed}"
        )

        if (
            elapsed
            >= REVIEW_TIMEOUT_SECONDS
        ):

            print(
                "HUMAN_REVIEW_STATE=TIMEOUT"
            )

            print(
                "EXECUTION_DISPATCH=DISABLED"
            )

            return 6

        time.sleep(
            POLL_SECONDS
        )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
