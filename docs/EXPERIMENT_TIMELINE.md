# Cloud-HPC Experiment Timeline

## Repository provenance

This Git repository was initialized retrospectively after the initial
Cloud-to-HPC portability and Kube2HPC-Lite experiments had already begun.

Earlier validated source artifacts were imported from their frozen experiment
locations. Therefore, the first Git commits represent a reconstruction of
validated milestones rather than the original chronological development
history.

## Completed milestones

### Tier A
Manual Kubernetes-to-MN5 portability validation.

### B1.1
Dedicated unattended Kubernetes control-plane -> HPC SSH/Slurm control channel.

### B1.2
Kube2HPC-Lite controller primitives:
preflight, submit, status, logs, result, cancel.

### B1.3
Automated Tier-A replay through Kube2HPC-Lite.

### B1.4
One-command lifecycle:
preflight -> submit -> monitor -> retrieve -> validate -> summary.

### B1.5A
Kubernetes Pod -> MN5 network connectivity validation.

Next:
B1.5B Kubernetes-managed machine identity and controller execution.
