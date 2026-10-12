# Queue Based GPU Autoscaler

Status: durable local queue, CPU fixture autoscaler, graceful-drain workers and Prometheus exporter implemented and tested. KEDA integration template supplied. GPU node provisioning, Kubernetes reconciliation and spot interruption remain pending.

The queue uses SQLite WAL with atomic claims across connections. Job IDs provide idempotent enqueueing. Every attempt has a fenced lease epoch; stale workers cannot renew or complete reassigned jobs. Expired jobs return to the queue and move to a dead-letter state after three attempts. Draining workers cannot claim more work, but can renew and finish their current lease. The local autoscaler starts actual fixture worker threads and stops them when work empties and cooldown expires.

Run from the repository root:

```powershell
python queue_control.py demo --db results/queue-demo.db
python queue_control.py metrics --db results/queue.db --port 9091
```

In separate terminals, use the same absolute database path:

```powershell
python queue_control.py autoscale --db results/queue.db --tokens-per-worker 2048 --max-workers 8 --cooldown 10
python queue_control.py enqueue --db results/queue.db --id example-1 --tokens 4096 --seconds 2
python queue_control.py snapshot --db results/queue.db
```

Workers perform timed CPU fixture tasks, not model inference. Token counts are estimates, not observed generation. Ctrl+C stops admission and drains the active fixture jobs. Fixture duration is capped at 120 seconds. Recovery is at-least-once execution with fenced acknowledgement, not exactly-once external side effects. Real workers must renew leases and deduplicate side effects. Stopped IDs are not reused; use a new worker ID after restart. SQLite is single-host storage, not a shared multi-node database.

The loopback `/metrics` exporter exposes pending/leased tokens, job counts, busy workers and drain counts. The KEDA template includes pending and leased tokens so an empty waiting queue does not imply an idle cluster. Missing Prometheus data is treated as a scaler error.

## Cluster integration

[keda-scaledobject.json](keda-scaledobject.json) names an `inference-worker` Deployment and a Prometheus service. They must be supplied by the target cluster. The loopback exporter is not directly reachable from Kubernetes; replace it with a cluster-accessible authenticated exporter backed by a multi-node queue. Configure termination grace longer than admitted work and a pre-stop drain hook. Do not use shared multi-node SQLite storage. No resources are deployed by these commands.

Tests cover atomic concurrent claims, restart persistence, idempotency conflicts, stale lease fencing, dead lettering, HTTP metrics, active-worker shutdown, cooldown, real fixture scale-up and scale-to-zero. The JSON contract test checks basic fields and metric names, not Kubernetes API validation.

References: [KEDA Prometheus](https://keda.sh/docs/2.21/scalers/prometheus/), [ScaledObject](https://keda.sh/docs/2.21/reference/scaledobject-spec/).
