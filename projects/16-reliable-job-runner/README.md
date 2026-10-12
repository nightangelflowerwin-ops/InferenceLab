# Reliable text job runner

Milestone 1: real UTF-8 text reports, durable SQLite jobs and separate worker processes. Crash recovery is tested by killing an actual child process while it owns a job, then waiting for its lease to expire and another process to complete the report. The dashboard is being built next.

```powershell
python job_runner.py demo --out results/job-runner/recovery.json
python job_runner.py submit --folder YOUR_TEXT_FOLDER
python job_runner.py worker
python job_runner.py status
```

Jobs produce word/line/character counts, top words and a content checksum. CLI batches accept 1 to 100 UTF-8 `.txt` files, at most 1 MiB per file and 10 MiB per batch. File contents remain in the local SQLite database under ignored `results/`; no model, external service or GPU is used.

Leases renew while work runs. A crashed worker's job returns to the queue; after three claims without successful completion it becomes dead. Epoch fencing rejects stale completion. This permits repeated computation but commits one result for a job. It does not guarantee exactly-once external side effects. The coordinator and SQLite storage live on one host, so this is worker-process fault tolerance, not replicated multi-machine storage.
