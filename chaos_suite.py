"""Isolated CPU fixture faults and checkpoint recovery; never injects real GPU errors."""
import argparse
import hashlib
import json
from pathlib import Path
import time


class FixtureFault(RuntimeError):
    pass


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FixtureReplica:
    def __init__(self, name):
        self.name, self.cache, self.failure = name, {}, None

    def admit(self, key, context):
        if self.failure:
            raise FixtureFault(self.failure)
        hit = key in self.cache
        if not hit:
            # A CPU fixture state, not real KV tensors.
            self.cache[key] = {"context": context, "processed_units": len(context)}
        return dict(replica=self.name, fixture_cache_hit=hit,
                    answer_checksum=checksum(context), fixture=True)


class RecoveringCluster:
    def __init__(self, names=("a", "b"), store=None):
        self.replicas = {name: FixtureReplica(name) for name in names}
        self.healthy = set(names)
        self.checkpoints, self.completed = {}, {}
        self.events = []
        self.store = store

    def inject(self, replica, fault):
        if fault not in ("partition", "simulated_ecc", "simulated_oom"):
            raise ValueError("Only isolated fixture fault types are supported")
        self.replicas[replica].failure = fault

    def restore(self, replica):
        node = self.replicas[replica]
        node.failure = None
        node.cache.clear()
        self.healthy.add(replica)
        self.events.append(dict(event="restored_empty", replica=replica))

    def execute(self, request_id, context, preferred="a"):
        if not isinstance(context, str):
            raise ValueError("String fixture context required")
        key = checksum(context)
        if self.store and request_id not in self.completed:
            persisted = self.store.read('completions', request_id)
            if persisted is not None:
                self.completed[request_id] = persisted
        if request_id in self.completed:
            old = self.completed[request_id]
            if old["answer_checksum"] != key:
                raise ValueError("Request ID reused with different context")
            return old
        candidates = sorted(self.healthy, key=lambda name: (name != preferred, name))
        for name in candidates:
            node = self.replicas[name]
            start = time.perf_counter()
            try:
                checkpoint = self.checkpoints.get(key)
                if self.store and checkpoint is None:
                    from recovery_store import CorruptState
                    try:
                        checkpoint = self.store.read('checkpoints', key)
                    except CorruptState:
                        self.events.append(dict(event='durable_checkpoint_rejected', replica=name))
                if checkpoint and key not in node.cache:
                    content = json.loads(checkpoint["data"])
                    if checksum(content) == checkpoint["checksum"]:
                        node.cache[key] = content
                        self.events.append(dict(event="fixture_state_migrated", replica=name))
                    else:
                        self.events.append(dict(event="checkpoint_rejected", replica=name))
                response = node.admit(key, context)
                state = node.cache[key]
                self.checkpoints[key] = dict(data=json.dumps(state), checksum=checksum(state))
                if self.store:
                    response = self.store.record_success(request_id, response, key, self.checkpoints[key])
                self.completed[request_id] = response
                self.events.append(dict(event="completed", replica=name,
                                        local_elapsed_ms=(time.perf_counter() - start) * 1000))
                return response
            except FixtureFault as error:
                self.healthy.discard(name)
                # Never migrate a tainted source after an ECC signal. The controller's
                # already verified checkpoint is separate from that source's cache.
                if node.failure == "simulated_ecc":
                    node.cache.clear()
                self.events.append(dict(event="fault_detected", replica=name, fault=str(error)))
        raise FixtureFault("No healthy fixture replica remains")


def run_suite():
    cases = []
    for fault in ("partition", "simulated_ecc", "simulated_oom"):
        cluster = RecoveringCluster()
        context = "shared fixture context"
        original = cluster.execute("warm", context)
        cluster.inject("a", fault)
        result = cluster.execute("after-fault", context)
        assert result["replica"] == "b"
        assert result["answer_checksum"] == original["answer_checksum"]
        assert result["fixture_cache_hit"]
        assert cluster.execute("after-fault", context) == result
        cluster.restore("a")
        assert not cluster.replicas["a"].cache
        cases.append(dict(fault=fault, recovered=True, events=cluster.events))
    return dict(evidence="CPU fixture faults only; no GPU ECC injection, OOM allocation, real network partition, or GPU KV migration", cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_suite()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Passed three isolated fixture fault/recovery scenarios")


if __name__ == "__main__":
    main()
