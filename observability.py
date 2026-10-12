"""Prometheus snapshot checks and a loopback fixture exporter.

Field availability and genuine hardware provenance must be verified separately.
"""
import argparse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import time


SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+(\S+))?$')
LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)')


def parse_samples(text):
    samples, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError("Unsupported Prometheus sample line")
        name, label_text, number, timestamp = match.groups()
        labels, offset = {}, 0
        while label_text and offset < len(label_text):
            item = LABEL.match(label_text, offset)
            if not item or item[1] in labels:
                raise ValueError("Malformed or duplicate label")
            labels[item[1]] = json.loads('"' + item[2] + '"')
            offset = item.end()
        key = name, tuple(sorted(labels.items()))
        if key in seen:
            raise ValueError("Duplicate sample identity")
        seen.add(key)
        value = float(number)
        sample_time = float(timestamp) / 1000 if timestamp is not None else None
        if sample_time is not None and not math.isfinite(sample_time):
            raise ValueError("Invalid timestamp")
        samples.append({"name": name, "labels": labels, "value": value, "timestamp_seconds": sample_time})
    return samples


def assess(samples, now=None, stale_seconds=60, temperature_limit=80):
    now = time.time() if now is None else now
    groups = defaultdict(list)
    for sample in samples:
        gpu = sample["labels"].get("UUID", sample["labels"].get("gpu", "unlabeled"))
        for dimension in ("GPU_I_ID", "GPU_CI_ID"):
            if dimension in sample["labels"]:
                gpu += "|" + dimension + "=" + sample["labels"][dimension]
        groups[gpu].append(sample)
    results = []
    required = ("DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_PROF_SM_OCCUPANCY", "DCGM_FI_PROF_SM_ACTIVE",
                "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_FREE")
    for gpu, series in sorted(groups.items()):
        values, alerts, invalid, timestamped, ambiguous = {}, [], [], [], set()
        for sample in series:
            value, name = sample["value"], sample["name"]
            # DCGM sentinel/blank fields must not look like valid zero utilization.
            if not math.isfinite(value) or abs(value) >= 1e18:
                invalid.append(name)
                continue
            if name in values:
                # Multi-dimensional event series are analyzed individually below.
                if name != "DCGM_EXP_CLOCK_EVENTS_COUNT":
                    ambiguous.add(name)
                    values.pop(name, None)
                    alerts.append("ambiguous_series:" + name)
            elif name not in ambiguous:
                values[name] = value
            if sample["timestamp_seconds"] is not None:
                timestamped.append(sample["timestamp_seconds"])
            if name == "DCGM_FI_DEV_GPU_TEMP" and value >= temperature_limit:
                alerts.append("temperature_above_configured_threshold")
            if name == "DCGM_EXP_CLOCK_EVENTS_COUNT" and sample["labels"].get("clock_event") and value > 0:
                alerts.append("clock_event_reported:" + sample["labels"]["clock_event"])
            if name == "inference_gpu_oom_total" and value > 0:
                alerts.append("software_oom_counter_nonzero")
        missing = [name for name in required if name not in values]
        if missing or invalid or ambiguous:
            alerts.append("telemetry_incomplete")
        if timestamped and now - min(timestamped) > stale_seconds:
            alerts.append("timestamped_samples_stale")
        for name in ("DCGM_FI_PROF_SM_OCCUPANCY", "DCGM_FI_PROF_SM_ACTIVE"):
            if name in values and not 0 <= values[name] <= 1:
                alerts.append("invalid_ratio:" + name)
        used, free = values.get("DCGM_FI_DEV_FB_USED"), values.get("DCGM_FI_DEV_FB_FREE")
        memory_fraction = used / (used + free) if used is not None and free is not None and used >= 0 and free >= 0 and used + free > 0 else None
        if memory_fraction is not None and memory_fraction >= .95:
            alerts.append("framebuffer_utilization_above_95_percent")
        results.append({"gpu_identity": gpu, "alerts": sorted(set(alerts)), "missing_metrics": missing,
                        "invalid_metrics": sorted(set(invalid)), "framebuffer_used_fraction": memory_fraction,
                        "values": values, "measurement_timestamps_present": bool(timestamped),
                        "fixture": any(sample["labels"].get("source") == "fixture" for sample in series)})
    return {"devices": results, "evidence": "Analysis of supplied metric snapshots. Missing/sentinel values stay unknown; a temperature threshold or low occupancy alone does not diagnose hardware throttling."}


def fixture_handler(path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path not in ("/metrics", "/health"):
                self.send_error(404)
                return
            if self.path == "/metrics":
                data = path.read_bytes()
                parse_samples(data.decode())
                mime = "text/plain; version=0.0.4"
            else:
                data, mime = b'{"fixture":true,"gpu_hardware":false}', "application/json"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["assess", "fixture-exporter"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1",
                        help="0.0.0.0 is for the isolated Docker fixture network")
    args = parser.parse_args()
    if args.action == "fixture-exporter":
        print("SYNTHETIC fixture exporter; no GPU hardware readings", flush=True)
        ThreadingHTTPServer((args.host, args.port), fixture_handler(args.input)).serve_forever()
        return
    result = assess(parse_samples(args.input.read_text(encoding="utf-8")))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
