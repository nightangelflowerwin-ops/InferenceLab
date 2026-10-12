"""vLLM tuning runner. Planning works anywhere; execution requires Linux/CUDA."""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import subprocess
import time
import urllib.request


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def command(config, candidate, out, repeat, workload):
    serve = ["vllm", "serve", config["model"], "--revision", config["revision"],
             "--host", "127.0.0.1", "--port", str(config["port"]),
             "--dtype", "half", "--max-model-len", str(config["max_model_len"]),
             "--gpu-memory-utilization", str(candidate["memory"]),
             "--max-num-seqs", str(candidate["seqs"]),
             "--max-num-batched-tokens", str(candidate["tokens"]),
             "--enable-chunked-prefill" if candidate["chunked"] else "--no-enable-chunked-prefill",
             "--no-enable-prefix-caching"]
    bench = ["vllm", "bench", "serve", "--backend", "vllm", "--model", config["model"],
             "--tokenizer", config["model"], "--dataset-name", "random",
             "--base-url", f"http://127.0.0.1:{config['port']}",
             "--endpoint", "/v1/completions", "--random-input-len", str(workload["input"]),
             "--random-output-len", str(workload["output"]), "--random-range-ratio", "0",
             "--num-prompts", str(config["requests"]),
             "--max-concurrency", str(workload["concurrency"]),
             "--request-rate", str(workload["request_rate"]), "--seed", str(config["seed"]),
             "--ignore-eos", "--num-warmups", "8", "--percentile-metrics", "ttft,tpot,itl,e2el",
             "--metric-percentiles", "50,95,99", "--save-result", "--save-detailed",
             "--result-dir", str(out), "--result-filename", f"metrics-{repeat}.json"]
    return serve, bench


def make_plan(config, root):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", config["revision"]):
        raise ValueError("Set revision to an immutable model commit hash before planning.")
    grid = config["grid"]
    jobs = []
    for seqs, tokens, memory, chunked in itertools.product(
            grid["seqs"], grid["tokens"], grid["memory"], grid["chunked"]):
        if not chunked and tokens < config["max_model_len"]:
            continue
        candidate = dict(seqs=seqs, tokens=tokens, memory=memory, chunked=chunked)
        for workload in config["workloads"]:
            if workload["input"] + workload["output"] > config["max_model_len"]:
                raise ValueError("Workload exceeds max_model_len")
            name = f"s{seqs}-t{tokens}-m{memory}-c{int(chunked)}-{workload['name']}"
            out = root / name
            jobs.append(dict(name=name, candidate=candidate, workload=workload,
                             serve=command(config, candidate, out, 0, workload)[0],
                             benchmarks=[command(config, candidate, out, r, workload)[1]
                                         for r in range(config["repeats"])]))
    random.Random(config["seed"]).shuffle(jobs)
    return jobs


def ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before readiness; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("Server readiness timed out")


def stop(process):
    if process is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(config, jobs, root):
    if platform.system() != "Linux" or not shutil.which("nvidia-smi") or not shutil.which("vllm"):
        raise RuntimeError("Execution needs Linux, nvidia-smi and installed vLLM. Planning is available on this laptop.")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv"], text=True)
    if config["gpu_name_contains"].lower() not in gpu.lower():
        raise RuntimeError("GPU differs from configured target architecture")
    # Do not attach benchmark traffic to an unrelated server already using the port.
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", config["port"]))
    dump(root / "environment.json", dict(gpu=gpu, platform=platform.platform(),
         vllm=subprocess.check_output(["vllm", "--version"], text=True), config=config,
         timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    for job in jobs:
        out = root / job["name"]
        out.mkdir()
        dump(out / "job.json", job)
        server = telemetry = None
        status = {"status": "failed"}
        print("Running", job["name"], flush=True)
        with (out / "server.log").open("w") as log, (out / "gpu.csv").open("w") as gpu_log:
            try:
                server = subprocess.Popen(job["serve"], stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                ready(server, config["port"], config["startup_timeout"])
                telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv", "-l", "1"], stdout=gpu_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                for r, cmd in enumerate(job["benchmarks"]):
                    with (out / f"bench-{r}.log").open("w") as bench_log:
                        benchmark = subprocess.Popen(cmd, stdout=bench_log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        try:
                            code = benchmark.wait(timeout=config["benchmark_timeout"])
                            if code:
                                raise subprocess.CalledProcessError(code, cmd)
                        finally:
                            stop(benchmark)
                    data = json.loads((out / f"metrics-{r}.json").read_text())
                    if data.get("completed") != config["requests"]:
                        raise RuntimeError("Incomplete requests; exclude this candidate")
                status = {"status": "passed"}
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                status["error"] = str(error)
            finally:
                stop(telemetry)
                stop(server)
                dump(out / "status.json", status)


def summarize(root, ttft_slo):
    rows = []
    for status_file in sorted(root.glob("*/status.json")):
        if json.loads(status_file.read_text())["status"] != "passed":
            continue
        for file in sorted(status_file.parent.glob("metrics-*.json")):
            data = json.loads(file.read_text())
            ttft = data.get("p95_ttft_ms")
            throughput = data.get("output_throughput")
            if not isinstance(ttft, (int, float)) or not isinstance(throughput, (int, float)):
                continue
            if not math.isfinite(ttft) or not math.isfinite(throughput) or ttft < 0 or throughput <= 0:
                continue
            rows.append(dict(candidate=file.parent.name, repeat=file.stem,
                             output_tokens_s=throughput, p95_ttft_ms=ttft,
                             p95_itl_ms=data.get("p95_itl_ms"), completed=data.get("completed"),
                             within_ttft_slo=ttft <= ttft_slo))
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["candidate", "repeat", "output_tokens_s",
                    "p95_ttft_ms", "p95_itl_ms", "completed", "within_ttft_slo"])
        writer.writeheader()
        writer.writerows(rows)
    # Require every repeat to meet the SLO; compare median throughput across repeats.
    import statistics
    groups = {}
    for row in rows:
        groups.setdefault(row["candidate"], []).append(row)
    plan = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else []
    expected = {j["name"]: len(j["benchmarks"]) for j in plan}
    winners = []
    for name, group in groups.items():
        if name in expected and len(group) == expected[name] and all(r["within_ttft_slo"] for r in group):
            winners.append(dict(candidate=name, median_output_tokens_s=statistics.median(
                r["output_tokens_s"] for r in group)))
    # Different workload shapes are not comparable; select separately for each workload.
    workload_names = {j["name"]: j["workload"]["name"] for j in plan}
    best = {}
    for entry in winners:
        workload = workload_names[entry["candidate"]]
        if workload not in best or entry["median_output_tokens_s"] > best[workload]["median_output_tokens_s"]:
            best[workload] = entry
    dump(root / "recommendations.json", {"ttft_slo_ms": ttft_slo, "best_per_workload": best,
        "note": "No recommendation means no complete eligible measurements. These are workload-specific results."})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run", "report"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    args = parser.parse_args()
    if args.action == "report":
        print("Reported", len(summarize(args.out, args.ttft_slo_ms)), "measured repeats")
        return
    config = json.loads(Path(args.config).read_text())
    jobs = make_plan(config, args.out.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "plan.json", jobs)
    if args.action == "run":
        run(config, jobs, args.out.resolve())
        summarize(args.out, args.ttft_slo_ms)
    print(f"{len(jobs)} configurations/workloads; {sum(len(j['benchmarks']) for j in jobs)} benchmark runs")


if __name__ == "__main__":
    main()
