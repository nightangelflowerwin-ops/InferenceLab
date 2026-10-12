# DCGM Observability and Throttling Dashboard

Status: metric snapshot parser/assessment, CPU fixture exporter, Grafana dashboard, Prometheus alert rules and a local Compose stack implemented. Local parser/exporter tests pass. CI is configured to validate real Grafana import, Compose, Prometheus scraping and rules. Real DCGM hardware telemetry remains pending.

```powershell
python observability.py assess projects/11-dcgm-dashboard/fixture.prom --out results/telemetry-fixture.json
python observability.py fixture-exporter projects/11-dcgm-dashboard/fixture.prom --port 9400
```

The fixture is explicitly synthetic and includes a source label on every sample. Missing, nonfinite and DCGM sentinel values stay unknown rather than being converted to zero. Snapshot checks distinguish SM activity from occupancy, flag configured temperature/framebuffer thresholds, report application OOM evidence and detect old explicit timestamps. The parser supports ordinary Prometheus samples/labels, not the full OpenMetrics/exemplar grammar.

On a host with Docker Compose:

```powershell
$env:GRAFANA_PASSWORD = "CHOOSE_A_LOCAL_DEMO_PASSWORD"
docker compose -f projects/11-dcgm-dashboard/compose.yml up -d
```

Open `http://127.0.0.1:3000`, sign in as admin with the selected local password, and open the InferenceLab dashboard. The stack runs a Python fixture exporter, Prometheus and Grafana; no NVIDIA device or cloud account is used. Grafana/Prometheus ports bind to loopback. The exporter binds to its isolated container network. Stop with the matching `docker compose ... down` command.

Nine panels cover temperature, SM active/occupancy, framebuffer usage, software OOMs, PCIe TX/RX, clock-event sample counts and an explicit fixture indicator. PCIe profiling gauges are bytes/sec for the referenced exporter field definitions. Clock-event window counts are sample counts, not unique incidents. `inference_gpu_oom_total` is an application counter the serving software must emit; it is not a native DCGM OOM counter. No-data panels deliberately have no zero fallback.

For real GPU use configure Prometheus to scrape an owned DCGM exporter, enable the supported fields in `dcgm-counters.example.csv`, verify units/driver support, and emit the software OOM counter from serving code. Profiling and exporter-owned clock metrics may be unavailable or require explicit enablement. A temperature threshold or low occupancy alone is not proof of throttling. No hardware event or SLO correlation was measured on this laptop.

CI separately imports and reads back the dashboard via real Grafana, validates Compose, checks Prometheus rules and scrapes real HTTP fixture data. These checks do not claim browser rendering or DCGM GPU validation. [DCGM metric definitions](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html), [exporter field units](https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv), [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).
