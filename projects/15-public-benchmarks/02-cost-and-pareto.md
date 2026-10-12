# Cost, latency and throughput are different objectives

This methodology deep dive uses synthetic plotting fixtures. Every price in the figure is invented; no quote, invoice, GPU measurement or spending action is represented.

![Synthetic cost and prefill comparison](reports/context-2-pareto.png)

## Use observed output and billed footprint

Cost per million output tokens comes from total-footprint hourly cost, billed seconds and successful observed output tokens. The footprint must include all compared replicas/nodes. Billed time must cover the benchmark window and should include billed startup, warmup and idle time. Document region, date, spot/discount conditions and excluded fees.

Requested maximum output multiplied by request count is not observed work. EOS, cancellation and failed requests change the denominator. This reporter uses infrastructure footprint cost, not provider token billing. Separate input/output charges when analyzing APIs.

## Read a frontier correctly

A candidate is dominated if another eligible candidate is no worse on every objective and strictly better on at least one. JSON identifies the three-objective cost/TTFT/throughput frontier. Dashed figure curves are two-objective frontiers in each projection. A two-axis winner can lose on the third axis.

Pareto selection does not establish answer quality or a production recommendation. Quantization can improve performance while changing answers. Use [project 6's paired quality harness](../06-fp8-calibration/README.md), document service objectives and apply quality constraints before selecting a deployment.

The invented dataset includes a dominated control configuration to test frontier arithmetic. Its position is not evidence against any real engine.
