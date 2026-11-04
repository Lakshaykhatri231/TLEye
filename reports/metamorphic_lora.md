# LoRA metamorphic checks (script-generated; follows reports/stage1_predeclaration.md)

Noise floor (same rows rebatched, n=256): median |change| 0.002, 99th percentile 0.012 logit.

## val split

| change | n | problems | mean logit change [95% CI] | mean abs change | by loop depth 0 / 1 / 2 / 3+ | wrong-direction share at 0 / 0.1 / 0.5 logit |
|---|---|---|---|---|---|---|
| n_div100 | 2383 | 107 | -0.266 [-0.394, -0.136] | 0.523 | -0.12 / -0.035 / -0.617 / -0.142 | 42.1% / 26.1% / 6.7% |
| n_div10 | 2383 | 107 | -0.229 [-0.319, -0.145] | 0.317 | -0.181 / -0.12 / -0.413 / -0.112 | 25.1% / 8.1% / 1.9% |
| n_x10 | 2383 | 107 | +0.070 [+0.020, +0.114] | 0.163 | 0.068 / 0.075 / 0.069 / 0.055 | 39.0% / 11.3% / 1.3% |
| n_x100 | 2383 | 107 | +0.162 [+0.101, +0.231] | 0.267 | 0.162 / 0.157 / 0.168 / 0.165 | 29.1% / 12.9% / 2.0% |
| n_x1000 | 2383 | 107 | +0.280 [+0.219, +0.363] | 0.367 | 0.299 / 0.296 / 0.248 / 0.306 | 28.1% / 14.3% / 1.3% |
| tl_x2 | 3000 | 148 | +0.001 [+0.000, +0.002] | 0.008 | 0.006 / 0.002 / -0.0 / -0.001 | 52.1% / 0.1% / 0.0% |
| tl_div2 | 3000 | 148 | +0.001 [-0.001, +0.002] | 0.009 | 0.011 / 0.001 / -0.002 / -0.005 | 47.6% / 0.3% / 0.0% |
| placebo_reorder | 1892 | 92 | +0.024 [+0.003, +0.044] | 0.103 | 0.014 / 0.036 / 0.016 / 0.011 | - |
| placebo_tl | 3000 | 148 | +0.002 [-0.001, +0.004] | 0.025 | 0.016 / 0.001 / -0.001 / -0.004 | - |

## test split

| change | n | problems | mean logit change [95% CI] | mean abs change | by loop depth 0 / 1 / 2 / 3+ | wrong-direction share at 0 / 0.1 / 0.5 logit |
|---|---|---|---|---|---|---|
| n_x10 | 4325 | 248 | +0.142 [+0.093, +0.203] | 0.222 | 0.156 / 0.051 / 0.222 / 0.196 | 41.8% / 12.0% / 1.1% |
| tl_x2 | 5000 | 295 | +0.001 [+0.001, +0.002] | 0.008 | 0.005 / 0.001 / 0.001 / 0.003 | 51.4% / 0.1% / 0.0% |
| placebo_reorder | 3443 | 203 | +0.001 [-0.016, +0.018] | 0.114 | 0.027 / 0.003 / -0.004 / -0.016 | - |
| placebo_tl | 5000 | 295 | +0.004 [+0.002, +0.006] | 0.027 | 0.01 / 0.005 / 0.002 / 0.001 | - |

## Claims by the pre-declared rules

- At depth >= 2, N x10 mean logit change on test: +0.216 [+0.138, +0.294]. Reorder placebo mean absolute change: 0.114.
- C1 responds to N: **True**. C2 beyond noise: **True**. C3 responds to the time limit: **False**.
- Stage 3 gates (val): N patching **False** (N x1000 at depth >= 2: +0.264), time-limit patching **False** (time limit x2: +0.001).

## N x10 on the same test rows: LoRA vs B2m (logit scale)

| model | n | mean logit change | wrong-direction share at 0 / 0.1 / 0.5 logit |
|---|---|---|---|
| LoRA (seed 0) | 4325 | +0.142 | 41.8% / 12.0% / 1.1% |
| B2m (monotone by construction) | 4325 | +0.158 | 0.0% / 0.0% / 0.0% |

## Prefix-edit pre-test (val, adapter P, descriptive only)

n=2000; logit SD 3.43.

| edit | AUROC | mean change | mean abs change | flips at logit 0 |
|---|---|---|---|---|
| true analysis | 0.886 | - | - | - |
| neutral | 0.885 | -0.090 | 0.118 | 2.2% |
| depth_up | 0.887 | +0.136 | 0.162 | 1.8% |
| depth_down | 0.885 | -0.063 | 0.072 | 1.3% |
| donor | 0.886 | +0.019 | 0.103 | 1.1% |

