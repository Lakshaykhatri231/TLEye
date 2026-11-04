# Linear probes (script-generated; follows reports/stage2_predeclaration.md)

Probes show what is **linearly decodable** from the model's internal state, not what it uses to decide.

## Validation: chosen sites (committed before test was opened)

| target | tuned site | tuned CV score | base site | base CV score |
|---|---|---|---|---|
| depth_ge2 | pool12 | 0.952 | pool12 | 0.956 |
| recursive | pool12 | 0.969 | pool12 | 0.970 |
| has_sort | pool12 | 0.921 | pool12 | 0.939 |
| sort_in_loop | last24 | 0.925 | pool12 | 0.874 |
| listed_any | last24 | 0.885 | pool12 | 0.874 |
| loops_log1p | pool12 | 0.840 | pool12 | 0.850 |
| log10_n | pool12 | 0.215 | pool24 | 0.061 |
| est | pool12 | 0.261 | pool12 | 0.110 |
| lang_python | last8 | 1.000 | last24 | 1.000 |
| tle | last24 | 0.887 | pool12 | 0.789 |
| log_tl | last8 | -0.018 | last24 | -0.027 |

Pipeline checks: {"layer0_last_max_var_tuned": 0.0, "layer0_last_max_var_base": 0.0, "tuned_last24_r2_for_logit": 1.0}

## Test split (8000 rows, 296 problems; AUROC for binary targets, R2 for numeric; problem-clustered 95% CI)

| target | tuned | base | metadata | bag of tokens | shuffled control | tuned minus metadata | tuned minus bag of tokens | tuned minus base | demeaned corr (tuned) |
|---|---|---|---|---|---|---|---|---|---|
| depth_ge2 | 0.959 [0.953, 0.965] | 0.961 [0.955, 0.967] | 0.646 | 0.913 | 0.760 | +0.313 [+0.288, +0.337] | +0.046 [+0.038, +0.054] | -0.002 [-0.005, +0.000] | 0.691 [0.674, 0.706] |
| recursive | 0.965 [0.960, 0.971] | 0.966 [0.960, 0.971] | 0.798 | 0.919 | 0.773 | +0.167 [+0.148, +0.187] | +0.046 [+0.040, +0.055] | -0.001 [-0.002, +0.001] | 0.728 [0.715, 0.744] |
| has_sort | 0.925 [0.912, 0.938] | 0.951 [0.941, 0.960] | 0.589 | 0.929 | 0.778 | +0.336 [+0.304, +0.371] | -0.004 [-0.015, +0.007] | -0.026 [-0.034, -0.018] | 0.514 [0.484, 0.544] |
| sort_in_loop | 0.926 [0.903, 0.946] | 0.884 [0.856, 0.909] | 0.639 | 0.835 | 0.801 | +0.287 [+0.234, +0.346] | +0.091 [+0.058, +0.123] | +0.042 [+0.014, +0.072] | 0.282 [0.231, 0.336] |
| listed_any | 0.885 [0.873, 0.896] | 0.859 [0.843, 0.874] | 0.630 | 0.812 | 0.765 | +0.255 [+0.233, +0.278] | +0.073 [+0.060, +0.088] | +0.027 [+0.015, +0.038] | 0.510 [0.488, 0.530] |
| lang_python | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.625 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.996 [0.996, 0.997] |
| tle | 0.865 [0.848, 0.883] | 0.749 [0.729, 0.770] | 0.562 | 0.686 | 0.678 | +0.303 [+0.280, +0.328] | +0.179 [+0.160, +0.199] | +0.116 [+0.099, +0.132] | 0.625 [0.599, 0.660] |
| loops_log1p | 0.866 [0.855, 0.876] | 0.869 [0.858, 0.879] | 0.296 | 0.779 | 0.302 | +0.570 [+0.543, +0.603] | +0.086 [+0.077, +0.097] | -0.003 [-0.007, +0.000] | 0.894 [0.888, 0.900] |
| log10_n | 0.383 [0.228, 0.464] | 0.063 [-0.225, 0.167] | 0.004 | 0.133 | -0.031 | +0.379 [+0.264, +0.460] | +0.250 [+0.184, +0.325] | +0.319 [+0.234, +0.482] | - |
| est | 0.342 [0.268, 0.391] | 0.132 [0.032, 0.209] | 0.017 | 0.126 | -0.055 | +0.325 [+0.256, +0.375] | +0.216 [+0.167, +0.258] | +0.210 [+0.151, +0.276] | - |
| log_tl | -0.002 [-0.074, 0.047] | -0.007 [-0.065, 0.022] | 0.028 | -0.007 | 0.000 | -0.030 [-0.075, +0.003] | +0.005 [-0.033, +0.046] | +0.005 [-0.029, +0.038] | - |

## Claims by the pre-declared rules

| target | decodable beyond metadata | beyond bag of tokens | fine-tuning changed it (Holm p) |
|---|---|---|---|
| depth_ge2 | yes | yes | no (0.268, -) |
| recursive | yes | yes | no (0.318, -) |
| has_sort | yes | no | yes (0.016, -) |
| sort_in_loop | yes | yes | yes (0.016, +) |
| listed_any | yes | yes | yes (0.016, +) |
| loops_log1p | yes | yes | no (0.138, -) |
| log10_n | yes | yes | yes (0.016, +) |
| est | yes | yes | yes (0.016, +) |

Caveat: the base model was never trained to summarize at the final `:`, so a tuned gain at the last token may only mean information moved there.

