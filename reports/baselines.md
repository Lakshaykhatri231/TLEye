# Baselines (B1 rule, B2 LightGBM) - test split, problem-disjoint

Point estimate [95% CI]; CIs from a problem-clustered bootstrap (300 resamples).
Prevalence = share of TLE in the CAPPED sample (40 per class per problem/language): AUPRC, F1 and ECE depend on it and must not be read as real-world rates; AUROC does not.

## Main (train on all 3 languages)
| model                                            | scope   |     n |   problems |   prevalence | auroc                | auprc                | f1                   | ece                  |
|:-------------------------------------------------|:--------|------:|-----------:|-------------:|:---------------------|:---------------------|:---------------------|:---------------------|
| B1 rule                                          | ALL     | 42990 |        296 |        0.401 | 0.632 [0.613, 0.650] | 0.504 [0.483, 0.526] | 0.585 [0.571, 0.598] | n/a                  |
| B1 rule                                          | C++     | 19779 |        293 |        0.423 | 0.607 [0.586, 0.626] | 0.515 [0.495, 0.537] | 0.591 [0.576, 0.604] | n/a                  |
| B1 rule                                          | Java    |  9759 |        198 |        0.336 | 0.650 [0.614, 0.683] | 0.442 [0.405, 0.481] | 0.531 [0.493, 0.566] | n/a                  |
| B1 rule                                          | Python  | 13452 |        240 |        0.418 | 0.646 [0.624, 0.670] | 0.531 [0.505, 0.559] | 0.603 [0.586, 0.620] | n/a                  |
| B2 LightGBM (AST+constraints) +Platt             | ALL     | 42990 |        296 |        0.401 | 0.759 [0.738, 0.775] | 0.675 [0.643, 0.703] | 0.650 [0.637, 0.664] | 0.020 [0.013, 0.033] |
| B2 LightGBM (AST+constraints) +Platt             | C++     | 19779 |        293 |        0.423 | 0.744 [0.727, 0.763] | 0.677 [0.648, 0.702] | 0.653 [0.640, 0.664] | 0.031 [0.021, 0.042] |
| B2 LightGBM (AST+constraints) +Platt             | Java    |  9759 |        198 |        0.336 | 0.780 [0.753, 0.806] | 0.658 [0.605, 0.709] | 0.618 [0.585, 0.648] | 0.030 [0.022, 0.050] |
| B2 LightGBM (AST+constraints) +Platt             | Python  | 13452 |        240 |        0.418 | 0.757 [0.733, 0.781] | 0.684 [0.654, 0.712] | 0.662 [0.643, 0.682] | 0.016 [0.018, 0.035] |
| B2 LightGBM (AST+constraints) raw (uncalibrated) | ALL     | 42990 |        296 |        0.401 | 0.759 [0.738, 0.775] | 0.675 [0.643, 0.703] | 0.648 [0.635, 0.662] | 0.008 [0.008, 0.021] |
| B2 LightGBM (AST only) +Platt                    | ALL     | 42990 |        296 |        0.401 | 0.727 [0.707, 0.746] | 0.620 [0.594, 0.645] | 0.630 [0.615, 0.644] | 0.025 [0.018, 0.041] |
| B2 LightGBM (AST only) +Platt                    | C++     | 19779 |        293 |        0.423 | 0.721 [0.703, 0.738] | 0.643 [0.621, 0.663] | 0.642 [0.628, 0.654] | 0.032 [0.023, 0.043] |
| B2 LightGBM (AST only) +Platt                    | Java    |  9759 |        198 |        0.336 | 0.740 [0.708, 0.770] | 0.580 [0.529, 0.628] | 0.569 [0.528, 0.604] | 0.042 [0.028, 0.064] |
| B2 LightGBM (AST only) +Platt                    | Python  | 13452 |        240 |        0.418 | 0.717 [0.694, 0.741] | 0.610 [0.575, 0.643] | 0.641 [0.621, 0.662] | 0.023 [0.017, 0.042] |

## Leave-one-language-out (train+val on 2 languages, test on the 3rd; language one-hots removed)
| model                   | scope           |     n |   problems |   prevalence | auroc                | auprc                | f1                   | ece                  |
|:------------------------|:----------------|------:|-----------:|-------------:|:---------------------|:---------------------|:---------------------|:---------------------|
| B1 rule                 | held-out C++    | 19779 |        293 |        0.423 | 0.607 [0.586, 0.626] | 0.515 [0.495, 0.537] | 0.523 [0.495, 0.545] | n/a                  |
| B2 LightGBM LOLO +Platt | held-out C++    | 19779 |        293 |        0.423 | 0.707 [0.688, 0.726] | 0.625 [0.599, 0.650] | 0.622 [0.607, 0.636] | 0.094 [0.082, 0.106] |
| B1 rule                 | held-out Java   |  9759 |        198 |        0.336 | 0.650 [0.614, 0.683] | 0.442 [0.405, 0.481] | 0.508 [0.478, 0.537] | n/a                  |
| B2 LightGBM LOLO +Platt | held-out Java   |  9759 |        198 |        0.336 | 0.733 [0.703, 0.761] | 0.604 [0.554, 0.659] | 0.551 [0.520, 0.580] | 0.105 [0.086, 0.129] |
| B1 rule                 | held-out Python | 13452 |        240 |        0.418 | 0.644 [0.621, 0.669] | 0.521 [0.497, 0.550] | 0.553 [0.524, 0.587] | n/a                  |
| B2 LightGBM LOLO +Platt | held-out Python | 13452 |        240 |        0.418 | 0.693 [0.670, 0.717] | 0.606 [0.574, 0.643] | 0.623 [0.603, 0.644] | 0.075 [0.060, 0.093] |

## B2 details
```
{
 "B2_top_features_gain": {
  "est_log_ops": 0.1175,
  "n_loops": 0.1055,
  "n_nodes": 0.0824,
  "n_funcs": 0.0648,
  "max_loop_depth": 0.0632,
  "log10_primary": 0.0562,
  "b1_score": 0.0546,
  "log10_size": 0.0416,
  "log10_max_bound": 0.0408,
  "n_rec_calls": 0.0369,
  "n_calls_in_loop": 0.0354,
  "recursive": 0.0351
 },
 "B2_best_iter": 337
}
```
B1 thresholds (val-tuned, per runtime group): {"PyPy": 4.699, "Python": 3.097, "C++": 2.21, "Java": 5.699}