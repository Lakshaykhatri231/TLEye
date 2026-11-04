# Metamorphic checks (test split, script-generated)

n = 42990 rows, 296 problems. Violation = probability moves the wrong way by > 1e-09.

| model                      |   test_auroc |   best_iter | N x10 lowers P(TLE)     |   N x10 lowers P(TLE) (mean abs dP) | time limit x2 raises P(TLE)   |   time limit x2 raises P(TLE) (mean abs dP) |
|:---------------------------|-------------:|------------:|:------------------------|------------------------------------:|:------------------------------|--------------------------------------------:|
| B2 (plain)                 |       0.7592 |         337 | 36.53% [33.26%, 39.97%] |                               0.053 | 71.56% [68.27%, 74.76%]       |                                      0.0382 |
| B2m (monotone constraints) |       0.7562 |         403 | 0.02% [0.01%, 0.04%]    |                               0.03  | 0.00% [0.00%, 0.00%]          |                                      0.0056 |

Paired AUROC change from adding monotone constraints (B2m - B2): -0.0030 [-0.0062, +0.0012]
