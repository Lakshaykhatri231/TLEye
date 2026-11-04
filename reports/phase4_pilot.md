# Phase 4 certificate pilots (script-generated)

Untrusted code ran only in the Docker sandbox on an arm64 M2 via Colima (no network, read-only root, 1 CPU, 1 GB). 'Certified' = CPU time >= the problem limit on some generated input, ON OUR HARDWARE, not the contest judge.
Only problems the generator fully supports were sampled (42.2% of test problems, see reports/cert_coverage.md). Intervals are Wilson and ignore clustering by problem (too narrow). 24 problems per pilot.

## Headline (rule: cpu >= limit)

| pilot | problems | certificate yield on true TLE | false certificate on true AC |
|---|---|---|---|
| 1 (rule changed afterwards, re-scored strictly) | 24 | 91/135 = 67.4% [59.1%, 74.7%] | 0/154 = 0.0% [0.0%, 2.4%] |
| 2 (fresh problems, held out) | 24 | 69/128 = 53.9% [45.3%, 62.3%] | 0/140 = 0.0% [0.0%, 2.7%] |
| 3 (newly supported problems; used to find engine bugs: DEVELOPMENT set) | 23 | 80/152 = 52.6% [44.7%, 60.4%] | 0/159 = 0.0% [0.0%, 2.4%] |
| 4 (fresh problems, engine after bounds fixes) | 29 | 104/171 = 60.8% [53.3%, 67.8%] | 0/181 = 0.0% [0.0%, 2.1%] |
| 5 (problems newly supported by relation handling; exposed missing rules: DEVELOPMENT set) | 29 | 95/170 = 55.9% [48.4%, 63.1%] | 2/165 = 1.2% [0.3%, 4.3%] |
| 6 (fresh problems, engine with the unmodeled-prose rule: the cleanest evaluation) | 30 | 109/179 = 60.9% [53.6%, 67.7%] | 1/196 = 0.5% [0.1%, 2.8%] |

### Same pilots restricted to the problems the FINAL classification supports (post hoc: the classification was tightened after seeing pilots 3, 5, 6)

| pilot | problems kept / run | certificate yield on true TLE | false certificate on true AC |
|---|---|---|---|
| 1 | 20 / 24 | 72/109 = 66.1% [56.8%, 74.3%] | 0/129 = 0.0% [0.0%, 2.9%] |
| 2 | 18 / 24 | 50/100 = 50.0% [40.4%, 59.6%] | 0/106 = 0.0% [0.0%, 3.5%] |
| 3 | 23 / 23 | 80/152 = 52.6% [44.7%, 60.4%] | 0/159 = 0.0% [0.0%, 2.4%] |
| 4 | 28 / 29 | 96/163 = 58.9% [51.2%, 66.2%] | 0/173 = 0.0% [0.0%, 2.2%] |
| 5 | 22 / 29 | 72/131 = 55.0% [46.4%, 63.2%] | 0/125 = 0.0% [0.0%, 3.0%] |
| 6 | 29 / 30 | 108/171 = 63.2% [55.7%, 70.0%] | 0/188 = 0.0% [0.0%, 2.0%] |

Pilot 1 under its ORIGINAL pre-registered rule (wall-timeout OR cpu>=limit): 93 certified TLE of 135; 1 false certificate of 154 (it relied on a wall-clock kill at 1.05 s CPU).

## Status table, pilot 2

| y_tle    |   certified |   compile_error |   inconclusive |   not_certified |
|:---------|------------:|----------------:|---------------:|----------------:|
| true AC  |           0 |               0 |             11 |             129 |
| true TLE |          69 |               3 |             18 |              41 |

## By runtime, pilot 2 (rule: cpu >= limit)

| runtime   | TLE certified   | AC falsely certified   |
|:----------|:----------------|:-----------------------|
| C++       | 33/45           | 0/48                   |
| Java      | 13/23           | 0/24                   |
| PyPy      | 11/28           | 0/35                   |
| Python    | 12/32           | 0/33                   |

## Sensitivity: judge-speed factor (threshold = limit x factor)

Per-runtime factors = median(our max CPU / judge CPU) on Accepted pilot-1 submissions with judge cpu >= 100 ms: {'C++': 0.25, 'Java': 0.64, 'PyPy': 0.37, 'Python': 0.59}. Inputs differ from the judge's hidden tests, so these are rough. Scored on HELD-OUT pilot 2; the plain rule above remains the headline.

| threshold | TLE yield | false cert on AC |
|---|---|---|
| limit x 1.00 (headline) | 69/128 = 53.9% [45.3%, 62.3%] | 0/140 = 0.0% [0.0%, 2.7%] |
| limit x 0.75 | 75/128 = 58.6% [49.9%, 66.8%] | 0/140 = 0.0% [0.0%, 2.7%] |
| limit x 0.50 | 78/128 = 60.9% [52.3%, 69.0%] | 1/140 = 0.7% [0.1%, 3.9%] |
| limit x 0.35 | 80/128 = 62.5% [53.9%, 70.4%] | 1/140 = 0.7% [0.1%, 3.9%] |
| per-runtime factors | 76/128 = 59.4% [50.7%, 67.5%] | 2/140 = 1.4% [0.4%, 5.1%] |

## Reading the numbers
- Speed scaling lifts held-out yield only a few points, so machine speed is NOT the main reason ~40% of true TLE submissions are not certified: they run fast on our generated inputs.
- 29 of 29 inconclusive pilot-2 submissions crashed (non-zero exit) on generated inputs, consistent with inputs violating constraints stated only in prose.
- Not measured here: precision of certificates among MODEL-flagged submissions, coverage beyond `simple` problems, repeat-run timing noise.
