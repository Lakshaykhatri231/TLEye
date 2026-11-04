# Phase 1 data feasibility spike (Gate 1)

All numbers computed by `tleye/data/spike.py` + `spike_report.py` from CodeNet metadata. Scope: languages C++ / Java / Python, statuses Accepted vs Time Limit Exceeded only.

## 1. Dataset basics
- Problems listed: 4053 ({'AIZU': 2534, 'AtCoder': 1519}); time_limit missing: 56
- Statements found: 3999 / 4053

## 2. Label quality: cpu_time / limit ratio (the key finding)
Ratio bins: <0, 0-.1, .1-.25, .25-.5, .5-.8, .8-.95, .95-1, 1-1.05, 1.05-1.5, 1.5-3, >3

| dataset | lang | status | n | share >= 1.0x limit | histogram |
|---|---|---|---:|---:|---|
| AIZU | C++ | Accepted | 525003 | 0.02% | [0, 464993, 35460, 14113, 6674, 2819, 859, 84, 0, 1, 0] |
| AIZU | C++ | Time Limit Exceeded | 49508 | 99.55% | [0, 1, 2, 7, 46, 75, 93, 1763, 9171, 18271, 20079] |
| AIZU | Java | Accepted | 82888 | 2.52% | [0, 53771, 16612, 6714, 2827, 684, 189, 159, 1213, 719, 0] |
| AIZU | Java | Time Limit Exceeded | 6996 | 99.89% | [0, 1, 0, 0, 5, 0, 2, 0, 4, 2135, 4849] |
| AIZU | Python | Accepted | 130933 | 7.18% | [0, 102058, 6969, 5921, 4485, 1563, 535, 444, 2231, 3394, 3333] |
| AIZU | Python | Time Limit Exceeded | 7413 | 100.00% | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7413] |
| AtCoder | C++ | Accepted | 3816212 | 0.00% | [0, 3527210, 193839, 58822, 22293, 8901, 5021, 125, 1, 0, 0] |
| AtCoder | C++ | Time Limit Exceeded | 275834 | 100.00% | [0, 0, 0, 0, 0, 0, 0, 2436, 272875, 516, 7] |
| AtCoder | Java | Accepted | 265474 | 0.01% | [0, 192920, 37650, 23880, 7878, 2199, 924, 23, 0, 0, 0] |
| AtCoder | Java | Time Limit Exceeded | 29714 | 100.00% | [0, 0, 0, 0, 0, 0, 0, 391, 29322, 1, 0] |
| AtCoder | Python | Accepted | 1663146 | 0.01% | [0, 1331241, 181963, 88630, 40031, 14740, 6379, 162, 0, 0, 0] |
| AtCoder | Python | Time Limit Exceeded | 290841 | 100.00% | [0, 0, 0, 0, 0, 0, 0, 2904, 287771, 159, 7] |

## 3. Problems and submissions available (problems having both AC and TLE)
| dataset   | language   |   problems |   with_tle |   both |   both_ge5 |   ac_sub |   tle_sub |
|:----------|:-----------|-----------:|-----------:|-------:|-----------:|---------:|----------:|
| AIZU      | C++        |       2508 |       1759 |   1755 |       1177 |   536837 |     50506 |
| AIZU      | Java       |       1811 |        667 |    643 |        231 |    89508 |      7161 |
| AIZU      | Python     |       1737 |        404 |    379 |        157 |   133417 |      7462 |
| AtCoder   | C++        |       1513 |       1467 |   1467 |       1321 |  3816212 |    275834 |
| AtCoder   | Java       |       1458 |        966 |    958 |        615 |   265474 |     29714 |
| AtCoder   | Python     |       1423 |       1171 |   1149 |        870 |  1663146 |    290842 |

## 4. Python runtime split (TLE submissions)
| dataset   | pypy   |      n |
|:----------|:-------|-------:|
| AIZU      | False  |   7462 |
| AtCoder   | False  | 210457 |
| AtCoder   | True   |  80385 |

## 5. Statement parseability (crude regex probe, NOT the Phase 2 parser)
| dataset   |   problems |   with_stmt |   constraints_section |   numeric_bound |   contains_cjk |
|:----------|-----------:|------------:|----------------------:|----------------:|---------------:|
| AIZU      |       2534 |        2480 |                   552 |             476 |           1352 |
| AtCoder   |       1519 |        1519 |                  1502 |            1432 |              4 |

## 6. Borderline AC rule: share of AC rows with cpu_time >= 0.8x limit
- AIZU C++: 0.72% of 525003
- AIZU Java: 3.58% of 82888
- AIZU Python: 8.78% of 130933
- AtCoder C++: 0.37% of 3816212
- AtCoder Java: 1.19% of 265474
- AtCoder Python: 1.28% of 1663146
