# Certificates among the solutions the LoRA model flags (script-generated)

Flag = P(TLE) >= 0.381 (F1-optimal on val, capped prevalence). Rows = test-split submissions from the two certificate pilots (supported problems only). Sandbox outcome strict rule (CPU >= limit). Wilson intervals ignore problem clustering; sample TLE share is by design ~47%, not real-world.

## pilot 1

289 submissions (24 problems), 141 flagged by the model.

| question | answer |
|---|---|
| share of true TLE solutions the model flags (recall) | 119/135 = 88.1% [81.6%, 92.6%] |
| share of true Accepted solutions the model flags (false alarms) | 22/154 = 14.3% [9.6%, 20.7%] |
| flagged AND truly TLE: confirmed by a failing input | 81/119 = 68.1% [59.2%, 75.8%] |
| flagged BUT truly Accepted: wrongly certified (false certificate) | 0/22 = 0.0% [0.0%, 14.9%] |
| all flagged: certified share | 81/141 = 57.4% [49.2%, 65.3%] |
| certified-TLE precision among flagged (certified that are truly TLE) | 81/81 = 100.0% [95.5%, 100.0%] |
| flagged TLE solutions left 'suspected only' (not confirmed) | 38/119 = 31.9% [24.2%, 40.8%] |

## pilot 2 (fresh problems, final battery)

268 submissions (24 problems), 143 flagged by the model.

| question | answer |
|---|---|
| share of true TLE solutions the model flags (recall) | 95/128 = 74.2% [66.0%, 81.0%] |
| share of true Accepted solutions the model flags (false alarms) | 48/140 = 34.3% [26.9%, 42.5%] |
| flagged AND truly TLE: confirmed by a failing input | 50/95 = 52.6% [42.7%, 62.4%] |
| flagged BUT truly Accepted: wrongly certified (false certificate) | 0/48 = 0.0% [0.0%, 7.4%] |
| all flagged: certified share | 50/143 = 35.0% [27.6%, 43.1%] |
| certified-TLE precision among flagged (certified that are truly TLE) | 50/50 = 100.0% [92.9%, 100.0%] |
| flagged TLE solutions left 'suspected only' (not confirmed) | 45/95 = 47.4% [37.6%, 57.3%] |

## both pilots pooled

557 submissions (48 problems), 284 flagged by the model.

| question | answer |
|---|---|
| share of true TLE solutions the model flags (recall) | 214/263 = 81.4% [76.2%, 85.6%] |
| share of true Accepted solutions the model flags (false alarms) | 70/294 = 23.8% [19.3%, 29.0%] |
| flagged AND truly TLE: confirmed by a failing input | 131/214 = 61.2% [54.5%, 67.5%] |
| flagged BUT truly Accepted: wrongly certified (false certificate) | 0/70 = 0.0% [0.0%, 5.2%] |
| all flagged: certified share | 131/284 = 46.1% [40.4%, 51.9%] |
| certified-TLE precision among flagged (certified that are truly TLE) | 131/131 = 100.0% [97.2%, 100.0%] |
| flagged TLE solutions left 'suspected only' (not confirmed) | 83/214 = 38.8% [32.5%, 45.5%] |

## Reading it
- A flag alone has many false alarms; the certificate is what supplies precision. Certified solutions are almost all truly TLE, but a large share of true TLE solutions stay 'suspected only' because no generated input slowed them down.
- No claim about real-world precision: the sample is label-balanced by design and covers only problems the generator fully supports.
