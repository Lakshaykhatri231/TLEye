# Phase 3 benchmark (Kaggle, script-generated from reports/kaggle_bench/bench.json)

GPU: Tesla T4 (n_gpu visible = 2, benchmark used ONE). max_len=1024, batch=4, fp16 autocast, LoRA r=16 on all attention+MLP projections, gradient checkpointing ON.

Only 10 timed steps per model after 2 warmup steps: treat timings as rough (+/- tens of percent), not precise.

| model | trainable params (M) | sec/step | rows/s | real tokens/s | peak GPU mem (GB) | h per 40k rows | h per full 142,400 train rows |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B | 8.8 | 0.943 | 4.24 | 1701 | 1.86 | 2.62 | 9.33 |
| Qwen/Qwen2.5-Coder-1.5B | 18.47 | 2.267 | 1.76 | 707 | 4.67 | 6.31 | 22.47 |

Prompt length (tokens, 300 train rows): median 331, p90 1065, 11% of rows exceed 1024 and would be truncated.

Loss values over the 12 steps are noisy (0.5B: 6.135, 5.732, 2.102, 2.242, 0.012, 0.0, 7.605, 3.301, 9.772, 7.004, 3.448, 4.416) - 12 steps with a freshly initialised head say nothing about final quality; they only suggest the head init / learning rate needs care.

Not measured: multi-GPU speedup, effect of dropping gradient checkpointing, length-bucketed batching, any validation metric.
