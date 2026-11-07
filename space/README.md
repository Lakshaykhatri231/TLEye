---
title: TLEye
emoji: ⏱️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - LakshayKhatri231/tleye-lora-0.5b
short_description: Predicts Time Limit Exceeded for contest solutions
---

# TLEye demo

Paste a C++, Java or Python competitive-programming solution and the problem's constraints; a fine-tuned Qwen2.5-Coder-0.5B (LoRA)
predicts whether it will get **Time Limit Exceeded**.

This hosted demo gives the verdict only. Sandbox certificates (running the code on generated worst-case inputs) need Docker and run
locally: see the [code repository](https://github.com/Lakshaykhatri231/TLEye). Model card: [tleye-lora-0.5b](https://huggingface.co/LakshayKhatri231/tleye-lora-0.5b).

The model ignores the time limit and reacts only weakly to the input size N; probabilities are calibrated to a training set with about
40% TLE. An estimate, never a proof that a solution is fast enough.
