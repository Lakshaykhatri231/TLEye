"""Training loss vs learning rate for the Kaggle runs (script-generated: reports/lr_vs_loss.md).
Loss = mean BCE of the last 50 micro-batches at each logged step. Constant-guess BCE at the train TLE rate p is -(p ln p + (1-p) ln(1-p))."""
import json
import math
from pathlib import Path

import pandas as pd

P = 0.395   # train TLE rate of the 40k subset (from the Kaggle logs)
CONST = -(P * math.log(P) + (1 - P) * math.log(1 - P))
rows = []
for name, f in [("run1 (no prefix)", "reports/kaggle_train/train_log.json"), ("run2 (prefix, aborted at step 750)", "reports/kaggle_train_sym/train_log.json")]:
    d = pd.DataFrame(json.loads(Path(f).read_text()))
    d["lr_bin"] = pd.cut(d.lr, [0, 3e-5, 6e-5, 1e-4, 1.5e-4, 2.1e-4], labels=["<3e-5", "3-6e-5", "6e-5-1e-4", "1-1.5e-4", "1.5-2.1e-4"])
    g = d.groupby("lr_bin", observed=True).agg(logged_points=("loss", "size"), mean_loss=("loss", "mean"), last_step=("step", "max")).reset_index()
    g.insert(0, "run", name); rows.append(g)
t = pd.concat(rows)
out = ["# Training loss by learning-rate range (script-generated)", "",
       f"Constant-guess BCE at the training TLE rate {P}: {CONST:.3f}. Values clearly below this mean the model learned something.", "",
       t.round(4).to_markdown(index=False), "",
       "Note: within one run lr and training progress are confounded (lr decays as training proceeds), so this table is suggestive, not proof of an lr effect."]
Path("reports/lr_vs_loss.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
