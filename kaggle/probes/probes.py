"""STAGE 2 probes (pre-declared in reports/stage2_predeclaration.md). CPU only.
PHASE 'val': choose one site per target and model by grouped CV on val; test shards are NOT opened. PHASE 'test': fit on val with the committed choices and predict test once."""
import glob, json, os, sys, time

PHASE = "val"
CHOICES = None                     # pasted from the committed val_choices.json before the test phase
import numpy as np
import pandas as pd

find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)  # noqa: E731
sys.path.insert(0, os.path.dirname(find("probe_lib.py")[0]))
import kernel_lib as kl
import probe_lib as pl

OUT = "/kaggle/working"; T0 = time.time()
LAYERS, POOL = [0, 4, 8, 12, 16, 20, 24], [0, 12, 24]


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def load_states(name):
    fs = sorted(glob.glob(f"/kaggle/input/**/hidden/{name}_*.npz", recursive=True)); assert fs, name
    ids, last, pool, lg = [], [], [], []
    for f in fs:
        z = np.load(f); ids.append(z["submission_id"]); last.append(z["last"].astype(np.float32)); pool.append(z["pool"].astype(np.float32)); lg.append(z["logit"])
    return np.concatenate(ids), np.concatenate(last), np.concatenate(pool), np.concatenate(lg)


pack = pd.read_parquet(find("tleye_data.parquet")[0], columns=["submission_id", "problem_id", "language", "y_tle", "time_limit_ms"])
side = pd.read_parquet(find("sft_targets.parquet")[0])
meta = pack.merge(side, on="submission_id", validate="one_to_one").set_index("submission_id")


def table(ids):
    d = meta.loc[ids]
    est = (d.vis_loop_depth.clip(lower=1) * d.log10_n).clip(upper=30)
    t = {"depth_ge2": ("binary", (d.vis_loop_depth >= 2).astype(float).values), "recursive": ("binary", d.vis_recursive.astype(float).values), "has_sort": ("binary", (d.vis_sorts > 0).astype(float).values),
         "sort_in_loop": ("binary", (d.vis_sort_in_loop > 0).astype(float).values), "listed_any": ("binary", (d.vis_listed > 0).astype(float).values), "loops_log1p": ("numeric", np.log1p(d.vis_loops.values.astype(float))),
         "log10_n": ("numeric", d.log10_n.values.astype(float)), "est": ("numeric", est.values.astype(float)), "lang_python": ("binary", (d.language == "Python").astype(float).values),
         "tle": ("binary", d.y_tle.astype(float).values), "log_tl": ("numeric", np.log10(d.time_limit_ms.values / 1000.0))}
    X_meta = np.c_[np.log10(d.n_prompt_tokens.values.astype(float)), d.truncated.astype(float).values, (d.language == "C++").astype(float).values, (d.language == "Java").astype(float).values]
    return d, t, X_meta


PROBLEM_LEVEL = {"log10_n", "est", "log_tl"}
ids_v, last_v, pool_v, lg_v = load_states("tuned_val"); ids_b, last_bv, pool_bv, _ = load_states("base_val")
assert (ids_v == ids_b).all(), "tuned and base val rows differ"
dv, tv, Xmv = table(ids_v); gv = dv.problem_id.values
checks = {"layer0_last_max_var_tuned": float(last_v[:, 0].var(0).max()), "layer0_last_max_var_base": float(last_bv[:, 0].var(0).max())}
assert checks["layer0_last_max_var_tuned"] == 0 and checks["layer0_last_max_var_base"] == 0, "last token at layer 0 should be constant"


def site_features(last, pool):
    s = {f"last{L}": last[:, i] for i, L in enumerate(LAYERS) if L != 0}
    s.update({f"pool{L}": pool[:, i] for i, L in enumerate(POOL) if L != 0})
    return s


sites = {"tuned": site_features(last_v, pool_v), "base": site_features(last_bv, pool_bv)}
if PHASE == "val":
    rows, choices = [], {}
    for tname, (kind, y) in tv.items():
        ok = np.isfinite(y); keep = 50 if tname in PROBLEM_LEVEL else None; choices[tname] = {}
        for model in ("tuned", "base"):
            best = None
            for site, X in sites[model].items():
                a, sc, oof = pl.cv_select(X[ok], y[ok], gv[ok], kind, keep=keep)
                dm = pl.demeaned_corr(y[ok], oof, gv[ok]) if tname not in PROBLEM_LEVEL else float("nan")
                rows.append({"target": tname, "model": model, "site": site, "alpha": a, "cv_score": sc, "cv_demeaned_corr": dm, "n": int(ok.sum()), "problems": len(set(gv[ok]))})
                if not np.isnan(sc) and (best is None or sc > best[1]):
                    best = (site, sc, a)
            choices[tname][model] = {"site": best[0], "alpha": best[2], "cv_score": best[1]}
            log(tname, model, choices[tname][model])
    _, sc, oof = pl.cv_select(sites["tuned"]["last24"], lg_v, gv, "numeric")
    checks["tuned_last24_r2_for_logit"] = sc
    assert sc > 0.9, "tuned layer 24 last-token probe should track the logit it feeds"
    pd.DataFrame(rows).to_csv(f"{OUT}/val_scores.csv", index=False)
    kl.atomic_json(f"{OUT}/val_choices.json", {"choices": choices, "checks": checks})
else:
    assert CHOICES, "paste the committed val choices first"
    ids_t, last_t, pool_t, lg_t = load_states("tuned_test"); ids_tb, last_bt, pool_bt, _ = load_states("base_test")
    assert (ids_t == ids_tb).all() and not set(meta.loc[ids_t].problem_id) & set(gv), "test rows must be aligned and problem-disjoint from val"
    dt, tt, Xmt = table(ids_t)
    test_sites = {"tuned": site_features(last_t, pool_t), "base": site_features(last_bt, pool_bt)}
    out = {"submission_id": ids_t.astype(str)}
    for tname, (kind, y) in tv.items():
        ok, okt = np.isfinite(y), np.isfinite(tt[tname][1]); keep = 50 if tname in PROBLEM_LEVEL else None
        P = lambda Xa, Xb, yy, al: np.where(okt, pl.Ridge(Xa[ok], keep).predict(yy, np.nan_to_num(Xb), al), np.nan)  # noqa: E731
        for model in ("tuned", "base"):
            c = CHOICES[tname][model]; Xa, Xb = sites[model][c["site"]], test_sites[model][c["site"]]
            out[f"{tname}|{model}"] = P(Xa, Xb, y[ok], c["alpha"])
            yc = pl.permute_across_groups(y[ok], gv[ok]) if tname in PROBLEM_LEVEL else pl.shuffle_within_groups(y[ok], gv[ok])
            out[f"{tname}|{model}_shuffled"] = P(Xa, Xb, yc, c["alpha"])
        mk = pl.Ridge(Xmv[ok]); out[f"{tname}|metadata"] = np.where(okt, mk.predict(y[ok], Xmt, 1.0), np.nan)
        bow_v, bow_t = pool_bv[:, 0], pool_bt[:, 0]
        a, _, _ = pl.cv_select(bow_v[ok], y[ok], gv[ok], kind, keep=keep)
        out[f"{tname}|bag_of_tokens"] = np.where(okt, pl.Ridge(bow_v[ok], keep).predict(y[ok], bow_t, a), np.nan); log("test", tname)
    kl.atomic_savez(f"{OUT}/test_preds.npz", **out)
    kl.atomic_json(f"{OUT}/test_run.json", {"choices": CHOICES})
kl.atomic_json(f"{OUT}/run_{PHASE}.json", {"phase": PHASE, "minutes": round((time.time() - T0) / 60, 1), "script_sha256": kl.sha(os.path.abspath(__file__)), "lib_sha256": kl.sha(os.path.join(os.path.dirname(pl.__file__), "probe_lib.py"))})
log("DONE")
