import pandas as pd
import pytest

from tleye.data.splits import make_split


def test_each_problem_in_exactly_one_split():
    ids = [f"p{i:05d}" for i in range(1000)] * 3  # duplicates (many submissions per problem)
    sp = make_split(ids, seed=0)
    assert sp["problem_id"].is_unique
    assert set(sp["split"]) == {"train", "val", "test"}
    assert len(sp) == 1000


def test_split_is_deterministic_and_seed_sensitive():
    ids = [f"p{i:05d}" for i in range(500)]
    a, b, c = make_split(ids, 0), make_split(ids, 0), make_split(ids, 1)
    assert a.equals(b)
    # compare the problem -> split MAPPING (the split column alone is always train,val,test in order)
    map_a, map_c = a.set_index("problem_id")["split"], c.set_index("problem_id")["split"]
    assert not map_a.equals(map_c.loc[map_a.index])


def test_fractions_close_to_70_10_20():
    sp = make_split([f"p{i}" for i in range(1000)], 0)
    frac = sp["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.7, abs=0.01)
    assert frac["test"] == pytest.approx(0.2, abs=0.01)


def test_real_data_no_problem_overlap_and_all_languages_covered():
    sel_p, sp_p = "data/processed/selected.parquet", "data/processed/splits.parquet"
    try:
        sel, sp = pd.read_parquet(sel_p), pd.read_parquet(sp_p)
    except FileNotFoundError:
        pytest.skip("processed data not built")
    m = sel.merge(sp, on="problem_id")
    per_split_problems = m.groupby("split")["problem_id"].apply(set)
    assert not (per_split_problems["train"] & per_split_problems["test"])
    assert not (per_split_problems["train"] & per_split_problems["val"])
    assert not (per_split_problems["val"] & per_split_problems["test"])
    counts = m.groupby(["split", "language", "y_tle"]).size()
    for split in ["train", "val", "test"]:
        for lang in ["C++", "Java", "Python"]:
            for y in [0, 1]:
                assert counts.get((split, lang, y), 0) >= 100, (split, lang, y)
