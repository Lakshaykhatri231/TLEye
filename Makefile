.PHONY: setup test lint eval data sandbox-test headline

PY=.venv/bin/python -I -m

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

sandbox-test:  ## needs Docker/Colima + images (gcc:13, eclipse-temurin:21-jdk, python:3.11-slim, pypy:3.10-slim)
	.venv/bin/python -m pytest -q tests/test_runner.py

headline:
	$(PY) tleye.eval.headline

# Reproduce every reported result table from the processed data (needs `make data` outputs and, for the LoRA row, reports/kaggle_train/).
eval:
	$(PY) tleye.models.baselines
	$(PY) tleye.eval.metamorphic
	$(PY) tleye.certify.coverage
	$(PY) tleye.certify.pilot_report
	$(PY) tleye.constraints.eval_parser
	$(PY) tleye.eval.headline

# Data pipeline (downloads the 7.8 GB CodeNet archive once; see README). Raw data is UNTRUSTED text and is never executed on the host.
data:
	mkdir -p data/raw && aria2c -x16 -s16 -k 8M -c -d data/raw -o Project_CodeNet.tar.gz "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz"
	$(PY) tleye.data.stream_extract --tar data/raw/Project_CodeNet.tar.gz --out data/raw/extracted --meta-only
	$(PY) tleye.data.spike && $(PY) tleye.data.spike_report
	$(PY) tleye.data.select --cap 40
	$(PY) tleye.data.stream_extract --tar data/raw/Project_CodeNet.tar.gz --out data/raw/extracted --selected data/processed/selected.parquet
	$(PY) tleye.data.splits && $(PY) tleye.features.build && $(PY) tleye.data.clean && $(PY) tleye.data.kaggle_pack
