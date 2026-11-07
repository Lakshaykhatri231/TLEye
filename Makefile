.PHONY: setup test lint eval data sandbox-test sandbox-images headline serve image data-manifest verify-data weights space

PY=.venv/bin/python -I -m

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

sandbox-images:  ## the images the certificate runner uses (tleye/certify/runner.py: IMAGES)
	docker build -t tleye-python:3.11 sandbox/python
	docker pull gcc:13
	docker pull eclipse-temurin:21-jdk
	docker pull pypy:3.10-slim

sandbox-test:  ## needs Docker/Colima and `make sandbox-images`
	.venv/bin/python -m pytest -q tests/test_runner.py

headline:
	$(PY) tleye.eval.headline

# Regenerate every report and the README tables. Needs `make data` outputs (data/processed/) and the committed Kaggle outputs in reports/kaggle_*.
# No GPU: the Kaggle runs are NOT repeated here; their stored predictions are re-scored.
eval:
	$(PY) tleye.models.baselines
	$(PY) tleye.eval.metamorphic
	$(PY) tleye.certify.coverage
	$(PY) tleye.certify.pilot_report
	$(PY) tleye.certify.flagged_report
	$(PY) tleye.constraints.eval_parser
	$(PY) tleye.eval.sweep_report
	$(PY) tleye.eval.confirm_report
	$(PY) tleye.eval.seeds_report
	$(PY) tleye.eval.lolo_report
	$(PY) tleye.eval.b3_report
	$(PY) tleye.eval.metamorphic_lora
	$(PY) tleye.eval.probe_report
	$(PY) tleye.eval.sft_report
	$(PY) tleye.eval.sft_final_report
	$(PY) tleye.eval.risk_coverage
	$(PY) tleye.eval.model_card
	$(PY) tleye.eval.headline

# Data pipeline (downloads the 7.8 GB CodeNet archive once; see README). Raw data is UNTRUSTED text and is never executed on the host.
data:
	mkdir -p data/raw && aria2c -x16 -s16 -k 8M -c -d data/raw -o Project_CodeNet.tar.gz "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz"
	$(PY) tleye.data.stream_extract --tar data/raw/Project_CodeNet.tar.gz --out data/raw/extracted --meta-only
	$(PY) tleye.data.spike && $(PY) tleye.data.spike_report
	$(PY) tleye.data.select --cap 40
	$(PY) tleye.data.stream_extract --tar data/raw/Project_CodeNet.tar.gz --out data/raw/extracted --selected data/processed/selected.parquet
	$(PY) tleye.data.splits && $(PY) tleye.features.build && $(PY) tleye.data.clean && $(PY) tleye.data.kaggle_pack

data-manifest:  ## record the exact data version (checksums) the results come from -> ci/data_manifest.json
	$(PY) tleye.data.manifest

verify-data:    ## check a rebuilt dataset against the committed manifest
	$(PY) tleye.data.manifest --check

serve:          ## HTTP service on http://127.0.0.1:8000 (docs at /docs)
	.venv/bin/uvicorn tleye.serve:app --host 127.0.0.1 --port 8000

image:          ## container with the CPU LightGBM model; add --build-arg WITH_LORA=1 for the LoRA model
	docker build -t tleye .

weights:        ## verdict model adapter from Hugging Face at the pinned revision, checksums verified -> models/lora/
	$(PY) tleye.models.fetch

space:          ## assemble the Hugging Face Space (needs a PRO account to host Gradio) -> data/scratch/space
	rm -rf data/scratch/space && mkdir -p data/scratch/space/kaggle/lib data/scratch/space/models
	cp space/* LICENSE data/scratch/space/ && cp kaggle/lib/kernel_lib.py data/scratch/space/kaggle/lib/
	rsync -a --exclude __pycache__ tleye data/scratch/space/ && rsync -a models/b2 data/scratch/space/models/
