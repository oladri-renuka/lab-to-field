.PHONY: setup data audit shortcut test smoke sweep compute generative report deploy app api docker
PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt

data:
	bash scripts/download_data.sh
	$(PY) -m src.data.prepare

audit:
	$(PY) -m src.data.audit

shortcut:
	for s in 0 1 2; do echo "===== shortcut seed $$s/2 ====="; $(PY) -m src.shortcut --config configs/audit/shortcut.yaml seed=$$s || exit 1; done

test:
	$(PY) -m pytest -q -m "not slow"

smoke:
	$(PY) -m pytest -q

generative:
	$(PY) -m src.train_cyclegan && $(PY) -m src.generate cyclegan
	$(PY) -m src.train_cvae && $(PY) -m src.generate cvae
	$(PY) -m src.fid

sweep:
	bash scripts/run_experiments.sh

compute:
	bash scripts/compute_experiments.sh

report:
	$(PY) -m src.report

deploy:
	$(PY) -m serve.export --config configs/deploy.yaml

app:
	$(PY) -m serve.app

api:
	uvicorn serve.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -f serve/Dockerfile -t crop-disease .
