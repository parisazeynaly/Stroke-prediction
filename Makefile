.PHONY: help setup train eval reproduce run-api

PYTHON := python
PIP := pip

help:
	@echo "Available commands:"
	@echo "  make setup      Install dependencies"
	@echo "  make train      Train model"
	@echo "  make eval       Evaluate model and save metrics/plots"
	@echo "  make reproduce  Train + evaluate (full pipeline)"
	@echo "  make run-api    Run Flask API"

setup:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m stroke_prediction.train

eval:
	$(PYTHON) -m stroke_prediction.evaluate

reproduce: train eval

run-api:
	$(PYTHON) app/app.py

