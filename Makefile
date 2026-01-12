.PHONY: install test train-all eval-all clean

install:
	pip install -r requirements.txt
	chmod +x scripts/run_experiments.sh

test:
	python -m pytest tests/ -v

train-all:
	bash scripts/run_experiments.sh

eval-all:
	python scripts/evaluate.py
	python scripts/eval_boundedness.py
	python scripts/eval_cluster.py
	python scripts/eval_counterfactual.py
	python scripts/compare_baselines.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

help:
	@echo "MoTS-RL Makefile Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make train-all  - Run all experiments"
	@echo "  make eval-all   - Run all evaluations"
	@echo "  make clean      - Clean generated files"
