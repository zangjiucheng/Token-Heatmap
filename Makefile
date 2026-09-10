# llm-token-heatmap — common tasks.
#
#   make            list the targets
#   make install    set up everything (Python venv + frontend deps)
#
# Every target works from a clean checkout. Python commands prefer the project
# venv at .venv when it exists, so you do not have to `source` anything first;
# they fall back to whatever is on PATH otherwise (e.g. an active conda env).
# Point them somewhere else with PY=, which is how you drive an environment that
# is not .venv — the HPC box, for instance:
#
#   make test-py PY=/work/j7zang/th-gpu/bin/python

VENV := .venv
PY   := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
TH   := $(if $(wildcard $(VENV)/bin/token-heatmap),$(VENV)/bin/token-heatmap,token-heatmap)
NPM  := npm --prefix app

# Which experiment `make trace` / `make hpc` run. Override on the command line:
#   make trace CONFIG=configs/nqueens-verify.yaml
CONFIG ?= configs/example.yaml
GPU    ?= l40s

.DEFAULT_GOAL := help

.PHONY: help install install-py install-app test test-py test-app e2e lint \
        format typecheck build app dev dev-app docs docs-serve trace hpc \
        clean clean-all

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  CONFIG=<path> selects the experiment for trace/hpc (default: $(CONFIG))"

# ---------------------------------------------------------------- install ---

install: ## Full dev setup: Python venv + package + frontend deps
	@./scripts/setup.sh

install-py: ## Python package only, editable, with dev extras
	$(PY) -m pip install -e ".[dev,models]"

install-app: ## Frontend dependencies only
	$(NPM) install

# ------------------------------------------------------------------ check ---

test: test-py test-app ## Run both test suites

test-py: ## Python tests
	$(PY) -m pytest tests/ -q

test-app: ## Frontend unit tests
	$(NPM) run test -- --run

e2e: ## Frontend end-to-end tests (Playwright)
	$(NPM) run e2e

typecheck: ## TypeScript type check
	$(NPM) run typecheck

lint: ## Lint Python (ruff) and the frontend (eslint + tsc)
	@$(PY) -m ruff --version >/dev/null 2>&1 || { \
		echo "ruff is missing from $(PY) — it lives in the dev extra."; \
		echo "Install it with:  make install-py PY=$(PY)"; exit 1; }
	$(PY) -m ruff check llm_token_heatmap tests
	$(NPM) run lint
	$(NPM) run typecheck

format: ## Auto-format Python (ruff) and the frontend (prettier)
	@$(PY) -m ruff --version >/dev/null 2>&1 || { \
		echo "ruff is missing from $(PY) — it lives in the dev extra."; \
		echo "Install it with:  make install-py PY=$(PY)"; exit 1; }
	$(PY) -m ruff format llm_token_heatmap tests
	$(PY) -m ruff check --fix llm_token_heatmap tests
	$(NPM) run format

# ------------------------------------------------------------------ build ---

build: ## Build the frontend bundle (app/dist)
	$(NPM) run build

app: ## Build the Tauri desktop installer for this OS
	$(NPM) run build:app

dev: ## Frontend dev server (http://localhost:5173)
	$(NPM) run dev

dev-app: ## Tauri desktop dev window
	$(NPM) run dev:app

docs: ## Build the documentation site
	$(PY) -m mkdocs build

docs-serve: ## Serve the docs with live reload
	$(PY) -m mkdocs serve

# ------------------------------------------------------------------- runs ---

trace: ## Run an experiment locally (CONFIG=configs/<name>.yaml)
	$(TH) trace --config $(CONFIG)

hpc: ## Run an experiment on the HPC and rsync the results back
	$(TH) hpc run $(CONFIG) --gpu $(GPU)

# ------------------------------------------------------------------ clean ---

clean: ## Remove build output and caches (keeps .venv and outputs/)
	rm -rf app/dist dist site build *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './app/node_modules/*' -exec rm -rf {} +
	find . -name '.DS_Store' -not -path './app/node_modules/*' -delete

clean-all: clean ## Also remove .venv and node_modules (full reset)
	rm -rf $(VENV) app/node_modules
