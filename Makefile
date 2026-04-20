# amphiphile — developer commands.
#
#   make check    — lint + type-check + fast tests (run before any commit)
#   make test     — full pytest suite including slow markers
#   make train    — train baseline model end-to-end
#   make serve    — run the FastAPI service locally (requires trained artifacts)
#   make docker   — build the production Docker image
#   make clean    — remove caches, not data or models

.DEFAULT_GOAL := help

.PHONY: help check lint typecheck test test-fast test-slow train serve docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "%-12s %s\n", $$1, $$2}'

check: lint typecheck test-fast ## lint + types + fast tests

lint: ## ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck: ## mypy strict
	uv run mypy src

test: test-fast test-slow ## full test suite

test-fast: ## fast tests only (skip Chemprop training)
	uv run pytest -m 'not slow'

test-slow: ## slow tests (Chemprop)
	uv run pytest -m 'slow'

train: ## train baseline ensemble end-to-end
	uv run logd train --model baseline

train-chemprop: ## train Chemprop D-MPNN ensemble
	uv run logd train --model chemprop

serve: ## run the FastAPI service on localhost:8000
	uv run uvicorn logd.api:app --host 0.0.0.0 --port 8000

docker: ## build production image
	docker build -t amphiphile .

docker-run: docker ## build + run locally with mounted models
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models amphiphile

clean: ## remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
