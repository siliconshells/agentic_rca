.DEFAULT_GOAL := help
UV := uv run

.PHONY: help install lint fmt typecheck test check scenarios demo mcp api web up down clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install everything
	uv venv && uv pip install -e ".[dev,charts]"

lint: ## Ruff check
	$(UV) ruff check src mcp_server evals tests

fmt: ## Ruff format + autofix
	$(UV) ruff format src mcp_server evals tests && $(UV) ruff check --fix src mcp_server evals tests

typecheck: ## mypy strict
	$(UV) mypy src mcp_server

test: ## Hermetic test suite (no API key, no spend)
	$(UV) pytest

check: lint typecheck test ## Everything CI runs

scenarios: ## Generate the eval scenario set with ground truth
	$(UV) python -m mcp_server.env.generator --scenarios 60 --out evals/scenarios.jsonl

demo: ## One offline run against the mock model
	$(UV) aegis run --scenario evals/scenarios.jsonl:3 --mock

eval-mock: ## Run the eval pipeline offline (a pipeline check, not a measurement)
	$(UV) python evals/run_eval.py --scenarios evals/scenarios.jsonl --arms full,single_agent,no_verifier --mock

eval-live: ## Run a real eval (COSTS MONEY; --max-cost is the ceiling)
	$(UV) python evals/run_eval.py --scenarios evals/scenarios.jsonl --arms full,single_agent,no_verifier --limit 30 --max-cost 25

charts: ## Render the README charts from the latest eval summary
	$(UV) python evals/charts.py --summary evals/results/summary.json --out docs/charts

mcp: ## Run the MCP server over Streamable HTTP
	$(UV) python -m mcp_server.server --transport http --port 8765

api: ## Run the FastAPI app with reload
	$(UV) uvicorn aegis.api.app:app --reload --port 8000

web: ## Run the dashboard dev server
	cd web && npm run dev

up: ## docker compose up (api + mcp + jaeger)
	docker compose up --build -d

down: ## docker compose down
	docker compose down -v

clean: ## Remove caches and local state
	rm -rf .pytest_cache .mypy_cache .ruff_cache data evals/results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
