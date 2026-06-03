.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

run: ## Start the app with uvicorn
	uv run --env-file .env uvicorn open_climate_service.main:app --reload --reload-include "*.yaml" --reload-include "*.yml" --port 8002