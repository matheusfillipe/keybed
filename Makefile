BUN := bun --cwd=web
UV := uv run --project tools

.DEFAULT_GOAL := help
.PHONY: help install fix precommit check \
        tools-fix tools-format-check tools-lint tools-typecheck tools-test tools-coverage \
        tools-dead-code tools-unused-deps tools-security tools-audit build \
        web-typecheck web-lint web-fix web-test web-build dev clean

help: ## list available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## install dependencies (bun, uv) and git hooks
	$(BUN) install
	uv sync --project tools
	cd .opencode && bun install
	$(UV) pre-commit install

fix: tools-fix web-fix ## autofix formatting and lint issues (ruff, biome)

precommit: fix ## hook entry: same as fix

# --- checks (verify, never produce artifacts) ---

check: tools-format-check tools-lint tools-typecheck tools-test web-typecheck web-lint web-test web-build ## run all checks (the pre-commit gate)

quality: check tools-dead-code tools-unused-deps tools-security tools-audit tools-coverage build ## run the full quality gate
	@echo "quality gate passed"

tools-fix: ## autofix python formatting and lint (ruff)
	cd tools && uv run ruff format src tests && uv run ruff check --fix src tests

tools-format-check: ## check python formatting (ruff format)
	cd tools && uv run ruff format --check src tests

tools-lint: ## lint python (ruff)
	cd tools && uv run ruff check src tests

tools-typecheck: ## typecheck python (mypy strict)
	cd tools && uv run mypy

tools-test: ## run python tests (pytest)
	cd tools && uv run pytest

tools-coverage: ## run python tests with coverage (pytest-cov)
	cd tools && uv run pytest --cov --cov-report=term-missing

tools-dead-code: ## detect dead python code (vulture)
	cd tools && uv run vulture src/kvt tests

tools-unused-deps: ## detect unused python dependencies (deptry)
	cd tools && uv run deptry .

tools-security: ## scan python for security issues (bandit)
	cd tools && uv run bandit -c pyproject.toml -r src/kvt

tools-audit: ## audit python dependencies for vulnerabilities (pip-audit)
	cd tools && uv run --with pip pip-audit

build: ## build the python package (uv build)
	uv build --project tools

web-typecheck: ## typecheck web (tsc)
	$(BUN) run typecheck

web-lint: ## lint web (biome check)
	$(BUN) run lint

web-fix: ## autofix web formatting and lint (biome)
	$(BUN) run fix

web-test: ## run web tests (vitest)
	$(BUN) run test

web-build: ## bundle the web app (vite build)
	$(BUN) run build

dev: ## run the web dev server (vite)
	$(BUN) run dev

clean: ## remove local caches and build artifacts
	rm -rf tools/.ruff_cache tools/.mypy_cache tools/.pytest_cache tools/.coverage tools/.coverage.* tools/.vulture web/dist
