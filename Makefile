.PHONY: setup test lint format run clean help

help:
	@echo "Available targets:"
	@echo "  make setup     - Install dependencies via uv"
	@echo "  make test      - Run tests (if available)"
	@echo "  make lint      - Format and lint code with ruff"
	@echo "  make format    - Format code with ruff"
	@echo "  make run       - Run the training notebook"
	@echo "  make clean     - Remove cache and build artifacts"

setup:
	uv sync

test:
	@echo "No tests yet"

lint:
	uv run ruff check src/ 2>/dev/null || echo "ruff not in dependencies; skipping"

format:
	uv run ruff format src/ 2>/dev/null || echo "ruff not in dependencies; skipping"

run:
	uv run jupyter nbconvert --to notebook --execute notebooks/01-training.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
