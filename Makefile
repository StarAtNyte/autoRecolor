.PHONY: install install-dev run clean lint

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

run:
	python -m autoRecolor

clean:
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
