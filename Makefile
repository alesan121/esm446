.PHONY: help install demo test lint bench clean

help:
	@echo "install   install dependencies with Poetry"
	@echo "demo      generate a scene, run the node, score it against ground truth"
	@echo "test      run the test suite with coverage"
	@echo "lint      run black, ruff, pflake8 and bandit"
	@echo "bench     measure channeliser and full-node throughput"
	@echo "clean     remove generated output"

install:
	poetry install --with dev

# The point of this target: no SDR, no configuration, no arguments. Someone evaluating the
# repository can see the system work before deciding whether to read any of it.
demo:
	poetry run esm446-demo --quiet

test:
	poetry run pytest --cov=esm446 --cov-report=term-missing --cov-fail-under=80

lint:
	poetry run black --check .
	poetry run ruff check .
	poetry run pflake8 .
	poetry run bandit -c pyproject.toml -r .

bench:
	poetry run esm446-bench

clean:
	rm -rf out .pytest_cache .ruff_cache .coverage coverage.xml
