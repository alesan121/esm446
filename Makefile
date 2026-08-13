.PHONY: help install demo report test lint bench clean

help:
	@echo "install   install dependencies with Poetry"
	@echo "demo      generate a scene, run the node, score it against ground truth"
	@echo "report    demo plus the waterfall, the dashboard and the V&V figures"
	@echo "test      run the test suite with coverage"
	@echo "lint      run black, ruff, pflake8 and bandit"
	@echo "bench     measure channeliser and full-node throughput"
	@echo "clean     remove generated output"

install:
	poetry install --with dev

# The point of this target: no SDR, no configuration, no arguments. Someone evaluating the
# repository can see the system work before deciding whether to read any of it.
demo:
	poetry run esm446-demo --quiet --out out

# Everything a reader might want to look at, regenerated from the system rather than
# committed and left to rot: the band picture, the order-of-battle dashboard, and every
# figure in the V&V report.
#
# Note that this rewrites the committed figures under docs/figures, so `git status` will show
# them modified afterwards. That is the point -- it is how the report is refreshed -- but the
# throughput figures move a little between runs, so commit them only when refreshing on
# purpose.
report:
	poetry run esm446-demo --quiet --out out
	poetry run esm446-vv --output docs/figures
	@echo ""
	@echo "open out/dashboard.html and out/waterfall.png"

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
