PYTHON ?= python
UV ?= uv

.PHONY: help install wheel dist clean clean-docs test coverage docs format lint

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  install      install PennyLane-IonQ in the active environment"
	@echo "  wheel        build a wheel under dist/"
	@echo "  dist         build wheel + sdist under dist/"
	@echo "  clean        remove build artifacts and caches"
	@echo "  clean-docs   remove built documentation"
	@echo "  test         run the test suite"
	@echo "  coverage     run the test suite with a coverage report"
	@echo "  docs         build the HTML documentation"
	@echo "  format       run ruff format"
	@echo "  lint         run ruff check"

install:
	$(UV) pip install -e .

wheel:
	$(UV) build --wheel

dist:
	$(UV) build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .coverage coverage_html_report htmlcov coverage.xml
	find pennylane_ionq tests -type d -name __pycache__ -exec rm -rf {} +

clean-docs:
	$(MAKE) -C doc clean

test:
	$(PYTHON) -m pytest tests --tb=short

coverage:
	$(PYTHON) -m pytest tests --cov=pennylane_ionq --cov-report=term-missing --cov-report=xml --cov-report=html

docs:
	$(MAKE) -C doc html

format:
	$(UV) run ruff format pennylane_ionq tests

lint:
	$(UV) run ruff check pennylane_ionq tests
	$(UV) run ruff format --check pennylane_ionq tests
