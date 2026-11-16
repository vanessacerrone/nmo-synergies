# Reproduce the paper from the repository root. PY is the Python with the dependencies of pyproject.toml.
PY ?= python3
RESULTS = vacuum deepcore analytic orca bands direction solar grid release measured joint
FIGURES = $(basename $(notdir $(wildcard figures/fig*.py)))

.PHONY: data results validate figures numbers test all

all: results figures numbers

data:
	$(PY) scripts/download_data.py

results:
	for r in $(RESULTS); do $(PY) -m compute.$$r || exit 1; done

validate:
	$(PY) -m compute.icecube_fit

figures:
	for f in $(FIGURES); do $(PY) -m figures.$$f || exit 1; done

numbers:
	$(PY) paper_numbers.py

test:
	$(PY) -m pytest -q
