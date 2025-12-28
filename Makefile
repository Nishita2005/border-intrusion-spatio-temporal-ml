PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: install dev test format lint precommit-install bootstrap

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r dev-requirements.txt

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m flake8

precommit-install:
	$(PYTHON) -m pre_commit install

bootstrap: install dev precommit-install
