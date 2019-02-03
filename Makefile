SHELL	:= /bin/bash
PYTHON	?= python
SETUP	:= setup.py

SUBDIRS := $(wildcard tvl_backends/*/.)

clean:
	$(PYTHON) $(SETUP) clean --all
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) clean --all && popd || break; \
	done

build:
	$(PYTHON) $(SETUP) build_ext --inplace
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) build_ext --inplace && popd || break; \
	done

dist: build
	$(PYTHON) $(SETUP) sdist bdist_wheel
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) sdist bdist_wheel && cp dist/* ../../dist/ && popd || break; \
	done

install-dev:
	pip install -e .
	for dir in $(SUBDIRS); do \
		pushd $$dir && pip install -e . && popd || break; \
	done

test:
	pytest tests
	for dir in $(SUBDIRS); do \
		pushd $$dir && pytest tests && popd || break; \
	done

.PHONY: clean build dist install-dev test
