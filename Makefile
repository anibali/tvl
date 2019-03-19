SHELL	:= /bin/bash
PYTHON	?= python
SETUP	:= setup.py

SUBDIRS := $(wildcard tvl_backends/*/.)

clean:
	$(PYTHON) $(SETUP) clean --all
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) clean --all && popd || break; \
	done

clean-dist: clean
	rm -rf dist
	for dir in $(SUBDIRS); do \
		pushd $$dir && rm -rf dist && popd || break; \
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

uninstall:
	pip uninstall tvl tvl-backends-nvdec tvl-backends-opencv tvl-backends-pyav

test: build
	pytest -s tests
	for dir in $(SUBDIRS); do \
		pushd $$dir && pytest -s tests && popd || break; \
	done

.PHONY: clean build dist install-dev uninstall test
