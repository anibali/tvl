SHELL	:= /bin/bash
PYTHON	?= python
SETUP	:= setup.py

SUBDIRS := $(wildcard tvl_backends/tvl-backends-*/.)

clean:
	$(PYTHON) $(SETUP) clean --all
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) clean --all && popd || exit 1; \
	done

clean-dist: clean
	rm -rf dist
	for dir in $(SUBDIRS); do \
		pushd $$dir && rm -rf dist && popd || exit 1; \
	done

build:
	$(PYTHON) $(SETUP) build_ext --inplace
	for dir in $(SUBDIRS); do \
		pushd $$dir && $(PYTHON) $(SETUP) build_ext --inplace && popd || exit 1; \
	done

dist: build
	$(PYTHON) $(SETUP) sdist bdist_wheel
	for dir in $(SUBDIRS); do \
		pushd $$dir \
		&& $(PYTHON) $(SETUP) sdist \
		&& $(PYTHON) $(SETUP) bdist_wheel \
		&& cp dist/* ../../dist/ \
		&& popd \
		|| exit 1; \
	done

install-dev:
	pip install -e .
	for dir in $(SUBDIRS); do \
		pushd $$dir && pip install -e . && popd || exit 1; \
	done
	# TODO: Also need to set up symlinks for build outputs (eg. tvlnv.py, _tvlnv.*.so)

uninstall:
	pip uninstall tvl tvl-backends-fffr tvl-backends-nvdec tvl-backends-opencv tvl-backends-pyav

test:
	pytest -s tests
	for dir in $(SUBDIRS); do \
		if [ -d "$$dir/tests" ]; then \
			pushd $$dir && pytest -s tests && popd || exit 1; \
		fi \
	done
	pushd tvl_backends && pytest -s tests && popd || exit 1;

.PHONY: clean build dist install-dev uninstall test
