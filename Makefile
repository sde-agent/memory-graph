# Memory Graph Engine — task runner.
# The engine is local-first; only `setup` is allowed network (to install packages).

PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: setup refresh init doctor test lint fix clean publish-check app _require-venv

# Fail with a clear message if the venv is missing, instead of a cryptic
# "no such file or directory: .venv/bin/python".
_require-venv:
	@test -x $(PY) || { \
	  echo "error: $(PY) not found. Run 'make setup' first to create the venv."; \
	  exit 1; \
	}

setup:  ## create the venv and install the package (the only networked target)
	python3 -m venv .venv
	$(PIP) install -e ".[dev]"

refresh:  ## reinstall the editable package after pulling repository updates
	$(PIP) install -e ".[dev]"

init: _require-venv  ## create the database and data directories
	$(PY) -m memory_graph.cli init

doctor: _require-venv  ## report config and environment health
	$(PY) -m memory_graph.cli doctor

test: _require-venv  ## run the test suite (sockets disabled)
	$(PY) -m pytest -q

lint: _require-venv  ## check style with ruff
	.venv/bin/ruff check .

fix: _require-venv  ## auto-fix style issues
	.venv/bin/ruff check --fix .

publish-check: _require-venv  ## fail-closed PII scan — run before any commit/push to a public repo
	$(PY) scripts/publish_check.py

app:  ## launch the local reflection app (Streamlit search bar; needs the 'app' extra)
	$(PIP) install -q streamlit
	.venv/bin/streamlit run app.py

clean:  ## remove caches (keeps data/ and the venv)
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
