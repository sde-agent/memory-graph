#!/bin/zsh
set -eu

failed() {
  print -u2 "BUILD FAILED"
}
trap failed ERR

.venv/bin/python -m pytest -q
.venv/bin/ruff check .

trap - ERR
print "BUILD SUCCEEDED"
