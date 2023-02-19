#!/bin/bash
set -e
set -v

# unit tests
python tests/testsuite.py

# linter
pylint simdkalman

# documentation
cd doc
make html
make doctest
