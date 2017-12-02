#!/bin/bash
set -e
set -v

# unit tests
nosetests

# linter
pylint simdkalman

# documentation
cd doc
make html
make doctest
