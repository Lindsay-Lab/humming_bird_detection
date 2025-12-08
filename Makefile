#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = humming-bird-detection
VERSION = 0.0.1
PYTHON_INTERPRETER = python

SHELL := /bin/bash
ENV_NAME := hb-yolo
ENV_FILE := environment.yml
# PHONY TARGETS                                                                 #
#################################################################################

.PHONY: clean test env install check all conda-clean help conda-setup

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Run tests
test:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	  conda activate $(ENV_NAME) && \
	  python -m pytest tests

## Optional: global conda cleanup (cache)
conda-clean:
	conda clean --all --yes

## One-time recommended conda config (solver + channels)
conda-setup:
	conda config --set solver libmamba
	conda config --set channel_priority strict

## Create or update the conda environment
env:
	conda env create -f $(ENV_FILE) || conda env update -f $(ENV_FILE) --prune

## Run your pip installs (ultralytics --no-deps, etc.) inside the env
install:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	  conda activate $(ENV_NAME) && \
	  ./install_packages.sh

## Run environment checks
check:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	  conda activate $(ENV_NAME) && \
	  python check_env.py

## Full setup: create/update env + install pip deps + run checks
all: env install check

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := all

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
