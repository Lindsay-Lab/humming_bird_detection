#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = humming_bird_detection
PROJECT_ABRV = humming_bird_detection
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
USERNAME = rg5022
PROXY_HOST = 
CLUSTER_HOST =


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean_py
clean_py:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all snakemake logs
.PHONY: clean_snakemake
clean_snakemake:
	find . -type d -name ".snakemake" -exec rm -rv {} +

## Delete all slurm logs
.PHONY: clean_slurm
clean_slurm:
		find . -type f -name "*.err" -exec rm -rv {} +
		find . -type f -name "*.out" -exec rm -rv {} +


## Sync code to cluster
.PHONY: sync_code
sync_code:
	rsync -va \
			  --exclude-from='.gitignore' \
			  --prune-empty-dirs --copy-links \
			  ./ \
			  ${CLUSTER_HOST_PROFILE}:~/${PROJECT_NAME}/

## Sync data to local
.PHONY: sync_data
sync_data:
	rsync \
		-va  \
		--include='reports/***' \
		--include='models/***' \
		--exclude='*' --prune-empty-dirs --copy-links \
		'${USERNAME}@${CLUSTER_HOST}:~/${PROJECT_NAME}/' \
		./


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 humming_bird_detection
	black --check --config pyproject.toml humming_bird_detection


## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml humming_bird_detection




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) humming_bird_detection/data/make_dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)