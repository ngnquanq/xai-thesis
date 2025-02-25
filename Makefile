.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = xai-thesis
PYTHON_INTERPRETER = python
KAGGLE = kaggle

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: 
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt; 


## Make Dataset
data: 
	mkdir -p data/raw data/processed data/external data/internal

	# Download telecom_churn.csv from Kaggle
	if [ ! -f data/raw/telco_churn.csv ]; then $(KAGGLE) datasets download kashnitsky/mlcourse -f telecom_churn.csv -p data/raw ; fi

	# Download cell2celltrain.csv from Kaggle
	if [ ! -f data/raw/cell2celltrain.csv ]; then $(KAGGLE) datasets download jpacse/datasets-for-churn-telecom -f cell2celltrain.csv -p data/raw; fi

	# Download telco_customer_churn.csv from Kaggle
	if [ ! -f data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv ]; then $(KAGGLE) datasets download nithinreddy90/wa-fn-usec-telco-customer-churn -f WA_Fn-UseC_-Telco-Customer-Churn.csv -p data/raw; fi

	# Unzip files
	if [ -f data/raw/telecom_churn.csv.zip ]; then unzip data/raw/telecom_churn.csv.zip -d data/raw && rm data/raw/telecom_churn.csv.zip; fi
	if [ -f data/raw/cell2celltrain.csv.zip ]; then unzip data/raw/cell2celltrain.csv.zip -d data/raw && rm data/raw/cell2celltrain.csv.zip; fi
	if [ -f data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.zip ]; then unzip data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.zip -d data/raw && mv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_customer_churn.csv && rm data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.zip; fi


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src



## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.9 -y
else
	conda create --name $(PROJECT_NAME) python=3.9 -y
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
