.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 

################################################################################
# GLOBALS                                                                      #
################################################################################
PROJECT_NAME = eskd_model_update
PYTHON_VERSION = 3.12.0
PYTHON_INTERPRETER = python
VENV_DIR = venv_equi312
CONDA_ENV_NAME = conda_equi_312
PROJECT_DIRECTORY = aki_model


PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
MLFLOW_TRACKING_URI ?= sqlite:///$(PROJECT_DIR)/mlflow.db
export MLFLOW_TRACKING_URI
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = equiboots
PYTHON_INTERPRETER = python3


ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif


### general usage notes
### 2>&1 | tee ==>pipe operation to save model output from terminal to .txt file

############################## Training Globals ################################

# Define variables for looping
RAW_FILE = ELAIA-1_deidentified_data_10-6-2020.csv
RAW_DATA = data/raw/$(RAW_FILE)
OUTCOMES = Label_ESKD_2_years
# PIPELINES = orig under over orig_rfe under_rfe over_rfe
PIPELINES = orig over orig_rfe over_rfe
SCORING = average_precision
PRETRAINED ?= 0  # 0 if you want to train the models, 1 if calibrate pretrained

#################################################################################
# COMMANDS                                                                      #
#################################################################################

################################################################################
############## Setting up a Virtual Environment and Dependencies ###############
################################################################################
# virtual environment set-up (local)
venv:
	$(PYTHON_INTERPRETER) -m venv equi_venv
	source equi_venv/bin/activate

## Install Python Dependencies
requirements_local:	
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements_prod.txt	

venv_dep_setup_local: venv requirements_local	# for local set-up
venv_dep_setup_gpu: venv requirements_gpu     # for server/gpu set-up

################################################################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf public_data/

## Create folder/file paths
create_folders:
	mkdir -p public_data

## Prediction Generation
pred_generation:
	$(PYTHON_INTERPRETER) -m py_scripts.adult_income_xgbearly 


#################################################################################
# Instantiate MLFlow                                                            #
#################################################################################

.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5501


################################################################################
.PHONY: data_prep_preprocessing_training
data_prep_preprocessing_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
	--input-data-file $(RAW_DATA) \
	--output-data-file data/processed/df_sans_zero.parquet \
	--stage training \
	$(TRACK_FLAG)

################################################################################
############################ Feature generation ################################
################################################################################
 
.PHONY: feat_gen_training
feat_gen_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
	--input-data-file data/processed/df_sans_zero.parquet \
	--stage training \
	$(TREATMENT_FLAG) \
	$(TRACK_FLAG)

preproc_pipeline: data_prep_preprocessing_training feat_gen_training