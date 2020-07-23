"""

@Author:

@Date: 29/11/2018

Constant python configuration script.

"""

import os

NAME_LOG = 'daily_log.log'

# Algorithms

SUPPORTED_LEARNERS = ['sentence_wise', 'context_wise']
SUPPORTED_ALGORITHMS = {
    'experimental_basic_memn2n_v2': {
        'save_suffix': 'experimental_basic_memn2n_v2',
        'model_type': 'context_wise'
    },
    'experimental_baseline_lstm_v2': {
        'save_suffix': 'experimental_baseline_lstm_v2',
        'model_type': 'sentence_wise'
    },
    'experimental_baseline_cnn_v2': {
        'save_suffix': 'experimental_baseline_cnn_v2',
        'model_type': 'sentence_wise'
    },
    'experimental_baseline_sum_v2': {
        'save_suffix': 'experimental_baseline_sum_v2',
        'model_type': 'sentence_wise'
    },
}

MODEL_CONFIG = {
    'experimental_basic_memn2n_v2': {
        'processor': 'text_kb_supervision_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'kb_supervision_converter'
    },
    'experimental_baseline_lstm_v2': {
        'processor': 'ttext_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'base_converter'
    },
    'experimental_baseline_cnn_v2': {
        'processor': 'text_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'base_converter'
    },
    'experimental_baseline_sum_v2': {
        'processor': 'text_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'base_converter'
    },
}

ALGORITHM_CONFIG = {
    'sentence_wise': {
        'processor': 'text_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'base_converter'
    },
    'context_wise': {
        'processor': 'text_processor',
        'tokenizer': 'keras_tokenizer',
        'converter': 'base_converter'
    }
}

# DEFAULT DIRECTORIES
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')
PATH_LOG = os.path.join(PROJECT_DIR, 'log')
RUNNABLES_DIR = os.path.join(PROJECT_DIR, 'runnables')
LOCAL_DATASETS_DIR = os.path.join(PROJECT_DIR, 'local_database')
CV_DIR = os.path.join(PROJECT_DIR, 'cv_test')
EMBEDDING_MODELS_DIR = os.path.join(PROJECT_DIR, 'embedding_models')
TOS_100_DIR = os.path.join(LOCAL_DATASETS_DIR, 'ToS_100')
KB_DIR = os.path.join(LOCAL_DATASETS_DIR, 'KB')
PREBUILT_FOLDS_DIR = os.path.join(PROJECT_DIR, 'prebuilt_folds')
TESTS_DATA_DIR = os.path.join(PROJECT_DIR, 'tests_data')

DATASET_PATHS = {
    'tos_100': TOS_100_DIR
}

# JSON FILES
JSON_CALLBACKS_NAME = 'callbacks.json'
JSON_DISTRIBUTED_MODEL_CONFIG_NAME = 'distributed_model_config.json'
JSON_TRAINING_CONFIG_NAME = 'training_config.json'
JSON_CV_TEST_CONFIG_NAME = 'cv_test_config.json'
JSON_DATA_LOADER_CONFIG_NAME = 'data_loader.json'
JSON_VALIDATION_INFO_NAME = 'validation_info.json'
JSON_TEST_INFO_NAME = 'test_info.json'
JSON_PREDICTIONS_NAME = 'predictions.json'
JSON_DISTRIBUTED_CONFIG_NAME = 'distributed_config.json'
JSON_MODEL_DATA_CONFIGS_NAME = 'data_configs.json'
JSON_QUICK_SETUP_CONFIG_NAME = "quick_setup_config.json"
