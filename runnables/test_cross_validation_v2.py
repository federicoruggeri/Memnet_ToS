"""

@Author:
@Date: 22/07/2019

[Repeated] Cross-validation routine.

** BEFORE RUNNING **

1. Check the following configuration files:
    distributed_config.json, callbacks.json, data_loader.json, model_config.json, training_config.json

"""

import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import tensorflow as tf

import const_define as cd
from custom_callbacks_v2 import EarlyStopping, TrainingLogger
from data_loader import DataLoaderFactory
from utility.cross_validation_utils import PrebuiltCV
from utility.distributed_test_utils import cross_validation
from utility.json_utils import save_json, load_json
from utility.log_utils import get_logger
from utility.python_utils import merge

logger = get_logger(__name__)

if __name__ == '__main__':

    # Ensure TF2
    assert tf.version.VERSION.startswith('2.')
    # tf.config.experimental_run_functions_eagerly(True)

    cv_test_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_CV_TEST_CONFIG_NAME))

    model_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))[
        cv_test_config['model_type']]

    training_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_TRAINING_CONFIG_NAME))

    # Loading data
    data_loader_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME))
    data_loader_type = data_loader_config['type']
    data_loader_info = data_loader_config['configs'][data_loader_type]
    loader_additional_info = {key: value['value'] for key, value in model_config.items()
                              if 'data_loader' in value['flags']}
    data_loader_info = merge(data_loader_info, loader_additional_info)

    data_loader = DataLoaderFactory.factory(data_loader_type)
    data_handle = data_loader.load(**data_loader_info)

    # Distributed info
    distributed_info = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DISTRIBUTED_CONFIG_NAME))

    # CV
    cv = PrebuiltCV(n_splits=10, shuffle=True, random_state=None, held_out_key=cv_test_config['cv_held_out_key'])
    folds_path = os.path.join(cd.PREBUILT_FOLDS_DIR, '{}.json'.format(cv_test_config['prebuilt_folds']))
    cv.load_folds(load_path=folds_path)

    if cv_test_config['pre_loaded_model'] is None:
        current_date = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
        save_base_path = os.path.join(cd.CV_DIR, cv_test_config['model_type'], current_date)

        if cv_test_config['save_model']:
            os.makedirs(save_base_path)
    else:
        save_base_path = os.path.join(cd.CV_DIR, cv_test_config['model_type'], cv_test_config['pre_loaded_model'])
        if not os.path.isdir(save_base_path):
            msg = "Can't find given pre-trained model. Got: {}".format(save_base_path)
            logger.exception(msg)
            raise RuntimeError(msg)

    # Callbacks
    # TODO: automatize callbacks creation
    callbacks_data = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_CALLBACKS_NAME))
    early_stopper = EarlyStopping(**callbacks_data['earlystopping'])
    training_logger = TrainingLogger(filepath=save_base_path, suffix=None, save_model=cv_test_config['save_model'])
    callbacks = [
        early_stopper,
    ]

    if cv_test_config['save_model']:
        callbacks.append(training_logger)

    scores = cross_validation(validation_percentage=cv_test_config['validation_percentage'],
                              data_handle=data_handle,
                              network_args=deepcopy(model_config),
                              data_loader_info=data_loader_info,
                              model_type=cv_test_config['model_type'],
                              error_metrics=cv_test_config['error_metrics'],
                              error_metrics_additional_info=cv_test_config['error_metrics_additional_info'],
                              error_metrics_nicknames=cv_test_config['error_metrics_nicknames'],
                              training_config=training_config,
                              repetitions=cv_test_config['repetitions'],
                              cv=cv,
                              save_model=cv_test_config['save_model'],
                              test_path=save_base_path,
                              callbacks=callbacks,
                              save_predictions=True,
                              use_tensorboard=cv_test_config['use_tensorboard'],
                              split_key=cv_test_config['split_key'],
                              distributed_info=distributed_info,
                              preloaded=cv_test_config['pre_loaded'],
                              preloaded_model=cv_test_config["pre_loaded_model"],
                              load_externally=cv_test_config['load_externally']
                              )
    # Validation
    if cv_test_config['repetitions'] > 1:
        logger.info('Average validation scores: {}'.format(
            {key: np.mean(item) for key, item in scores['validation_info'].items() if key.startswith('avg')}))
    else:
        logger.info('Average validation scores: {}'.format(
            {key: np.mean(item) for key, item in scores['validation_info'].items() if not key.startswith('avg')}))

    # Test
    if cv_test_config['repetitions'] > 1:
        logger.info('Average test scores: {}'.format(
            {key: np.mean(item) for key, item in scores['test_info'].items() if key.startswith('avg')}))
    else:
        logger.info('Average test scores: {}'.format(
            {key: np.mean(item) for key, item in scores['test_info'].items() if not key.startswith('avg')}))

    if cv_test_config['save_model']:
        save_json(os.path.join(save_base_path, cd.JSON_VALIDATION_INFO_NAME), scores['validation_info'])
        save_json(os.path.join(save_base_path, cd.JSON_TEST_INFO_NAME), scores['test_info'])
        save_json(os.path.join(save_base_path, cd.JSON_PREDICTIONS_NAME), scores['predictions'])
        save_json(os.path.join(save_base_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME), model_config)
        save_json(os.path.join(save_base_path, cd.JSON_TRAINING_CONFIG_NAME), training_config)
        save_json(os.path.join(save_base_path, cd.JSON_CV_TEST_CONFIG_NAME), data=cv_test_config)
        save_json(os.path.join(save_base_path, cd.JSON_CALLBACKS_NAME), data=callbacks_data)
        save_json(os.path.join(save_base_path, cd.JSON_DATA_LOADER_CONFIG_NAME), data=data_loader_config)
        save_json(os.path.join(save_base_path, cd.JSON_DISTRIBUTED_CONFIG_NAME), data=distributed_info)
