"""

Re-runs pre-trained model on the same folds to gather attention weights via callback

@Author:

@Date: 11/09/2019

** BEFORE RUNNING **

1. Change the test unique identifier 'test_name' variable.
2. Change the 'model_type' variable as to match your tested model.
3. [Optional] Set 'repetition_ids' variable to a list of indexes in order to consider specific repetitions.
   Setting it to None allows best repetition retrieval based on 'retrieval_metric' variable (validation set).
4. Set best-K configuration parameters as follows:
    A. Retrieve top-K models attention: current_K = None, top_K = K
    B. Retrieve specific model attention: current_K = J

"""

import os

import numpy as np
import tensorflow as tf

import const_define as cd
from custom_callbacks_v2 import AttentionRetriever
from data_loader import DataLoaderFactory
from utility.distributed_test_utils import cross_validation_forward_pass
from utility.cross_validation_utils import PrebuiltCV
from utility.json_utils import save_json, load_json
from utility.log_utils import get_logger
from utility.python_utils import merge
from copy import deepcopy

logger = get_logger(__name__)

if __name__ == '__main__':

    # Ensure TF2
    assert tf.version.VERSION.startswith('2.')
    # tf.config.experimental_run_functions_eagerly(True)

    # Inputs

    model_type = "experimental_basic_memn2n_v2"
    test_name = ""
    save_scores = False
    # Retrieves the best model weights for each fold.
    repetition_ids = None
    retrieval_metric = 'f1_score'
    top_K = 3
    current_K = None
    #

    model_path = os.path.join(cd.CV_DIR, model_type, test_name)

    if not os.path.isdir(model_path):
        raise RuntimeError('Could not find test folder. Got: {}'.format(model_path))

    cv_test_config = load_json(os.path.join(model_path, cd.JSON_CV_TEST_CONFIG_NAME))

    model_config = load_json(os.path.join(model_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))

    training_config = load_json(os.path.join(model_path, cd.JSON_TRAINING_CONFIG_NAME))

    # Loading data
    data_loader_config = load_json(os.path.join(model_path, cd.JSON_DATA_LOADER_CONFIG_NAME))
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

    if current_K is None and top_K >= 1:
        save_scores = False

        for current_K in range(top_K):
            # Callbacks
            # TODO: automatize callbacks creation
            callbacks_data = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_CALLBACKS_NAME))
            attention_retriever = AttentionRetriever(model_path, save_suffix='top{}'.format(current_K + 1))
            callbacks = [
                attention_retriever
            ]

            scores = cross_validation_forward_pass(validation_percentage=cv_test_config['validation_percentage'],
                                                   data_handle=data_handle,
                                                   network_args=deepcopy(model_config),
                                                   data_loader_info=data_loader_info,
                                                   model_type=cv_test_config['model_type'],
                                                   error_metrics=cv_test_config['error_metrics'],
                                                   error_metrics_additional_info=cv_test_config[
                                                       'error_metrics_additional_info'],
                                                   error_metrics_nicknames=cv_test_config['error_metrics_nicknames'],
                                                   training_config=training_config,
                                                   cv=cv,
                                                   test_path=model_path,
                                                   callbacks=callbacks,
                                                   save_predictions=True,
                                                   repetition_ids=repetition_ids,
                                                   retrieval_metric=retrieval_metric,
                                                   split_key=cv_test_config['split_key'],
                                                   distributed_info=distributed_info,
                                                   top_K=top_K,
                                                   current_K=current_K)

            logger.info('Average validation scores: {}'.format(
                {key: np.mean(item) for key, item in scores['validation_info'].items() if
                 not key.startswith('avg')}))

            logger.info('Average test scores: {}'.format(
                {key: np.mean(item) for key, item in scores['test_info'].items() if not key.startswith('avg')}))

    else:

        assert current_K is not None
        assert type(current_K) == int

        # Callbacks
        # TODO: automatize callbacks creation
        callbacks_data = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_CALLBACKS_NAME))
        attention_retriever = AttentionRetriever(model_path, save_suffix='top{}'.format(current_K + 1))
        callbacks = [
            attention_retriever
        ]

        scores = cross_validation_forward_pass(validation_percentage=cv_test_config['validation_percentage'],
                                               data_handle=data_handle,
                                               network_args=model_config,
                                               data_loader_info=data_loader_info,
                                               model_type=cv_test_config['model_type'],
                                               error_metrics=cv_test_config['error_metrics'],
                                               error_metrics_additional_info=cv_test_config[
                                                   'error_metrics_additional_info'],
                                               error_metrics_nicknames=cv_test_config['error_metrics_nicknames'],
                                               training_config=training_config,
                                               cv=cv,
                                               test_path=model_path,
                                               callbacks=callbacks,
                                               save_predictions=True,
                                               repetition_ids=repetition_ids,
                                               retrieval_metric=retrieval_metric,
                                               split_key=cv_test_config['split_key'],
                                               distributed_info=distributed_info,
                                               top_K=top_K,
                                               current_K=current_K)

        logger.info('Average validation scores: {}'.format(
            {key: np.mean(item) for key, item in scores['validation_info'].items() if
             not key.startswith('avg')}))

        logger.info('Average test scores: {}'.format(
            {key: np.mean(item) for key, item in scores['test_info'].items() if not key.startswith('avg')}))

        if save_scores:
            # Validation

            save_json(os.path.join(model_path, cd.JSON_VALIDATION_INFO_NAME), scores['validation_info'])

            # Test
            save_json(os.path.join(model_path, cd.JSON_TEST_INFO_NAME), scores['test_info'])

            # Predictions
            save_json(os.path.join(model_path, cd.JSON_PREDICTIONS_NAME), scores['predictions'])
