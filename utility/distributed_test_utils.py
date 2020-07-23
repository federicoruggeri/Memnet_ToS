"""

@Author:

@Date: 24/09/2019

"""

from __future__ import division

import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import const_define as cd
from custom_callbacks_v2 import TensorBoard
from data_converter import DataConverterFactory
from data_processor import ProcessorFactory
from nn_models_v2 import ModelFactory
from tokenization import TokenizerFactory
from utility.cross_validation_utils import build_metrics, compute_iteration_validation_error, update_cv_validation_info
from utility.data_utils import get_data_config_id
from utility.json_utils import save_json, load_json
from utility.log_utils import get_logger
from utility.pipeline_utils import get_dataset_fn
from utility.python_utils import flatten
from utility.python_utils import merge

logger = get_logger(__name__)


def cross_validation(validation_percentage, data_handle, cv, data_loader_info,
                     model_type, network_args, training_config, test_path,
                     error_metrics, error_metrics_additional_info=None, error_metrics_nicknames=None,
                     callbacks=None, compute_test_info=True, save_predictions=False,
                     use_tensorboard=False, repetitions=1, save_model=False,
                     split_key=None, checkpoint=None, distributed_info=None,
                     preloaded=False, preloaded_model=None, load_externally=False):
    """
    [Repeated] Cross-validation routine:

        1. [For each repetition]

        2. For each fold:
            2A. Load fold data: a DataHandle (tf_data_loader.py) object is used to retrieved cv fold data.
            2B. Pre-processing: a preprocessor (experimental_preprocessing.py) is used to parse text input.
            2C. Conversion: a converter (converters.py) is used to convert text to numerical format.
            2D. Train/Val/Test split: a splitter (splitters.py) is used to defined train/val/test sets
            2E. Model definition: a network model (nn_models_v2.py) is built
            2F. Model training: the network is trained.
            2G. Model evaluation on val/test sets: trained model is evaluated on val/test sets

        3. Results post-processing: macro-average values are computed.
    """

    if repetitions < 1:
        message = 'Repetitions should be at least 1! Got: {}'.format(repetitions)
        logger.error(message)
        raise AttributeError(message)

    # Step 0: build metrics
    parsed_metrics = build_metrics(error_metrics=error_metrics)

    # Step 0: add tensorboard visualization
    if use_tensorboard:
        test_name = os.path.split(test_path)[-1]
        tensorboard_base_dir = os.path.join(cd.PROJECT_DIR, 'logs', test_name)
        os.makedirs(tensorboard_base_dir)

    # Step 2: cross validation
    total_validation_info = OrderedDict()
    total_test_info = OrderedDict()
    total_preds = OrderedDict()

    # Associates an ID to each combination for easy file naming while maintaining whole info
    config_args = {key: arg['value']
                   for key, arg in network_args.items()
                   if 'processor' in arg['flags']
                   or 'tokenizer' in arg['flags']
                   or 'converter' in arg['flags']}
    config_args = flatten(config_args)
    config_args = merge(config_args, data_loader_info)
    config_args_tuple = [(key, value) for key, value in config_args.items()]
    config_args_tuple = sorted(config_args_tuple, key=lambda item: item[0])

    config_name = '_'.join(['{0}-{1}'.format(name, value) for name, value in config_args_tuple])
    model_base_path = os.path.join(cd.TESTS_DATA_DIR,
                                   data_handle.data_name,
                                   model_type)
    config_id = get_data_config_id(filepath=model_base_path, config=config_name)
    model_path = os.path.join(model_base_path, str(config_id))

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Build pipeline: each step here is guaranteed to be idempotent (make sure of it!)

    # Build processor
    processor_type = cd.MODEL_CONFIG[model_type]['processor']
    processor_args = {key: arg['value'] for key, arg in network_args.items() if 'processor' in arg['flags']}
    processor_args['loader_info'] = data_handle.get_additional_info()
    processor = ProcessorFactory.factory(processor_type, **processor_args)

    # Build tokenizer
    tokenizer_type = cd.MODEL_CONFIG[model_type]['tokenizer']
    if preloaded and load_externally:
        tokenizer_class = TokenizerFactory.supported_tokenizers[tokenizer_type]
        tokenizer = tokenizer_class.from_pretrained(network_args['preloaded_name']['value'])
    else:
        tokenizer_args = {key: arg['value'] for key, arg in network_args.items() if 'tokenizer' in arg['flags']}
        tokenizer = TokenizerFactory.factory(tokenizer_type, **tokenizer_args)

    # Build converter
    converter_type = cd.MODEL_CONFIG[model_type]['converter']
    converter_args = {key: arg['value'] for key, arg in network_args.items() if 'converter' in arg['flags']}
    converter = DataConverterFactory.factory(converter_type, **converter_args)
    converter_instance_args = converter.get_instance_args()

    for repetition in range(repetitions):
        logger.info('Repetition {0}/{1}'.format(repetition + 1, repetitions))

        validation_info = OrderedDict()
        test_info = OrderedDict()
        all_preds = OrderedDict()

        for fold_idx, (train_indexes, test_indexes) in enumerate(cv.split(None)):
            logger.info('Starting Fold {0}/{1}'.format(fold_idx + 1, cv.n_splits))

            train_df, val_df, test_df = data_handle.get_split(key=split_key,
                                                              key_values=test_indexes,
                                                              validation_percentage=validation_percentage)

            train_filepath = os.path.join(model_path, 'train_data_fold_{}'.format(fold_idx))
            val_filepath = os.path.join(model_path, 'val_data_fold_{}'.format(fold_idx))
            test_filepath = os.path.join(model_path, 'test_data_fold_{}'.format(fold_idx))

            save_prefix = 'fold_{}'.format(fold_idx)

            if not os.path.isfile(test_filepath):
                logger.info('Dataset not found! Building new one from scratch....it may require some minutes')

                # Processor

                train_data = processor.get_train_examples(data=train_df, ids=np.arange(train_df.shape[0]))
                val_data = processor.get_dev_examples(data=val_df, ids=np.arange(val_df.shape[0]))
                test_data = processor.get_test_examples(data=test_df, ids=np.arange(test_df.shape[0]))

                # Tokenizer

                train_texts = train_data.get_data()
                tokenizer.build_vocab(data=train_texts, filepath=model_path, prefix=save_prefix)
                tokenizer.save_info(filepath=model_path, prefix=save_prefix)
                tokenizer_info = tokenizer.get_info()

                # Conversion

                # WARNING: suffers multi-threading (what if another processing is building the same data?)
                # This may happen only the first time an input pipeline is used. Usually calibration is on
                # model parameters
                converter.convert_data(examples=train_data,
                                       label_list=processor.get_labels(),
                                       output_file=train_filepath,
                                       tokenizer=tokenizer,
                                       checkpoint=checkpoint,
                                       is_training=True)
                converter.save_conversion_args(filepath=model_path, prefix=save_prefix)
                converter_info = converter.get_conversion_args()

                converter.convert_data(examples=val_data,
                                       label_list=processor.get_labels(),
                                       output_file=val_filepath,
                                       tokenizer=tokenizer,
                                       checkpoint=checkpoint)

                converter.convert_data(examples=test_data,
                                       label_list=processor.get_labels(),
                                       output_file=test_filepath,
                                       tokenizer=tokenizer,
                                       checkpoint=checkpoint)
            else:
                tokenizer_info = TokenizerFactory.supported_tokenizers[tokenizer_type].load_info(filepath=model_path,
                                                                                                 prefix=save_prefix)
                converter_info = DataConverterFactory.supported_data_converters[converter_type].load_conversion_args(
                    filepath=model_path,
                    prefix=save_prefix)

            # Debug
            logger.info('Tokenizer info: \n{}'.format(tokenizer_info))
            logger.info('Converter info: \n{}'.format(converter_info))

            # Create Datasets

            train_data = get_dataset_fn(filepath=train_filepath,
                                        batch_size=training_config['batch_size'],
                                        name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                              converter_instance_args),
                                        selector=converter.feature_class.get_dataset_selector(),
                                        is_training=True,
                                        shuffle_amount=distributed_info['shuffle_amount'],
                                        reshuffle_each_iteration=distributed_info['reshuffle_each_iteration'],
                                        prefetch_amount=distributed_info['prefetch_amount'])

            fixed_train_data = get_dataset_fn(filepath=train_filepath,
                                              batch_size=training_config['batch_size'],
                                              name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                                    converter_instance_args),
                                              selector=converter.feature_class.get_dataset_selector(),
                                              is_training=False,
                                              prefetch_amount=distributed_info['prefetch_amount'])

            val_data = get_dataset_fn(filepath=val_filepath,
                                      batch_size=training_config['batch_size'],
                                      name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                            converter_instance_args),
                                      selector=converter.feature_class.get_dataset_selector(),
                                      is_training=False,
                                      prefetch_amount=distributed_info['prefetch_amount'])

            test_data = get_dataset_fn(filepath=test_filepath,
                                       batch_size=training_config['batch_size'],
                                       name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                             converter_instance_args),
                                       selector=converter.feature_class.get_dataset_selector(),
                                       is_training=False,
                                       prefetch_amount=distributed_info['prefetch_amount'])

            # Build model

            network_retrieved_args = {key: value['value'] for key, value in network_args.items()
                                      if 'model_class' in value['flags']}
            network_retrieved_args['additional_data'] = data_handle.get_additional_info()
            network_retrieved_args['name'] = '{0}_repetition_{1}_fold_{2}'.format(
                cd.SUPPORTED_ALGORITHMS[model_type]['save_suffix'], repetition, fold_idx)
            network = ModelFactory.factory(cl_type=model_type, **network_retrieved_args)

            # Useful stuff
            train_steps = int(np.ceil(train_df.shape[0] / training_config['batch_size']))
            eval_steps = int(np.ceil(val_df.shape[0] / training_config['batch_size']))
            test_steps = int(np.ceil(test_df.shape[0] / training_config['batch_size']))

            np_train_y = np.concatenate([item for item in fixed_train_data().map(lambda x, y: y).take(train_steps)])
            np_val_y = np.concatenate([item for item in val_data().map(lambda x, y: y)])
            np_test_y = np.concatenate([item for item in test_data().map(lambda x, y: y)])

            logger.info('Total train steps: {}'.format(train_steps))
            logger.info('Total eval steps: {}'.format(eval_steps))
            logger.info('Total test steps: {}'.format(test_steps))

            # computing positive label weights (for unbalanced dataset)
            network.compute_output_weights(y_train=np_train_y, num_classes=data_handle.num_classes)

            # Custom callbacks only
            for callback in callbacks:
                if hasattr(callback, 'on_build_model_begin'):
                    callback.on_build_model_begin(logs={'network': network})

            text_info = merge(tokenizer_info, converter_info)
            network.build_model(text_info=text_info)

            # Custom callbacks only
            for callback in callbacks:
                if hasattr(callback, 'on_build_model_end'):
                    callback.on_build_model_end(logs={'network': network})

            if use_tensorboard:
                fold_log_dir = os.path.join(tensorboard_base_dir,
                                            'repetition_{}'.format(repetition),
                                            'fold_{}'.format(fold_idx))
                os.makedirs(fold_log_dir)
                tensorboard = TensorBoard(batch_size=training_config['batch_size'],
                                          log_dir=fold_log_dir,
                                          )
                fold_callbacks = callbacks + [tensorboard]
            else:
                fold_callbacks = callbacks

            if preloaded:
                network.predict(x=iter(val_data()), steps=1)

                initial_weights = [layer.get_weights() for layer in network.model.layers]

                if load_externally:
                    if 'preloaded_name' in network_args:
                        preloaded_name = network_args['preloaded_name']['value']
                    else:
                        preloaded_name = model_type
                    network.load(network.from_pretrained_weights(preloaded_name), by_name=True)
                else:
                    network.load(
                        os.path.join(cd.CV_DIR, preloaded_model, '{0}_repetition_{1}_fold_{2}.h5'.format(model_type,
                                                                                                         repetition,
                                                                                                         fold_idx)))

                # Correct loading check (inherently makes sure that restore ops are run)
                for layer, initial in zip(network.model.layers, initial_weights):
                    weights = layer.get_weights()
                    if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                        logger.info('Checkpoint contained no weights for layer {}!'.format(layer.name))

            # Training
            network.fit(train_data=train_data,
                        fixed_train_data=fixed_train_data,
                        epochs=training_config['epochs'], verbose=training_config['verbose'],
                        callbacks=fold_callbacks, validation_data=val_data,
                        step_checkpoint=training_config['step_checkpoint'],
                        metrics=training_config['metrics'],
                        additional_metrics_info=training_config['additional_metrics_info'],
                        metrics_nicknames=training_config['metrics_nicknames'],
                        train_num_batches=train_steps,
                        eval_num_batches=eval_steps,
                        np_val_y=np_val_y,
                        np_train_y=np_train_y)

            # Inference
            val_predictions = network.predict(x=iter(val_data()),
                                              steps=eval_steps,
                                              callbacks=fold_callbacks)

            iteration_validation_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                            true_values=np_val_y,
                                                                            predicted_values=val_predictions,
                                                                            error_metrics_additional_info=error_metrics_additional_info,
                                                                            error_metrics_nicknames=error_metrics_nicknames)

            validation_info = update_cv_validation_info(test_validation_info=validation_info,
                                                        iteration_validation_info=iteration_validation_error)

            logger.info('Iteration validation info: {}'.format(iteration_validation_error))

            test_predictions = network.predict(x=iter(test_data()),
                                               steps=test_steps,
                                               callbacks=callbacks)

            iteration_test_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                      true_values=np_test_y,
                                                                      predicted_values=test_predictions,
                                                                      error_metrics_additional_info=error_metrics_additional_info,
                                                                      error_metrics_nicknames=error_metrics_nicknames)

            if compute_test_info:
                test_info = update_cv_validation_info(test_validation_info=test_info,
                                                      iteration_validation_info=iteration_test_error)
                logger.info('Iteration test info: {}'.format(iteration_test_error))

                if save_predictions:
                    all_preds[fold_idx] = test_predictions.ravel()

            # Save model
            if save_model:
                filepath = os.path.join(test_path,
                                        '{0}_repetition_{1}_fold_{2}'.format(
                                            cd.SUPPORTED_ALGORITHMS[model_type]['save_suffix'],
                                            repetition,
                                            fold_idx))
                network.save(filepath=filepath)

                filepath = os.path.join(test_path, 'y_test_fold_{}.json'.format(fold_idx))
                if not os.path.isfile(filepath):
                    save_json(filepath=filepath, data=np_test_y)

            # Flush
            K.clear_session()

        for key, item in validation_info.items():
            total_validation_info.setdefault(key, []).append(item)
        for key, item in test_info.items():
            total_test_info.setdefault(key, []).append(item)
        for key, item in all_preds.items():
            total_preds.setdefault(key, []).append(item)

    if repetitions == 1:
        total_validation_info = {key: np.mean(item, 0) for key, item in total_validation_info.items()}
        total_test_info = {key: np.mean(item, 0) for key, item in total_test_info.items()}
        total_preds = {key: np.mean(item, 0) for key, item in total_preds.items()}
    else:
        avg_validation_info = {}
        for key, item in total_validation_info.items():
            avg_validation_info['avg_{}'.format(key)] = np.mean(item, 0)
        total_validation_info = merge(total_validation_info, avg_validation_info)

        avg_test_info = {}
        for key, item in total_test_info.items():
            avg_test_info['avg_{}'.format(key)] = np.mean(item, 0)
        total_test_info = merge(total_test_info, avg_test_info)

    result = {
        'validation_info': total_validation_info,
    }

    if compute_test_info:
        result['test_info'] = total_test_info
        if save_predictions:
            result['predictions'] = total_preds
    else:
        if save_predictions:
            result['predictions'] = total_preds

    return result


def cross_validation_forward_pass(validation_percentage, data_handle, cv, data_loader_info,
                                  model_type, network_args, training_config, test_path,
                                  error_metrics, error_metrics_additional_info=None, error_metrics_nicknames=None,
                                  callbacks=None, compute_test_info=True, save_predictions=False,
                                  repetition_ids=None, retrieval_metric=None,
                                  split_key=None, distributed_info=None, top_K=None, current_K=None):
    """
    Simple CV variant that retrieves a pre-trained model to do the forward pass for each val/test fold sets.
    """

    # Step 0: build metrics
    parsed_metrics = build_metrics(error_metrics=error_metrics)

    # Step 1: cross validation

    # Associates an ID to each combination for easy file naming while maintaining whole info
    config_args = {key: arg['value']
                   for key, arg in network_args.items()
                   if 'processor' in arg['flags']
                   or 'tokenizer' in arg['flags']
                   or 'converter' in arg['flags']}
    config_args = flatten(config_args)
    config_args = merge(config_args, data_loader_info)
    config_args_tuple = [(key, value) for key, value in config_args.items()]
    config_args_tuple = sorted(config_args_tuple, key=lambda item: item[0])

    config_name = '_'.join(['{0}-{1}'.format(name, value) for name, value in config_args_tuple])
    model_base_path = os.path.join(cd.TESTS_DATA_DIR,
                                   data_handle.data_name,
                                   model_type)
    config_id = get_data_config_id(filepath=model_base_path, config=config_name)
    model_path = os.path.join(model_base_path, str(config_id))

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Build pipeline: each step here is guaranteed to be idempotent (make sure of it!)

    # Build processor
    processor_type = cd.MODEL_CONFIG[model_type]['processor']
    processor_args = {key: arg['value'] for key, arg in network_args.items() if 'processor' in arg['flags']}
    processor_args['loader_info'] = data_handle.get_additional_info()
    processor = ProcessorFactory.factory(processor_type, **processor_args)

    # Build tokenizer
    tokenizer_type = cd.MODEL_CONFIG[model_type]['tokenizer']
    tokenizer_args = {key: arg['value'] for key, arg in network_args.items() if 'tokenizer' in arg['flags']}
    tokenizer = TokenizerFactory.factory(tokenizer_type, **tokenizer_args)

    # Build converter
    converter_type = cd.MODEL_CONFIG[model_type]['converter']
    converter_args = {key: arg['value'] for key, arg in network_args.items() if 'converter' in arg['flags']}
    converter = DataConverterFactory.factory(converter_type, **converter_args)
    converter_instance_args = converter.get_instance_args()

    # Retrieve repetition ids if not given
    if repetition_ids is None:
        if retrieval_metric is None or retrieval_metric not in error_metrics:
            raise AttributeError('Invalid retrieval metric! It is required in'
                                 ' order to determine best folds. Got: {}'.format(retrieval_metric))
        loaded_val_results = load_json(os.path.join(test_path, cd.JSON_VALIDATION_INFO_NAME))
        metric_val_results = loaded_val_results[retrieval_metric]
        metric_val_results = np.array(metric_val_results)

        if len(metric_val_results.shape) == 1:
            metric_val_results = metric_val_results[np.newaxis, :]

        if top_K is None:
            repetition_ids = np.argmax(metric_val_results, axis=0)
        else:
            if current_K != 0:
                metric_val_results = metric_val_results.transpose()
                repetition_ids = np.argsort(metric_val_results, axis=1)[:, ::-1]
                repetition_ids = repetition_ids[:, :top_K][:, current_K]
            else:
                # Ensure that greedy result equals top 1 result (avoid multiple matches)
                repetition_ids = np.argmax(metric_val_results, axis=0)

    validation_info = OrderedDict()
    test_info = OrderedDict()
    all_preds = OrderedDict()

    for fold_idx, (train_indexes, test_indexes) in enumerate(cv.split(None)):
        logger.info('Starting Fold {0}/{1}'.format(fold_idx + 1, cv.n_splits))

        train_df, val_df, test_df = data_handle.get_split(key=split_key,
                                                          key_values=test_indexes,
                                                          validation_percentage=validation_percentage)

        test_df.to_csv(os.path.join(test_path, 'test_df_fold_{}.csv'.format(fold_idx)), index=None)

        train_filepath = os.path.join(model_path, 'train_data_fold_{}'.format(fold_idx))
        val_filepath = os.path.join(model_path, 'val_data_fold_{}'.format(fold_idx))
        test_filepath = os.path.join(model_path, 'test_data_fold_{}'.format(fold_idx))

        save_prefix = 'fold_{}'.format(fold_idx)

        if not os.path.isfile(test_filepath):
            logger.info('Dataset not found! Building new one from scratch....it may require some minutes')

            # Processor

            train_data = processor.get_train_examples(data=train_df, ids=np.arange(train_df.shape[0]))
            val_data = processor.get_dev_examples(data=val_df, ids=np.arange(val_df.shape[0]))
            test_data = processor.get_test_examples(data=test_df, ids=np.arange(test_df.shape[0]))

            # Tokenizer

            train_texts = train_data.get_data()
            tokenizer.build_vocab(data=train_texts, filepath=model_path, prefix=save_prefix)
            tokenizer.save_info(filepath=model_path, prefix=save_prefix)
            tokenizer_info = tokenizer.get_info()

            # Conversion

            # WARNING: suffers multi-threading (what if another processing is building the same data?)
            # This may happen only the first time an input pipeline is used. Usually calibration is on
            # model parameters
            converter.convert_data(examples=train_data,
                                   label_list=processor.get_labels(),
                                   output_file=train_filepath,
                                   tokenizer=tokenizer,
                                   is_training=True)
            converter.save_conversion_args(filepath=model_path, prefix=save_prefix)
            converter_info = converter.get_conversion_args()

            converter.convert_data(examples=val_data,
                                   label_list=processor.get_labels(),
                                   output_file=val_filepath,
                                   tokenizer=tokenizer)

            converter.convert_data(examples=test_data,
                                   label_list=processor.get_labels(),
                                   output_file=test_filepath,
                                   tokenizer=tokenizer)
        else:
            tokenizer_info = TokenizerFactory.supported_tokenizers[tokenizer_type].load_info(filepath=model_path,
                                                                                             prefix=save_prefix)
            converter_info = DataConverterFactory.supported_data_converters[converter_type].load_conversion_args(
                filepath=model_path,
                prefix=save_prefix)

        # Debug
        logger.info('Tokenizer info: \n{}'.format(tokenizer_info))
        logger.info('Converter info: \n{}'.format(converter_info))

        # Create Datasets

        val_data = get_dataset_fn(filepath=val_filepath,
                                  batch_size=training_config['batch_size'],
                                  name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                        converter_instance_args),
                                  selector=converter.feature_class.get_dataset_selector(),
                                  is_training=False,
                                  prefetch_amount=distributed_info['prefetch_amount'])

        test_data = get_dataset_fn(filepath=test_filepath,
                                   batch_size=training_config['batch_size'],
                                   name_to_features=converter.feature_class.get_mappings(converter_info,
                                                                                         converter_instance_args),
                                   selector=converter.feature_class.get_dataset_selector(),
                                   is_training=False,
                                   prefetch_amount=distributed_info['prefetch_amount'])

        # Build model

        network_retrieved_args = {key: value['value'] for key, value in network_args.items()
                                  if 'model_class' in value['flags']}
        network_retrieved_args['additional_data'] = data_handle.get_additional_info()
        network_retrieved_args['name'] = '{0}_repetition_{1}_fold_{2}'.format(
            cd.SUPPORTED_ALGORITHMS[model_type]['save_suffix'], repetition_ids[fold_idx], fold_idx)
        network = ModelFactory.factory(cl_type=model_type, **network_retrieved_args)

        # Useful stuff
        eval_steps = int(np.ceil(val_df.shape[0] / training_config['batch_size']))
        test_steps = int(np.ceil(test_df.shape[0] / training_config['batch_size']))

        np_val_y = np.concatenate([item for item in val_data().map(lambda x, y: y)])
        np_test_y = np.concatenate([item for item in test_data().map(lambda x, y: y)])

        logger.info('Total eval steps: {}'.format(eval_steps))
        logger.info('Total test steps: {}'.format(test_steps))

        # Custom callbacks only
        for callback in callbacks:
            if hasattr(callback, 'on_build_model_begin'):
                callback.on_build_model_begin(logs={'network': network})

        text_info = merge(tokenizer_info, converter_info)
        network.build_model(text_info=text_info)

        # Custom callbacks only
        for callback in callbacks:
            if hasattr(callback, 'on_build_model_end'):
                callback.on_build_model_end(logs={'network': network})

        # Setup model by feeding an input
        network.predict(x=iter(val_data()), steps=eval_steps)

        # load pre-trained weights
        current_weight_filename = '{0}_repetition_{1}_fold_{2}.h5'.format(model_type,
                                                                          repetition_ids[fold_idx],
                                                                          fold_idx)
        network.load(os.path.join(test_path, current_weight_filename))

        # Inference
        val_predictions = network.predict(x=iter(val_data()),
                                          steps=eval_steps)

        iteration_validation_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                        true_values=np_val_y,
                                                                        predicted_values=val_predictions,
                                                                        error_metrics_additional_info=error_metrics_additional_info,
                                                                        error_metrics_nicknames=error_metrics_nicknames)

        validation_info = update_cv_validation_info(test_validation_info=validation_info,
                                                    iteration_validation_info=iteration_validation_error)

        logger.info('Iteration validation info: {}'.format(iteration_validation_error))

        test_predictions = network.predict(x=iter(test_data()),
                                           steps=test_steps,
                                           callbacks=callbacks)

        iteration_test_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                  true_values=np_test_y,
                                                                  predicted_values=test_predictions,
                                                                  error_metrics_additional_info=error_metrics_additional_info,
                                                                  error_metrics_nicknames=error_metrics_nicknames)

        if compute_test_info:
            test_info = update_cv_validation_info(test_validation_info=test_info,
                                                  iteration_validation_info=iteration_test_error)
            logger.info('Iteration test info: {}'.format(iteration_test_error))

            if save_predictions:
                all_preds[fold_idx] = test_predictions.ravel()

        # Flush
        K.clear_session()

    result = {
        'validation_info': validation_info,
    }

    if compute_test_info:
        result['test_info'] = test_info
        if save_predictions:
            result['predictions'] = all_preds
    else:
        if save_predictions:
            result['predictions'] = all_preds

    return result
