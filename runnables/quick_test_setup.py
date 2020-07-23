"""

Simple test interface for quick and no error-prone setup

"""

from utility.json_utils import load_json, save_json
import os
import const_define as cd
from utility.log_utils import get_logger

logger = get_logger(__name__)


if __name__ == '__main__':

    # Any model name
    model_type = "experimental_basic_memn2n_v2"

    # Leave this blank
    model_dataset_type = ""

    # This should be fixed
    dataset_name = "tos_100"

    # If using KB -> task1_kb - If not using KB -> task1
    task_type = 'task1_kb'

    # A, CH, CR, TER, LTD
    category = 'CH'

    # This should be fixed
    test_type = "cv_test"

    # Leave this blank
    architecture_type = ""

    # If you want to save results and do a proper test -> True
    is_valid_test = False

    batch_size = 256
    step_checkpoint = None

    quick_settings = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_QUICK_SETUP_CONFIG_NAME))

    logger.info('Quick setup start!')

    # Data loader

    data_loader_settings = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME))
    data_loader_settings['type'] = task_type
    data_loader_settings['configs'][task_type]['category'] = category
    if architecture_type in quick_settings['input_specific_settings']:

        logger.info('Uploading data loader settings...')

        for key, value in quick_settings['input_specific_settings'][architecture_type].items():
            data_loader_settings['configs'][task_type][key] = value

    else:
        logger.warning('Could not find architecture type specific settings...Is it ok?')

    logger.info('Data loader settings uploaded! Do not forget to check loader specific args!!')

    save_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME), data_loader_settings)

    # Specific test settings

    if model_dataset_type:
        actual_model_type = model_dataset_type + '_' + model_type
    else:
        actual_model_type = model_type
    test_settings = load_json(os.path.join(cd.CONFIGS_DIR, '{}_config.json'.format(test_type)))
    test_settings['model_type'] = actual_model_type

    if dataset_name in quick_settings['data_specific_settings']:

        logger.info('Uploading {} settings...'.format(test_type))

        for key, value in quick_settings['data_specific_settings'][dataset_name].items():
            test_settings[key] = value
    else:
        logger.warning('Could not find dataset test specific settings...Aborting..')
        exit(-1)

    if is_valid_test:
        if test_type in quick_settings['valid_test_settings'] and dataset_name in quick_settings['valid_test_settings'][test_type]:
            for key, value in quick_settings['valid_test_settings'][test_type][dataset_name].items():
                test_settings[key] = value

        test_settings['save_model'] = True
        test_settings['compute_test_info'] = True
    else:
        if test_type in quick_settings['default_test_settings'] and dataset_name in quick_settings['default_test_settings'][test_type]:
            for key, value in quick_settings['default_test_settings'][test_type][dataset_name].items():
                test_settings[key] = value

        test_settings['save_model'] = False

    if model_type in quick_settings['model_specific_settings']:
        for key, value in quick_settings['model_specific_settings'][model_type].items():
            test_settings[key] = value
    else:
        logger.warning('Could not find model specific settings! Is this ok?')

    save_json(os.path.join(cd.CONFIGS_DIR, '{}_config.json'.format(test_type)), test_settings)

    # Training settings

    training_path = os.path.join(cd.CONFIGS_DIR, cd.JSON_TRAINING_CONFIG_NAME)
    training_settings = load_json(training_path)

    logger.info('Uploading {} training settings...'.format(test_type))

    training_settings['batch_size'] = batch_size
    training_settings['metrics'] = quick_settings['data_specific_settings'][dataset_name]['error_metrics']
    training_settings['additional_metrics_info'] = quick_settings['data_specific_settings'][dataset_name]['error_metrics_additional_info']
    training_settings['metrics_nicknames'] = quick_settings['data_specific_settings'][dataset_name]['error_metrics_nicknames']

    save_json(training_path, training_settings)

    logger.info('Quick setup upload done! Check loader specific settings and your model config before running the test!')