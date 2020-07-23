from utility.json_utils import load_json, save_json
import os
import const_define as cd


def get_data_config_id(filepath, config):
    config_path = os.path.join(filepath, cd.JSON_MODEL_DATA_CONFIGS_NAME)

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    if os.path.isfile(config_path):
        data_config = load_json(config_path)
    else:
        data_config = {}

    if config in data_config:
        return int(data_config[config])
    else:
        max_config = list(map(lambda item: int(item), data_config.values()))

        if len(max_config) > 0:
            max_config = max(max_config)
        else:
            max_config = -1

        data_config[config] = max_config + 1

        save_json(config_path, data_config)

        return max_config + 1
