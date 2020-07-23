"""

@Author:
@Date: 22/03/2019

TODO: merge compute_metrics and compute_iteration_validation_error

"""

import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection._split import _BaseKFold

from utility.json_utils import save_json, load_json
from utility.log_utils import get_logger

logger = get_logger(__name__)

supported_custom_metrics = {

}


def build_metrics(error_metrics):
    """
    Build validation metrics from given metrics name and problem type

    :param error_metrics: list of error metrics names (strings)
    :return: list of pointers to error metrics functions
    """

    parsed_metrics = []

    for metric in error_metrics:
        if hasattr(metrics, metric):
            parsed_metrics.append(getattr(metrics, metric))
        elif metric in supported_custom_metrics:
            parsed_metrics.append(supported_custom_metrics[metric])
        else:
            logger.warn('Unknown metric given (got {})! Skipping...'.format(metric))

    # Empty metrics list
    if not parsed_metrics:
        message = 'No valid metrics found! Aborting...'
        logger.error(message)
        raise RuntimeError(message)

    return parsed_metrics


def compute_metrics(parsed_metrics, true_values, predicted_values,
                    additional_metrics_info, metrics_nicknames=None, suffix=None, prefix=None):
    """
    Computes each given metric value, given true and predicted values.
    """

    if len(true_values.shape) > 1 and true_values.shape[1] > 1:
        true_values = np.argmax(true_values, axis=1)
        predicted_values = np.argmax(predicted_values, axis=1)

    if metrics_nicknames is None:
        metrics_nicknames = [metric.__name__ for metric in parsed_metrics]

    metrics_result = {}
    for metric, metric_info, metric_name in zip(parsed_metrics, additional_metrics_info, metrics_nicknames):

        signal_error = metric(y_true=true_values, y_pred=predicted_values,
                              **metric_info)

        if suffix is not None:
            metric_name = '{0}_{1}'.format(metric_name, suffix)
        if prefix is not None:
            metric_name = '{0}_{1}'.format(prefix, metric_name)

        metrics_result[metric_name] = signal_error

    return metrics_result


def compute_iteration_validation_error(parsed_metrics, true_values, predicted_values,
                                       error_metrics_additional_info=None,
                                       error_metrics_nicknames=None):
    """
    Computes each given metric value, given true and predicted values.

    :param parsed_metrics: list of metric functions (typically sci-kit learn metrics)
    :param true_values: ground-truth values
    :param predicted_values: model predicted values
    :param error_metrics_additional_info: additional arguments for each metric
    :param error_metrics_nicknames: custom nicknames for each metric
    :return: dict as follows:

        key: metric.__name__
        value: computed metric value
    """

    error_metrics_additional_info = error_metrics_additional_info or {}
    fold_error_info = {}

    if len(true_values.shape) > 1 and true_values.shape[1] > 1:
        true_values = np.argmax(true_values, axis=1)
        predicted_values = np.argmax(predicted_values, axis=1)

    if error_metrics_nicknames is None:
        error_metrics_nicknames = [metric.__name__ for metric in parsed_metrics]

    for metric, metric_info, metric_name in zip(parsed_metrics,
                                                error_metrics_additional_info,
                                                error_metrics_nicknames):
        signal_error = metric(y_true=true_values, y_pred=predicted_values,
                              **metric_info)
        fold_error_info.setdefault(metric_name, signal_error)

    return fold_error_info


def update_cv_validation_info(test_validation_info, iteration_validation_info):
    """
    Updates a dictionary with given values
    """

    test_validation_info = test_validation_info or {}

    for metric in iteration_validation_info:
        test_validation_info.setdefault(metric, []).append(iteration_validation_info[metric])

    return test_validation_info


class PrebuiltCV(_BaseKFold):
    """
    Simple CV wrapper for custom fold definition.
    """

    def __init__(self, cv_type='kfold', held_out_key='validation', **kwargs):
        super(PrebuiltCV, self).__init__(**kwargs)
        self.folds = None
        self.held_out_key = held_out_key

        if cv_type == 'kfold':
            self.cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        else:
            self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)

    def build_folds(self, X, y):
        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(X, y)):
            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes,
                self.held_out_key: held_out_indexes
            }

    def load_dataset_list(self, load_path):
        with open(load_path, 'r') as f:
            dataset_list = [item.strip() for item in f.readlines()]

        return dataset_list

    def save_folds(self, save_path, tolist=False):

        if tolist:
            to_save = {}
            for fold_key in self.folds:
                for split_set in self.folds[fold_key]:
                    to_save.setdefault(fold_key, {}).setdefault(split_set, self.folds[fold_key][split_set].tolist())
            save_json(save_path, to_save)
        else:
            save_json(save_path, self.folds)

    def load_folds(self, load_path):
        self.folds = load_json(load_path)

    def _iter_test_indices(self, X=None, y=None, groups=None):

        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            yield self.folds[fold][self.held_out_key]

    def split(self, X, y=None, groups=None):

        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            yield self.folds[fold]['train'], self.folds[fold][self.held_out_key]
