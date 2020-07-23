"""

Simple script that computes useful info about cross-validation results (v2 only)

1. Average and standard deviation (over all repetitions)
2. Per-fold best metrics

** BEFORE RUNNING **

1. Change the test unique identifier 'test_name' variable.
2. Change the 'model_type' variable as to match your tested model.
3. Make sure the first element if 'metrics' variable is the metric you wish to find the best repetitions for.

"""

import os
import const_define as cd
from utility.json_utils import load_json
import numpy as np
import matplotlib.pyplot as plt
from utility.cross_validation_utils import build_metrics, compute_iteration_validation_error, update_cv_validation_info
import math


def plot_avg_and_std(metric_data):
    if type(metric_data) is not np.ndarray:
        metric_data = np.array(metric_data)

    fig, axs = plt.subplots()
    axs.set_title('Per fold average and standard deviation', fontsize=30)
    axs.set_xlabel('Fold', fontsize=30)
    axs.tick_params(axis='x', labelsize=30)
    axs.tick_params(axis='y', labelsize=30)

    axs.boxplot(metric_data.transpose().tolist())


def plot_per_fold_max(metric_data, max_indexes=None):
    if type(metric_data) is not np.ndarray:
        metric_data = np.array(metric_data)

    if len(metric_data.shape) == 1:
        metric_data = metric_data[np.newaxis, :]

    metric_data = metric_data.transpose()

    if max_indexes is None:
        maximums = np.max(metric_data, axis=1)
    else:
        maximums = metric_data[np.arange(metric_data.shape[0]), max_indexes]

    # print('==> Best values per fold (according to validation set): ', maximums)
    avg_result = np.mean(maximums)

    fig, axs = plt.subplots()
    axs.set_title('Per fold maximums', fontsize=30)
    axs.set_xlabel('Fold', fontsize=30)

    axs.plot(np.arange(maximums.shape[0]) + 1, maximums, color='c', linewidth=5)
    axs.axhline(avg_result, color='r', linewidth=5)
    axs.tick_params(axis='x', labelsize=30)
    axs.tick_params(axis='y', labelsize=30)
    axs.set_xticks(np.arange(maximums.shape[0]) + 1)

    return avg_result, np.argmax(metric_data, axis=1)


# NOTE!: argmax gives incosistent results with argsort when ties
def get_per_fold_top_K_indexes(metric_data, K):
    if type(metric_data) is not np.ndarray:
        metric_data = np.array(metric_data)

    metric_data = metric_data.transpose()

    if len(metric_data.shape) == 1:
        metric_data = metric_data[np.newaxis, :]

    best_indexes = np.argsort(metric_data, axis=1)[:, ::-1]
    best_indexes = best_indexes[:, :K]

    return best_indexes


model_type = 'experimental_basic_memn2n_v2'
test_name = ''

# up to the number of repetitions done
voting_members = 1

print('Evaluating model: {0} -- test: {1} -- voting members: {2}'.format(model_type, test_name, voting_members))

# best indexes are computed for the first metric only and re-used for successive ones.
metrics = ['f1_score', 'precision_score', 'recall_score']

test_path = os.path.join(cd.CV_DIR, model_type, test_name)
test_info_path = os.path.join(test_path, cd.JSON_TEST_INFO_NAME)
val_info_path = os.path.join(test_path, cd.JSON_VALIDATION_INFO_NAME)

test_info = load_json(test_info_path)
val_info = load_json(val_info_path)

best_indexes = None

for metric in metrics:
    print('Metric: ', metric)

    # Validation
    val_metric_data = val_info[metric]

    # Compute avg and mean per fold
    plot_avg_and_std(val_metric_data)

    # Compute per fold max
    if best_indexes is None:
        val_best_avg_res, best_indexes = plot_per_fold_max(val_metric_data)
    else:
        val_best_avg_res, _ = plot_per_fold_max(val_metric_data)
    print('[Validation] [{}] Best Average result: '.format(metric), val_best_avg_res)

    # Compute average result
    if 'avg_{}'.format(metric) in val_info:
        print('[Validation] [{}] Average result: '.format(metric), np.mean(val_info['avg_{}'.format(metric)]))

    # Test

    # Compute average result
    if 'avg_{}'.format(metric) in test_info:
        print('[Test] [{}] Average result: '.format(metric), np.mean(test_info['avg_{}'.format(metric)]))

    # Compute per fold max (using validation indexes)
    test_metric_data = test_info[metric]
    test_best_avg_res, _ = plot_per_fold_max(test_metric_data, best_indexes)
    print('[Test] [{}] Best Average result: '.format(metric), test_best_avg_res)

    print('*' * 50)

# plt.show()

# Majority Voting (6 out of 10)
predictions = load_json(os.path.join(test_path, cd.JSON_PREDICTIONS_NAME))

if len(predictions['0'].shape) > 1:

    if voting_members < 0:
        voting_members = len(predictions[list(predictions.keys())[0]])

    majority_info = {}
    built_metrics = build_metrics(metrics)
    metrics_additional_info = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_CV_TEST_CONFIG_NAME))[
        'error_metrics_additional_info']

    best_K_indexes = get_per_fold_top_K_indexes(val_info[metrics[0]], voting_members)

    # Show top K test info
    for idx, best_set in enumerate(np.transpose(best_K_indexes)):
        for metric in metrics:
            test_metric_data = test_info[metric]
            test_best_avg_res, _ = plot_per_fold_max(test_metric_data, best_set)
            print('[Test Top {0}] [{1}] Best Average result: '.format(idx + 1, metric), test_best_avg_res)
        print('-' * 20)

    print('*' * 50)

    # assert np.equal(best_indexes, best_K_indexes[:, 0]).all()

    for fold_key, fold_preds in predictions.items():
        print('Fold: ', fold_key)
        total_reps = len(fold_preds)

        if voting_members > 0:
            total_reps = min(total_reps, voting_members)

        if total_reps % 2 == 0:
            threshold = total_reps / 2 + 1
        else:
            threshold = math.ceil(total_reps / 2)

        fold_preds = np.array(fold_preds)

        fold_majority = np.zeros(shape=[len(fold_preds[0]), voting_members], dtype=np.float32)
        for idx, rep_preds in enumerate(fold_preds[best_K_indexes[int(fold_key)], :]):
            fold_majority[:, idx] = rep_preds

        fold_majority = np.sum(fold_majority, axis=1)
        fold_majority[fold_majority < threshold] = 0
        fold_majority[fold_majority >= threshold] = 1

        fold_true_values = load_json(os.path.join(test_path, 'y_test_fold_{}.json'.format(fold_key)))

        majority_metrics = compute_iteration_validation_error(parsed_metrics=built_metrics,
                                                              true_values=fold_true_values,
                                                              predicted_values=fold_majority,
                                                              error_metrics_additional_info=metrics_additional_info)

        majority_info = update_cv_validation_info(test_validation_info=majority_info,
                                                  iteration_validation_info=majority_metrics)

        # Debug
        compute_metric = lambda preds: compute_iteration_validation_error(parsed_metrics=built_metrics,
                                                                          true_values=fold_true_values,
                                                                          predicted_values=preds,
                                                                          error_metrics_additional_info=metrics_additional_info)
        for rep_preds in fold_preds:
            print(compute_metric(rep_preds))

        print('==> Majority metrics:', majority_metrics)

        print('*' * 50)

    for metric, values in majority_info.items():
        print('Metric: ', metric)
        print('All: ', values)
        print('Average: ', np.mean(values))
        print('=' * 70)
else:
    print('Skipping majority metrics since more than 1 repetition is required...')