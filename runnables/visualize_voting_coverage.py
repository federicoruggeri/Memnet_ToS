"""

@Author:
@Date: 13/12/2019


BEFORE RUNNING:

1. Make sure "model_type" is set to an existing architecture
2. Make sure 'test_name' is set to an existing model test folder (check cv_test folder)
3. Make sure the retrieval metric is a supported metric and the one of your choice
4.

OUTPUT:

1. For each selected repetition ('top_K') it prints out all memory usage statistics (see paper)
2. Top R hits for CP ranking is controlled by 'ranking' variable

"""

import os

import numpy as np
import pandas as pd

import const_define as cd
from utility.json_utils import load_json
from utility.plot_utils import show_target_coverage, show_voting_coverage, MemoryMetrics


def compute_coverage_info(attention_weights, K_repetition_ids, R=3):
    total_usage, total_hits, total_correct, total_correct_and_hit, \
    total_correct_and_no_hit, total_samples, \
    avg_memory_percentage = 0, 0, 0, 0, 0, 0, 0
    total_top_R_hits = np.zeros(R)

    for attention_weight in attention_weights:
        sub_path = os.path.join(model_path, attention_weight)
        loaded_weights = load_json(sub_path)
        fold_name = attention_weight.split('fold')[1].split('_')[1]

        fold_test_df = pd.read_csv(os.path.join(model_path, 'test_df_fold_{}.csv'.format(fold_name)))
        repetition_id = K_repetition_ids[int(fold_name)]
        fold_stats = show_target_coverage(
            attention=loaded_weights,
            test_df=fold_test_df,
            fold_name=fold_name,
            predictions=predictions[fold_name][repetition_id],
            category=category,
            attention_mode=attention_mode,
            verbose=0,
            R=R)

        total_usage += fold_stats.usage
        total_hits += fold_stats.hits
        total_correct += fold_stats.correct
        total_correct_and_hit += fold_stats.correct_and_hit
        total_correct_and_no_hit += fold_stats.correct_and_no_hit
        total_samples += fold_stats.samples
        total_top_R_hits += fold_stats.top_R_hits
        avg_memory_percentage += fold_stats.avg_memory_percentage

    memory_usage = total_usage / total_samples
    coverage = total_hits / total_samples
    recall = total_correct / total_samples
    supervision_precision = total_correct_and_hit / total_samples
    non_memory_accuracy = (total_correct - total_correct_and_hit - total_correct_and_no_hit) / total_samples

    try:
        coverage_precision = total_hits / total_usage
        memory_precision = total_correct_and_hit / total_usage
        avg_memory_percentage = avg_memory_percentage / total_usage
        top_R_coverage_precision = total_top_R_hits / total_usage
    except ZeroDivisionError:
        coverage_precision = None
        memory_precision = None
        avg_memory_percentage = None
        top_R_coverage_precision = None

    metrics = MemoryMetrics(memory_usage,
                            coverage,
                            coverage_precision,
                            recall,
                            memory_precision,
                            supervision_precision,
                            non_memory_accuracy,
                            top_R_coverage_precision,
                            avg_memory_percentage)
    return metrics


def get_per_fold_top_K_indexes(metric_data, K):
    if type(metric_data) is not np.ndarray:
        metric_data = np.array(metric_data)

    metric_data = metric_data.transpose()

    if K == 1:
        return np.argmax(metric_data, axis=1)

    best_indexes = np.argsort(metric_data, axis=1)[:, ::-1]
    best_indexes = best_indexes[:, :K]

    return best_indexes


model_type = "experimental_basic_memn2n_v2"
test_name = ""
retrieval_metric = 'f1_score'

# up to the number of repetitions done
top_K = 3
ranking = 3

model_path = os.path.join(cd.CV_DIR, model_type, test_name)

loader_info = load_json(os.path.join(model_path, cd.JSON_DATA_LOADER_CONFIG_NAME))
category = loader_info['configs'][loader_info['type']]["category"]

attention_weights = [name for name in os.listdir(model_path)
                     if 'attention_weights' in name.lower()]

predictions = load_json(os.path.join(model_path, cd.JSON_PREDICTIONS_NAME))

if len(predictions['0'].shape) == 1:
    predictions = {key: value[np.newaxis, :] for key, value in predictions.items()}

loaded_val_results = load_json(os.path.join(model_path, cd.JSON_VALIDATION_INFO_NAME))
metric_val_results = loaded_val_results[retrieval_metric]

if len(metric_val_results.shape) == 1:
    metric_val_results = metric_val_results[np.newaxis, :]

repetition_ids = get_per_fold_top_K_indexes(metric_val_results, top_K)

if len(repetition_ids.shape) == 1:
    repetition_ids = repetition_ids[:, np.newaxis]

true_value_name = 'y_test_fold_{}.json'
true_values = [load_json(os.path.join(model_path, true_value_name.format(fold)))
               for fold in range(len(attention_weights) // top_K)]

memory_explanations = pd.read_csv(os.path.join(cd.KB_DIR, '{}_explanations.csv'.format(category)))
memory_labels = {idx: exp for idx, exp in zip(memory_explanations.index.values, memory_explanations.Id.values)}

attention_mode = load_json(os.path.join(model_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))['extraction_info']['value'][
    'mode']

for K in range(top_K):
    K_attention_weights = [name for name in attention_weights
                           if 'top{}'.format(K + 1) in name]
    K_metrics = compute_coverage_info(
        K_attention_weights,
        repetition_ids[:, K],
        R=ranking)

    for key, value in K_metrics._asdict().items():
        print('[Top {0}] {1}: {2}'.format(K, key, value))

    print('*' * 50)
    print()

grouped_attention_weights = [[name for name in attention_weights if 'fold_{}'.format(fold) in name]
                             for fold in range(len(true_values))]

total_usage_union, total_hits_union, \
total_correct_and_hit_union, total_correct_and_no_hit_union, \
avg_memory_percentage_union = 0, 0, 0, 0, 0
total_top_R_hits_union = np.zeros(ranking)

total_usage_intersection, total_hits_intersection, \
total_correct_and_hit_intersection, total_correct_and_no_hit_intersection, \
avg_memory_percentage_int = 0, 0, 0, 0, 0
total_top_R_hits_intersection = np.zeros(ranking)

total_correct, total_samples = 0, 0
for attention_group in grouped_attention_weights:
    group_weights = []
    for attention_weight in attention_group:
        sub_path = os.path.join(model_path, attention_weight)
        loaded_weights = load_json(sub_path)
        fold_name = attention_weight.split('fold')[1].split('_')[1]
        group_weights.append(loaded_weights)

    group_weights = np.array(group_weights)

    fold_test_df = pd.read_csv(os.path.join(model_path, 'test_df_fold_{}.csv'.format(fold_name)))
    fold_K_repetition_ids = repetition_ids[int(fold_name)]
    K_predictions = np.array(predictions[fold_name])[fold_K_repetition_ids].transpose()

    fold_union_stats, fold_intersection_stats = show_voting_coverage(K_attentions=group_weights, test_df=fold_test_df,
                                                                     category=category,
                                                                     fold_name=fold_name, K_predictions=K_predictions,
                                                                     attention_mode=attention_mode,
                                                                     verbose=0,
                                                                     R=ranking)

    # Union
    total_usage_union += fold_union_stats.usage
    total_hits_union += fold_union_stats.hits
    total_correct_and_hit_union += fold_union_stats.correct_and_hit
    total_correct_and_no_hit_union += fold_union_stats.correct_and_no_hit
    total_top_R_hits_union += fold_union_stats.top_R_hits
    avg_memory_percentage_union += fold_union_stats.avg_memory_percentage

    # Intersection
    total_usage_intersection += fold_intersection_stats.usage
    total_hits_intersection += fold_intersection_stats.hits
    total_correct_and_hit_intersection += fold_intersection_stats.correct_and_hit
    total_correct_and_no_hit_intersection += fold_intersection_stats.correct_and_no_hit
    total_top_R_hits_intersection += fold_intersection_stats.top_R_hits
    avg_memory_percentage_int += fold_intersection_stats.avg_memory_percentage

    # Overall
    total_samples += fold_union_stats.samples
    total_correct += fold_union_stats.correct

# Union
memory_usage_union = total_usage_union / total_samples
coverage_union = total_hits_union / total_samples
supervision_precision_union = total_correct_and_hit_union / total_samples
non_memory_accuracy_union = (
                                        total_correct - total_correct_and_hit_union - total_correct_and_no_hit_union) / total_samples
try:
    coverage_precision_union = total_hits_union / total_usage_union
    top_R_coverage_precision_union = total_top_R_hits_union / total_usage_union
    memory_precision_union = total_correct_and_hit_union / total_usage_union
    avg_memory_percentage_union = avg_memory_percentage_union / total_usage_union
except ZeroDivisionError:
    coverage_precision_union = None
    top_R_coverage_precision_union = None
    memory_precision_union = None
    avg_memory_percentage_union = None

print('[{0}] stats:\n'
      ' Memory usage: {1}/{2} ({3})\n'
      ' Coverage (correct memory over all samples): {4}/{2} ({5})\n'
      ' Coverage precision (correct memory over memory usage): {4}/{1} ({6})\n'
      ' Top R coverage precision (correct memory and no other memories over memory usage): {12}/{4} ({13})\n'
      ' Memory precision (correct and correct memory over memory usage: {7}/{1} ({8})\n'
      ' Supervision precision (correct and correct memory over all samples): {7}/{2} ({9})\n'
      ' Non-memory precision (correct and no memory over all samples): {10}/{2} ({11})\n'
      ' Average memory percentage (average percentage of selected memory): {14}\n'
      .format("union",
              total_usage_union,
              total_samples,
              memory_usage_union,
              total_hits_union,
              coverage_union,
              coverage_precision_union,
              total_correct_and_hit_union,
              memory_precision_union,
              supervision_precision_union,
              total_correct - total_correct_and_hit_union - total_correct_and_no_hit_union,
              non_memory_accuracy_union,
              total_top_R_hits_union,
              top_R_coverage_precision_union,
              avg_memory_percentage_union))

# Intersection
memory_usage_intersection = total_usage_intersection / total_samples
coverage_intersection = total_hits_intersection / total_samples
supervision_precision_intersection = total_correct_and_hit_intersection / total_samples
non_memory_accuracy_intersection = (
                                           total_correct - total_correct_and_hit_intersection - total_correct_and_no_hit_intersection) / total_samples
try:
    coverage_precision_intersection = total_hits_intersection / total_usage_intersection
    top_R_coverage_precision_intersection = total_top_R_hits_intersection / total_usage_intersection
    memory_precision_intersection = total_correct_and_hit_intersection / total_usage_intersection
    avg_memory_percentage_int = avg_memory_percentage_int / total_usage_intersection
except ZeroDivisionError:
    coverage_precision_intersection = None
    top_R_coverage_precision_intersection = None
    memory_precision_intersection = None
    avg_memory_percentage_int = None

print('[{0}] stats:\n'
      ' Memory usage: {1}/{2} ({3})\n'
      ' Coverage (correct memory over all samples): {4}/{2} ({5})\n'
      ' Coverage precision (correct memory over memory usage): {4}/{1} ({6})\n'
      ' Top R coverage precision (correct memory and no other memories over memory usage): {12}/{4} ({13})\n'
      ' Memory precision (correct and correct memory over memory usage: {7}/{1} ({8})\n'
      ' Supervision precision (correct and correct memory over all samples): {7}/{2} ({9})\n'
      ' Non-memory precision (correct and no memory over all samples): {10}/{2} ({11})\n'
      ' Average memory percentage (average percentage of selected memory): {14}\n'
      .format("intersection",
              total_usage_intersection,
              total_samples,
              memory_usage_intersection,
              total_hits_intersection,
              coverage_intersection,
              coverage_precision_intersection,
              total_correct_and_hit_intersection,
              memory_precision_intersection,
              supervision_precision_intersection,
              total_correct - total_correct_and_hit_intersection - total_correct_and_no_hit_intersection,
              non_memory_accuracy_intersection,
              total_top_R_hits_intersection,
              top_R_coverage_precision_intersection,
              avg_memory_percentage_int))

# Overall
precision = total_correct / total_samples

print('Recall: ', precision)
