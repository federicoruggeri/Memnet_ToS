"""

@Author:

@Date: 10/01/2019


"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import namedtuple

from utility.json_utils import load_json
import functools

MemoryStats = namedtuple('MemoryStats',
                         'usage'
                         ' hits'
                         ' correct'
                         ' correct_and_hit'
                         ' correct_and_no_hit'
                         ' samples'
                         ' top_R_hits'
                         ' avg_memory_percentage')

MemoryMetrics = namedtuple('MemoryMetrics',
                           'memory_usage'
                           ' coverage'
                           ' coverage_precision'
                           ' recall'
                           ' memory_precision'
                           ' supervision_precision'
                           ' non_memory_precision'
                           ' top_R_coverage_precision'
                           ' avg_memory_percentage')


def plot_attention_contour_v2(attention, name=None):
    """
    Plots Memory Networks attention distribution as a contour plot.
    """

    name = name or ""

    samples = attention.shape[0]
    memory_size = attention.shape[2]

    figures = []

    for hop in range(attention.shape[1]):
        # for hop in range(1):
        xpos = np.arange(memory_size)
        ypos = np.arange(samples)
        xpos, ypos = np.meshgrid(xpos, ypos)
        zpos = np.reshape(attention[:, hop, :], newshape=(samples, memory_size))

        fig, ax = plt.subplots(1, 1)
        ax.set_title('{0} Hop {1}'.format(name, hop + 1))

        CS = ax.contourf(xpos, ypos, zpos, 15, levels=np.arange(0, np.max(attention), np.max(attention) * 0.001),
                         cmap=cm.coolwarm)
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
        fig.add_axes(cax)
        cbar = plt.colorbar(CS, cax=cax, orientation='horizontal')

        # Show grid
        ax.grid(True)
        ax.set_xticks(np.arange(memory_size))

        # Labeling
        ax.set_xlabel('Memory slots')
        ax.set_ylabel('Samples')

        figures.append(fig)

    return figures


def memory_histogram(attention_weights, model_path, filter_unfair=False,
                     true_values=None, memory_labels=None, attention_mode='softmax',
                     show_max_only=True):
    """
    Plots Memory Network attention distribution as histogram.
    """

    def get_predictions(attention_values, attention_mode):
        if attention_mode == 'softmax':
            return np.argmax(attention_values, axis=2).ravel()[:, np.newaxis]
        elif attention_mode == 'sigmoid':
            if not show_max_only:
                attention_values = np.round(attention_values).astype(np.int32)
                return np.where(attention_values)[-1][:, np.newaxis]
            else:
                # attention_values = np.round(attention_values).astype(np.int32)
                filtered_attention_values = []
                for mem_values in attention_values:
                    mem_indexes = [np.argmax(values) for values in mem_values if np.max(values) >= 0.5]
                    if mem_indexes:
                        filtered_attention_values.append(mem_indexes)
                return filtered_attention_values
        else:
            raise RuntimeError('Invalid attention mode! Got: {}'.format(attention_mode))

    memory_size = None
    counter = None
    total_unfair = 0
    for idx, weight_name in enumerate(attention_weights):
        sub_path = os.path.join(model_path, weight_name)
        fold_name = weight_name.split('fold')[1].split('_')[1]

        loaded_weights = load_json(sub_path)

        if idx == 0:
            counter = {slot: 0 for slot in range(loaded_weights.shape[-1])}

        if memory_size is None:
            memory_size = loaded_weights.shape[2]

        total_unfair += np.sum(true_values[int(fold_name)])

        if filter_unfair:
            loaded_weights = loaded_weights[np.argwhere(true_values[int(fold_name)]).ravel()]

        # selected_memories = np.argmax(loaded_weights, axis=2)
        selected_memories = get_predictions(loaded_weights, attention_mode)
        if len(selected_memories) > 0:
            selected_memories = [list(set(item)) for item in selected_memories]
            flat_selections = [item for seq in selected_memories for item in seq]

            fold_counts = Counter(flat_selections)
            for key, item in fold_counts.items():
                counter[key] += item

            print('Fold {0} counts: {1}'.format(fold_name, fold_counts))
        else:
            print('Fold {0} counts: {1}'.format(fold_name, {key: 0 for key in range(loaded_weights.shape[-1])}))

    print('Distinct memories uses: ', counter)
    print("'Total unfair: ", total_unfair)

    fig, ax = plt.subplots()
    ax.set_title('Memory usage', fontsize=32)

    memory_indexes = np.arange(len(counter)) + 1
    used_memories = np.array(list(counter.keys()))
    if memory_labels is not None:
        used_memories = [memory_labels[item] for item in used_memories]
    counter_values = [counter[key] for key in memory_indexes - 1]

    ax.bar(memory_indexes, counter_values, align='center')

    ax.set_xlabel('Memory slots', fontsize=32)
    ax.set_ylabel('Selections amount', fontsize=32)
    ax.set_xticks(memory_indexes)
    ax.set_xticklabels(used_memories)
    ax.tick_params(axis='both', labelsize=24)

    for idx, value in enumerate(counter_values):
        if value > 0:
            ax.text(idx + 0.88, value + 3, str(value), color='k', fontweight='bold', fontsize=24)

    if memory_labels is not None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(75)

    return fig


def compute_trajectories_distribution(attention):
    """
    Counts memory selection (hard-attention) trajectories for Memory Network models.
    """

    attention = np.argmax(attention, axis=2)

    # Convert to string by concatenating
    str_operation = lambda seq: '-'.join(seq.astype(np.str))
    attention_str = np.apply_along_axis(func1d=str_operation, axis=1, arr=attention)

    counter = Counter(attention_str)
    return counter


def visualize_unfair_trajectories(unfair_info, fold_name, key, aggregate=False):
    """
    Counts memory selection (hard-attention) trajectories for Memory Network models.
    Memory trajectories are restricted to positive samples only.
    """

    str_op = lambda seq: '-'.join(seq.astype(np.str))
    if not aggregate:
        info_values = [item['attention'] for _, item in unfair_info.items()]
        info_values = np.array(info_values)
        info_values_str = np.apply_along_axis(func1d=str_op, axis=1, arr=info_values)
    else:
        info_values = [np.unique(item['attention']) for _, item in unfair_info.items()]
        info_values_str = [str_op(item) if len(item) > 1 else str(item[0]) for item in info_values]
    counter = Counter(info_values_str)
    print('{0} trajectories distribution fold {1}: {2}'.format(key, fold_name, counter))


def plot_attention_trajectories(unfair_info, fold_name, key):
    """
    Shows memory trajectories graphs, where nodes are selected memories (hard attention) along with
    their count (number of samples that use that memory for a given memory lookup operation, i.e. hop)
    """

    info_values = [item['attention'] for _, item in unfair_info.items()]
    info_values = np.array(info_values)
    info_values = info_values + 1

    fig, ax = plt.subplots()
    ax.set_title('{0}_trajectories_fold_{1}'.format(key, fold_name))

    hops = np.arange(len(info_values[0])) + 1

    for traj in info_values:
        ax.plot(traj, hops, linewidth=1, marker='D', markersize=6, linestyle='dashed', c='r')

    ax.set_xlabel('Memory slots')
    ax.set_ylabel('Memory iterations')
    ax.set_xticks(np.arange(np.max(info_values)) + 1)

    # Annotate points
    for hop in hops:
        hop_counter = Counter(info_values[:, hop - 1])
        for value, count in hop_counter.items():
            ax.annotate(count, (value + 0.2, hop), size=16)


def plot_memory_selections(attention, width=0.35, name=None):
    """
    Plots Memory Networks attention distribution as a bar plot.
    """

    name = name or ""

    # [#samples, #hops, memory_size]
    assert len(attention.shape) == 3

    memory_size = attention.shape[2]
    hops = attention.shape[1]

    # [#samples, #hops]
    attention = np.argmax(attention, axis=2)

    hop_counts = []
    for hop in range(hops):
        c = Counter(attention[:, hop])
        hop_values = list(c.values())
        hop_keys = list(c.keys())
        adjusted = np.zeros(memory_size, dtype=np.int32)
        for key, value in zip(hop_keys, hop_values):
            adjusted[key] = value
        hop_counts.append(adjusted)

    hop_counts = np.array(hop_counts)
    memory_indexes = np.arange(memory_size) + 1

    fig, ax = plt.subplots(1, 1)
    ax.set_title('{0}'.format(name))

    for hop in range(hops):
        if hop == 0:
            ax.bar(memory_indexes, hop_counts[hop], width=width)
        else:
            ax.bar(memory_indexes, hop_counts[hop], width=width, bottom=np.sum(hop_counts[:hop], axis=0))

    ax.set_xlabel('Memory slots')
    ax.set_ylabel('Selections amount')
    ax.set_xticks(memory_indexes)

    return fig


def show_target_coverage(attention, test_df, category, fold_name, predictions, attention_mode='softmax',
                         verbose=0, R=3):
    """
    Given memory targets, computes the memory target coverage. In particular, for each sample, the method
    shows: (i) selected memories (hard attention), (ii) target memories, (iii) predicted label, (iv) true label.
    """

    def get_hits(target, predicted):
        target = set(target)
        predicted = set(predicted)
        intersection = predicted.intersection(target)
        hits = len(intersection)
        missed = target.difference(intersection)
        others = predicted.difference(intersection)
        return hits, missed, others

    def get_predictions(attention_values, attention_mode):
        if attention_mode == 'softmax':
            return np.argmax(attention_values, axis=1).ravel().tolist()
        elif attention_mode == 'sigmoid':
            return np.where(attention_values >= 0.5)[-1].tolist()
        else:
            raise RuntimeError('Invalid attention mode! Got: {}'.format(attention_mode))

    def get_top_K_predictions(attention_values, attention_mode, K):

        if attention_mode == 'softmax':
            best_indexes = np.argsort(attention_values, axis=1)[:, ::-1]
            best_indexes = best_indexes[:, :K]
            return best_indexes.ravel().tolist()
        elif attention_mode == 'sigmoid':
            best_indexes = np.argsort(attention_values, axis=1)[:, ::-1]
            valid_mask = attention_values < 0.5
            sorted_valid_mask = np.take_along_axis(valid_mask, best_indexes, axis=1)
            best_indexes[sorted_valid_mask] = -1
            best_indexes = best_indexes[:, :K]
            return best_indexes.ravel().tolist()
        else:
            raise RuntimeError('Invalid attention mode! Got: {}'.format(attention_mode))

    unfair_data = test_df[test_df[category] == 1]

    # Get targets attention weights
    total_usage = 0
    total_hits = 0
    total_top_R_hits = [0] * R
    total_correct = 0
    total_correct_and_hit = 0
    total_correct_and_no_hit = 0
    avg_memory_percentage = 0
    for row_id, row in unfair_data.iterrows():
        target_id = row_id
        target_values = row['{}_targets'.format(category)]
        target_values = [int(item) for item in target_values[1:-1].split(',')]
        predicted_label = predictions[target_id]
        true_label = 1
        predicted_values = get_predictions(attention[target_id], attention_mode)

        hits, missed, others = get_hits(target=target_values, predicted=predicted_values)

        if len(predicted_values) > 0:
            total_usage += 1

            top_R_predictions = get_top_K_predictions(attention[target_id], attention_mode=attention_mode, K=R)
            total_top_R_hits = [count + 1 if len(set(top_R_predictions[:idx + 1]).intersection(set(target_values)))
                                else count
                                for idx, count in enumerate(total_top_R_hits)]

            avg_memory_percentage += len(set(predicted_values)) / attention.shape[-1]

            if hits > 0:
                total_hits += 1
                total_correct_and_hit += int(predicted_label == true_label)
            else:
                total_correct_and_no_hit += int(predicted_label == true_label)

        total_correct += int(predicted_label == true_label)

        if verbose:
            print('*' * 20)
            print('Sample ', target_id, ': ', row.text)
            print('Hits: ', hits,
                  ' Missed: ', missed,
                  ' Others: ', others,
                  'Predicted Label: ', predicted_label,
                  'True Label: ', true_label)
            print('*' * 20)

    total_top_R_hits = np.array(total_top_R_hits)

    if verbose:
        memory_usage = total_usage / unfair_data.shape[0]
        coverage = total_hits / unfair_data.shape[0]
        accuracy = total_correct / unfair_data.shape[0]
        supervision_accuracy = total_correct_and_hit / unfair_data.shape[0]
        non_memory_accuracy = (total_correct - total_correct_and_hit - total_correct_and_no_hit) / \
                              unfair_data.shape[0]
        try:
            coverage_precision = total_hits / total_usage
            top_R_coverage_precision = total_top_R_hits / total_hits
            memory_accuracy = total_correct_and_hit / total_usage
            fold_memory_percentage = avg_memory_percentage / total_usage
        except ZeroDivisionError:
            coverage_precision = None
            top_R_coverage_precision = None
            memory_accuracy = None
            fold_memory_percentage = None

        print('Fold {0} stats:\n'
              ' Memory usage: {1}/{2} ({3})\n'
              ' Coverage (correct memory over all samples): {4}/{2} ({5})\n'
              ' Coverage precision (correct memory over memory usage): {4}/{1} ({6})\n'
              ' Top R coverage precision (correct memory and no other memories over memory usage): {14}/{4} ({15})',
              ' Precision (correct over all samples): {7}/{2} ({8})\n'
              ' Memory precision (correct and correct memory over memory usage: {9}/{1} ({10})\n'
              ' Supervision precision (correct and correct memory over all samples): {9}/{2} ({11})\n'
              ' Non-memory precision (correct and no memory over all samples): {12}/{2} ({13})\n'
              ' Average memory selection amount (percentage): {16}\n'
              .format(fold_name,
                      total_usage,
                      unfair_data.shape[0],
                      memory_usage,
                      total_hits,
                      coverage,
                      coverage_precision,
                      total_correct,
                      accuracy,
                      total_correct_and_hit,
                      memory_accuracy,
                      supervision_accuracy,
                      total_correct - total_correct_and_hit - total_correct_and_no_hit,
                      non_memory_accuracy,
                      total_top_R_hits,
                      top_R_coverage_precision,
                      fold_memory_percentage))

    stats = MemoryStats(total_usage, total_hits, total_correct, total_correct_and_hit, total_correct_and_no_hit,
                        unfair_data.shape[0], total_top_R_hits, avg_memory_percentage)
    return stats


def show_voting_coverage(K_attentions, test_df, category, fold_name, K_predictions,
                         attention_mode='softmax', verbose=0, R=3):
    # K_attention: [K, #samples, hops, mem_size]
    # K_predictions: [#samples, K]

    K = K_predictions.shape[1]
    threshold = np.ceil(K / 2)

    assert K % 2 != 0

    def get_voting_label(K_predictions):
        if np.sum(K_predictions) >= threshold:
            return 1, np.where(K_predictions)[0]
        else:
            return 0, np.where(K_predictions == 0)[0]

    def get_attention(attention_values, attention_mode):
        if attention_mode == 'softmax':
            return np.argmax(attention_values, axis=1).ravel().tolist()
        elif attention_mode == 'sigmoid':
            return np.where(attention_values >= 0.5)[-1].tolist()
        else:
            raise RuntimeError('Invalid attention mode! Got: {}'.format(attention_mode))

    def get_hits(target, predicted):
        target = set(target)
        predicted = set(predicted)
        intersection = predicted.intersection(target)
        hits = len(intersection)
        missed = target.difference(intersection)
        others = predicted.difference(intersection)
        return hits, missed, others

    def get_top_K_predictions(attention_values, attention_mode, K):

        if attention_mode == 'softmax':
            best_indexes = np.argsort(attention_values, axis=1)[:, ::-1]
            best_indexes = best_indexes[:, :K]
            return best_indexes.ravel().tolist()
        elif attention_mode == 'sigmoid':
            best_indexes = np.argsort(attention_values, axis=1)[:, ::-1]
            valid_mask = attention_values < 0.5
            sorted_valid_mask = np.take_along_axis(valid_mask, best_indexes, axis=1)
            best_indexes[sorted_valid_mask] = -1
            best_indexes = best_indexes[:, :K]
            return best_indexes.ravel().tolist()
        else:
            raise RuntimeError('Invalid attention mode! Got: {}'.format(attention_mode))

    unfair_data = test_df[test_df[category] == 1]

    # Get targets attention weights
    total_usage_union, total_hits_union, \
    total_correct_and_hit_union, total_correct_and_no_hit_union, \
    total_exclusive_hits_union, avg_memory_percentage_union = 0, 0, 0, 0, 0, 0
    total_top_R_hits_union = [0] * R

    total_usage_intersection, total_hits_intersection, \
    total_correct_and_hit_intersection, total_correct_and_no_hit_intersection, \
    total_exclusive_hits_intersection, avg_memory_percentage_int = 0, 0, 0, 0, 0, 0
    total_top_R_hits_intersection = [0] * R

    total_correct = 0

    for row_id, row in unfair_data.iterrows():
        target_id = row_id
        target_values = row['{}_targets'.format(category)]
        target_values = [int(item) for item in target_values[1:-1].split(',')]
        predicted_label, winning_members = get_voting_label(K_predictions[target_id])
        true_label = 1
        members_attentions = [get_attention(K_attentions[winner][target_id], attention_mode) for winner in
                              winning_members]

        union_attention = np.concatenate(members_attentions, axis=-1)
        intersection_attention = functools.reduce(lambda a, b: set(a).intersection(set(b)), members_attentions)

        union_hits, union_missed, union_others = get_hits(target=target_values, predicted=union_attention)
        intersection_hits, intersection_missed, intersection_others = get_hits(target=target_values,
                                                                               predicted=intersection_attention)

        if len(union_attention) > 0:
            total_usage_union += 1

            avg_memory_percentage_union += len(set(union_attention)) / K_attentions[0].shape[-1]

            # Top R
            union_attention_values = np.concatenate([K_attentions[winner][target_id] for winner in winning_members],
                                                    axis=-1)
            top_R_predictions_union = get_top_K_predictions(union_attention_values, attention_mode=attention_mode, K=R)
            top_R_predictions_union = [item % K_attentions[0].shape[-1] if item != -1 else item for item in top_R_predictions_union]
            total_top_R_hits_union = [
                count + 1 if len(set(top_R_predictions_union[:idx + 1]).intersection(set(target_values)))
                else count
                for idx, count in enumerate(total_top_R_hits_union)]

            if union_hits > 0:
                total_correct_and_hit_union += int(predicted_label == true_label)
                total_hits_union += 1

                if len(set(union_attention).difference(target_values)) == 0:
                    total_exclusive_hits_union += 1
            else:
                total_correct_and_no_hit_union += int(predicted_label == true_label)

        if len(intersection_attention) > 0:
            total_usage_intersection += 1

            avg_memory_percentage_int += len(set(intersection_attention)) / K_attentions[0].shape[-1]

            # Top R
            intersection_attention_values = [np.take_along_axis(K_attentions[winner][target_id],
                                                                np.array(list(intersection_attention),
                                                                         dtype=np.int32)[np.newaxis, :],
                                                                axis=1)
                                             for winner in winning_members]
            intersection_attention_values = np.stack(intersection_attention_values, axis=-1)
            intersection_attention_values = np.mean(intersection_attention_values, axis=-1)
            top_R_predictions_intersection = get_top_K_predictions(intersection_attention_values,
                                                                   attention_mode=attention_mode, K=R)
            total_top_R_hits_intersection = [
                count + 1 if len(set(top_R_predictions_intersection[:idx + 1]).intersection(set(target_values)))
                else count
                for idx, count in enumerate(total_top_R_hits_intersection)]

            if intersection_hits > 0:
                total_correct_and_hit_intersection += int(predicted_label == true_label)
                total_hits_intersection += 1

                if len(set(intersection_attention).difference(target_values)) == 0:
                    total_exclusive_hits_intersection += 1
            else:
                total_correct_and_no_hit_intersection += int(predicted_label == true_label)

        total_correct += int(predicted_label == true_label)

        if verbose:
            print('*' * 20)
            print('Sample ', target_id, ': ', row.text)
            print('Hits (Union): ', union_hits,
                  ' Missed (Union): ', union_missed,
                  ' Others (Union): ', union_others,
                  'Predicted Label: ', predicted_label,
                  'True Label: ', true_label)
            print('Hits (Intersection): ', intersection_hits,
                  ' Missed (Intersection): ', intersection_missed,
                  ' Others (Intersection): ', intersection_others,
                  'Predicted Label: ', predicted_label,
                  'True Label: ', true_label)
            print('*' * 20)

    union_stats = MemoryStats(total_usage_union, total_hits_union, total_correct, total_correct_and_hit_union,
                              total_correct_and_no_hit_union, unfair_data.shape[0], total_top_R_hits_union,
                              avg_memory_percentage_union)
    intersection_stats = MemoryStats(total_usage_intersection, total_hits_intersection, total_correct,
                                     total_correct_and_hit_intersection,
                                     total_correct_and_no_hit_intersection, unfair_data.shape[0],
                                     total_top_R_hits_intersection, avg_memory_percentage_int)

    return union_stats, intersection_stats
