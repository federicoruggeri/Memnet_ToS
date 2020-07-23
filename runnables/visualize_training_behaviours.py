"""

Reads TrainingLogger data from pre-trained models and plots it

Uncomment supervision losses and related operators when considering a strong supervised model

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import const_define as cd
from utility.json_utils import load_json


def plot_block(axs, info_block, op_block, info, title, info_colours, best_values=None):
    legend_names = []

    if best_values is not None:
        for key, value in best_values.items():
            axs.axvline(x=value, ymin=0, ymax=2, linestyle='--', c=info_colours[key])
            legend_names.append('{}_best'.format(key))

    for item, item_op in zip(info_block, op_block):
        axs.plot(info[item], c=info_colours[item])
        legend_names.append(item)

    axs.legend(legend_names)
    axs.set_title(title)


def get_cmap(n, name='Set1'):
    return plt.cm.get_cmap(name, n)


model_type = "experimental_basic_memn2n_v2"
test_type = ""
plot_info = [
    [
        # "train_supervision_loss",
        # "val_supervision_loss",
        "train_loss",
        "val_loss",
    ],
    [
        "val_f1_score"
    ]
]
line_op = [
    [
        # np.argmin,
        # np.argmin,
        np.argmin,
        np.argmin
    ],
    [
        np.argmax
    ]
]
filter_best_model = True
retrieval_metric = 'f1_score'

model_path = os.path.join(cd.CV_DIR, model_type, test_type)
info_files = [name for name in os.listdir(model_path) if name.endswith('_info.npy')]

print("Found {0} files".format(len(info_files)))

best_info = {key: op for key, op in [(item, op)
                                     for seq_info, seq_op in zip(plot_info, line_op)
                                     for item, op in zip(seq_info, seq_op)]
             }
best_info_keys = set(list(best_info.keys()))
cmap = get_cmap(len(best_info_keys))
info_colours = {key: cmap(idx) for idx, key in enumerate(best_info_keys)}

if filter_best_model:
    # Retrieve repetition ids if not given
        loaded_val_results = load_json(os.path.join(model_path, cd.JSON_VALIDATION_INFO_NAME))
        metric_val_results = loaded_val_results[retrieval_metric]
        metric_val_results = np.array(metric_val_results)

        if len(metric_val_results.shape) == 1:
            metric_val_results = metric_val_results[np.newaxis, :]

        repetition_ids = np.argmax(metric_val_results, axis=0)

        filtered_names = ['repetition_{0}_fold_{1}'.format(rep_id, fold_idx) for rep_id, fold_idx in zip(repetition_ids,
                                                                                                         np.arange(10))]
        info_files = [name for name in info_files if any([filt_name in name for filt_name in filtered_names])]

        print("Filtered to {0} files".format(len(info_files)))

for filename in info_files:
    curr_info = np.load(os.path.join(model_path, filename), allow_pickle=True).item()
    fig, axs = plt.subplots(len(plot_info), 1)

    best_values = {key: op(curr_info[key]) for key, op in best_info.items()}
    for info_block, op_block, ax in zip(plot_info, line_op, axs):
        plot_block(axs=ax,
                   info_block=info_block,
                   op_block=op_block,
                   info=curr_info,
                   title=filename,
                   info_colours=info_colours,
                   best_values=best_values)

plt.show()