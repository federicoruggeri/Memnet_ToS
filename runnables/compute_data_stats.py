"""

Shows sentence length stats for given dataset

"""

import numpy as np
from data_loader import Task1Loader
from utility.preprocessing_utils import filter_line

loader = Task1Loader()

data_handle = loader.load(dataset='tos_100', category=None)
dataset = data_handle.data
filter = 'TER'

if filter:
    print('Using filter: ', filter)
    texts = dataset[dataset[filter] == 1].text.values
else:
    texts = dataset.text.values

print("Total texts: ", texts.shape)

texts = list(map(lambda sent: filter_line(sent), texts))

lengths = [len(sent.split()) for sent in texts]

print("Word Level Statistics")

print('Mean sentence size: ', np.mean(lengths))
print('Max sentence size: ', np.max(lengths))
print('Min sentence size: ', np.min(lengths))
print('99% sentence size quantile: ', np.quantile(lengths, q=0.99))
print('95% sentence size quantile: ', np.quantile(lengths, q=0.95))

