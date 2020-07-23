"""

@Author:

@Date: 22/07/2019

"""

import os

import numpy as np
import pandas as pd

import const_define as cd


def load_file_data(file_path):
    """
    Reads a file line by line.

    :param file_path: path to the file
    :return: list of sentences (string)
    """

    sentences = []

    with open(file_path, 'r') as f:
        for line in f:
            sentences.append(line)

    return sentences


def load_dataset(df_path):
    """
    Loads the ToS dataset given the DataFrame path.

    :param df_path: path to the .csv file
    :return pandas.DataFrame
    """

    df = pd.read_csv(df_path)
    return df


class Task1Loader(object):
    """
    Basic DataFrame-compliant loader for ToS Task 1: unfair clauses classification
    """

    def load(self, dataset, category=None):
        df_path = os.path.join(cd.DATASET_PATHS[dataset], 'dataset.csv')
        df = load_dataset(df_path=df_path)
        df['category'] = [category] * df.shape[0]

        return DataHandle(data=df, num_classes=2, category=category, data_name='{}_task1'.format(dataset),
                          inference_labels=[0, 1])


class Task1KBLoader(object):
    """
    ToS Task 1 loader is extended by loading an external KB.
    This loader is used by Memory Networks.
    """

    def load(self, dataset, category=None):
        df_path = os.path.join(cd.DATASET_PATHS[dataset], 'dataset.csv')
        df = load_dataset(df_path=df_path)
        df['category'] = [category] * df.shape[0]

        if category:
            kb_name = '{}_KB.txt'.format(category)
        else:
            kb_name = 'KB.txt'

        kb_path = os.path.join(cd.KB_DIR, kb_name)
        kb_data = {category: load_file_data(kb_path)}

        return KBDataHandle(data=df,
                            data_name='{}_task1kb'.format(dataset),
                            num_classes=2,
                            category=category,
                            kb_data=kb_data,
                            inference_labels=[0, 1])


class DataLoaderFactory(object):
    supported_data_loaders = {
        'task1': Task1Loader,
        'task1_kb': Task1KBLoader
    }

    @staticmethod
    def factory(cl_type, **kwargs):
        """
        Returns an instance of specified type, built with given, if any, parameters.

        :param cl_type: string name of the classifier class (not case sensitive)
        :param kwargs: additional __init__ parameters
        :return: classifier instance
        """

        key = cl_type.lower()
        if DataLoaderFactory.supported_data_loaders[key]:
            return DataLoaderFactory.supported_data_loaders[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))


class DataHandle(object):
    """
    General dataset wrapper. Additional behaviour can be expressed via attributes.
    """

    def __init__(self, data, data_name, num_classes, inference_labels,
                 category=None, data_keys=['text'], to_upper=True):
        self.data = data
        self.data_name = data_name
        self.num_classes = num_classes
        self.to_upper = to_upper
        self.inference_labels = inference_labels

        if category:
            category = category.upper() if to_upper else category
            self.label = category
        else:
            self.label = 'label'

        self.data_keys = data_keys

    def get_split(self, key_values, key=None, validation_percentage=None):
        if key is None:
            key = 'Unnamed: 0' if self.data.index.name is None else self.data.index.name

        train_df = self.data[np.logical_not(self.data[key].isin(key_values))]
        test_df = self.data[self.data[key].isin(key_values)]

        if validation_percentage is not None:
            validation_amount = int(len(train_df) * validation_percentage)
            train_amount = len(train_df) - validation_amount
            val_df = train_df[train_amount:]
            train_df = train_df[:train_amount]
        else:
            raise NotImplementedError('No validation data right now! ;)')

        return train_df, val_df, test_df

    def get_additional_info(self):
        return {
            'label': self.label,
            'inference_labels': self.inference_labels,
            'data_keys': self.data_keys
        }


class TrainAndTestDataHandle(object):

    def __init__(self, train_data, test_data, data_name, num_classes, inference_labels,
                 label, data_keys, validation_data=None, build_validation=True, pairwise_labels=None):
        self.train_data = train_data
        self.test_data = test_data
        self.data_name = data_name
        self.num_classes = num_classes
        self.inference_labels = inference_labels
        self.label = label
        self.validation_data = validation_data
        self.data_keys = data_keys
        self.pairwise_labels = pairwise_labels
        self.build_validation = build_validation

    def get_data(self, validation_percentage=None):

        train_df = self.train_data
        test_df = self.test_data

        if self.build_validation:
            if validation_percentage is None:
                val_df = self.validation_data
            else:
                validation_amount = int(len(train_df) * validation_percentage)
                train_amount = len(train_df) - validation_amount
                val_df = train_df[train_amount:]
                train_df = train_df[:train_amount]
        else:
            val_df = None

        return train_df, val_df, test_df

    def get_additional_info(self):
        return {'label': self.label,
                'inference_labels': self.inference_labels,
                'pairwise_labels': self.pairwise_labels,
                'data_keys': self.data_keys
                }


class KBDataHandle(DataHandle):
    """
    DataHandle wrapper that stores a KB as additional input.
    """

    def __init__(self, kb_data, **kwargs):
        super(KBDataHandle, self).__init__(**kwargs)
        self.kb_data = kb_data

    def get_additional_info(self):
        info = super(KBDataHandle, self).get_additional_info()
        info['kb'] = self.kb_data
        return info
