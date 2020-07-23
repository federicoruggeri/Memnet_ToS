"""

@Author:

@Date: 18/09/19

Data-set wrapper and pre-processor

"""

import pandas as pd
from tqdm import tqdm
from utility import preprocessing_utils

from sample_wrappers import TextExampleList, \
    TextKBExampleList, TextExample, TextKBExample, TextKBSupervisionExample


# TODO: add preprocessing filter functions
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, loader_info, filter_names=None, retrieve_label=True):
        self.loader_info = loader_info
        self.filter_names = filter_names if filter_names is not None else preprocessing_utils.filter_methods
        self.retrieve_label = retrieve_label

    def get_train_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.loader_info['inference_labels'] if self.retrieve_label else None

    def get_processor_name(self):
        """Gets the string identifier of the processor."""
        return self.loader_info['data_name']

    @classmethod
    def read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file)
        return df


class TextProcessor(DataProcessor):

    def _get_examples_from_df(self, df, suffix):
        examples = TextExampleList()
        for row_id, row in tqdm(df.iterrows()):
            guid = '{0}-{1}'.format(suffix, row_id)
            text = preprocessing_utils.filter_line(row.text, function_names=self.filter_names)
            if self.retrieve_label:
                label = row[self.loader_info['label']]
            else:
                label = None
            example = TextExample(guid=guid, text=text, label=label)
            examples.append(example)

        return examples

    def get_train_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='train')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='train')

    def get_dev_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='dev')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='dev')

    def get_test_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='test')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='test')


class TextKBProcessor(TextProcessor):

    def __init__(self, **kwargs):
        super(TextKBProcessor, self).__init__(**kwargs)

        kb = self.loader_info['kb']
        self.kb = list(map(lambda item: preprocessing_utils.filter_line(item, function_names=self.filter_names),
                           kb[self.loader_info['label']]))

    def _get_examples_from_df(self, df, suffix):
        examples = TextKBExampleList()
        for row_id, row in tqdm(df.iterrows()):
            guid = '{0}-{1}'.format(suffix, row_id)
            text = preprocessing_utils.filter_line(row.text, function_names=self.filter_names)
            if self.retrieve_label:
                label = row[self.loader_info['label']]
            else:
                label = None
            example = TextKBExample(guid=guid, text=text, label=label, kb=self.kb)
            examples.append(example)

        return examples


class TextKBSupervisionProcessor(TextKBProcessor):

    def __init__(self, partial_supervision_info, **kwargs):
        super(TextKBSupervisionProcessor, self).__init__(**kwargs)
        self.partial_supervision_info = partial_supervision_info

    def _get_examples_from_df(self, df, suffix):
        examples = TextKBExampleList()
        for row_id, row in tqdm(df.iterrows()):
            guid = '{0}-{1}'.format(suffix, row_id)
            text = preprocessing_utils.filter_line(row.text, function_names=self.filter_names)
            if self.retrieve_label:
                label = row[self.loader_info['label']]
            else:
                label = None

            if self.partial_supervision_info['flag']:
                targets = row['{}_targets'.format(self.loader_info['label'])]
            else:
                targets = None
            example = TextKBSupervisionExample(guid=guid, text=text, label=label, kb=self.kb, targets=targets)
            examples.append(example)

        return examples


class ProcessorFactory(object):
    supported_processors = {
        'text_processor': TextProcessor,
        'text_kb_processor': TextKBProcessor,
        'text_kb_supervision_processor': TextKBSupervisionProcessor,
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
        if ProcessorFactory.supported_processors[key]:
            return ProcessorFactory.supported_processors[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))
