import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from sample_wrappers import Features, TextExampleList, TextKBExampleList, FeatureFactory
from utility.log_utils import get_logger

logger = get_logger(__name__)


class BaseConverter(object):

    def __init__(self, feature_class, max_tokens_limit=None, **kwargs):
        self.feature_class = FeatureFactory.supported_features[feature_class]
        self.max_tokens_limit = max_tokens_limit if max_tokens_limit else 100000

    def convert_data(self, examples, tokenizer, label_list, output_file, checkpoint=None,
                     is_training=False, has_labels=True):

        assert issubclass(self.feature_class, Features)

        if is_training:
            logger.info('Retrieving training set info...this may take a while...')
            self.training_preparation(examples=examples,
                                      label_list=label_list,
                                      tokenizer=tokenizer)

        writer = tf.io.TFRecordWriter(output_file)

        for ex_index, example in enumerate(tqdm(examples, leave=True, position=0)):
            if checkpoint is not None and ex_index % checkpoint == 0:
                logger.info('Writing example {0} of {1}'.format(ex_index, len(examples)))

            feature = self.feature_class.from_example(example,
                                                      label_list,
                                                      tokenizer=tokenizer,
                                                      has_labels=has_labels,
                                                      conversion_args=self.get_conversion_args(),
                                                      converter_args=self.get_instance_args())

            features = self.feature_class.get_feature_records(feature, self.get_instance_args())

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()

    def get_instance_args(self):
        return {
            "max_tokens_limit": self.max_tokens_limit
        }

    def get_conversion_args(self):
        return {
            'max_seq_length': self.max_seq_length,
            'label_map': self.label_map,
            'num_labels': self.num_labels,
            'feature_class': self.feature_class
        }

    # Computing max text length
    # TODO: we can do better here by defining attribute-defining functions: one for each conversion_args key
    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, TextExampleList)

        max_seq_length = None
        num_labels = None
        for example in tqdm(examples):

            features = self.feature_class.convert_example(example,
                                                          label_list,
                                                          tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args())
            text_ids, label_id, label_map = features

            features_ids_len = len(text_ids)

            if label_id is not None:
                num_labels = len(label_id)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.label_map = label_map

    def save_conversion_args(self, filepath, prefix=None):
        filepath = os.path.join(filepath, 'converter_info_{}.npy'.format(prefix))
        np.save(filepath, self.get_conversion_args())

    @staticmethod
    def load_conversion_args(filepath, prefix=None):
        filepath = os.path.join(filepath, 'converter_info_{}.npy'.format(prefix))
        return np.load(filepath, allow_pickle=True).item()


class KBConverter(BaseConverter):

    def get_conversion_args(self):
        conversion_args = super(KBConverter, self).get_conversion_args()
        conversion_args['max_kb_length'] = self.max_kb_length
        conversion_args['max_kb_seq_length'] = self.max_kb_seq_length

        return conversion_args

    # Computing max text length and kb info
    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, TextExampleList)

        max_seq_length = None
        max_kb_length = None
        max_kb_seq_length = None
        num_labels = None
        for example in tqdm(examples):

            is_valid_example = True

            features = self.feature_class.convert_example(example,
                                                          label_list,
                                                          tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args())
            text_ids, kb_ids, label_id, label_map = features
            features_ids_len = len(text_ids)
            features_kb_len = len(kb_ids)
            features_kb_text_len = max([len(item) for item in kb_ids])

            if label_id is not None:
                num_labels = len(label_id)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

            if features_ids_len > self.max_tokens_limit:
                is_valid_example = False

            if max_kb_length is None and is_valid_example:
                max_kb_length = features_kb_len
            elif max_kb_length < features_kb_len and is_valid_example:
                max_kb_length = features_kb_len

            if max_kb_seq_length is None and is_valid_example:
                max_kb_seq_length = features_kb_text_len
            elif max_kb_seq_length < features_kb_text_len and is_valid_example:
                max_kb_seq_length = features_kb_text_len

        self.max_seq_length = max_seq_length
        self.max_kb_length = max_kb_length
        self.max_kb_seq_length = min(max_kb_seq_length, self.max_tokens_limit)
        self.num_labels = num_labels
        self.label_map = label_map


class KBSupervisionConverter(KBConverter):

    def __init__(self, partial_supervision_info, **kwargs):
        super(KBSupervisionConverter, self).__init__(**kwargs)
        self.partial_supervision_info = partial_supervision_info

    def get_conversion_args(self):
        conversion_args = super(KBSupervisionConverter, self).get_conversion_args()

        if self.partial_supervision_info['flag']:
            conversion_args['max_supervision_padding'] = self.max_supervision_padding

        return conversion_args

    def get_instance_args(self):
        instance_args = super(KBSupervisionConverter, self).get_instance_args()
        instance_args['partial_supervision'] = self.partial_supervision_info['flag']
        return instance_args

    # Computing max text length, kb_info and supervision targets
    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, TextKBExampleList)

        max_seq_length = None
        max_kb_length = None
        max_kb_seq_length = None
        max_supervision_padding = None
        num_labels = None

        for example in tqdm(examples):

            is_valid_example = True

            features = self.feature_class.convert_example(example,
                                                          label_list,
                                                          tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args())
            text_ids, kb_ids, target_ids, label_id, label_map = features
            features_ids_len = len(text_ids)
            features_kb_len = len(kb_ids)
            features_kb_text_len = max([len(item) for item in kb_ids])

            target_amount = len(np.where(np.asanyarray(target_ids))[0])
            non_target_amount = len(np.where(np.asanyarray(target_ids) == 0)[0])
            features_supervision_padding = max(target_amount, non_target_amount)

            if label_id is not None:
                num_labels = len(label_id)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

            if features_ids_len > self.max_tokens_limit:
                is_valid_example = False

            if max_kb_length is None and is_valid_example:
                max_kb_length = features_kb_len
            elif max_kb_length < features_kb_len and is_valid_example:
                max_kb_length = features_kb_len

            if max_kb_seq_length is None and is_valid_example:
                max_kb_seq_length = features_kb_text_len
            elif max_kb_seq_length < features_kb_text_len and is_valid_example:
                max_kb_seq_length = features_kb_text_len

            if max_supervision_padding is None and is_valid_example:
                max_supervision_padding = features_supervision_padding
            elif max_supervision_padding < features_supervision_padding and is_valid_example:
                max_supervision_padding = features_supervision_padding

        self.max_seq_length = max_seq_length
        self.max_kb_length = max_kb_length
        self.max_kb_seq_length = min(max_kb_seq_length, self.max_tokens_limit)
        self.max_supervision_padding = max_supervision_padding
        self.num_labels = num_labels
        self.label_map = label_map


class DataConverterFactory(object):
    supported_data_converters = {
        'base_converter': BaseConverter,
        'kb_converter': KBConverter,
        'kb_supervision_converter': KBSupervisionConverter
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
        if DataConverterFactory.supported_data_converters[key]:
            return DataConverterFactory.supported_data_converters[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))
