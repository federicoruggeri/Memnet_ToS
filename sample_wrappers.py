"""

Simple wrappers for each dataset example

"""

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from utility.log_utils import get_logger
from utility.wrapping_utils import create_int_feature

logger = get_logger(__name__)


# Examples


class TextExample(object):

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

    def get_data(self):
        return self.text


class TextKBExample(TextExample):

    def __init__(self, kb, **kwargs):
        super(TextKBExample, self).__init__(**kwargs)
        self.kb = kb

    def get_data(self):
        return self.text


class TextKBSupervisionExample(TextKBExample):

    def __init__(self, targets, **kwargs):
        super(TextKBSupervisionExample, self).__init__(**kwargs)
        self.targets = targets


class TextExampleList(object):

    def __init__(self):
        self.content = []

    def __iter__(self):
        return self.content.__iter__()

    def append(self, item):
        self.content.append(item)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        return self.content[item]

    def get_data(self):
        return [item.get_data() for item in self.content]


class TextKBExampleList(TextExampleList):

    def get_data(self):
        texts = super(TextKBExampleList, self).get_data()
        return texts + self.content[0].kb


# Features


class Features(object):

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        raise NotImplementedError()

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        raise NotImplementedError()

    @classmethod
    def get_dataset_selector(cls):
        raise NotImplementedError()

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args,
                     has_labels=True, converter_args=None):
        raise NotImplementedError()

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None):
        raise NotImplementedError()


class TextFeatures(Features):

    def __init__(self, text_ids, label_id, is_real_example=True):
        self.text_ids = text_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']
        num_labels = conversion_args['num_labels']

        mappings = dict()
        mappings['text_ids'] = tf.io.FixedLenFeature([max_seq_length], tf.int64)

        if has_labels:
            mappings['label_id'] = tf.io.FixedLenFeature([num_labels], tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):

        features = OrderedDict()
        features['text_ids'] = create_int_feature(feature.text_ids)

        if feature.label_id is not None:
            features['label_id'] = create_int_feature(feature.label_id)

        features['is_real_example'] = create_int_feature([int(feature.is_real_example)])

        return features

    @classmethod
    def get_dataset_selector(cls):

        def _selector(record):
            x = {
                'text': record['text_ids'],
            }

            if 'label_id' in record:
                y = record['label_id']
                return x, y
            else:
                return x

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None):

        label_map = {}
        if label_list is not None or has_labels:
            for (i, label) in enumerate(label_list):
                hot_encoding = [0] * len(label_list)
                hot_encoding[i] = 1
                label_map[label] = hot_encoding

            label_id = label_map[example.label]
        else:
            label_id = None

        tokens = tokenizer.tokenize(example.text)
        text_ids = tokenizer.convert_tokens_to_ids(tokens)

        return text_ids, label_id, label_map

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None):

        if not isinstance(example, TextExample):
            raise AttributeError('Expected TextExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']

        text_ids, label_id, label_map = TextFeatures.convert_example(example=example,
                                                                     label_list=label_list,
                                                                     tokenizer=tokenizer,
                                                                     has_labels=has_labels,
                                                                     converter_args=converter_args)

        # Padding
        text_ids += [0] * (max_seq_length - len(text_ids))
        text_ids = text_ids[:max_seq_length]

        assert len(text_ids) == max_seq_length

        feature = TextFeatures(text_ids=text_ids, label_id=label_id, is_real_example=True)
        return feature


class TextKBFeatures(TextFeatures):

    def __init__(self, kb_ids, kb_mask, **kwargs):
        super(TextKBFeatures, self).__init__(**kwargs)
        self.kb_ids = kb_ids
        self.kb_mask = kb_mask

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_kb_seq_length = conversion_args['max_kb_seq_length']
        max_kb_length = conversion_args['max_kb_length']

        mappings = super(TextKBFeatures, cls).get_mappings(conversion_args, converter_args, has_labels=has_labels)
        mappings['kb_ids'] = tf.io.FixedLenFeature([max_kb_seq_length * max_kb_length], dtype=tf.int64)
        mappings['kb_mask'] = tf.io.FixedLenFeature([max_kb_length], dtype=tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        features = super(TextKBFeatures, cls).get_feature_records(feature)

        features['kb_ids'] = create_int_feature(feature.kb_ids)
        features['kb_mask'] = create_int_feature(feature.kb_mask)

        return features

    @classmethod
    def get_dataset_selector(cls):

        def _selector(record):
            x = {
                'text': record['text_ids'],
                'context': record['kb_ids'],
                'context_mask': record['kb_mask']
            }

            if 'label_id' in record:
                y = record['label_id']
                return x, y
            else:
                return x

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None):

        label_map = {}
        if label_list is not None or has_labels:
            for (i, label) in enumerate(label_list):
                hot_encoding = [0] * len(label_list)
                hot_encoding[i] = 1
                label_map[label] = hot_encoding

            label_id = label_map[example.label]
        else:
            label_id = None

        text_tokens = tokenizer.tokenize(example.text)
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        kb_tokens = list(map(lambda t: tokenizer.tokenize(t), example.kb))
        kb_ids = tokenizer.convert_tokens_to_ids(kb_tokens)

        return text_ids, kb_ids, label_id, label_map

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None):

        if not isinstance(example, TextKBExample):
            raise AttributeError('Expected TextKBExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']
        max_kb_seq_length = conversion_args['max_kb_seq_length']
        max_kb_length = conversion_args['max_kb_length']

        text_ids, kb_ids, label_id, label_map = TextKBFeatures.convert_example(example,
                                                                               label_list,
                                                                               tokenizer,
                                                                               converter_args=converter_args,
                                                                               has_labels=has_labels)

        # Padding
        if len(text_ids) < max_seq_length:
            text_ids += [0] * (max_seq_length - len(text_ids))

        text_ids = text_ids[:max_seq_length]

        kb_mask = np.zeros(len(kb_ids), dtype=np.int64)

        kb_ids = list(map(lambda t: t + [0] * (max_kb_seq_length - len(t)), kb_ids))
        kb_ids = list(map(lambda t: t[:max_kb_seq_length], kb_ids))

        if len(kb_ids) < max_kb_length:
            kb_ids += [[0] * max_kb_seq_length for _ in range(max_kb_length - len(kb_ids))]

        assert len(text_ids) == max_seq_length
        for id_seq in kb_ids:
            assert len(id_seq) == max_kb_seq_length

        # Flatten KB for TFRecord saving
        kb_ids = [item for seq in kb_ids for item in seq]

        feature = TextKBFeatures(text_ids=text_ids, label_id=label_id, kb_ids=kb_ids, is_real_example=True,
                                 kb_mask=kb_mask)
        return feature


class TextKBSupervisionFeatures(TextKBFeatures):

    def __init__(self, positive_indexes=None, negative_indexes=None, mask_indexes=None, **kwargs):
        super(TextKBSupervisionFeatures, self).__init__(**kwargs)
        self.positive_indexes = positive_indexes
        self.negative_indexes = negative_indexes
        self.mask_indexes = mask_indexes

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        mappings = super(TextKBSupervisionFeatures, cls).get_mappings(conversion_args=conversion_args,
                                                                      converter_args=converter_args,
                                                                      has_labels=has_labels)

        if converter_args['partial_supervision']:
            padding_amount = conversion_args['max_supervision_padding']
            mappings['positive_indexes'] = tf.io.FixedLenFeature([padding_amount], dtype=tf.int64)
            mappings['negative_indexes'] = tf.io.FixedLenFeature([padding_amount], dtype=tf.int64)
            mappings['mask_indexes'] = tf.io.FixedLenFeature([padding_amount], dtype=tf.int64)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        features = super(TextKBSupervisionFeatures, cls).get_feature_records(feature, converter_args)

        if converter_args['partial_supervision']:
            features['positive_indexes'] = create_int_feature(feature.positive_indexes)
            features['negative_indexes'] = create_int_feature(feature.negative_indexes)
            features['mask_indexes'] = create_int_feature(feature.mask_indexes)

        return features

    @classmethod
    def get_dataset_selector(cls):

        def _selector(record):
            x = {
                'text': record['text_ids'],
                'context': record['kb_ids'],
                'context_mask': record['kb_mask'],

            }
            if 'positive_indexes' in record:
                x['positive_indexes'] = record['positive_indexes']
                x['negative_indexes'] = record['negative_indexes']
                x['mask_indexes'] = record['mask_indexes']

            if 'label_id' in record:
                y = record['label_id']
                return x, y
            else:
                return x

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None):

        label_map = {}
        if label_list is not None or has_labels:
            for (i, label) in enumerate(label_list):
                hot_encoding = [0] * len(label_list)
                hot_encoding[i] = 1
                label_map[label] = hot_encoding

            label_id = label_map[example.label]
        else:
            label_id = None

        text_tokens = tokenizer.tokenize(example.text)
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        kb_tokens = list(map(lambda t: tokenizer.tokenize(t), example.kb))
        kb_ids = tokenizer.convert_tokens_to_ids(kb_tokens)

        if type(example.targets) is not float and example.targets is not None:
            targets = [int(item) for item in example.targets[1:-1].split(',')]
            target_ids = [1 if idx in targets else 0
                          for idx in range(len(kb_ids))]
        else:
            target_ids = [0]

        return text_ids, kb_ids, target_ids, label_id, label_map

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None):

        if not isinstance(example, TextKBSupervisionExample):
            raise AttributeError('Expected TextKBSupervisionExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']
        max_kb_seq_length = conversion_args['max_kb_seq_length']
        max_kb_length = conversion_args['max_kb_length']
        if converter_args['partial_supervision']:
            max_supervision_padding = conversion_args['max_supervision_padding']

        text_ids, kb_ids, target_ids, label_id, label_map = TextKBSupervisionFeatures.convert_example(example,
                                                                                                      label_list,
                                                                                                      tokenizer,
                                                                                                      has_labels=has_labels,
                                                                                                      converter_args=converter_args)

        # Padding
        if len(text_ids) < max_seq_length:
            text_ids += [0] * (max_seq_length - len(text_ids))

        text_ids = text_ids[:max_seq_length]

        kb_mask = np.zeros(len(kb_ids), dtype=np.int64)

        kb_ids = list(map(lambda t: t + [0] * (max_kb_seq_length - len(t)), kb_ids))
        kb_ids = list(map(lambda t: t[:max_kb_seq_length], kb_ids))

        if len(kb_ids) < max_kb_length:
            kb_ids += [[0] * max_kb_seq_length for _ in range(max_kb_length - len(kb_ids))]

        assert len(text_ids) == max_seq_length
        for id_seq in kb_ids:
            assert len(id_seq) == max_kb_seq_length

        # Flatten KB for TFRecord saving
        kb_ids = [item for seq in kb_ids for item in seq]

        if converter_args['partial_supervision']:
            positive_indexes = np.zeros(max_supervision_padding, dtype=np.int32)
            negative_indexes = np.zeros(max_supervision_padding, dtype=np.int32)
            mask_indexes = np.zeros(max_supervision_padding, dtype=np.int32)

            target_ids += [0] * (max_kb_length - len(target_ids))
            target_ids = np.array(target_ids)
            target_amount = sum(target_ids)
            target_positions = np.where(target_ids)[0]
            non_target_positions = np.where(target_ids == 0)[0]
            non_target_amount = len(non_target_positions)

            if target_amount > 0:

                # Mask padding positive values
                mask_indexes[:target_amount] = np.ones(target_amount)

                positive_indexes[:target_amount] = target_positions
                remaining_pos = max_supervision_padding - target_amount
                if remaining_pos > 0:
                    to_add = np.random.choice(a=target_positions, size=remaining_pos, replace=True)
                    positive_indexes[target_amount:] = to_add

                negative_indexes[:non_target_amount] = non_target_positions
                remaining_neg = max_supervision_padding - non_target_amount
                if remaining_neg > 0:
                    to_add = np.random.choice(a=non_target_positions, size=remaining_neg, replace=True)
                    negative_indexes[-remaining_neg:] = to_add
            else:
                mask_indexes = np.zeros(shape=max_supervision_padding, dtype=np.int32)
        else:
            positive_indexes = None
            negative_indexes = None
            mask_indexes = None

        feature = TextKBSupervisionFeatures(text_ids=text_ids, label_id=label_id, kb_ids=kb_ids, is_real_example=True,
                                            kb_mask=kb_mask, positive_indexes=positive_indexes,
                                            negative_indexes=negative_indexes, mask_indexes=mask_indexes)
        return feature


class FeatureFactory(object):
    supported_features = {
        'text_features': TextFeatures,
        'text_kb_features': TextKBFeatures,
        'text_kb_supervision_features': TextKBSupervisionFeatures,
    }
