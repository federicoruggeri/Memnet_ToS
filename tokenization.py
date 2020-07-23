# coding=utf-8
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tokenization classes implementation.
The file is forked from:
https://github.com/google-research/bert/blob/master/tokenization.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os

import numpy as np
import tensorflow as tf
from utility.embedding_utils import build_embeddings_matrix, load_embedding_model
from utility.log_utils import get_logger

logger = get_logger(__name__)


class Tokenizer(object):

    def __init__(self, build_embedding_matrix=False, embedding_dimension=None,
                 embedding_model_type=None):
        if build_embedding_matrix:
            assert embedding_model_type is not None
            assert embedding_dimension is not None and type(embedding_dimension) == int

        self.build_embedding_matrix = build_embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.embedding_model = None
        self.embedding_matrix = None
        self.vocab = None

    def build_vocab(self, data, **kwargs):
        raise NotImplementedError()

    def initalize_with_vocab(self, vocab):
        raise NotImplementedError()

    def tokenize(self, text):
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens):
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids):
        raise NotImplementedError()

    def get_info(self):
        return {
            'build_embedding_matrix': self.build_embedding_matrix,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model_type': self.embedding_model_type,
            'embedding_matrix': self.embedding_matrix,
            'embedding_model': self.embedding_model,
            'vocab_size': len(self.vocab) + 1,
            'vocab': self.vocab
        }

    def save_info(self, filepath, prefix=None):
        filepath = os.path.join(filepath, 'tokenizer_info_{}.npy'.format(prefix))
        np.save(filepath, self.get_info())

    def show_info(self, info=None):

        info = info if info is not None else self.get_info()
        info = {key: value for key, value in info.items() if key != 'vocab'}

        logger.info('Tokenizer info: {}'.format(info))

    @staticmethod
    def load_info(filepath, prefix=None):
        filepath = os.path.join(filepath, 'tokenizer_info_{}.npy'.format(prefix))
        return np.load(filepath, allow_pickle=True).item()


class KerasTokenizer(Tokenizer):

    def __init__(self, tokenizer_args=None, **kwargs):
        super(KerasTokenizer, self).__init__(**kwargs)

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args

        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args

    def build_vocab(self, data, **kwargs):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)
        self.vocab = self.tokenizer.word_index

        if self.build_embedding_matrix:
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            self.embedding_matrix = build_embeddings_matrix(vocab_size=len(self.vocab) + 1,
                                                            embedding_model=self.embedding_model,
                                                            embedding_dimension=self.embedding_dimension,
                                                            word_to_idx=self.vocab)

    def initalize_with_vocab(self, vocab):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.word_index = vocab

        if self.build_embedding_matrix:
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            self.embedding_matrix, self.vocab = build_embeddings_matrix(vocab_size=len(vocab) + 1,
                                                                        embedding_model=self.embedding_model,
                                                                        embedding_dimension=self.embedding_dimension,
                                                                        word_to_idx=vocab)

            self.tokenizer.word_index = self.vocab

    def get_info(self):
        info = super(KerasTokenizer, self).get_info()
        info['vocab_size'] = len(self.vocab) + 1

        return info

    def tokenize(self, text):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)


class TokenizerFactory(object):
    supported_tokenizers = {
        'keras_tokenizer': KerasTokenizer,
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
        if TokenizerFactory.supported_tokenizers[key]:
            return TokenizerFactory.supported_tokenizers[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))
