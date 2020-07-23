"""

@Author:

@Date: 29/11/2018

"""

import codecs
import os

import gensim
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

import const_define as cd
from utility.log_utils import get_logger

logger = get_logger(__name__)


def build_embeddings_matrix(vocab_size, word_to_idx, embedding_model, embedding_dimension=300):
    """
    Builds embedding matrix given the pre-loaded embedding model.
    """

    embedding_matrix = np.zeros((vocab_size, embedding_dimension))

    for word, i in word_to_idx.items():
        try:
            if type(word) != str:
                word = str(word)
            embedding_vector = embedding_model[word]

            # Check for any possible invalid term
            if embedding_vector.shape[0] != embedding_dimension:
                embedding_vector = np.zeros(embedding_dimension)
        except KeyError:
            embedding_vector = np.zeros(embedding_dimension)

        embedding_matrix[i] = embedding_vector

    return embedding_matrix


def pad_data(data, padding_length=None, padding='post', dtype=np.int32):
    """
    Pads input data with zeros.
    """

    padded = pad_sequences(data, maxlen=padding_length, padding=padding, dtype=dtype)
    return padded


def _load_fasttext(path):
    """
    Loads FastText embedding model: https://fasttext.cc/
    """

    embeddings_model = {}

    f = codecs.open(path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_model[word] = coefs
    f.close()

    return embeddings_model


def _load_word2vec(path):
    """
    Loads GoogleNews pre-trained Word2Vec model via gensim.
    """

    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


def _load_glove(path):
    """
    Loads GloVe pre-trained embedding model via gensim
    """

    tmp_file = get_tmpfile('temp_glove_w2v_format.txt')
    glove2word2vec(path, tmp_file)

    return gensim.models.KeyedVectors.load_word2vec_format(tmp_file)


def load_embedding_model(model_type, embedding_dimension=300):
    """
    Loads one supported pre-trained embeddings model.
    Currently, the following embeddings models are supported:

        1) FastText
        2) Word2Vec
        3) GloVe
    """

    if model_type == 'fasttext':
        logger.info('Loading FastText embedding model..')
        path = os.path.join(cd.EMBEDDING_MODELS_DIR,
                            'wiki.en',
                            'wiki.en.vec')
        return _load_fasttext(path=path)
    if model_type == 'word2vec':
        logger.info('Loading Word2Vec embedding model..')
        path = os.path.join(cd.EMBEDDING_MODELS_DIR,
                            'GoogleNews-vectors-negative{}.bin'.format(embedding_dimension),
                            'GoogleNews-vectors-negative{}.bin'.format(embedding_dimension))
        return _load_word2vec(path=path)
    if model_type == 'glove':
        logger.info('Loading GloVe embedding model..')
        path = os.path.join(cd.EMBEDDING_MODELS_DIR,
                            'glove.6B',
                            'glove.6B.{}d.txt'.format(embedding_dimension))
        return _load_glove(path)

    logger.exception("""Invalid embedding model type! Got: {}
    Supported model types: ['fasttext', 'word2vec', 'glove']""".format(model_type))
