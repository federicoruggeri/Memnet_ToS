"""

@Author:

@Date: 24/07/19

Core part of Tensorflow 2 models. Each network defined in nn_models_v2.py has a corresponding model here.

"""

import tensorflow as tf
import tensorflow_addons as tfa

from utility.tensorflow_utils_v2 import positional_encoding


######################################################
####################### MODELS #######################
######################################################

class M_Basic_Memn2n(tf.keras.Model):
    """
    Basic Memory Network implementation. The model is comprised of three steps:
        1. Memory Lookup: a similarity operation is computed between query and memory cells.
         A content vector is extracted.

        2. Memory Reasoning: extracted content vector is used along with the query to build a new query.

        3. Lookup and Reasoning can be iterated multiple times.

        4. Eventually, an answer module takes care of formulating a prediction.
    """

    def __init__(self, query_size, sentence_size, embedding_info, memory_lookup_info,
                 extraction_info, reasoning_info, partial_supervision_info,
                 vocab_size, embedding_dimension=32, hops=1,
                 answer_weights=[], dropout_rate=0.2, l2_regularization=0.,
                 embedding_matrix=None, padding_amount=None,
                 accumulate_attention=False, position_encoding=False,
                 output_size=1, **kwargs):

        super(M_Basic_Memn2n, self).__init__(**kwargs)
        self.sentence_size = sentence_size
        self.query_size = query_size
        self.partial_supervision_info = partial_supervision_info
        self.accumulate_attention = accumulate_attention
        self.use_positional_encoding = position_encoding
        self.output_size = output_size
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.reasoning_info = reasoning_info

        if self.use_positional_encoding:
            self.pos_encoding = positional_encoding(vocab_size, embedding_dimension)

        self.query_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dimension,
                                                         input_length=query_size,
                                                         weights=embedding_matrix if embedding_matrix is None else [
                                                             embedding_matrix],
                                                         mask_zero=True,
                                                         name='query_embedding')
        self.memory_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                          output_dim=embedding_dimension,
                                                          input_length=sentence_size,
                                                          weights=embedding_matrix if embedding_matrix is None else [
                                                              embedding_matrix],
                                                          mask_zero=True,
                                                          name='memory_embedding')

        self.sentence_embedder = MemorySentenceEmbedder(embedding_info=embedding_info,
                                                        embedding_dimension=embedding_dimension)

        # Can't share parameters if concat mode is enabled with mlp similarity
        if reasoning_info['mode'] == 'concat' and memory_lookup_info['mode'] == 'mlp':
            self.memory_lookup_blocks = [
                MemorySentenceLookup(memory_lookup_info=memory_lookup_info,
                                     dropout_rate=dropout_rate,
                                     l2_regularization=l2_regularization,
                                     reasoning_info=reasoning_info)
            ]
        else:
            lookup_block = MemorySentenceLookup(memory_lookup_info=memory_lookup_info,
                                                dropout_rate=dropout_rate,
                                                l2_regularization=l2_regularization,
                                                reasoning_info=reasoning_info)
            self.memory_lookup_blocks = [lookup_block for _ in range(hops)]

        extraction_block = SentenceMemoryExtraction(extraction_info=extraction_info,
                                                    partial_supervision_info=partial_supervision_info,
                                                    padding_amount=padding_amount)
        self.extraction_blocks = [extraction_block for _ in range(hops)]

        reasoning_block = SentenceMemoryReasoning(reasoning_info=reasoning_info)
        self.memory_reasoning_blocks = [reasoning_block for _ in range(hops)]

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.answer_blocks.append(tf.keras.layers.Dense(units=self.output_size,
                                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))

    def call(self, inputs, state='training', training=False, **kwargs):

        # [batch_size, query_size]
        query = inputs['text']

        # [batch_size, memory_size, sentence_size]
        # or [batch_size * memory_size, sentence_size]
        memory = inputs['context']

        # [batch_size, memory_size]
        memory_mask = inputs['context_mask']

        if len(memory.shape) == 2:
            memory = tf.reshape(memory, [query.shape[0], -1, self.sentence_size])

        query_emb = self.query_embedding(query)

        memory_emb = self.memory_embedding(memory)

        if self.use_positional_encoding:
            query_emb += self.pos_encoding[:, :tf.shape(query_emb)[1], :]
            memory_emb = tf.reshape(memory_emb, [-1, memory.shape[2], memory_emb.shape[-1]])
            memory_emb += self.pos_encoding[:, :tf.shape(memory_emb)[1], :]
            memory_emb = tf.reshape(memory_emb, memory.shape + (memory_emb.shape[-1],))

        memory_emb = tf.reshape(memory_emb, [-1, memory_emb.shape[2], memory_emb.shape[3]])
        memory_emb = self.sentence_embedder(memory_emb)
        memory_emb = tf.reshape(memory_emb, [-1, memory.shape[1], memory_emb.shape[-1]])

        query_emb = self.sentence_embedder(query_emb)

        upd_query, upd_memory = query_emb, memory_emb
        additional_inputs = {key: item for key, item in inputs.items() if key not in ['text',
                                                                                      'context',
                                                                                      'context_mask',
                                                                                      'targets']} \
            if self.partial_supervision_info['flag'] else {}

        if self.accumulate_attention:
            memory_attention = []
        else:
            memory_attention = None

        if self.partial_supervision_info:
            supervision_losses = []
        else:
            supervision_losses = None

        for lookup_block, extraction_block, reasoning_block in zip(self.memory_lookup_blocks,
                                                                   self.extraction_blocks,
                                                                   self.memory_reasoning_blocks):
            similarities = lookup_block({'query': upd_query, 'memory': upd_memory},
                                        training=training)

            extraction_input = {key: value for key, value in additional_inputs.items()}
            extraction_input['similarities'] = similarities
            extraction_input['context_mask'] = memory_mask

            probabilities, hop_supervision_loss = extraction_block(extraction_input,
                                                                   state=state,
                                                                   training=training)

            if self.accumulate_attention:
                memory_attention.append(probabilities)

            if self.partial_supervision_info['flag']:
                supervision_losses.append(hop_supervision_loss)

            memory_search = tf.reduce_sum(memory_emb * tf.expand_dims(probabilities, axis=-1), axis=1)

            upd_query = reasoning_block({'query': upd_query, 'memory_search': memory_search},
                                        state=state,
                                        training=training)

        answer = upd_query
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        return answer, {'memory_attention': memory_attention,
                        'supervision_losses': supervision_losses}


######################################################
####################### LAYERS #######################
######################################################

# Memory Network layers

# Embedding

class MemorySentenceEmbedder(tf.keras.layers.Layer):

    def __init__(self, embedding_info, embedding_dimension, **kwargs):
        super(MemorySentenceEmbedder, self).__init__(**kwargs)
        self.embedding_info = embedding_info

        if self.embedding_info['mode'] == 'recurrent_embedding':
            if self.embedding_info['bidirectional']:
                self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.embedding_info['hidden_size'],
                                                                             # activation=tf.nn.sigmoid
                                                                             ))
            else:
                self.rnn = tf.keras.layers.GRU(self.embedding_info['hidden_size'],
                                               # activation=tf.nn.sigmoid
                                               )

    def call(self, inputs, training=None, state='training', mask=None, **kwargs):
        # [# samples, max_sequence_length, embedding_dim]

        if self.embedding_info['mode'] == 'sum':
            return tf.reduce_sum(inputs, axis=1)
        elif self.embedding_info['mode'] == 'mean':
            return tf.reduce_mean(inputs, axis=1)
        elif self.embedding_info['mode'] == 'generalized_pooling':
            pooled = self.pooling(inputs, training=training, state=state)
            return tf.squeeze(pooled)
        elif self.embedding_info['mode'] == 'recurrent_embedding':
            if self.embedding_info['bidirectional']:
                rnn = self.rnn(inputs)
            else:
                rnn = self.rnn(inputs)
            return rnn
        else:
            raise RuntimeError('Invalid sentence embedding mode! Got: {}'.format(self.embedding_info['mode']))


# Lookup

class MemorySentenceLookup(tf.keras.layers.Layer):
    """
    Basic Memory Lookup layer. Query to memory cells similarity is computed either via doc product or via
    a FNN. The content vector is computed by weighting memory cells according to their corresponding computed
    similarities.

    Moreover, the layer is sensitive to strong supervision loss, since the latter is implemented via a max-margin loss
    on computed similarity distribution (generally a softmax).
    """

    def __init__(self,
                 reasoning_info,
                 memory_lookup_info,
                 dropout_rate=0.2,
                 l2_regularization=0.,
                 **kwargs):
        super(MemorySentenceLookup, self).__init__(**kwargs)
        self.memory_lookup_info = memory_lookup_info
        self.reasoning_info = reasoning_info
        self.dropout_rate = dropout_rate

        if self.memory_lookup_info['mode'] == 'mlp':
            self.mlp_weights = []
            for weight in self.memory_lookup_info['weights']:
                self.mlp_weights.append(tf.keras.layers.Dense(units=weight,
                                                              activation=tf.tanh,
                                                              kernel_regularizer=tf.keras.regularizers.l2(
                                                                  l2_regularization)))
                self.mlp_weights.append(tf.keras.layers.Dropout(rate=dropout_rate))

            self.mlp_weights.append(tf.keras.layers.Dense(units=1,
                                                          kernel_regularizer=tf.keras.regularizers.l2(
                                                              l2_regularization)))

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, training=False, state='training', mask=None, **kwargs):
        # [batch_size, embedding_dim]
        query = inputs['query']

        # [batch_size, mem_size, embedding_dim]
        memory = inputs['memory']

        if self.memory_lookup_info['mode'] == 'dot_product':
            q_temp = tf.expand_dims(query, axis=1)
            memory = tf.tile(input=memory,
                             multiples=[1, 1, tf.cast(query.shape[-1] / memory.shape[-1], tf.int32)])
            dotted = tf.reduce_sum(memory * q_temp, axis=2)
        elif self.memory_lookup_info['mode'] == 'scaled_dot_product':
            # [batch_size, mem_size, embedding_dim]
            q_temp = tf.expand_dims(query, axis=1)
            memory = tf.tile(input=memory,
                             multiples=[1, 1, tf.cast(query.shape[-1].value / memory.shape[-1].value, tf.int32)])
            dotted = tf.reduce_sum(memory * q_temp, axis=2) * self.embedding_dimension ** 0.5
        elif self.memory_lookup_info['mode'] == 'mlp':

            if self.reasoning_info['mode'] == 'concat':
                query_dimension = query.shape[-1]
                repeat_amount = query_dimension // memory.shape[-1]
                memory = tf.tile(memory, multiples=[1, 1, repeat_amount])

            # [batch_size, mem_size, embedding_dim]
            repeated_query = tf.expand_dims(query, axis=1)
            repeated_query = tf.tile(repeated_query, [1, memory.shape[1], 1])

            # [batch_size, mem_size, embedding_dim * 2]
            att_input = tf.concat((memory, repeated_query), axis=-1)
            att_input = tf.reshape(att_input, [-1, att_input.shape[-1]])

            dotted = att_input
            for block in self.mlp_weights:
                dotted = block(dotted, training=training)

            dotted = tf.reshape(dotted, [-1, memory.shape[1]])
        else:
            raise RuntimeError('Invalid similarity operation! Got: {}'.format(self.memory_lookup_info['mode']))

        # [batch_size, memory_size]
        return dotted


# Extraction

class SentenceMemoryExtraction(tf.keras.layers.Layer):

    def __init__(self, extraction_info, partial_supervision_info, padding_amount=None, **kwargs):
        super(SentenceMemoryExtraction, self).__init__(**kwargs)
        self.extraction_info = extraction_info
        self.partial_supervision_info = partial_supervision_info
        self.padding_amount = padding_amount

        self.supervision_loss = None

    def _add_supervision_loss(self, prob_dist, positive_idxs, negative_idxs, mask_idxs):

        # Repeat mask for each positive element in each sample memory
        # Mask_idxs shape: [batch_size, padding_amount]
        # Mask res shape: [batch_size * padding_amount, padding_amount]
        mask_res = tf.tile(mask_idxs, multiples=[1, self.padding_amount])
        mask_res = tf.reshape(mask_res, [-1, self.padding_amount, self.padding_amount])
        mask_res = tf.transpose(mask_res, [0, 2, 1])
        mask_res = tf.reshape(mask_res, [-1, self.padding_amount])

        # Split each similarity score for a target into a separate sample
        # similarities shape: [batch_size, memory_max_length]
        # positive_idxs shape: [batch_size, padding_amount]
        # gather_nd shape: [batch_size, padding_amount]
        # pos_scores shape: [batch_size * padding_amount, 1]
        pos_scores = tf.gather(prob_dist, positive_idxs, batch_dims=1)
        pos_scores = tf.reshape(pos_scores, [-1, 1])

        # Repeat similarity scores for non-target memories for each positive score
        # similarities shape: [batch_size, memory_max_length]
        # negative_idxs shape: [batch_size, padding_amount]
        # neg_scores shape: [batch_size * padding_amount, padding_amount]
        neg_scores = tf.gather(prob_dist, negative_idxs, batch_dims=1)
        neg_scores = tf.tile(neg_scores, multiples=[1, self.padding_amount])
        neg_scores = tf.reshape(neg_scores, [-1, self.padding_amount])

        # Compare each single positive score with all corresponding negative scores
        # [batch_size * padding_amount, padding_amount]
        # [batch_size, padding_amount]
        # [batch_size, 1]
        # Samples without supervision are ignored by applying a zero mask (mask_res)
        hop_supervision_loss = tf.maximum(0., self.partial_supervision_info['margin'] - pos_scores + neg_scores)
        hop_supervision_loss = hop_supervision_loss * tf.cast(mask_res, dtype=hop_supervision_loss.dtype)
        hop_supervision_loss = tf.reshape(hop_supervision_loss, [-1, self.padding_amount, self.padding_amount])

        hop_supervision_loss = tf.reduce_sum(hop_supervision_loss, axis=[1, 2])
        # hop_supervision_loss = tf.reduce_max(hop_supervision_loss, axis=1)
        normalization_factor = tf.cast(tf.reshape(mask_res, [-1, self.padding_amount, self.padding_amount]),
                                       hop_supervision_loss.dtype)
        normalization_factor = tf.reduce_sum(normalization_factor, axis=[1, 2])
        normalization_factor = tf.maximum(normalization_factor, tf.ones_like(normalization_factor))
        hop_supervision_loss = tf.reduce_sum(hop_supervision_loss / normalization_factor)

        # Normalize by number of unfair examples
        valid_examples = tf.reduce_sum(mask_idxs, axis=1)
        valid_examples = tf.cast(valid_examples, tf.float32)
        valid_examples = tf.minimum(valid_examples, 1.0)
        valid_examples = tf.reduce_sum(valid_examples)
        valid_examples = tf.maximum(valid_examples, 1.0)
        hop_supervision_loss = hop_supervision_loss / valid_examples

        return hop_supervision_loss

    def call(self, inputs, state='training', training=False, mask=None, **kwargs):

        similarities = inputs['similarities']
        context_mask = inputs['context_mask']

        context_mask = tf.cast(context_mask, similarities.dtype)
        similarities += (context_mask * -1e9)

        if self.extraction_info['mode'] == 'softmax':
            probs = tf.nn.softmax(similarities, axis=1)
        elif self.extraction_info['mode'] == 'sparsemax':
            probs = tfa.activations.sparsemax(similarities, axis=1)
        elif self.extraction_info['mode'] == 'sigmoid':
            probs = tf.nn.sigmoid(similarities)
        else:
            raise RuntimeError('Invalid extraction mode! Got: {}'.format(self.extraction_info['mode']))

        supervision_loss = None
        if self.partial_supervision_info['flag'] and state != 'prediction':
            supervision_loss = self._add_supervision_loss(prob_dist=probs,
                                                          positive_idxs=inputs['positive_indexes'],
                                                          negative_idxs=inputs['negative_indexes'],
                                                          mask_idxs=inputs['mask_indexes'])

        return probs, supervision_loss


# Reasoning

class SentenceMemoryReasoning(tf.keras.layers.Layer):
    """
    Basic Memory Reasoning layer. The new query is computed simply by summing the content vector to current query
    """

    def __init__(self, reasoning_info, **kwargs):
        super(SentenceMemoryReasoning, self).__init__(**kwargs)
        self.reasoning_info = reasoning_info

    def call(self, inputs, training=False, state='training', **kwargs):
        query = inputs['query']
        memory_search = inputs['memory_search']

        if self.reasoning_info['mode'] == 'sum':
            upd_query = query + memory_search
        elif self.reasoning_info['mode'] == 'concat':
            upd_query = tf.concat((query, memory_search), axis=1)
        elif self.reasoning_info['mode'] == 'rnn':
            cell = tf.keras.layers.GRUCell(query.shape[-1])
            upq_query, _ = cell(memory_search, [query])
        elif self.reasoning_info['mode'] == 'mlp':
            upd_query = tf.keras.layers.Dense(query.shape[-1],
                                              activation=tf.nn.relu)(tf.concat((query, memory_search), axis=1))
        else:
            raise RuntimeError(
                'Invalid aggregation mode! Got: {} -- Supported: [sum, concat]'.format(self.reasoning_info['mode']))

        return upd_query


######################################################
###################### BASELINES #####################
######################################################


class M_Baseline_LSTM(tf.keras.Model):
    """
    LSTM baseline for ToS Task 1. This is a simple stacked-LSTMs model.
    """

    def __init__(self, sentence_size, vocab_size, lstm_weights,
                 answer_weights, embedding_dimension,
                 l2_regularization=0., dropout_rate=0.2,
                 embedding_matrix=None, output_size=1):
        super(M_Baseline_LSTM, self).__init__()
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension

        self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dimension,
                                                         input_length=sentence_size,
                                                         weights=embedding_matrix,
                                                         mask_zero=True,
                                                         name='query_embedding')

        self.lstm_blocks = []
        for weight in self.lstm_weights[:-1]:
            self.lstm_blocks.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(weight,
                                                                                       return_sequences=True,
                                                                                       kernel_regularizer=tf.keras.regularizers.l2(
                                                                                           self.l2_regularization))))
            self.lstm_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.lstm_blocks.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_weights[-1],
                                                                                   return_sequences=False,
                                                                                   kernel_regularizer=tf.keras.regularizers.l2(
                                                                                       self.l2_regularization))))

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.answer_blocks.append(tf.keras.layers.Dense(units=output_size,
                                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))

    def call(self, inputs, training=False, **kwargs):
        sentence = inputs['text']

        sentence_emb = self.input_embedding(sentence)

        lstm_input = sentence_emb
        for block in self.lstm_blocks:
            lstm_input = block(lstm_input, training=training)

        answer = lstm_input
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        return answer, None


class M_Baseline_CNN(tf.keras.Model):
    """
    CNN baseline for ToS Task 1. This is a simple stacked-CNNs model.
    """

    def __init__(self, sentence_size, vocab_size, cnn_weights,
                 answer_weights, embedding_dimension,
                 l2_regularization=0., dropout_rate=0.2,
                 embedding_matrix=None, output_size=1):
        super(M_Baseline_CNN, self).__init__()
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.cnn_weights = cnn_weights
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension

        self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dimension,
                                                         input_length=sentence_size,
                                                         weights=embedding_matrix,
                                                         mask_zero=True,
                                                         name='query_embedding')

        self.cnn_blocks = []
        for filters, kernel in self.cnn_weights:
            self.cnn_blocks.append(tf.keras.layers.Conv1D(filters,
                                                          kernel,
                                                          kernel_regularizer=tf.keras.regularizers.l2(
                                                              self.l2_regularization)))
            self.cnn_blocks.append(tf.keras.layers.MaxPool1D())
            self.cnn_blocks.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.cnn_blocks.append(tf.keras.layers.Flatten())

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.answer_blocks.append(tf.keras.layers.Dense(units=output_size,
                                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))

    def call(self, inputs, training=False, **kwargs):
        sentence = inputs['text']

        sentence_emb = self.input_embedding(sentence)

        cnn_input = sentence_emb
        for block in self.cnn_blocks:
            cnn_input = block(cnn_input, training=training)

        answer = cnn_input
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        return answer, None


class M_Baseline_Sum(tf.keras.Model):

    def __init__(self, sentence_size, vocab_size,
                 answer_weights, embedding_dimension,
                 l2_regularization=0., dropout_rate=0.2,
                 embedding_matrix=None, output_size=1):
        super(M_Baseline_Sum, self).__init__()
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.answer_weights = answer_weights
        self.l2_regularization = l2_regularization
        self.dropout_rate = dropout_rate
        self.embedding_dimension = embedding_dimension

        self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dimension,
                                                         input_length=sentence_size,
                                                         weights=embedding_matrix,
                                                         mask_zero=True,
                                                         name='query_embedding')

        self.answer_blocks = []
        for weight in answer_weights:
            self.answer_blocks.append(tf.keras.layers.Dense(units=weight,
                                                            activation=tf.nn.leaky_relu,
                                                            kernel_regularizer=tf.keras.regularizers.l2(
                                                                l2_regularization)))
            self.answer_blocks.append(tf.keras.layers.Dropout(rate=dropout_rate))

        self.answer_blocks.append(tf.keras.layers.Dense(units=output_size,
                                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)))

    def call(self, inputs, training=False, **kwargs):
        sentence = inputs['text']

        sentence_emb = self.input_embedding(sentence)

        sentence_encoding = tf.reduce_sum(sentence_emb, axis=1)

        answer = sentence_encoding
        for block in self.answer_blocks:
            answer = block(answer, training=training)

        return answer, None
