"""

@Author:

@Date: 18/09/19

"""

import tensorflow as tf

# Limiting GPU access
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import numpy as np

from experimental_custom_tf_v2 import M_Basic_Memn2n, M_Baseline_LSTM, M_Baseline_CNN, M_Baseline_Sum
from utility.cross_validation_utils import build_metrics, compute_metrics
from utility.log_utils import get_logger
from utility.python_utils import merge
from utility.tensorflow_utils_v2 import add_gradient_noise
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

logger = get_logger(__name__)


class Network(object):

    def __init__(self, embedding_dimension, name='network', additional_data=None,
                 is_pretrained=False):
        self.embedding_dimension = embedding_dimension
        self.model = None
        self.optimizer = None
        self.name = name
        self.additional_data = additional_data
        self.is_pretrained = is_pretrained

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, filepath, overwrite=True):
        if not filepath.endswith('.h5'):
            filepath += '.h5'
        self.model.save_weights(filepath)

    def load(self, filepath, **kwargs):
        self.model.load_weights(filepath=filepath, **kwargs)

    def get_attentions_weights(self, x, batch_size=32):
        return None

    def build_tensorboard_info(self):
        pass

    def compute_output_weights(self, y_train, num_classes, mode='multi-class'):

        if mode != 'multi-class':
            raise NotImplementedError('Only multi-class scenario is supported right now! Sorry ;)')

        # Multi-class
        if len(y_train.shape) > 1:
            if y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
            else:
                y_train = y_train.ravel()

        self.num_classes = num_classes
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        # class_weights = [len(y_train[y_train != cat]) / len(y_train[y_train == cat])
        #                  for cat in range(num_classes)]
        self.class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}

    def predict(self, x, steps, callbacks=None):

        callbacks = callbacks or []

        total_preds = []

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_begin'):
                if not hasattr(callback, 'model'):
                    callback.set_model(model=self)
                callback.on_prediction_begin(logs=None)

        for batch_idx in tqdm(range(steps), leave=True, position=0):

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_begin'):
                    callback.on_batch_prediction_end(batch=batch_idx, logs=None)

            batch = next(x)
            if type(batch) in [tuple, list]:
                batch = batch[0]
            preds, model_additional_info = self.batch_predict(x=batch)
            preds = preds.numpy()

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_end'):
                    callback.on_batch_prediction_end(batch=batch_idx, logs={'predictions': preds,
                                                                            'model_additional_info': model_additional_info})

            if type(preds) is list:
                if batch_idx == 0:
                    total_preds.extend(preds)
                else:
                    for idx, pred in enumerate(preds):
                        total_preds[idx] = np.append(total_preds[idx], pred, axis=0)
            else:
                total_preds.extend(preds)

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_end'):
                callback.on_prediction_end(logs=None)

        return np.array(total_preds)

    def evaluate(self, data, steps):
        total_loss = {}

        for batch_idx in tqdm(range(steps), leave=True, position=0):

            batch_additional_info = self._get_additional_info()
            batch_info = self.batch_evaluate(*next(data), batch_additional_info)
            batch_info = {key: item.numpy() for key, item in batch_info.items()}

            for key, item in batch_info.items():
                if key not in total_loss:
                    total_loss[key] = item
                else:
                    total_loss[key] += item

        total_loss = {key: item / steps for key, item in total_loss.items()}
        return total_loss

    def distributed_evaluate(self, data, steps, strategy):
        total_loss = {}

        for batch_idx in tqdm(range(steps), leave=True, position=0):

            batch_additional_info = self._get_additional_info()
            batch_info = self.distributed_batch_evaluate(inputs=list(next(data)) + [batch_additional_info],
                                                         strategy=strategy)
            batch_info = {key: item.numpy() for key, item in batch_info.items()}

            for key, item in batch_info.items():
                if key not in total_loss:
                    total_loss[key] = item
                else:
                    total_loss[key] += item

        total_loss = {key: item / steps for key, item in total_loss.items()}
        return total_loss

    def _compute_metrics(self, data, steps, metrics, true_values,
                         additional_metrics_info=None, metrics_nicknames=None,
                         prefix=None):

        predictions = self.predict(iter(data()), steps=steps)
        predictions = predictions.reshape(true_values.shape).astype(true_values.dtype)

        all_metrics = compute_metrics(metrics,
                                      true_values=true_values,
                                      predicted_values=predictions,
                                      additional_metrics_info=additional_metrics_info,
                                      metrics_nicknames=metrics_nicknames,
                                      prefix=prefix)
        metrics_str_result = [' -- '.join(['{0}: {1}'.format(key, value)
                                           for key, value in all_metrics.items()])]

        return all_metrics, metrics_str_result

    # TODO: evaluate should return predictions as well in order to avoid double forward pass
    def fit(self, train_data=None, fixed_train_data=None,
            epochs=1, verbose=1,
            callbacks=None, validation_data=None,
            step_checkpoint=None,
            metrics=None, additional_metrics_info=None, metrics_nicknames=None,
            train_num_batches=None, eval_num_batches=None,
            np_val_y=None,
            np_train_y=None):

        # self.validation_data = validation_data
        callbacks = callbacks or []

        for callback in callbacks:
            callback.set_model(model=self)
            res = callback.on_train_begin(logs={'epochs': epochs,
                                                'steps_per_epoch': train_num_batches})
            if res is not None and type(res) == dict and 'epochs' in res:
                epochs = res['epochs']

        if verbose:
            logger.info('Start Training!')

            if train_num_batches is not None:
                logger.info('Total batches: {}'.format(train_num_batches))

        if step_checkpoint is not None:
            if type(step_checkpoint) == float:
                step_checkpoint = int(train_num_batches * step_checkpoint)
                logger.info('Converting percentage step checkpoint to: {}'.format(step_checkpoint))
            else:
                if step_checkpoint > train_num_batches:
                    step_checkpoint = int(train_num_batches * 0.1)
                    logger.info('Setting step checkpoint to: {}'.format(step_checkpoint))

        parsed_metrics = None
        if metrics:
            parsed_metrics = build_metrics(metrics)

        train_data = iter(train_data())

        # Training
        for epoch in range(epochs):

            if hasattr(self.model, 'stop_training') and self.model.stop_training:
                break

            for callback in callbacks:
                callback.on_epoch_begin(epoch=epoch, logs={'epochs': epochs})

            train_loss = {}
            batch_idx = 0

            # Run epoch
            pbar = tqdm(total=train_num_batches)
            while batch_idx < train_num_batches:

                for callback in callbacks:
                    callback.on_batch_begin(batch=batch_idx, logs=None)

                batch_additional_info = self._get_additional_info()
                batch_info = self.batch_fit(*next(train_data), batch_additional_info)
                batch_info = {key: item.numpy() for key, item in batch_info.items()}

                for callback in callbacks:
                    callback.on_batch_end(batch=batch_idx, logs=batch_info)

                for key, item in batch_info.items():
                    if key in train_loss:
                        train_loss[key] += item
                    else:
                        train_loss[key] = item

                batch_idx += 1
                pbar.update(1)

            pbar.close()

            train_loss = {key: item / train_num_batches for key, item in train_loss.items()}

            val_info = None

            # Compute metrics at the end of each epoch
            callback_additional_args = {}

            if validation_data is not None:
                val_info = self.evaluate(data=iter(validation_data()), steps=eval_num_batches)

                # TODO: extend metrics for multi-labeling
                if metrics is not None:
                    all_val_metrics, \
                    val_metrics_str_result = self._compute_metrics(data=validation_data,
                                                                   steps=eval_num_batches,
                                                                   true_values=np_val_y,
                                                                   prefix='val',
                                                                   metrics=parsed_metrics,
                                                                   additional_metrics_info=additional_metrics_info,
                                                                   metrics_nicknames=metrics_nicknames)

                    all_train_metrics, \
                    train_metrics_str_result = self._compute_metrics(data=fixed_train_data,
                                                                     steps=train_num_batches,
                                                                     true_values=np_train_y,
                                                                     prefix='train',
                                                                     metrics=parsed_metrics,
                                                                     additional_metrics_info=additional_metrics_info,
                                                                     metrics_nicknames=metrics_nicknames)

                    logger.info('Epoch: {0} -- Train Loss: {1}'
                                ' -- Val Loss: {2}'
                                ' -- Val Metrics: {3}'
                                ' -- Train Metrics: {4}'.format(epoch + 1,
                                                                train_loss,
                                                                val_info,
                                                                ' -- '.join(val_metrics_str_result),
                                                                ' -- '.join(train_metrics_str_result)))
                    callback_additional_args = merge(all_train_metrics, all_val_metrics)
                else:
                    if verbose:
                        logger.info('Epoch: {0} -- Train Loss: {1} -- Val Loss: {2}'.format(epoch + 1,
                                                                                            train_loss,
                                                                                            val_info))
            else:
                logger.info('Epoch: {0} -- Train Loss: {1}'.format(epoch + 1,
                                                                   train_loss))

            for callback in callbacks:
                callback_args = merge(train_loss, val_info)
                callback_args = merge(callback_args,
                                      callback_additional_args,
                                      overwrite_conflict=False)
                callback.on_epoch_end(epoch=epoch, logs=callback_args)

        for callback in callbacks:
            callback.on_train_end(logs={'name': self.name})

    def _get_input_iterator(self, input_fn, strategy):
        """Returns distributed dataset iterator."""
        # When training with TPU pods, datasets needs to be cloned across
        # workers. Since Dataset instance cannot be cloned in eager mode, we instead
        # pass callable that returns a dataset.
        if not callable(input_fn):
            raise ValueError('`input_fn` should be a closure that returns a dataset.')
        iterator = iter(
            strategy.experimental_distribute_datasets_from_function(input_fn))
        return iterator

    def distribute_fit(self, train_data=None,
                       epochs=1, verbose=1, strategy=None,
                       callbacks=None, validation_data=None,
                       step_checkpoint=None,
                       metrics=None, additional_metrics_info=None, metrics_nicknames=None,
                       train_num_batches=None, eval_num_batches=None,
                       np_val_y=None):

        # self.validation_data = validation_data
        callbacks = callbacks or []

        for callback in callbacks:
            callback.set_model(model=self.model)
            callback.on_train_begin(logs=None)

        if verbose:
            logger.info('Start Training!')

            if train_num_batches is not None:
                logger.info('Total batches: {}'.format(train_num_batches))

        if step_checkpoint is not None:
            if type(step_checkpoint) == float:
                step_checkpoint = int(train_num_batches * step_checkpoint)
                logger.info('Converting percentage step checkpoint to: {}'.format(step_checkpoint))
            else:
                if step_checkpoint > train_num_batches:
                    step_checkpoint = int(train_num_batches * 0.1)
                    logger.info('Setting step checkpoint to: {}'.format(step_checkpoint))

        parsed_metrics = None
        if metrics:
            parsed_metrics = build_metrics(metrics)

        train_data = self._get_input_iterator(train_data, strategy)

        # Training
        for epoch in range(epochs):

            if hasattr(self.model, 'stop_training') and self.model.stop_training:
                break

            for callback in callbacks:
                callback.on_epoch_begin(epoch=epoch, logs={'epochs': epochs})

            train_loss = {}
            batch_idx = 0

            # Run epoch
            pbar = tqdm(total=train_num_batches)
            while batch_idx < train_num_batches:

                for callback in callbacks:
                    callback.on_batch_begin(batch=batch_idx, logs=None)

                batch_additional_info = self._get_additional_info()
                batch_info = self.distributed_batch_fit(inputs=list(next(train_data)) + [batch_additional_info],
                                                        strategy=strategy)
                batch_info = {key: item.numpy() for key, item in batch_info.items()}

                for callback in callbacks:
                    callback.on_batch_end(batch=batch_idx, logs=batch_info)

                for key, item in batch_info.items():
                    if key in train_loss:
                        train_loss[key] += item
                    else:
                        train_loss[key] = item

                batch_idx += 1
                pbar.update(1)

            pbar.close()

            train_loss = {key: item / train_num_batches for key, item in train_loss.items()}

            val_info = None

            # Compute metrics at the end of each epoch
            callback_additional_args = {}

            if validation_data is not None:
                val_info = self.distributed_evaluate(data=self._get_input_iterator(validation_data, strategy),
                                                     strategy=strategy,
                                                     steps=eval_num_batches)

                # TODO: extend metrics for multi-labeling
                if metrics is not None:
                    val_predictions = self.predict(self._get_input_iterator(validation_data, strategy),
                                                   steps=eval_num_batches)
                    val_predictions = val_predictions.reshape(np_val_y.shape).astype(np_val_y.dtype)

                    all_val_metrics = compute_metrics(parsed_metrics,
                                                      true_values=np_val_y,
                                                      predicted_values=val_predictions,
                                                      additional_metrics_info=additional_metrics_info,
                                                      metrics_nicknames=metrics_nicknames,
                                                      prefix='val')
                    val_metrics_str_result = [' -- '.join(['{0}: {1}'.format(key, value)
                                                           for key, value in all_val_metrics.items()])]
                    logger.info('Epoch: {0} -- Train Loss: {1}'
                                ' -- Val Loss: {2} -- Val Metrics: {3}'.format(epoch + 1,
                                                                               train_loss,
                                                                               val_info,
                                                                               ' -- '.join(
                                                                                   val_metrics_str_result)))
                    callback_additional_args = all_val_metrics
                else:
                    if verbose:
                        logger.info('Epoch: {0} -- Train Loss: {1} -- Val Loss: {2}'.format(epoch + 1,
                                                                                            train_loss,
                                                                                            val_info))
            else:
                logger.info('Epoch: {0} -- Train Loss: {1}'.format(epoch + 1,
                                                                   train_loss))

            for callback in callbacks:
                callback_args = train_loss
                callback_args = merge(callback_args, val_info)
                callback_args = merge(callback_args,
                                      callback_additional_args,
                                      overwrite_conflict=False)
                callback.on_epoch_end(epoch=epoch, logs=callback_args)

        for callback in callbacks:
            callback.on_train_end(logs=None)

    def _get_additional_info(self):
        return None

    def _parse_predictions(self, predictions):
        return predictions

    @tf.function
    def batch_fit(self, x, y, additional_info=None):
        loss, loss_info, grads = self.train_op(x, y, additional_info=additional_info)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        train_loss_info = {'train_{}'.format(key): item for key, item in loss_info.items()}
        train_loss_info['train_loss'] = loss
        return train_loss_info

    @tf.function
    def distributed_batch_fit(self, inputs, strategy):
        train_loss_info = strategy.experimental_run_v2(self.batch_fit, args=inputs)
        train_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                           for key, item in train_loss_info.items()}
        return train_loss_info

    @tf.function
    def batch_predict(self, x):
        predictions, model_additional_info = self.model(x, state='prediction', training=False)
        predictions = self._parse_predictions(predictions)
        return predictions, model_additional_info

    @tf.function
    def distributed_batch_predict(self, inputs, strategy):
        predictions = strategy.experimental_run_v2(self.batch_predict, args=inputs)
        return predictions

    @tf.function
    def distributed_batch_evaluate(self, inputs, strategy):
        val_loss_info = strategy.experimental_run_v2(self.batch_evaluate, args=inputs)
        val_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                         for key, item in val_loss_info.items()}
        return val_loss_info

    @tf.function
    def batch_evaluate(self, x, y, additional_info=None):
        loss, loss_info = self.loss_op(x, y, training=False, state='evaluation', additional_info=additional_info)
        val_loss_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info

    def train_op(self, x, y, additional_info):
        with tf.GradientTape() as tape:
            loss, loss_info = self.loss_op(x, y, training=True, additional_info=additional_info)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, loss_info, grads

    def build_model(self, text_info):
        raise NotImplementedError()

    def loss_op(self, x, targets, training=False, state='training', additional_info=None):
        raise NotImplementedError()


# Memory Networks #


class Basic_Memn2n(Network):
    """
    Basic version of the end-to-end memory network.
    The model architecture is defined as follows:
        1. Memory Lookup: stack of Memory Lookup layers
        2. Memory Reasoning: stack of Dense layers
        3. Memory reasoning is done only at the end of the sequence of memory lookups.
    """

    def __init__(self, hops, optimizer, optimizer_args, embedding_info,
                 memory_lookup_info, extraction_info, reasoning_info, partial_supervision_info, dropout_rate=.4,
                 positional_encoding=False, max_grad_norm=40.0, clip_gradient=True, add_gradient_noise=True,
                 l2_regularization=None, answer_weights=(256, 64), weight_predictions=True,
                 accumulate_attention=False, **kwargs):

        super(Basic_Memn2n, self).__init__(**kwargs)

        self.hops = hops
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.embedding_info = embedding_info
        self.extraction_info = extraction_info
        self.reasoning_info = reasoning_info
        self.memory_lookup_info = memory_lookup_info
        self.dropout_rate = dropout_rate
        self.max_grad_norm = max_grad_norm
        self.clip_gradient = clip_gradient
        self.add_gradient_noise = add_gradient_noise
        self.l2_regularization = l2_regularization
        self.answer_weights = answer_weights
        self.weight_predictions = weight_predictions
        self.partial_supervision_info = partial_supervision_info
        self.partial_supervision_info['supervision_alpha'] = 1. if 'mask_by_attention' in partial_supervision_info and \
                                                                   partial_supervision_info['mask_by_attention'] else 0.
        self.positional_encoding = positional_encoding
        self.accumulate_attention = accumulate_attention

    def _get_additional_info(self):
        return {
            'supervision_alpha': tf.convert_to_tensor(self.partial_supervision_info['supervision_alpha'])
        }

    def build_model(self, text_info):

        self.max_seq_length = text_info['max_seq_length']
        self.max_kb_seq_length = text_info['max_kb_seq_length']
        self.max_kb_length = text_info['max_kb_length']
        self.vocab_size = text_info['vocab_size']

        if self.partial_supervision_info['flag']:
            self.padding_amount = text_info['max_supervision_padding']
            self.accumulate_attention = True
        else:
            self.padding_amount = None

        self.model = M_Basic_Memn2n(query_size=self.max_seq_length,
                                    position_encoding=self.positional_encoding,
                                    sentence_size=self.max_kb_seq_length,
                                    accumulate_attention=self.accumulate_attention,
                                    vocab_size=self.vocab_size,
                                    embedding_dimension=self.embedding_dimension,
                                    hops=self.hops,
                                    embedding_info=self.embedding_info,
                                    answer_weights=self.answer_weights,
                                    dropout_rate=self.dropout_rate,
                                    l2_regularization=self.l2_regularization,
                                    embedding_matrix=text_info['embedding_matrix'],
                                    memory_lookup_info=self.memory_lookup_info,
                                    extraction_info=self.extraction_info,
                                    reasoning_info=self.reasoning_info,
                                    partial_supervision_info=self.partial_supervision_info,
                                    output_size=text_info['num_labels'],
                                    padding_amount=self.padding_amount)

        # TODO: build optimizer instance from args
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None):
        logits, model_additional_info = self.model(x, training=training, state=state)
        targets = tf.cast(targets, dtype=logits.dtype)
        targets = tf.reshape(targets, logits.shape)

        # Cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)

        # build weights for unbalanced classification
        if self.weight_predictions:
            weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            target_classes = tf.argmax(targets, axis=1)
            for cls, weight in self.class_weights.items():
                to_fill = tf.cast(tf.fill(weights.shape, value=weight), logits.dtype)
                weights = tf.where(target_classes == cls, to_fill, weights)

            cross_entropy *= weights
        cross_entropy = tf.reduce_mean(cross_entropy)

        total_loss = cross_entropy
        loss_info = {
            'cross_entropy': cross_entropy
        }

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['l2_regularization'] = additional_losses

        # Partial supervision
        if self.partial_supervision_info['flag']:
            # Masked Supervision by Attention

            # [batch_size, #hops]
            masked_supervision_loss = tf.stack(model_additional_info['supervision_losses'])

            # [batch_size, mem_size, #hops]
            # [batch_size, #hops]
            attentions = tf.stack(model_additional_info['memory_attention'], axis=2)
            attentions = tf.stop_gradient(attentions)
            attentions = tf.reduce_max(attentions, axis=1)
            attentions = tf.round(attentions)

            masked_supervision_loss = tf.reduce_mean(masked_supervision_loss * attentions)

            # Original Supervision

            # [batch_size, #hops]
            supervision_loss = tf.stack(model_additional_info['supervision_losses'])
            supervision_loss = tf.reduce_mean(supervision_loss)

            # Convex combination of both losses
            total_supervision = masked_supervision_loss * additional_info['supervision_alpha'] \
                                + (1 - additional_info['supervision_alpha']) * supervision_loss
            total_supervision = total_supervision * self.partial_supervision_info['coefficient']
            total_loss += total_supervision
            loss_info['supervision_loss'] = total_supervision

        return total_loss, loss_info

    def train_op(self, x, y, additional_info):
        with tf.GradientTape() as tape:
            loss, loss_info = self.loss_op(x, y, training=True, state='training', additional_info=additional_info)
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.clip_gradient:
            grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v)
                              for g, v in zip(grads, self.model.trainable_variables)]
            grads = [item[0] for item in grads_and_vars]
        if self.add_gradient_noise:
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in zip(grads, self.model.trainable_variables)]
            grads = [item[0] for item in grads_and_vars]
        return loss, loss_info, grads

    def _parse_predictions(self, predictions):
        depth = predictions.shape[-1]
        predictions = tf.math.argmax(predictions, axis=1)
        predictions = tf.one_hot(predictions, depth=depth)
        return predictions


# Baselines #

class Baseline_LSTM(Network):

    def __init__(self, lstm_weights, answer_weights,
                 optimizer_args=None, l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, weight_predictions=True, **kwargs):
        super(Baseline_LSTM, self).__init__(**kwargs)
        self.lstm_weights = lstm_weights
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.additional_data = additional_data
        self.weight_predictions = weight_predictions

    def build_model(self, text_info):
        self.sentence_size = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']

        self.model = M_Baseline_LSTM(sentence_size=self.sentence_size,
                                     vocab_size=self.vocab_size,
                                     lstm_weights=self.lstm_weights,
                                     answer_weights=self.answer_weights,
                                     embedding_dimension=self.embedding_dimension,
                                     l2_regularization=self.l2_regularization,
                                     dropout_rate=self.dropout_rate,
                                     embedding_matrix=text_info['embedding_matrix'],
                                     output_size=text_info['num_labels'])

        # TODO: build optimizer instance from args
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None):
        logits, _ = self.model(x, training=True)
        targets = tf.cast(targets, logits.dtype)
        targets = tf.reshape(targets, logits.shape)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)

        # build weights for unbalanced classification
        if self.weight_predictions:
            weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            target_classes = tf.argmax(targets, axis=1)
            for cls, weight in self.class_weights.items():
                to_fill = tf.cast(tf.fill(weights.shape, value=weight), logits.dtype)
                weights = tf.where(target_classes == cls, to_fill, weights)

            cross_entropy *= weights
        cross_entropy = tf.reduce_mean(cross_entropy)

        additional_losses = tf.reduce_sum(self.model.losses)

        total_loss = cross_entropy + additional_losses

        return total_loss, {'cross_entropy': cross_entropy,
                            'l2_regularization': additional_losses}

    def _parse_predictions(self, predictions):
        depth = predictions.shape[-1]
        predictions = tf.math.argmax(predictions, axis=1)
        predictions = tf.one_hot(predictions, depth=depth)
        return predictions


class Baseline_CNN(Network):

    def __init__(self, cnn_weights, answer_weights,
                 optimizer_args=None, l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, weight_predictions=True, **kwargs):
        super(Baseline_CNN, self).__init__(**kwargs)
        self.cnn_weights = cnn_weights
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.additional_data = additional_data
        self.weight_predictions = weight_predictions

    def build_model(self, text_info):
        self.sentence_size = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']

        self.model = M_Baseline_CNN(sentence_size=self.sentence_size,
                                    vocab_size=self.vocab_size,
                                    cnn_weights=self.cnn_weights,
                                    answer_weights=self.answer_weights,
                                    embedding_dimension=self.embedding_dimension,
                                    l2_regularization=self.l2_regularization,
                                    dropout_rate=self.dropout_rate,
                                    embedding_matrix=text_info['embedding_matrix'],
                                    output_size=text_info['num_labels'])

        # TODO: build optimizer instance from args
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None):
        logits, _ = self.model(x, training=True)
        targets = tf.cast(targets, logits.dtype)
        targets = tf.reshape(targets, logits.shape)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)

        # build weights for unbalanced classification
        if self.weight_predictions:
            weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            target_classes = tf.argmax(targets, axis=1)
            for cls, weight in self.class_weights.items():
                to_fill = tf.cast(tf.fill(weights.shape, value=weight), logits.dtype)
                weights = tf.where(target_classes == cls, to_fill, weights)

            cross_entropy *= weights
        cross_entropy = tf.reduce_mean(cross_entropy)

        additional_losses = tf.reduce_sum(self.model.losses)

        total_loss = cross_entropy + additional_losses

        return total_loss, {'cross_entropy': cross_entropy,
                            'l2_regularization': additional_losses}

    def _parse_predictions(self, predictions):
        depth = predictions.shape[-1]
        predictions = tf.math.argmax(predictions, axis=1)
        predictions = tf.one_hot(predictions, depth=depth)
        return predictions


class Baseline_Sum(Network):

    def __init__(self, answer_weights,
                 optimizer_args=None, l2_regularization=None, dropout_rate=0.2,
                 additional_data=None, weight_predictions=True, **kwargs):
        super(Baseline_Sum, self).__init__(**kwargs)
        self.answer_weights = answer_weights
        self.optimizer_args = optimizer_args
        self.l2_regularization = 0. if l2_regularization is None else l2_regularization
        self.dropout_rate = dropout_rate
        self.additional_data = additional_data
        self.weight_predictions = weight_predictions

    def loss_op(self, x, targets, training=False, state='training', additional_info=None):
        logits, _ = self.model(x, training=True)
        targets = tf.cast(targets, logits.dtype)
        targets = tf.reshape(targets, logits.shape)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)

        # build weights for unbalanced classification
        if self.weight_predictions:
            weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            target_classes = tf.argmax(targets, axis=1)
            for cls, weight in self.class_weights.items():
                to_fill = tf.cast(tf.fill(weights.shape, value=weight), logits.dtype)
                weights = tf.where(target_classes == cls, to_fill, weights)

            cross_entropy *= weights
        cross_entropy = tf.reduce_mean(cross_entropy)

        additional_losses = tf.reduce_sum(self.model.losses)

        total_loss = cross_entropy + additional_losses

        return total_loss, {'cross_entropy': cross_entropy,
                            'l2_regularization': additional_losses}

    def build_model(self, text_info):
        self.sentence_size = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']

        self.model = M_Baseline_Sum(sentence_size=self.sentence_size,
                                    vocab_size=self.vocab_size,
                                    answer_weights=self.answer_weights,
                                    embedding_dimension=self.embedding_dimension,
                                    l2_regularization=self.l2_regularization,
                                    dropout_rate=self.dropout_rate,
                                    embedding_matrix=text_info['embedding_matrix'],
                                    output_size=text_info['num_labels'])

        # TODO: build optimizer instance from args
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def _parse_predictions(self, predictions):
        depth = predictions.shape[-1]
        predictions = tf.math.argmax(predictions, axis=1)
        predictions = tf.one_hot(predictions, depth=depth)
        return predictions


class ModelFactory(object):
    supported_models = {
        'experimental_basic_memn2n_v2': Basic_Memn2n,

        'experimental_baseline_lstm_v2': Baseline_LSTM,
        'experimental_baseline_cnn_v2': Baseline_CNN,
        'experimental_baseline_sum_v2': Baseline_Sum,
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
        if ModelFactory.supported_models[key]:
            return ModelFactory.supported_models[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))
