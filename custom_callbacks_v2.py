"""

@Author:

@Date: 08/01/19

"""

import operator
import os
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import tf_logging as logging

from utility.json_utils import save_json
from utility.log_utils import get_logger

logger = get_logger(__name__)


class TensorBoard(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=False,
                 write_images=False,
                 embeddings_freq=0,
                 profile_batch=2,
                 embeddings_metadata=None,
                 update_freq='epoch',
                 **kwargs):
        super(TensorBoard, self).__init__()
        self._validate_kwargs(kwargs)

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata
        self.batch_size = batch_size
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq

        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._current_batch = 0
        self._total_batches_seen = 0
        self._total_val_batches_seen = 0

        self._writers = {}
        self.writer = summary_ops_v2.create_file_writer_v2(self.log_dir)

        self._profile_batch = profile_batch
        self._is_tracing = False
        self._chief_worke_only = True

    def _validate_kwargs(self, kwargs):
        """Handle arguments were supported in V1."""
        if kwargs.get('write_grads', False):
            logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                            'for the `TensorBoard` Callback.')
        if kwargs.get('batch_size', False):
            logging.warning('`batch_size` is no longer needed in the '
                            '`TensorBoard` Callback and will be ignored '
                            'in TensorFlow 2.0.')
        if kwargs.get('embeddings_layer_names', False):
            logging.warning('`embeddings_layer_names` is not supported in '
                            'TensorFlow 2.0. Instead, all `Embedding` layers '
                            'will be visualized.')
        if kwargs.get('embeddings_data', False):
            logging.warning('`embeddings_data` is not supported in TensorFlow '
                            '2.0. Instead, all `Embedding` variables will be '
                            'visualized.')

        unrecognized_kwargs = set(kwargs.keys()) - {
            'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
        }

        # Only allow kwargs that were supported in V1.
        if unrecognized_kwargs:
            raise ValueError('Unrecognized arguments in `TensorBoard` '
                             'Callback: ' + str(unrecognized_kwargs))

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        with context.eager_mode():
            # self._close_writers()
            if self.write_graph:
                with self.writer.as_default():
                    with summary_ops_v2.always_record_summaries():
                        if not model.run_eagerly:
                            summary_ops_v2.graph(K.get_graph(), step=0)

                        summary_writable = (
                                self.model.model._is_graph_network or  # pylint: disable=protected-access
                                self.model.model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
                        if summary_writable:
                            summary_ops_v2.keras_model('keras', self.model.model, step=0)

        if self.embeddings_freq:
            self._configure_embeddings()

    def _configure_embeddings(self):
        """Configure the Projector for embeddings."""
        from tensorflow.python.keras.layers import embeddings
        try:
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                              'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for layer in self.model.model.layers:
            if isinstance(layer, embeddings.Embedding):
                embedding = config.embeddings.add()
                embedding.tensor_name = layer.embeddings.name

                if self.embeddings_metadata is not None:
                    if isinstance(self.embeddings_metadata, str):
                        embedding.metadata_path = self.embeddings_metadata
                    else:
                        if layer.name in embedding.metadata_path:
                            embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

        if self.embeddings_metadata:
            raise ValueError('Unrecognized `Embedding` layer names passed to '
                             '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                             'argument: ' + str(self.embeddings_metadata.keys()))

        class DummyWriter(object):
            """Dummy writer to conform to `Projector` API."""

            def __init__(self, logdir):
                self.logdir = logdir

            def get_logdir(self):
                return self.logdir

        writer = DummyWriter(self.log_dir)
        projector.visualize_embeddings(writer, config)

    def _close_writers(self):
        """Close all remaining open file writers owned by this callback.
        If there are no such file writers, this is a no-op.
        """
        with context.eager_mode():
            self.writer.close()

    def on_train_begin(self, logs=None):
        if self._profile_batch == 1:
            summary_ops_v2.trace_on(graph=True, profiler=True)
            self._is_tracing = True

    def on_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.
        Performs profiling if current batch is in profiler_batches.
        Arguments:
          batch: Integer, index of batch within the current epoch.
          logs: Dict. Metric results for this batch.
        """
        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
            self._log_metrics(logs, prefix='batch_', step=self._total_batches_seen)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1
        if self._is_tracing:
            self._log_trace()
        elif (not self._is_tracing and
              self._total_batches_seen == self._profile_batch - 1):
            self._enable_trace()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        step = epoch if self.update_freq == 'epoch' else self._samples_seen
        # TODO: add logs control method
        self._log_metrics(logs, prefix='', step=step)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def on_train_end(self, logs=None):
        if self._is_tracing:
            self._log_trace()
        self._close_writers()

    def _enable_trace(self):
        if context.executing_eagerly():
            summary_ops_v2.trace_on(graph=True, profiler=True)
            self._is_tracing = True

    def _log_trace(self):
        if context.executing_eagerly():
            with self.writer.as_default(), \
                 summary_ops_v2.always_record_summaries():
                summary_ops_v2.trace_export(
                    name='batch_%d' % self._total_batches_seen,
                    step=self._total_batches_seen,
                    profiler_outdir=os.path.join(self.log_dir, 'train'))

            self._is_tracing = False

    def _log_metrics(self, logs, prefix, step):
        """Writes metrics out as custom scalar summaries.
        Arguments:
            logs: Dict. Keys are scalar summary names, values are NumPy scalars.
            prefix: String. The prefix to apply to the scalar summary names.
            step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        validation_prefix = 'val_'
        logs_by_writer = []
        for (name, value) in logs.items():
            if name in ('batch', 'size', 'num_steps'):
                # Scrub non-metric items.
                continue
            name = prefix + name  # assign batch or epoch prefix
            logs_by_writer.append((name, value))

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                if not logs_by_writer:
                    # Don't create a "validation" events file if we don't
                    # actually have any validation data.
                    pass
                with self.writer.as_default():
                    for (name, value) in logs_by_writer:
                        summary_ops_v2.scalar(name, value, step=step)

    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        with context.eager_mode(), \
             self.writer.as_default(), \
             summary_ops_v2.always_record_summaries():
            for layer in self.model.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(':', '_')
                    with ops.init_scope():
                        weight = K.get_value(weight)
                    summary_ops_v2.histogram(weight_name, weight, step=epoch)
                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, epoch)
            self.writer.flush()

    def _log_weight_as_image(self, weight, weight_name, epoch):
        """Logs a weight as a TensorBoard image."""
        w_img = array_ops.squeeze(weight)
        shape = K.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if K.image_data_format() == 'channels_last':
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

        shape = K.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            summary_ops_v2.image(weight_name, w_img, step=epoch)

    def _log_embeddings(self, epoch):
        embeddings_ckpt = os.path.join(self.log_dir, 'train',
                                       'keras_embedding.ckpt-{}'.format(epoch))
        self.model.model.save_weights(embeddings_ckpt)


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    Example:
    ```python
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    model.fit(data, labels, epochs=100, callbacks=[callback],
        validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            # logger.info('[EarlyStopping] New best value: {}'.format(self.best))
            if self.restore_best_weights:
                self.best_weights = self.model.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value


class AttentionRetriever(Callback):
    """
    Simple callback that allows to extract and save attention tensors during prediction phase.
    Extraction is simply implemented as attribute inspection.
    """

    def __init__(self, save_path, save_suffix=None):
        super(Callback, self).__init__()
        self.start_monitoring = False
        self.stored_memory_attention = None
        self.save_path = save_path
        self.save_suffix = save_suffix

    def set_model(self, model):
        self.model = model

    def on_build_model_begin(self, logs=None):
        self.network = logs['network']
        self.network.accumulate_attention = True

    def on_prediction_begin(self, logs=None):
        self.start_monitoring = True

    def on_batch_prediction_end(self, batch, logs=None):

        if self.start_monitoring:

            # [batch_size, hops, mem_size]
            model_additional_info = logs['model_additional_info']

            # Discriminative memn2n
            if type(model_additional_info) == dict:
                memory_attention = tf.stack(model_additional_info['memory_attention'], axis=1)
                if 'm__gated__memn2n' in self.model.model.name or 'm__res__memn2n' in self.model.model.name:
                    gating = tf.stack(model_additional_info['gating'], axis=1)
                    gating = tf.nn.sigmoid(gating)
                    gating = tf.round(gating)
                    memory_attention *= gating
                if 'm__discriminative__memn2n' in self.model.model.name:
                    gating = tf.stack(model_additional_info['gating'])
                    gating = tf.nn.sigmoid(gating)
                    gating = tf.round(gating)
                    gating = tf.expand_dims(gating, axis=-1)
                    memory_attention *= gating

                memory_attention = memory_attention.numpy()
            else:
                memory_attention = tf.stack(model_additional_info, axis=1).numpy()

            if batch == 0:
                # [batch_size, hops, mem_size]
                self.stored_memory_attention = memory_attention
            else:
                # [samples, hops, mem_size]
                self.stored_memory_attention = np.append(self.stored_memory_attention, memory_attention, axis=0)

    def on_prediction_end(self, logs=None):

        if self.start_monitoring:
            # Saving
            filepath = os.path.join(self.save_path,
                                    '{0}_{1}_attention_weights.json'.format(self.network.name, self.save_suffix))
            save_json(filepath=filepath, data=self.stored_memory_attention)

            # Resetting
            self.start_monitoring = None
            del self.stored_memory_attention


class TrainingLogger(Callback):

    def __init__(self, filepath, suffix=None, save_model=False, **kwargs):
        super(TrainingLogger, self).__init__(**kwargs)
        self.filepath = filepath
        self.suffix = suffix
        self.info = {}

        if save_model and not os.path.isdir(self.filepath):
            os.makedirs(self.filepath)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            assert type(logs) == dict

            for key, item in logs.items():
                self.info.setdefault(key, []).append(item)

    def on_train_end(self, logs=None):
        if logs is not None and 'name' in logs:
            self.suffix = logs['name']

        if self.suffix is None:
            filename = 'info.npy'
        else:
            filename = '{}_info.npy'.format(self.suffix)
        savepath = os.path.join(self.filepath, filename)

        np.save(savepath, self.info)

        self.info = {}


class CallbackFactory(object):
    supported_callbacks = {
        'earlystopping': EarlyStopping,
        'attentionretriever': AttentionRetriever
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
        if CallbackFactory.supported_callbacks[key]:
            return CallbackFactory.supported_callbacks[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))
