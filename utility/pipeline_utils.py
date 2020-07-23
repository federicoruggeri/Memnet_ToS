import tensorflow as tf


def decode_record(record, name_to_features):
    """
    TPU does not support int64
    """

    example = tf.io.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def load_single_dataset(filepath, name_to_features):
    data = tf.data.TFRecordDataset(filepath)
    data = data.map(lambda record: decode_record(record, name_to_features))

    # Disabling auto-sharding -> each worker receives the full dataset
    # if isinstance(filepath, str) or len(filepath) == 1:
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = (
    #     tf.data.experimental.AutoShardPolicy.OFF)
    # data = data.with_options(options)
    return data


def create_dataset(filepath, batch_size, name_to_features, selector, is_training=True,
                   input_pipeline_context=None, shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True):
    dataset = load_single_dataset(filepath=filepath, name_to_features=name_to_features)

    # Dataset is sharded by the number of hosts (num_input_pipelines == num_hosts)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    dataset = dataset.map(selector)

    if is_training:
        dataset = dataset.shuffle(shuffle_amount, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(prefetch_amount)
    return dataset


def get_dataset_fn(filepath, batch_size, name_to_features, selector, is_training=True,
                   shuffle_amount=10000, prefetch_amount=1024,
                   reshuffle_each_iteration=True):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset"""
        bs = ctx.get_per_replica_batch_size(batch_size) if ctx else batch_size
        dataset = create_dataset(filepath=filepath, batch_size=bs,
                                 name_to_features=name_to_features,
                                 selector=selector,
                                 is_training=is_training,
                                 input_pipeline_context=ctx,
                                 shuffle_amount=shuffle_amount,
                                 reshuffle_each_iteration=reshuffle_each_iteration,
                                 prefetch_amount=prefetch_amount)
        return dataset

    return _dataset_fn
