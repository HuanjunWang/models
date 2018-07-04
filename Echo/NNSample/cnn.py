import tensorflow as tf


def cnn_classifier(features, labels, mode):
    with tf.name_scope("Layer1_Cov"):
        cov1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=cov1, pool_size=(2, 2), strides=2)

    with tf.name_scope("Layer2_Cov"):
        cov2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3), padding='same',
                                activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=cov2, pool_size=(2, 2), strides=2)

    with tf.name_scope("Layer3_FC"):
        cshape = tf.shape(pool2)
        pool2_plain = tf.reshape(pool2, [-1, 8*8*64])
        fc3 = tf.layers.dense(inputs=pool2_plain, units=1024, activation=tf.nn.relu)

    with tf.name_scope("Layer4_FC"):
        fc4 = tf.layers.dense(inputs=fc3, units=256, activation=tf.nn.relu)

    with tf.name_scope("Layer5_logist"):
        logist5 = tf.layers.dense(inputs=fc4, units=10, activation=None)

    predicted_classes = tf.argmax(logist5, 1, name="PredictClass")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classss": predicted_classes,
            "probabilities": tf.nn.softmax(logist5)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope("CrossEntropy"):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logist5)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name="Accuracy")
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("Opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    metrics = {'accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
