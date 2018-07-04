import tensorflow as tf


def logits_classifier(features, labels, mode):
    with tf.name_scope("LogistLayer"):
        plain_input = tf.reshape(features, [-1, 32 * 32 * 3], name="Reshape")
        logist = tf.layers.dense(inputs=plain_input, units=10, activation=None,name="Linner")
        predicted_classes = tf.argmax(logist, 1,name="PredictClass")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classss": predicted_classes,
            "probabilities": tf.nn.softmax(logist)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope("Loss"):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logist)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
        tf.summary.scalar('accuracy', accuracy[1])


    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("Opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    metrics = {'accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
