import tensorflow as tf
import numpy as np

from logist import logits_classifier
from nn import nn_classifier
from cnn import  cnn_classifier

def main(arg):
    all_data = tf.keras.datasets.cifar10.load_data()
    [[train_data, train_label], [dev_data, dev_label]] = all_data
    train_data = train_data.astype(np.float32) / 255
    dev_data = dev_data.astype(np.float32) / 255

    input_fun_train = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_label.astype(np.int32),
        batch_size=128,
        shuffle=True,
        num_epochs=None
    )
    input_fun_eval = tf.estimator.inputs.numpy_input_fn(
        x=dev_data,
        y=dev_label.astype(np.int32),
        num_epochs=1,
        shuffle=False)

    classifier = tf.estimator.Estimator(model_fn=cnn_classifier, model_dir="/tmp/nn2/test6")

    for i in range(1000):
        classifier.train(input_fn=input_fun_train, steps=1000)
        evl_result = classifier.evaluate(input_fn=input_fun_eval)
        print("Evalurate Result: %.2f%%"%(evl_result['accuracy']*100))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
