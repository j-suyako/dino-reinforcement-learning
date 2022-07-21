from functools import partial
from tensorflow.keras import Model, layers
import tensorflow as tf


leaky_relu = partial(tf.nn.leaky_relu, alpha=0.01)


class DinoModel(Model):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.ACTIONS = 3
        self.conv1 = layers.Conv2D(32, 8, strides=4, kernel_initializer='he_uniform')
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv2D(64, 4, strides=2, kernel_initializer='he_uniform')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = layers.Conv2D(96, 3, strides=1, kernel_initializer='he_uniform')
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.active = layers.Activation(leaky_relu)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1153, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(1153, kernel_initializer='he_uniform')
        self.classifier = layers.Dense(self.ACTIONS, kernel_initializer='he_uniform')
        self.time_feature = layers.Embedding(20, 3456)

    def call(self, inputs, training=None, mask=None):
        x, step = inputs
        step = tf.minimum(step // 100, 19)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.active(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.active(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.active(x)
        x = self.flatten(x)
        x += tf.squeeze(self.time_feature(step))
        x = self.dense1(x)
        x = self.active(x)
        x = self.dense2(x)
        x = self.active(x)
        x = self.classifier(x)
        return x
