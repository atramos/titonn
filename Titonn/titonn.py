# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""MNIST classifier -- MODIFIED TO WORK ON ARBITRARY TEXT DATA VIA ONE-HOT ENCODING.
Originally from https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

def onehot(str):
    chars = list(map(ord, list(str)))
    return tf.one_hot(chars, 256)

#from tensorflow.examples.tutorials.mnist import input_data
# nah: use hard-coded input and labels for demonstration
import pandas
df = pandas.read_csv(sys.argv[1])
train_inputs = df['input'].map(onehot)
train_labels = df['output'].map(onehot)
print('INPUTS:\n' + str(train_inputs))
print('OUTPUTS:\n' + str(train_labels))
INPUT_SAMPLES = len(train_inputs)
INPUT_VARS = len(train_inputs[0])
OUTPUT_SAMPLES = len(train_labels)
OUTPUT_CLASSES = len(train_labels[0])
test_inputs = train_inputs
test_labels = train_labels

def next_batch(batch_size):
    return train_inputs, train_labels

# Create the model
x = tf.placeholder(tf.float32, [None, INPUT_VARS])
W = tf.Variable(tf.zeros([INPUT_VARS, OUTPUT_CLASSES]))
b = tf.Variable(tf.zeros([OUTPUT_CLASSES]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None])

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for epoch in range(1000):
  batch_xs, batch_ys = next_batch(100)
  _,cost = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if epoch % 200 == 0:
      print ("Cost after epoch %i: %f" % (epoch, cost))

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(
    accuracy, feed_dict={
        x: test_inputs,
        y_: test_labels
    }))

