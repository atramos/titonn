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
import numpy
sess = tf.Session()

FIRST = ord('a')
LAST = ord('f')
def onehot(str):
    chars = list(map(lambda c: max(0, ord(c) - FIRST), list(str)))
    twoDim = sess.run(tf.one_hot(chars, LAST - FIRST))
    # flatten the 2D array:
    return [item for sublist in twoDim for item in sublist]
    
import pandas
df = pandas.read_csv(sys.argv[1])
train_inputs = df['input'].map(onehot).values.tolist()
train_labels = df['output'].map(onehot).values.tolist()
sess.close()

print('INPUTS:\n' + str(train_inputs))
print('OUTPUTS:\n' + str(train_labels))

import code
#code.interact(local=locals())

INPUT_SAMPLES = len(train_inputs)
INPUT_VARS = len(train_inputs[0])
OUTPUT_SAMPLES = len(train_labels)
OUTPUT_CLASSES = len(train_labels[0])
HIDDEN=100
num_epochs = 100000
learning_rate = 0.001

print("is=%d iv=%d os=%d oc=%d" % (INPUT_SAMPLES, INPUT_VARS, OUTPUT_SAMPLES, OUTPUT_CLASSES))

# setup the Neural Network
X = tf.placeholder(tf.float32, shape=[None, INPUT_VARS])
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_CLASSES])
parameters = {
    'W1': tf.Variable(tf.random_normal([INPUT_VARS, HIDDEN])),
    'b1': tf.Variable(tf.random_normal([HIDDEN])),
    'W2': tf.Variable(tf.random_normal([HIDDEN, OUTPUT_CLASSES])),
    'b2': tf.Variable(tf.random_normal([OUTPUT_CLASSES]))
}

print('W1: ' + str(parameters['W1']))
print('b1: ' + str(parameters['b1']))
print('X: ' + str(X))
print('W2: ' + str(parameters['W2']))
print('b2: ' + str(parameters['b2']))
print('Y: ' + str(Y))

Z1 = tf.add(tf.matmul(X, parameters['W1']), parameters['b1'])
A2 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(A2, parameters['W2']), parameters['b2']) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2,  labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        _ , c = sess.run([optimizer, cost], feed_dict={X: train_inputs, Y: train_labels}) 
        if c <= 0.0005:
            break
        if epoch % 200 == 0:
            print ("Cost after epoch %i: %f" % (epoch, c))

    # Test predictions by computing the output using training set as input
    output = sess.run(Z2, feed_dict={X: training_x})
    print(np.array2string(output, precision=3))
