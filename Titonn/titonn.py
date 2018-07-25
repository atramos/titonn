# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

data   = pandas.read_csv('example.csv')


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

X =data['input'] 
values = array(X)
print(values)
# integer encode
label_encoder_X = LabelEncoder()
integer_encoded_X = label_encoder_X.fit_transform(values)
#print(integer_encoded)
# binary encode
onehot_encoder_X = OneHotEncoder(sparse=False)
integer_encoded_X = integer_encoded_X.reshape(len(integer_encoded_X), 1)
onehot_encoded_X = onehot_encoder_X.fit_transform(integer_encoded_X)
#print(onehot_encoded)
# invert first example
inverted_X = label_encoder_X.inverse_transform([argmax(onehot_encoded_X[0, :])])
print(inverted_X)

y =data['output'] 
values_y = array(y)
#print(values)
# integer encode
label_encoder_y = LabelEncoder()
integer_encoded_y = label_encoder_y.fit_transform(values_y)
#print(integer_encoded)
# binary encode
onehot_encoder_y = OneHotEncoder(sparse=False)
integer_encoded_y = integer_encoded_y.reshape(len(integer_encoded_y), 1)
onehot_encoded_y = onehot_encoder_y.fit_transform(integer_encoded_y)
#print(onehot_encoded)
# invert first example
inverted_y = label_encoder_y.inverse_transform([argmax(onehot_encoded_y[0, :])])
print(inverted_y)
def main():
    train_X=onehot_encoded_X
    test_X=onehot_encoded_X
    train_y=onehot_encoded_y
    test_y=onehot_encoded_y
    

    # Layer's sizes
    x_size = train_X.shape[0]   # Number of input nodes: 4 features and 1 bias
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[0]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(300):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    test  = ['fgh','cde','ghi','def','abc']
    #test2 = np.array(test)
    #print(test2)
    label_encoder_test = LabelEncoder()
    integer_encoded_test = label_encoder_test.fit_transform(test)
    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    onehot_encoder_X = OneHotEncoder(sparse=False)
    onehot_encoded_test  = onehot_encoder_X.fit_transform(integer_encoded_test)

    #test=onehot_encoded_X
    test_accuracy  = sess.run(predict, feed_dict={X: onehot_encoded_test})
    test_accuracy2 = label_encoder_y.inverse_transform(test_accuracy)
    print(test_accuracy2)
    sess.close()

if __name__ == '__main__':
    main()
