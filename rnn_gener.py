# coding: utf-8
from data_helper.batch_gener import BatchGener
import tensorflow as tf
import numpy as np

lr = 0.01
hidden_size = 128  # output from the LSTM
input_dim = 100  # word2vec size
batch_size = 1
sequence_length = 1500  # |ihello| == 6

sequence_lengths = [sequence_length for i in range(batch_size)]
print sequence_length
print(sequence_lengths)

X = tf.placeholder(tf.float32, [None, None, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, None])  # Y label

with tf.variable_scope("LSTM", reuse=tf.AUTO_REUSE):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    # initial_state: shape = (batch_size, cell.state_size)。
    outputs, _states = tf.nn.dynamic_rnn(cell, X, sequence_length=sequence_lengths, initial_state=initial_state,
                                         dtype=tf.float32)
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

with tf.variable_scope("Loss", reuse=None):
    weights = tf.ones([batch_size, sequence_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)

with tf.variable_scope("Train", reuse=tf.AUTO_REUSE):
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.variable_scope("Out", reuse=None):
    prediction = tf.argmax(outputs, axis=2)

B = BatchGener('/home/jiang/data/sorted10000', input_dim, batch_size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    x_batch, y_batch = B.generate_batch()
    # print x_batch, y_batch
    print len(y_batch[0])
    l, _ = sess.run([loss, train], feed_dict={X: x_batch, Y: y_batch})
    result = sess.run(prediction, feed_dict={X: x_batch})
    print(i, "loss:", l, "prediction: ", result, "true Y: ", y_batch)

    # print char using dic
    # result_str = [idx2char[c] for c in np.squeeze(result)]
    # print("\tPrediction str: ", ''.join(result_str))
