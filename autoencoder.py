import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


data_path = '/home/vincent/python/virtual_envs/keras/proj/tensorflow_mnist_cnn_example/MNIST_data'
mnist = input_data.read_data_sets(data_path)
train = mnist.train
test = mnist.test

print(train.images.shape)


def layer_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def layer_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def build_layer(input_tensor,output_dim):
    w = layer_weight([input_tensor.shape[1].value,output_dim])
    b = layer_bias([output_dim])
    layer = tf.matmul(input_tensor,w)+b
    return tf.nn.relu(layer)


#encode layer
x = tf.placeholder(dtype=tf.float32, shape=[None, train.images.shape[1]])
encode_l1 = build_layer(x, 300)
encode_l2 = build_layer(encode_l1, 200)
encode_l3 = build_layer(encode_l2, 100)
encode_out = build_layer(encode_l3, 20)


#decode layer
decode_input = tf.placeholder(dtype=tf.float32, shape=[None, encode_out.shape[1].value])
decode_l1 = build_layer(decode_input, 100)
decode_l2 = build_layer(decode_l1, 200)
decode_l3 = build_layer(decode_l2, 300)
decode_out = build_layer(decode_l3, x.shape[1].value)

loss = tf.reduce_mean(tf.pow(decode_out - x, 2))
opt = tf.train.RMSPropOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()

import cv2
import numpy as np
step = 0
with tf.Session() as sess:
    sess.run(init)

    batch, _ = train.next_batch(50)

    loss_out = 1
    #for i in range(2500):
    while loss_out > 0.03:
        code = sess.run(encode_out, feed_dict={x: batch})
        sess.run(opt, feed_dict={decode_input: code, x: batch})

        if step % 100 == 0:
            #tmp = sess.run(encode_out, feed_dict={x: batch})
            loss_out = sess.run(loss, feed_dict={decode_input: code, x: batch})
            print('step:{},loss:{}'.format(step, loss_out))

        batch, _ = train.next_batch(50)
        step += 1
        if step > 5000:
            break


    #test output using testset
    code = sess.run(encode_out, feed_dict={x: batch})
    res = sess.run(decode_out, feed_dict={decode_input:code})
    for i in range(10):
        img = res[i,:]
        img = img.reshape([28,28])
        cv2.imwrite('file{}.png'.format(i), img*255)


    #test out using random float vector
    test = np.ndarray((20,20))
    for i in range(20):
        test[i,:] = np.random.uniform(0,1,20)


    res = sess.run(decode_out, feed_dict={decode_input:test})
    for i in range(10):
        img = res[i,:]
        img = img.reshape([28,28])
        cv2.imwrite('rnd{}.png'.format(i), img*255)
