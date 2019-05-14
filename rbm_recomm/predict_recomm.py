# Created by Bhaskar at 26/12/18
import tensorflow as tf
from rbm import RBM, Config
import os

config = Config()
logs_dir = './logs'
with tf.variable_scope('RBM_recomm') as scope:
    rbm = RBM(n_visible=config.visibleUnits, n_hidden=config.hiddenUnits, k=config.k, momentum=False)

    x = tf.placeholder(tf.float32, shape=[None, config.visibleUnits])

    pred = rbm.predict(x)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
        print("restoring rbm recomm model")
        saver.restore(sess, checkpoint_path)
        print("model restored")


def pred(batch_x):
    ratings = sess.run(pred, feed_dict={x: batch_x})
    return ratings


