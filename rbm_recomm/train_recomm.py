# Created by Bhaskar at 26/12/18
import argparse

from scipy.sparse import coo_matrix
from sklearn.externals import joblib
import pandas as pd
import tensorflow as tf
import time
import numpy as np
from rbm import RBM
import os
TEST_SET_RATIO = 0.1


def main(args):
    # # Load the ratings dataset
    # views_df = pd.read_csv(args.filename, engine='python')
    # n_users = views_df.person_id.nunique()
    # n_items = views_df.attribute_id.nunique()
    # df_items = pd.DataFrame({'attribute_id': views_df.attribute_id.unique()})
    # df_sorted_items = df_items.sort_values('attribute_id').reset_index()
    # pds_items = df_sorted_items.attribute_id
    # df_users = pd.DataFrame({'person_id': views_df.person_id.unique()})
    # df_sorted_users = df_users.sort_values('person_id').reset_index()
    # pds_users = df_sorted_users.person_id
    # #
    # # preprocess data. df.groupby.agg sorts clientId and contentId
    # df_user_items = views_df.set_index(['person_id', 'attribute_id'])
    # # # create a list of (userId, itemId, timeOnPage) ratings, where userId and
    # # # clientId are 0-indexed
    # pv_ratings = []
    # print ("starting iteration to create user-item-index mapping")
    # for timeonpg in df_user_items.itertuples():
    #   user = timeonpg[0][0]
    #   item = timeonpg[0][1]
    #
    #
    #   # this search makes the preprocessing time O(r * i log(i)),
    #   # r = # ratings, i = # items
    #   ix = pds_items.searchsorted(item)[0]
    #   ux = pds_users.searchsorted(user)[0]

    #
    # # # convert ratings list and user map to np array
    # print ("Iteration completed")
    # pv_ratings = np.asarray(pv_ratings)
    model_dir = os.path.join(os.getcwd(), 'model')
    print (model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pds_users = np.load(os.path.join(model_dir, 'user_new.npy'))
    pds_items = np.load(os.path.join(model_dir, 'item_new.npy'))
    pv_ratings = np.load(os.path.join(model_dir, 'ratings.npy'))
    # pv_ratings =
    # np.save(os.path.join(model_dir, 'user'), pds_users.as_matrix())
    # np.save(os.path.join(model_dir, 'item'), pds_items.as_matrix())
    # np.save(os.path.join(model_dir, 'ratings'), pv_ratings)
    print(pds_users.shape)
    print(pds_items.shape)
    print(pv_ratings.shape)
    n_users = pds_users.shape[0]
    n_items = pds_items.shape[0]
    # test_set_size = int(len(pv_ratings) * TEST_SET_RATIO)
    # test_set_idx = np.random.choice(xrange(len(pv_ratings)),
    #                                 size=test_set_size, replace=False)
    # test_set_idx = sorted(test_set_idx)
    # # #
    # # # # sift ratings into train and test sets
    # ts_ratings = pv_ratings[test_set_idx]
    # tr_ratings = np.delete(pv_ratings, test_set_idx, axis=0)
    tr_ratings = pv_ratings

    # create training and test matrices as coo_matrix's
    u_tr, i_tr, r_tr = zip(*tr_ratings)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))
    tr_sparse = tr_sparse/10.0

    # u_ts, i_ts, r_ts = zip(*ts_ratings)
    # test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))
    print ("training_matrix_shape: {}".format(tr_sparse.shape))
    # print ("test_matrix_shape: {}".format(test_sparse.shape))

    #
    #
    # Setting the models Parameters
    visibleUnits = n_items
    print("Starting training")
    train(tr_sparse, int(args.epochs), int(args.batch), visibleUnits, int(args.hidden), int(args.k), None)
    print("Completing training")


def train(train_data, epochs, batch_size, visibleUnits, hiddenUnits, k, test_data=None):
    logs_dir = './logs'
    train_data = train_data.tocsr()

    if test_data is not None:
        test_data = test_data.tocsr()
    # samples_dir = './samples'

    with tf.variable_scope('RBM_recomm') as scope:
        rbm = RBM(n_visible=visibleUnits, n_hidden=hiddenUnits, k=k, momentum=False)

    x = tf.placeholder(tf.float32, shape=[None, visibleUnits])
    if test_data is not None:
        noise_x = test_data

    step = rbm.learn(x)
    pl = rbm.pseudo_likelihood(x)

    saver = tf.train.Saver()
    total_steps = train_data.shape[0]//batch_size
    print(total_steps)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        mean_cost = []
        epoch = 1
        for i in range(epochs):
            p = np.random.permutation(train_data.shape[0])
            training_data = train_data[p]
            start = time.time()
            for iter_step in range(total_steps):
                index = range(iter_step * batch_size, (iter_step + 1) * batch_size)
                # draw samples
                batch_x = training_data[index].A
                sess.run(step, feed_dict={x: batch_x, rbm.lr: 0.1})
                cost = sess.run(pl, feed_dict={x: batch_x})
                cost[np.isnan(cost)] = 0
                mean_cost.append(np.mean(cost))
            # save model
            # if i is not 0 and train_data.batch_index is 0:
            checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print('Saved Model.')
            # print pseudo likelihood
            # if i is not 0 and train_data.batch_index is 0:
            print('Epoch %d Train Cost %g' % (epoch, np.mean(mean_cost)))
            if test_data is not None:
                test_steps = noise_x.shape[0] // 1000
                test_losses = []
                for test_step in range(test_steps):
                    index = range(test_step * 1000, (test_step + 1) * 1000)
                    test_loss = sess.run(pl, feed_dict={x: noise_x[index].A})
                    test_loss[np.isnan(test_loss)] = 0
                    test_losses.append(test_loss)
                print('Epoch %d Test Cost %g' % (epoch, np.mean(test_losses)))
            print('Total epoch time: {}'.format(time.time() - start))
            mean_cost = []
            epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", dest="filename")
    parser.add_argument("-e", "--epochs", dest="epochs")
    parser.add_argument("-b", "--batch", dest="batch")
    parser.add_argument("-hd", "--hidden", dest="hidden")
    parser.add_argument("-k", "--k", dest="k")

    options = parser.parse_args()
    main(options)

