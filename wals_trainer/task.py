"""Job entry point for ML Engine."""

import argparse
import json
import os
import numpy as np

import tensorflow as tf

import model
import util
import wals
from sklearn.model_selection import KFold


def main(args):
  # process input file
  input_file = util.ensure_local_file(args['train_files'][0])
  if not args['crossvalidate']:
      print ("No CV")
      # user_map, item_map, tr_sparse, test_sparse = model.create_test_and_train_sets(
      #     args, input_file, args['data_type'])
      user_map, item_map, tr_sparse = model.create_test_and_train_sets(
          args, input_file, args['data_type'])


      # model_dir = os.path.join(args['output_dir'], 'model')
      # os.makedirs(model_dir)
      # np.save(os.path.join(model_dir, 'user'), user_map)
      # np.save(os.path.join(model_dir, 'item'), item_map)

      # train model
      output_row, output_col = model.train_model(args, tr_sparse, test_sparse=None)

      # save trained model to job directory
      model.save_model(args, user_map, item_map, output_row, output_col)

      # log results
      train_rmse = wals.get_rmse(output_row, output_col, tr_sparse)
      # test_rmse = wals.get_rmse(output_row, output_col, test_sparse)

      if args['hypertune']:
          pass
        # write test_rmse metric for hyperparam tuning
        # util.write_hptuning_metric(args, test_rmse)
      print ('FINAL: train RMSE = %.2f' % train_rmse)
      # print ('FINAL: test RMSE = %.2f' % test_rmse)
      tf.logging.info('FINAL: train RMSE = %.2f' % train_rmse)
      # tf.logging.info('FINAL: test RMSE = %.2f' % test_rmse)
  else:
      print ("CV")
      user_map, item_map, ratings = model.create_test_and_train_sets(
          args, input_file, args['data_type'], valid=args['crossvalidate'])
      # kf = KFold(n_splits=5)
      # split_iterations = list(kf.split(ratings))
      # training_error = []
      # test_error = []
      # arg_config = []
      # for latent_factor in [50, 100, 150, 200, 250, 300]:
      #     for reg in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
      #         for wt_factor in [50, 100, 150, 200, 250]:
      #             for unobs_wt in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
      args['latent_factors'] = 50
      args['regularization'] = 0.01
      args['unobs_weight'] = 0.0001
      args['feature_wt_factor'] = 50
      # arg_config.append(args)
                      # tr_errors = []
                      # tst_errors = []
                      # for train_idx, test_idx in split_iterations:
      tr_sparse, test_sparse = model.create_sparse_train_and_test(ratings,
                                                                  user_map.size,
                                                                  item_map.size)
      # tr_sparse, test_sparse = model.generate_sparse_train_and_test(ratings,
      #                                                               user_map.size,
      #                                                               item_map.size,
      #                                                               test_idx)
      output_row, output_col = model.train_model(args, tr_sparse, test_sparse)
                          # # save trained model to job directory
                          # # model.save_model(args, user_map, item_map, output_row, output_col)
                          #
                          # # log results
      train_rmse = wals.get_rmse(output_row, output_col, tr_sparse)
      test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
      print ("train error: {}, test_error: {}".format(train_rmse, test_rmse))
      # tr_errors.append(train_rmse)
      # tst_errors.append(test_rmse)
      # print (tr_errors, tst_errors)
      #                 tf.logging.info('FINAL: train RMSE = %.2f' % np.mean(tr_errors))
      #                 print ('FINAL: train RMSE = %.2f' % np.mean(tr_errors))
      #                 tf.logging.info('FINAL: test RMSE = %.2f' % np.mean(tst_errors))
      #                 print('FINAL: test RMSE = %.2f' % np.mean(tst_errors))
      #                 training_error.append(np.mean(tr_errors))
      #                 test_error.append((np.mean(tst_errors)))
      #                 print ("Config iteration ends")
      # print ("final")
      # print ("MIN training error: {}".format(np.min(training_error)))
      # print ("MIN testing error: {}".format(np.min(test_error)))
      # print ("MIN training config: {}".format(np.argmin(training_error)))
      # print ("MIN testing config: {}".format(np.argmin(test_error)))
      # model_dir = os.path.join(os.getcwd(), 'model')
      # if not os.path.exists(model_dir):
      #     os.makedirs(model_dir)
      # np.save(os.path.join(model_dir, 'args'), arg_config)
      # np.save(os.path.join(model_dir, 'test_error'), test_error)
      # np.save(os.path.join(model_dir, 'train_error'), training_error)

      # return training_error, test_error, arg_config
      return train_rmse, test_rmse


def parse_arguments():
  """Parse job arguments."""

  parser = argparse.ArgumentParser()
  # required input arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # hyper params for model
  parser.add_argument(
      '--latent_factors',
      type=int,
      help='Number of latent factors',
  )
  parser.add_argument(
      '--num_iters',
      type=int,
      help='Number of iterations for alternating least squares factorization',
  )
  parser.add_argument(
      '--regularization',
      type=float,
      help='L2 regularization factor',
  )
  parser.add_argument(
      '--unobs_weight',
      type=float,
      help='Weight for unobserved values',
  )
  parser.add_argument(
      '--wt_type',
      type=int,
      help='Rating weight type (0=linear, 1=log)',
      default=wals.LINEAR_RATINGS
  )
  parser.add_argument(
      '--feature_wt_factor',
      type=float,
      help='Feature weight factor (linear ratings)',
  )
  parser.add_argument(
      '--feature_wt_exp',
      type=float,
      help='Feature weight exponent (log ratings)',
  )

  # other args
  parser.add_argument(
      '--output-dir',
      help='GCS location to write model, overriding job-dir',
  )
  parser.add_argument(
      '--verbose-logging',
      default=False,
      action='store_true',
      help='Switch to turn on or off verbose logging and warnings'
  )
  parser.add_argument(
      '--hypertune',
      default=False,
      action='store_true',
      help='Switch to turn on or off hyperparam tuning'
  )
  parser.add_argument(
      '--crossvalidate',
      default=False,
      action='store_true',
      help='Switch to turn on or off crossvalidation'
  )
  parser.add_argument(
      '--data-type',
      type=str,
      default='ratings',
      help='Data type, one of ratings (e.g. MovieLens) or web_views (GA data)'
  )
  parser.add_argument(
      '--delimiter',
      type=str,
      default='\t',
      help='Delimiter for csv data files'
  )
  parser.add_argument(
      '--headers',
      default=False,
      action='store_true',
      help='Input file has a header row'
  )
  parser.add_argument(
      '--use-optimized',
      default=False,
      action='store_true',
      help='Use optimized hyperparameters'
  )

  args = parser.parse_args()
  arguments = args.__dict__

  # set job name as job directory name
  job_dir = args.job_dir
  job_dir = job_dir[:-1] if job_dir.endswith('/') else job_dir
  job_name = os.path.basename(job_dir)

  # set output directory for model
  if args.hypertune:
    # if tuning, join the trial number to the output path
    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    trial = config.get('task', {}).get('trial', '')
    output_dir = os.path.join(job_dir, trial)
  elif args.output_dir:
    output_dir = args.output_dir
  else:
    output_dir = job_dir

  if args.verbose_logging:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  # update default params with any args provided to task
  params = model.DEFAULT_PARAMS
  params.update({k: arg for k, arg in arguments.iteritems() if arg is not None})
  if args.use_optimized:
    if args.data_type == 'web_views':
      params.update(model.OPTIMIZED_PARAMS_WEB)
    else:
      params.update(model.OPTIMIZED_PARAMS)
  params.update(task_data)
  params.update({'output_dir': output_dir})
  params.update({'job_name': job_name})

  # For web_view data, default to using the exponential weight formula
  # with feature weight exp.
  # For movie lens data, default to the linear weight formula.
  if args.data_type == 'web_views':
    params.update({'wt_type': wals.LOG_RATINGS})

  return params


if __name__ == '__main__':
  job_args = parse_arguments()
  main(job_args)


