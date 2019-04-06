from __future__ import absolute_import
from __future__ import division

import os
import sys

# Append TensorBreeze root directory to sys
this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.append(root_dir)

import time
import datetime
from tqdm import tqdm

import yaml
import logging
import argparse
from collections import Mapping

import tensorflow as tf

from tensorbreeze import layers
from tensorbreeze.data.data_loaders import add_coco_loader_ops
from tensorbreeze.retinanet import add_retinanet_train_ops
from tensorbreeze.retinanet import load_pretrained_weights
from tensorbreeze.utils.context import Session
from tensorbreeze.utils.logging import setup_logger
from tensorbreeze.utils.weights_io import save_weights_to_file
from tensorbreeze.utils.collect_env import get_pretty_env_info

logger = logging.getLogger(__name__)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def prettify(obj):
    """ Transfroms a jsonfiable object into pretty string """
    return yaml.safe_dump(obj, default_flow_style=False)


def deep_cast_dict(obj):
    if isinstance(obj, Mapping):
        return {k: deep_cast_dict(v) for k, v in obj.items()}
    else:
        return obj


def parse_args(args):
    parser = argparse.ArgumentParser('Trainer for detection model')

    parser.add_argument(
        '-y', '--yaml-file', type=str,
        help='Path to yaml file containing script configs')

    parser.add_argument(
        '--model-config-file', type=str,
        help='Path to model config file')
    parser.add_argument(
        '--ann-files', type=str, nargs='+',
        help='Path to annotation files multiple files can be accepted')
    parser.add_argument(
        '--root-image-dirs', type=str, nargs='+',
        help='Path to image directories, '
        'should have same entries as ann files')

    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Size of batches during training')
    parser.add_argument(
        '--max-iter', type=int, default=1440000,
        help='Maximum number of iterations to perform during training')
    parser.add_argument(
        '--base-lr', type=float, default=0.00001,
        help='Learning rate to use during training, '
        'will be adjusted by batch size, rank and num gpus')
    parser.add_argument(
        '--warmup-iters', type=int, default=8000,
        help='Number of iterations for SGD warm up')

    parser.add_argument(
        '--min-image-size', type=int, default=800,
        help='Minimum image resized size')
    parser.add_argument(
        '--max-image-size', type=int, default=1333,
        help='Maximum image resized size')

    parser.add_argument(
        '--log-dir', type=str, default='./',
        help='Directory to store log files')
    parser.add_argument(
        '--checkpoint-dir', type=str, default='./',
        help='Directory to store checkpoint files')

    return parser.parse_args(args)


def config_args(args):
    """
    Does a number of things
    - Ensure that args are valid
    - Create directory and files
    - Set up logging
    """
    if args.yaml_file is not None:
        with open(args.yaml_file, 'r') as f:
            yaml_configs = yaml.load(f)
        for key, value in yaml_configs.items():
            assert hasattr(args, key), \
                '{} is an invalid option'.format(key)
            setattr(args, key, value)

    if isinstance(args.ann_files, str):
        args.ann_files = [args.ann_files]
    if isinstance(args.root_image_dirs, str):
        args.root_image_dirs = [args.root_image_dirs]

    assert len(args.ann_files) == len(args.root_image_dirs)

    assert args.batch_size > 0
    assert args.max_iter > 0
    assert args.base_lr > 0

    makedirs(args.log_dir)
    makedirs(args.checkpoint_dir)

    return args


def add_input_fn(sess, args):
    print('Creating data loader thread and processes')
    with tf.device('/cpu:0'):
        batch = add_coco_loader_ops(
            sess=sess,
            root_image_dirs=args.root_image_dirs,
            ann_files=args.ann_files,
            batch_size=args.batch_size,
            num_workers=args.batch_size * 2,
            drop_no_anns=True,
            mask=False,
            min_size=args.min_image_size,
            max_size=args.max_image_size,
            random_horizontal_flip=True
        )
    return batch


def add_loss_fn(batch, args):
    print('Adding model and loss ops')
    with tf.device('/gpu:0'):
        loss_dict, retinanet_config = add_retinanet_train_ops(
            batch['image'],
            batch['annotations'],
            config_file=args.model_config_file
        )
    return loss_dict, retinanet_config


def add_backprop_fn(sess, loss_dict, args):
    print('Adding backprop ops')
    with tf.device('/gpu:0'):
        lr = args.base_lr * args.batch_size
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        backprop_op = opt.minimize(loss_dict['total_loss'])
    return backprop_op


def init_variables(sess, retinanet_config, args):
    logger.info('Initializing weights')
    sess.run(tf.global_variables_initializer())
    load_pretrained_weights(retinanet_config.BACKBONE.TYPE, sess, verbosity=1)


def setup_tensorboard(sess, loss_dict, args):
    summary_writer = tf.summary.FileWriter(logdir=args.log_dir, graph=sess.graph)
    return summary_writer


def add_epoch_summary_ops(sess, loss_dict, args):
    with tf.variable_scope('epoch_summaries'):
        for key, value in loss_dict.items():
            tf.summary.scalar(key, value)
    epoch_summaries = tf.summary.merge_all(scope='epoch_summaries')
    return epoch_summaries


def log_configs(sess, summary_writer, model_config, args):
    with tf.variable_scope('text_summaries'):
        logger.info('Collecting env info')
        logger.info('\n{}\n'.format(get_pretty_env_info()))

        training_config_str = 'Training config: \n{}'.format(
            prettify(vars(args))
        )
        logger.info(training_config_str)
        tf.summary.text(
            'training_config',
            tf.constant(
                training_config_str.replace('\n', '<br/>'),
                dtype=tf.string
            )
        )

        model_config_dict = deep_cast_dict(model_config)
        model_config_str = 'Model config: \n{}'.format(
            prettify(model_config_dict)
        )
        logger.info(model_config_str)
        tf.summary.text(
            'model_config',
            tf.constant(
                model_config_str.replace('\n', '<br/>'),
                dtype=tf.string
            )
        )

    text_summaries = tf.summary.merge_all(scope='text_summaries')
    summary_writer.add_summary(sess.run(text_summaries))

    with open(os.path.join(args.log_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(model_config_dict, f, default_flow_style=False)


def save_checkpoint(checkpoint_dir, postfix):
    save_weights_to_file(os.path.join(
        checkpoint_dir,
        'model_{}.pkl'.format(postfix)
    ))


def do_train(
    sess,
    summary_writer,
    epoch_summaries,
    batch,
    loss_dict,
    backprop_op,
    args
):
    """
    TODO: Record queued losses
    """
    logger.info('Start training')
    start_training_time = time.time()
    checkpoint_period = 20000 // args.batch_size
    logging_period = 250
    loss_dict_ops = {k: v.op for k, v in loss_dict.items()}

    t0 = time.time()
    for iter_num in tqdm(range(1, args.max_iter + 1), ncols=60):
        batch_time = time.time() - t0
        t0 = time.time()

        if iter_num % logging_period != 0:
            sess.run([backprop_op, loss_dict_ops])

        else:
            epoch_summaries_buffer, losses, _ = \
                sess.run([epoch_summaries, loss_dict, backprop_op])
            summary_writer.add_summary(epoch_summaries_buffer, iter_num)

            time_elapsed = time.time() - start_training_time
            avg_batch_time = time_elapsed / iter_num
            eta_seconds = (args.max_iter - iter_num) * avg_batch_time
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            msg = dict()
            msg['eta'] = eta_string
            msg['iter'] = iter_num
            msg['batch_time'] = '{:.4f}s ({:.4f}s)'.format(batch_time, avg_batch_time)
            msg['losses'] = losses
            logger.info('{}'.format(msg))

        if iter_num % checkpoint_period == 0:
            save_checkpoint(args.checkpoint_dir, iter_num)

    save_checkpoint(args.checkpoint_dir, 'final')
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    msg = 'Total training time: {} ({:.4f}s / it)'.format(
        total_time_str,
        total_training_time / args.max_iter
    )
    logger.info(msg)


def main_(sess, args):
    setup_logger(__name__, os.path.join(args.log_dir, 'train.log'))

    # Create all tensors and ops
    batch = add_input_fn(sess, args)
    loss_dict, retinanet_config = add_loss_fn(batch, args)
    backprop_op = add_backprop_fn(sess, loss_dict, args)
    init_variables(sess, retinanet_config, args)
    meter_dict, _ = layers.meter_dict(
        loss_dict,
        init_values=sess.run(loss_dict)
    )

    # Prepare and perform training
    summary_writer = setup_tensorboard(sess, meter_dict, args)
    epoch_summaries = add_epoch_summary_ops(sess, meter_dict, args)
    log_configs(sess, summary_writer, retinanet_config, args)

    do_train(
        sess,
        summary_writer,
        epoch_summaries,
        batch,
        meter_dict,
        backprop_op,
        args
    )


def main(args):
    """ main will be just a session container, the real main is in main_ """
    with Session(allow_growth=True) as sess:
        main_(sess, args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args = config_args(args)
    main(args)
