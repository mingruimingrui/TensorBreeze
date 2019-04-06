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

import tensorflow as tf

from tensorbreeze import layers
from tensorbreeze.data.data_loaders import add_triplet_loader_ops
from tensorbreeze.resnet import add_resnet_ops
# from tensorbreeze.resnet import load_pretrained_weights
from tensorbreeze.utils.context import Session
from tensorbreeze.utils.weights_io import save_weights_to_file
from tensorbreeze.utils.collect_env import get_pretty_env_info
from tensorbreeze.utils.logging import setup_logger
from tensorbreeze.utils.utils import makedirs, prettify, deep_cast_dict

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = argparse.ArgumentParser('Trainer triplet training on image')

    parser.add_argument(
        '-y', '--yaml-file', type=str,
        help='Path to yaml file containing script configs')

    parser.add_argument(
        '--model-config-file', type=str,
        help='Path to model config file')
    parser.add_argument(
        '--root-image-dir', type=str,
        help='Path to directory containing class folders')

    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Size of batches during training')
    parser.add_argument(
        '--max-iter', type=int, default=1440000,
        help='Maximum number of iterations to perform during training')
    parser.add_argument(
        '--base-lr', type=float, default=1e-6,
        help='Learning rate to use during training, '
        'will be adjusted by batch size, rank and num gpus')

    parser.add_argument(
        '--image-height', type=int, default=224,
        help='Input image height')
    parser.add_argument(
        '--image-width', type=int, default=224,
        help='Input image width')

    parser.add_argument(
        '--log-dir', type=str, default='./',
        help='Directory to store log files')
    parser.add_argument(
        '--checkpoint-dir', type=str, default='./',
        help='Directory to store checkpoint files')

    return parser.parse_args(args)


def config_args(args):
    if args.yaml_file is not None:
        with open(args.yaml_file, 'r') as f:
            yaml_configs = yaml.load(f)
        for key, value in yaml_configs.items():
            assert hasattr(args, key), \
                '{} is an invalid option'.format(key)
            setattr(args, key, value)

    assert os.path.isdir(args.root_image_dir), \
        '{} is an invalid path'.format(args.root_image_dir)

    assert args.batch_size > 0
    assert args.max_iter > 0
    assert args.base_lr > 0

    makedirs(args.log_dir)
    makedirs(args.checkpoint_dir)

    setup_logger(
        __name__,
        os.path.join(args.log_dir, 'train.log'),
        log_to_stdout=True
    )

    return args


def add_input_fn(sess, args):
    logger.info('Creating data loader thread and processes')
    with tf.device('/cpu:0'):
        a, p, n = add_triplet_loader_ops(
            sess=sess,
            root_image_dir=args.root_image_dir,
            batch_size=args.batch_size,
            num_workers=2,
            height=args.image_height,
            width=args.image_width,
            random_horizontal_flip=False,
            random_vertical_flip=True
        )
        input_tensor = tf.concat([a, p, n], axis=0)
    return input_tensor


def add_loss_fn(input_tensor, args):
    logger.info('Adding model and loss ops')
    with tf.device('/gpu:0'):
        with tf.variable_scope('encoder'):
            output_tensor, resnet_config = add_resnet_ops(
                input_tensor,
                config_file=args.model_config_file
            )
            assert resnet_config.NO_TOP is False

        with tf.variable_scope('loss'):
            a_out = output_tensor[0:args.batch_size]
            p_out = output_tensor[args.batch_size:(args.batch_size * 2)]
            n_out = output_tensor[(args.batch_size * 2):]
            loss = layers.triplet_margin_loss(
                a_out, p_out, n_out,
                margin=0.2,
                use_cosine=True,
                swap=True,
                eps=1e-6,
                reduction=tf.losses.Reduction.MEAN
            )
    return loss, resnet_config


def add_backprop_fn(sess, loss, args):
    logger.info('Adding backprop ops')
    with tf.device('/gpu:0'):
        lr = args.base_lr * args.batch_size
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        backprop_op = opt.minimize(loss)
    return backprop_op


def init_variables(sess, args):
    logger.info('Initializing weights')
    sess.run(tf.global_variables_initializer())


def setup_tensorboard(sess, loss, args):
    summary_writer = tf.summary.FileWriter(logdir=args.log_dir, graph=sess.graph)
    return summary_writer


def add_epoch_summary_ops(sess, loss, args):
    with tf.variable_scope('epoch_summaries'):
        tf.summary.scalar('triplet_loss', loss)
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
    loss_tensor,
    backprop_op,
    args
):
    logger.info('Start training')
    start_training_time = time.time()
    checkpoint_period = 20000
    logging_period = 200
    loss_op = loss_tensor.op

    t0 = time.time()
    for iter_num in tqdm(range(1, args.max_iter + 1)):
        batch_time = time.time() - t0
        t0 = time.time()

        if iter_num % logging_period != 0:
            sess.run([backprop_op, loss_op])

        else:
            epoch_summaries_buffer, loss, _ = \
                sess.run([epoch_summaries, loss_tensor, backprop_op])
            summary_writer.add_summary(epoch_summaries_buffer, iter_num)

            time_elapsed = time.time() - start_training_time
            avg_batch_time = time_elapsed / iter_num
            eta_seconds = (args.max_iter - iter_num) * avg_batch_time
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            msg = {
                'eta': eta_string,
                'iter': iter_num,
                'batch_time': '{:.4f}s ({:.4f}s)'.format(batch_time, avg_batch_time),
                'loss': loss
            }
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
    # Create all tensors and ops
    input_tensor = add_input_fn(sess, args)
    loss, resnet_config = add_loss_fn(input_tensor, args)
    backprop_op = add_backprop_fn(sess, loss, args)
    init_variables(sess, args)
    meter, _ = layers.meter(loss, init_value=sess.run(loss))

    # Prepare and perform training
    summary_writer = setup_tensorboard(sess, meter, args)
    epoch_summaries = add_epoch_summary_ops(sess, meter, args)
    log_configs(sess, summary_writer, resnet_config, args)

    do_train(
        sess=sess,
        summary_writer=summary_writer,
        epoch_summaries=epoch_summaries,
        loss_tensor=meter,
        backprop_op=backprop_op,
        args=args
    )


def main(args):
    with Session(allow_growth=True) as sess:
        main_(sess, args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args = config_args(args)
    main(args)
