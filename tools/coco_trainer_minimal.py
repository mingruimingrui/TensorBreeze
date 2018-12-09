"""
This is a minimal training script describing what is required
"""

from __future__ import absolute_import
from __future__ import division

import os
import sys

# Append TensorBreeze root directory to sys
this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.append(root_dir)

import tensorflow as tf

from tensorbreeze.data.data_loaders import add_coco_loader_ops
from tensorbreeze.retinanet import add_retinanet_train_ops
from tensorbreeze.utils.context import Session

# There the minimum user defined variables
# You can and should define your own config file
root_image_dirs = ['Path to your train2017 folder']
ann_files = ['Path to your instances_train2017.json file']
config_file = os.path.join(root_dir, 'configs/retinanet_coco_config.yaml')
num_iter = 100


with Session(allow_growth=True) as sess:
    with tf.device('/cpu:0'):
        """
        This will start the multiprocessing and threaded queue loader which is
        powered by the torch library

        Running batch will fetch a new batch of data

        tf.data is a thing, but is rather memory inefficient and faces massive
        problems in multiprocessing. It also cannot be shared across other
        deep learning frameworks
        """
        batch = add_coco_loader_ops(
            sess=sess,
            root_image_dirs=root_image_dirs,
            ann_files=ann_files,
            batch_size=1,
            num_iter=num_iter
        )

    with tf.device('/gpu:0'):
        loss_dict, retinanet_config = add_retinanet_train_ops(
            batch['image'],
            batch['annotations'],
            config_file=config_file
        )

        optimizer = tf.train.AdamOptimizer()  # do change the default lr
        backwards_op = optimizer.minimize(loss_dict['total_loss'])
        sess.run(tf.global_variables_initializer())

    ##########################################################################
    # Recommended to perform the following actions here
    # - assign custom initial weights /
    #       weights from a previous checkpoint
    # - summary writer and summary ops
    # - log env info and
    ##########################################################################

    with tf.device('/gpu:0'):
        for _ in range(num_iter):
            sess.run(backwards_op)

            ##################################################################
            # Recommended to perform the following actions here
            # - checkpoints
            # - log training stats and run add_summary functions
            ##################################################################

    print('Training completed')

    ##########################################################################
    # Recommended to perform the following actions here
    # - log training stats and save final model
    ##########################################################################
