from __future__ import absolute_import
from __future__ import division

import sys
import threading
from six import integer_types

import tensorflow as tf
from torchvision import transforms
from torch.utils.data import DataLoader

from ..datasets import TripletDataset
from ..transforms import VGG_MEAN, VGG_STD


def make_triplet_data_loader(
    root_image_dir,
    num_iter,
    batch_size=1,
    num_workers=2,
    height=224,
    width=224,
    random_horizontal_flip=False,
    random_vertical_flip=False,
):
    """
    Triplet data loader that loads batches of triplets
    """
    assert isinstance(num_iter, integer_types), 'num_iter has to be am int'
    assert isinstance(batch_size, integer_types), 'batch_size has to be an int'
    assert isinstance(height, integer_types), 'height has to be an int'
    assert isinstance(width, integer_types), 'width has to be an int'

    # Create image transforms
    image_transforms = []

    if random_horizontal_flip:
        image_transforms.append(transforms.RandomHorizontalFlip)

    if random_vertical_flip:
        image_transforms.append(transforms.RandomVerticalFlip)

    image_transforms += [
        transforms.Resize(size=(int(height), int(width))),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[m / 255.0 for m in VGG_MEAN],
            std=[m / 255.0 for m in VGG_STD]
        )
    ]
    image_transforms = transforms.Compose(image_transforms)

    # Create dataset
    triplet_dataset = TripletDataset(
        root=root_image_dir,
        num_iter=num_iter * batch_size,
        transform=image_transforms
    )

    # Create data_loader
    data_loader = DataLoader(
        dataset=triplet_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return data_loader


def enqueue_thread_main(
    sess,
    enqueue_op,
    data_loader,
    anchor_placeholder,
    positive_placeholder,
    negative_placeholder
):
    data_iter = iter(data_loader)
    while True:
        # Get next batch
        try:
            (a, p, n), _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            (a, p, n) = next(data_iter)

        # Create feed dict
        feed_dict = {
            anchor_placeholder: a,
            positive_placeholder: p,
            negative_placeholder: n
        }

        # Enqueue
        try:
            sess.run(enqueue_op, feed_dict=feed_dict)
        except tf.errors.CancelledError:
            # Quit upon closing of session
            break


def add_triplet_loader_ops(
    sess,
    root_image_dir,
    batch_size=1,
    num_workers=2,
    height=224,
    width=224,
    random_horizontal_flip=False,
    random_vertical_flip=False,
    scope='data_loader'
):
    """
    Triplet data loader implemented with multiprocessing data loader and
    threaded enqueue dequeue
    """
    data_loader = make_triplet_data_loader(
        root_image_dir=root_image_dir,
        num_iter=2 ** 31 - 1,
        batch_size=batch_size,
        num_workers=num_workers,
        height=height,
        width=width,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    with tf.name_scope(scope):
        input_dtype = tf.float32
        input_shape = (batch_size, 3, height, width)

        # Create placeholder tensors
        anchor_placeholder = tf.placeholder(
            dtype=input_dtype, shape=input_shape)
        positive_placeholder = tf.placeholder(
            dtype=input_dtype, shape=input_shape)
        negative_placeholder = tf.placeholder(
            dtype=input_dtype, shape=input_shape)

        # Create queues
        batch_queue = tf.FIFOQueue(
            capacity=2,
            shapes=(input_shape,) * 3,
            dtypes=(input_dtype,) * 3
        )

        # Create enqueue dequeue ops
        enqueue_op = batch_queue.enqueue((
            anchor_placeholder,
            positive_placeholder,
            negative_placeholder
        ))
        anchor_tensor, positive_tensor, negative_tensor = \
            batch_queue.dequeue()

        # Start enqueue threads
        t = threading.Thread(target=enqueue_thread_main, kwargs={
            'sess': sess,
            'enqueue_op': enqueue_op,
            'data_loader': data_loader,
            'anchor_placeholder': anchor_placeholder,
            'positive_placeholder': positive_placeholder,
            'negative_placeholder': negative_placeholder
        })
        t.setDaemon(True)
        t.start()

    return anchor_tensor, positive_tensor, negative_tensor
