import sys
import threading

import torch
import torch.utils.data
import torchvision

import tensorflow as tf

from ..samplers import BalancedBatchSampler
from ..transforms.transforms import VGG_MEAN, VGG_STD


def make_image_folder_data_loader(
    root_image_dir,
    height=224,
    width=224,
    num_iter=None,
    method='random',
    batch_size=1,
    cls_per_batch=1,
    img_per_cls=1,
    num_workers=2,
    random_horizontal_flip=False,
    random_vertical_flip=False
):
    """
    Makes a dataset from a folder with the following structure

    <root_image_dir>
    |- <class 1>
    |   |-<class 1 img 1>
    |   |-<class 1 img 2>
    |   |...
    |- <class 2>
    |   |-<class 2 img 1>
    |   |-<class 2 img 2>
    |...

    Into a data loader that loads batches of images in the NCHW format

    Args:
        root_image_dir: The root directory storing all class folders
        height: The height of the images produced
        width: The width of the images produced
        num_iter: The number of iterations that data_loader can run
        method: One of 'random' and 'balanced'
            If 'random', batch_size will be used
            If 'balanced', cls_per_batch and img_per_cls will be used
        batch_size: The size of batches produced by data_loader
            Will only be used if method is 'random'
        cls_per_batch: The number of unique classes in each batch
            Will only be used if method is 'balanced'
        img_per_cls: The number of unique images per class in each batch
            Will only be used if method is 'balanced'
        num_workers: The number of workers used to generate data
        random_horizontal_flip: Should images be flipped horizontally at
            random?
        random_vertical_flip: Should image be flipped vertically at random?
    """
    # Make image transforms
    image_transforms = [torchvision.transforms.Resize(size=(height, width))]

    if random_horizontal_flip:
        image_transforms.append(torchvision.transforms.RandomHorizontalFlip())

    if random_vertical_flip:
        image_transforms.append(torchvision.transforms.RandomVerticalFlip())

    image_transforms += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=VGG_MEAN,
            std=VGG_STD
        )
    ]
    image_transforms = torchvision.transforms.Compose(image_transforms)

    # Make dataset
    image_folder_dataset = torchvision.datasets.ImageFolder(
        root=root_image_dir,
        transform=image_transforms
    )

    if method == 'random':
        # If random just shuffle
        sampler = torch.utils.data.RandomSampler(
            data_source=image_folder_dataset,
            replacement=True,
            num_samples=num_iter * batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=image_folder_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=sampler
        )

    else:
        # If balanced, use a batch sampler
        batch_sampler = BalancedBatchSampler(
            dataset=image_folder_dataset,
            num_iter=num_iter,
            cls_per_batch=cls_per_batch,
            img_per_cls=img_per_cls
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=image_folder_dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler
        )

    return data_loader, image_folder_dataset


def enqueue_thread(
    sess,
    enqueue_op,
    data_loader,
    image_placeholder,
    label_placeholder
):
    data_iter = iter(data_loader)

    while True:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        # Create feed dict
        feed_dict = {image_placeholder: batch[0]}
        if label_placeholder:
            feed_dict[label_placeholder] = batch[1]

        # Enqueue
        try:
            sess.run(enqueue_op, feed_dict=feed_dict)
        except tf.errors.CancelledError:
            break


def add_image_folder_loader_ops(
    sess,
    root_image_dir,
    height=224,
    width=224,
    method='random',
    batch_size=1,
    cls_per_batch=1,
    img_per_cls=1,
    num_workers=2,
    random_horizontal_flip=False,
    random_vertical_flip=False
):
    """
    Makes a data loader from a folder with the following structure

    <root_image_dir>
    |- <class 1>
    |   |-<class 1 img 1>
    |   |-<class 1 img 2>
    |   |...
    |- <class 2>
    |   |-<class 2 img 1>
    |   |-<class 2 img 2>
    |...

    Into a data loader that loads batches of images in the NCHW format

    Args:
        root_image_dir: The root directory storing all class folders
        height: The height of the images produced
        width: The width of the images produced
        num_iter: The number of iterations that data_loader can run
        method: One of 'random' and 'balanced'
            If 'random', batch_size will be used
            If 'balanced', cls_per_batch and img_per_cls will be used
        batch_size: The size of batches produced by data_loader
            Will only be used if method is 'random'
        cls_per_batch: The number of unique classes in each batch
            Will only be used if method is 'balanced'
        img_per_cls: The number of unique images per class in each batch
            Will only be used if method is 'balanced'
        num_workers: The number of workers used to generate data
        random_horizontal_flip: Should images be flipped horizontally at
            random?
        random_vertical_flip: Should image be flipped vertically at random?
    """
    data_loader = make_image_folder_data_loader(
        root_image_dir=root_image_dir,
        height=height,
        width=width,
        num_iter=sys.maxsize,
        method=method,
        batch_size=batch_size,
        cls_per_batch=cls_per_batch,
        img_per_cls=img_per_cls,
        num_workers=num_workers,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    # Create placeholder tensors
    image_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size, 3, height, width)
    )
    label_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size,)
    )

    # Create queue
    batch_queue = tf.FIFOQueue(
        capacity=2,
        dtypes=(tf.float32, tf.float32),
        shapes=[
            (batch_size, 3, height, width),
            (batch_size,)
        ]
    )

    # Create enqueue dequeue op
    enqueue_op = batch_queue.enqueue([image_placeholder, label_placeholder])
    image_tensor, label_tensor = batch_queue.dequeue()

    # Start enqueue threads
    t = threading.Thread(target=enqueue_thread, kwargs={
        'sess': sess,
        'enqueue_op': enqueue_op,
        'data_loader': data_loader,
        'image_placeholder': image_placeholder,
        'label_placeholder': label_placeholder
    })
    t.setDaemon(True)
    t.start()

    # Return batch
    return {
        'image': image_tensor,
        'label': label_tensor
    }
