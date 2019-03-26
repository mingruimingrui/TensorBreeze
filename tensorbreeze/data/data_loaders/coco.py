import sys
import threading
from six import string_types

import tensorflow as tf
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets import CocoDataset
from ..samplers import DetectionSampler
from ..collate import ImageCollate
from .. import transforms


def make_coco_data_loader(
    root_image_dirs,
    ann_files,
    num_iter=None,
    batch_size=1,
    num_workers=2,
    drop_no_anns=True,
    mask=False,
    min_size=800,
    max_size=1333,
    random_horizontal_flip=False,
    random_vertical_flip=False
):
    """
    Coco data loader implemented with
    multiprocessing data loader and batch queue

    Args:
        root_image_dirs: A directory storing all coco images
            (Or a list of directory)
        ann_files: Path to a coco annotation file
            (Or a list of paths) (If list, must have same number of entries as
            root_image_dirs)
        num_iters: The number of iterations that data_loader can run
        batch_size: The size of batches produced by data_loader
        num_workers: The number of workers used to generate batches
        drop_no_anns: Should images with no annotations be dropped?
        mask: Should masks be produced as well?
        min_size: Minimum size of image produced (max_size will be prioritized)
        max_size: Maximum size of image produced
        random_horizontal_flip: Should images be flipped horizontally at
            random?
        random_vertical_flip: Should image be flipped vertically at random?
    """
    if isinstance(root_image_dirs, string_types):
        assert isinstance(ann_files, string_types)
        root_image_dirs = [root_image_dirs]
        ann_files = [ann_files]

    # Has to be provided until eval method is exposed
    assert num_iter is not None, 'num_iter has to be provided'

    # Create image transforms
    image_transforms = []

    if random_horizontal_flip:
        image_transforms.append(transforms.RandomHorizontalFlip())

    if random_vertical_flip:
        image_transforms.append(transforms.RandomVerticalFlip())

    image_transforms += [
        transforms.ImageResize(min_size=min_size, max_size=max_size),
        transforms.ImageNormalization()
    ]
    image_transforms = transforms.Compose(image_transforms)

    # Create dataset
    datasets = []
    for root_image_dir, ann_file in zip(root_image_dirs, ann_files):
        datasets.append(CocoDataset(
            root_image_dir,
            ann_file,
            mask=mask,
            transforms=image_transforms
        ))
    coco_dataset = ConcatDataset(datasets)

    # Create image collate and batch sampler
    image_collate = ImageCollate()
    batch_sampler = DetectionSampler(
        dataset=coco_dataset,
        batch_size=batch_size,
        group_method='ratio',
        random_sample=True,
        num_iter=num_iter,
        drop_no_anns=drop_no_anns,
    )

    # Create data_loader
    data_loader = DataLoader(
        coco_dataset,
        collate_fn=image_collate,
        batch_sampler=batch_sampler,
        num_workers=num_workers
    )

    return data_loader


def enqueue_thread_main(
    sess,
    enqueue_op,
    enqueue_shape_op,
    data_loader,
    image_placeholder,
    annotation_placeholders,
):
    data_iter = iter(data_loader)

    while True:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        # Ensure that all images have annotations
        if not all([len(ann) > 0 for ann in batch['annotations']]):
            continue

        # Create feed dict
        feed_dict = dict(zip(annotation_placeholders, batch['annotations']))
        feed_dict[image_placeholder] = batch['image']

        # Enqueue
        try:
            sess.run([enqueue_op, enqueue_shape_op], feed_dict=feed_dict)
        except tf.errors.CancelledError:
            # Quit upon closing of session
            break


def add_coco_loader_ops(
    sess,
    root_image_dirs,
    ann_files,
    batch_size=1,
    num_workers=2,
    drop_no_anns=True,
    mask=False,
    min_size=800,
    max_size=1333,
    random_horizontal_flip=False,
    random_vertical_flip=False
):
    """
    Coco data loader implemented with
    multiprocessing data loader and batch queue

    Args:
        root_image_dirs: A directory storing all coco images
            (Or a list of directory)
        ann_files: Path to a coco annotation file
            (Or a list of paths) (If list, must have same number of entries as
            root_image_dirs)
        num_iters: The number of iterations that data_loader can run
        batch_size: The size of batches produced by data_loader
        num_workers: The number of workers used to generate batches
        drop_no_anns: Should images with no annotations be dropped?
        mask: Should masks be produced as well?
        min_size: Minimum size of image produced (max_size will be prioritized)
        max_size: Maximum size of image produced
        random_horizontal_flip: Should images be flipped horizontally at
            random?
        random_vertical_flip: Should image be flipped vertically at random?
    """
    data_loader = make_coco_data_loader(
        root_image_dirs=root_image_dirs,
        ann_files=ann_files,
        num_iter=sys.maxsize,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_no_anns=drop_no_anns,
        mask=mask,
        min_size=min_size,
        max_size=max_size,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    # Create placeholder tensors
    image_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size, 3, None, None)
    )
    annotation_placeholders = [tf.placeholder(
        dtype=tf.float32,
        shape=(None, 5)
    ) for i in range(batch_size)]
    input_placeholders = [image_placeholder] + annotation_placeholders

    # Create queues
    batch_queue = tf.FIFOQueue(
        capacity=2,
        dtypes=(tf.float32,) * len(input_placeholders)
    )
    batch_queue.size()
    image_shape_queue = tf.FIFOQueue(
        capacity=2,
        dtypes=tf.int32,
        shapes=(4,)
    )
    image_shape_queue.size()

    # Create enqueue dequeue ops
    enqueue_op = batch_queue.enqueue(input_placeholders)
    enqueue_shape_op = image_shape_queue.enqueue(tf.shape(image_placeholder))
    input_tensors = batch_queue.dequeue()

    # Create reshape ops
    image_shape_tensor = image_shape_queue.dequeue()
    image_tensor = tf.reshape(
        input_tensors[0],
        [image_shape_tensor[0], 3, image_shape_tensor[2], image_shape_tensor[3]]
    )
    annotations_tensor = [
        tf.reshape(ann, [-1, 5])
        for ann in input_tensors[1:]
    ]

    # Start enqueue threads
    t = threading.Thread(target=enqueue_thread_main, kwargs={
        'sess': sess,
        'enqueue_op': enqueue_op,
        'enqueue_shape_op': enqueue_shape_op,
        'data_loader': data_loader,
        'image_placeholder': image_placeholder,
        'annotation_placeholders': annotation_placeholders,
    })
    t.setDaemon(True)
    t.start()

    # Return batch
    return {
        'image': image_tensor,
        'annotations': annotations_tensor
    }
