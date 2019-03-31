import threading
from six import string_types

import tensorflow as tf
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets import WiderDataset
from ..samplers import DetectionSampler
from ..collate import ImageCollate
from .. import transforms


def make_wider_data_loader(
    root_image_dirs,
    ann_files,
    num_iter,
    batch_size=1,
    num_workers=2,
    min_size=1024,
    max_size=1700,
    random_horizontal_flip=False,
    random_vertical_flip=False
):
    """
    Wider multiprocessing and threaded enqueue data loader

    Args:
        root_image_dirs: List of root image directories
        ann_files: List of annotation files (.mat files)
        num_iter: The number of iterations that data_loader performs
        batch_size: The size of each batch that data_loader produces
        num_workers: The number of workers to generate batches with
        min_size/max_size: minimum and maximum size of images to produce
        random_X_flip: Should images be flipped at random?
    """
    if isinstance(root_image_dirs, string_types):
        assert isinstance(ann_files, string_types)
        root_image_dirs = [root_image_dirs]
        ann_files = [ann_files]

    image_transforms = []

    if random_horizontal_flip:
        image_transforms.append(transforms.RandomHorizontalFlip())

    if random_vertical_flip:
        image_transforms.append(transforms.RandomVerticalFlip())

    image_transforms += [
        # transforms.ImageResize(min_size=min_size, max_size=max_size),
        transforms.RandomCrop(height=min_size, width=max_size),
        transforms.RemoveInvalidAnnotations(data_format='channels_last'),
        transforms.ImageNormalization()
    ]
    image_transforms = transforms.Compose(image_transforms)
    image_collate = ImageCollate()

    datasets = []
    for root_image_dir, ann_file in zip(root_image_dirs, ann_files):
        datasets.append(WiderDataset(
            root_image_dir,
            ann_file,
            transforms=image_transforms
        ))

    wider_dataset = ConcatDataset(datasets)
    batch_sampler = DetectionSampler(
        dataset=wider_dataset,
        batch_size=batch_size,
        group_method='ratio',
        random_sample=True,
        num_iter=num_iter
    )

    data_loader = DataLoader(
        wider_dataset,
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


def add_wider_loader_ops(
    sess,
    root_image_dirs,
    ann_files,
    batch_size=1,
    num_workers=2,
    min_size=1024,
    max_size=1700,
    random_horizontal_flip=False,
    random_vertical_flip=False,
    scope='data_loader'
):
    """
    """
    data_loader = make_wider_data_loader(
        root_image_dirs=root_image_dirs,
        ann_files=ann_files,
        num_iter=2 ** 31 - 1,
        batch_size=batch_size,
        num_workers=num_workers,
        min_size=min_size,
        max_size=max_size,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    with tf.name_scope(scope):
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

        # Create enqueue ops
        enqueue_op = batch_queue.enqueue(input_placeholders)
        enqueue_shape_op = image_shape_queue.enqueue(tf.shape(image_placeholder))

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

        # Create dequeue + reshape ops
        input_tensors = batch_queue.dequeue()
        image_shape_tensor = image_shape_queue.dequeue()

        image_tensor = tf.reshape(
            input_tensors[0],
            [image_shape_tensor[0], 3, image_shape_tensor[2], image_shape_tensor[3]]
        )
        annotations_tensor = [
            tf.reshape(ann, [-1, 5])
            for ann in input_tensors[1:]
        ]

    return {
        'image': image_tensor,
        'annotations': annotations_tensor
    }
