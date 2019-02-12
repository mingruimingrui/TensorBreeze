import sys
import logging
import threading

import tensorflow as tf
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets import CocoDataset
from ..samplers import DetectionSampler
from ..collate import ImageCollate
from .. import transforms

logger = logging.getLogger(__name__)


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
    """
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
    image_collate = ImageCollate()

    datasets = []
    for root_image_dir, ann_file in zip(root_image_dirs, ann_files):
        datasets.append(CocoDataset(
            root_image_dir,
            ann_file,
            mask=mask,
            transforms=image_transforms
        ))

    coco_dataset = ConcatDataset(datasets)
    batch_sampler = DetectionSampler(
        dataset=coco_dataset,
        batch_size=batch_size,
        group_method='ratio',
        random_sample=True,
        num_iter=num_iter,
        drop_no_anns=drop_no_anns,
    )

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
    while True:
        try:
            for batch in data_loader:
                feed_dict = dict(zip(annotation_placeholders, batch['annotations']))
                feed_dict[image_placeholder] = batch['image']
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
    Coco data laoder implemented with multiprocessing data loader
    and an enqueue thread
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

    # Return batch
    return {
        'image': image_tensor,
        'annotations': annotations_tensor
    }
