import logging
import threading

import tensorflow as tf
from torch.utils.data import ConcatDataset, DataLoader

from ..datasets import CocoDataset
from ..samplers import DetectionSampler
from ..collate import ImageCollate
from .. import transforms

logger = logging.getLogger(__name__)


class CocoGenerator(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        for batch in self.data_loader:
            yield {
                'image': batch['image'],
                'annotations': tuple(batch['annotations'])
            }

    def __call__(self):
        return self


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
    data_loader,
    image_placeholder,
    annotation_placeholders
):
    for batch in data_loader:
        feed_dict = dict(zip(annotation_placeholders, batch['annotations']))
        feed_dict[image_placeholder] = batch['image']
        sess.run(enqueue_op, feed_dict=feed_dict)


def add_coco_loader_ops(
    sess,
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
    Coco data laoder implemented with multiprocessing data loader
    and an enqueue thread
    """
    data_loader = make_coco_data_loader(
        root_image_dirs=root_image_dirs,
        ann_files=ann_files,
        num_iter=num_iter,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_no_anns=drop_no_anns,
        mask=mask,
        min_size=min_size,
        max_size=max_size,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    # Create placeholder tensors to
    image_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size, 3, None, None)
    )
    annotation_placeholders = [tf.placeholder(
        dtype=tf.float32,
        shape=(None, 5)
    ) for i in range(batch_size)]
    input_placeholders = [image_placeholder] + annotation_placeholders

    batch_queue = tf.FIFOQueue(
        capacity=2,
        dtypes=(tf.float32,) * len(input_placeholders)
    )
    enqueue_op = batch_queue.enqueue(input_placeholders)

    t = threading.Thread(target=enqueue_thread_main, kwargs={
        'sess': sess,
        'enqueue_op': enqueue_op,
        'data_loader': data_loader,
        'image_placeholder': image_placeholder,
        'annotation_placeholders': annotation_placeholders
    })
    t.setDaemon(True)
    t.start()

    input_tensors = batch_queue.dequeue()
    return {
        'image': input_tensors[0],
        'annotations': input_tensors[1:]
    }


def add_coco_loader_ops_experimental(
    sess,
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
    Coco data loader implemented with tf.data
    TODO: Adapt entire tensorbreeze.data to use tf.data and tf.transforms ops
    """
    logger.warning('add_coco_loader_ops_experimental is still incomplete, efficiency is poor')
    data_loader = make_coco_data_loader(
        root_image_dirs=root_image_dirs,
        ann_files=ann_files,
        num_iter=num_iter,
        batch_size=batch_size,
        num_workers=0,
        drop_no_anns=drop_no_anns,
        mask=mask,
        min_size=min_size,
        max_size=max_size,
        random_horizontal_flip=random_horizontal_flip,
        random_vertical_flip=random_vertical_flip
    )

    coco_generator = CocoGenerator(data_loader)
    output_types = {
        'image': tf.float32,
        'annotations': (tf.float32,) * batch_size
    }
    output_shapes = {
        'image': (batch_size, 3, None, None),
        'annotations': ((None, 5),) * batch_size
    }

    tf_dataset = tf.data.Dataset.from_generator(
        coco_generator,
        output_types=output_types,
        output_shapes=output_shapes
    )

    tf_iterator = tf_dataset.make_one_shot_iterator()
    return tf_iterator.get_next()
