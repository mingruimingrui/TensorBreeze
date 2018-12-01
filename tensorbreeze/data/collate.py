import numpy as np


class ImageCollate(object):
    """
    Image datasets in kindler.data.datasets will output dictionary type items
    We need a collate function to instruct the data loader on how to colalte
    batches of image dicts
    """
    def __init__(self, pad_method='top-left', mode='constant', value=0, nchw_format=True):
        """
        Args
            pad_method (str, optional): For images of different sizes, states the
                way to pad images to create a batch of same size.
                Option of ['top-left', 'center']. Default 'top-left'.
            mode (str, optional): 'constant', 'reflect' or 'replicate'. Default: 'constant'
            value (int, optional): fill value for constant padding
        """
        assert pad_method in {'top-left', 'center'}
        assert mode in {'constant', 'reflect', 'replicate'}
        self.pad_method = pad_method
        self.mode = mode
        self.value = value
        self.nchw_format = nchw_format

    @staticmethod
    def _check_image_is_array(image):
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3

    def _enforce_nchw(self, image):
        if self.nchw_format:
            image = image.transpose(2, 0, 1)
        assert image.shape[0] == 3
        return image

    def _determine_pad(self, max_shape, image_shape):
        if self.pad_method == 'center':
            x_excess = max_shape[2] - image_shape[2]
            y_excess = max_shape[1] - image_shape[1]
            x1_pad = int(x_excess / 2)
            y1_pad = int(y_excess / 2)
            x2_pad = x_excess - x1_pad
            y2_pad = y_excess - y1_pad
        else:
            x1_pad = 0
            y1_pad = 0
            x2_pad = max_shape[2] - image_shape[2]
            y2_pad = max_shape[1] - image_shape[1]
        return (0, 0), (y1_pad, y2_pad), (x1_pad, x2_pad)

    def _pad_image(self, image, pad):
        return np.pad(
            image,
            pad_width=pad,
            mode=self.mode,
            constant_values=self.value
        )

    @staticmethod
    def _pad_annotations(annotations, pad):
        annotations[:, 0] += pad[2][0]
        annotations[:, 1] += pad[1][0]
        annotations[:, 2] += pad[2][0]
        annotations[:, 3] += pad[1][0]
        return annotations

    def __call__(self, raw_batch):
        for raw_item in raw_batch:
            self._check_image_is_array(raw_item['image'])
            raw_item['image'] = self._enforce_nchw(raw_item['image'])

        all_shapes = [item['image'].shape for item in raw_batch]
        max_shape = [max(s[i] for s in all_shapes) for i in range(3)]
        all_keys = raw_batch[0].keys()

        collated_batch = {k: [] for k in all_keys}
        assert 'image' in collated_batch

        for raw_item, item_shape in zip(raw_batch, all_shapes):
            pad = self._determine_pad(max_shape, item_shape)

            for k, v in raw_item.items():
                if k == 'image':
                    image = self._pad_image(v, pad)
                    collated_batch['image'].append(image)

                elif k == 'annotations':
                    annotations = self._pad_annotations(v, pad)
                    collated_batch['annotations'].append(annotations)

                elif k == 'masks':
                    masks = self._pad_image(v, pad)
                    collated_batch['masks'].append(masks)

                else:
                    collated_batch[k].append(v)

        collated_batch['image'] = np.stack(collated_batch['image'], axis=0)

        return collated_batch
