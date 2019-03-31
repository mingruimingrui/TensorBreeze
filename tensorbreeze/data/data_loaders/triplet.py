from __future__ import absolute_import
from __future__ import division

from six import string_types, integer_types

from torch.utils.data import DataLoader

from torchvision import transforms

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
