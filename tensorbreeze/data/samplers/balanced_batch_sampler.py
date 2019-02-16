import random
import torchvision


class BalancedBatchSampler(object):
    def __init__(self, dataset, num_iter=1, cls_per_batch=40, img_per_cls=20):
        assert isinstance(dataset, torchvision.datasets.DatasetFolder)
        class_idx_to_item_idx = []
        for c in dataset.classes:
            class_idx_to_item_idx.append([])

        for i, sample in enumerate(dataset.samples):
            class_idx_to_item_idx[sample[1]].append(i)

        self.num_iter = num_iter
        self.cls_per_batch = cls_per_batch
        self.img_per_cls = img_per_cls
        self.num_classes = len(dataset.classes)
        self.class_idx_to_item_idx = class_idx_to_item_idx

    def __iter__(self):
        for _ in range(self.num_iter):
            class_idx_choosen = random.sample(
                range(self.num_classes),
                k=self.cls_per_batch
            )

            batch = []
            for class_idx in class_idx_choosen:
                batch.extend(random.sample(
                    self.class_idx_to_item_idx[class_idx],
                    k=self.img_per_cls
                ))

            yield batch

    def __len__(self):
        return self.num_iter
