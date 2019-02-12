from __future__ import absolute_import
from __future__ import division

import os
from scipy.io import loadmat
from .detection_dataset import DetectionDataset


class WiderDataset(DetectionDataset):
    def __init__(self, root_image_dir, ann_file, transforms=None):
        self.root_dir = root_image_dir

        if transforms is not None:
            self.transforms = transforms

        self._load_wider_data(ann_file)
        self._check_inputs()

    def _load_wider_data(self, ann_file):
        all_image_files = []
        all_annotations = []

        wider_data = loadmat(ann_file)
        num_events = len(wider_data['event_list'])

        for e in range(num_events):
            event_name = wider_data['event_list'][e, 0][0]
            num_images = len(wider_data['file_list'][e, 0])

            for i in range(num_images):
                image_name = wider_data['file_list'][e, 0][i, 0][0]
                file_name = os.path.join(event_name, image_name) + '.jpg'
                num_anns = len(wider_data['face_bbx_list'][e, 0][i, 0])
                anns = []

                for j in range(num_anns):
                    invalid_label = wider_data['invalid_label_list'][e, 0][i, 0][j, 0]
                    if invalid_label == 1:
                        continue

                    # Extract bbox and convert to bbox + [class] format
                    # Also cast to list
                    bbox = wider_data['face_bbx_list'][e, 0][i, 0][j]
                    bbox = bbox.tolist() + [0]

                    # Convert to xyxy
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]

                    anns.append(bbox)

                all_image_files.append(file_name)
                all_annotations.append(anns)

        self.image_files = all_image_files
        self.annotations = all_annotations
        self._auto_fill_image_heights_widths()
        self._check_inputs()
