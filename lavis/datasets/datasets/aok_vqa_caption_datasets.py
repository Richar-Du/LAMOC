"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import torch

from PIL import Image

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

class AOKVQACapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann['question'])
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "question": question,
            "caption": caption,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

class AOKVQACapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann['question'])
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "question": question,
            "caption": caption,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }