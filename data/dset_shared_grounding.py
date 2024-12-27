import json
import os
import pdb
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import sys
sys.path.append('.')
from data.template import grounding_to_openai, grounding_to_openai_qwen, batch_add_answer, batch_add_answer_append

dataset_mapping = {
    "showui": "ShowUI-desktop",
}

"""
sample = {
        "img_url": "c12b572ebccfae5052fe62826615c58d.png",
        "img_size": [
            1920,
            1080
        ],
        "element": [
            {
                "instruction": "Galerie",
                "bbox": [
                    0.6125,
                    0.35648148148148145,
                    0.6817708333333333,
                    0.375
                ],
                "data_type": "text",
                "point": [
                    0.65,
                    0.37
                ]
            },
            {
                "instruction": "Coiffure",
                "bbox": [
                    0.30416666666666664,
                    0.35648148148148145,
                    0.3770833333333333,
                    0.375
                ],
                "data_type": "text",
                "point": [
                    0.34,
                    0.37
                ]
            }],
        "element_size": 2
"""

def random_crop_metadata(img, metadata, scale_range=(0.5, 1.0)):
    original_width, original_height = metadata['img_size']
    img_copy = img.copy()
    
    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)

    crop_width = int(original_width * scale_w)
    crop_height = int(original_height * scale_h)

    pad_x = pad_y = 0

    if crop_width > original_width or crop_height > original_height:
        pad_x = max(0, (crop_width - original_width) // 2)
        pad_y = max(0, (crop_height - original_height) // 2)

        padded_img = Image.new('RGB', (crop_width, crop_height), (255, 255, 255))
        padded_img.paste(img_copy, (pad_x, pad_y))

        img = padded_img
        img_width, img_height = crop_width, crop_height
    else:
        img_width, img_height = original_width, original_height

    crop_x_min = random.randint(0, img_width - crop_width)
    crop_y_min = random.randint(0, img_height - crop_height)
    crop_x_max = crop_x_min + crop_width
    crop_y_max = crop_y_min + crop_height

    cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

    new_elements = []
    for element in metadata['element']:
        bbox = element['bbox']
        point = element['point']

        bbox_abs = [int(bbox[0] * original_width) + pad_x, int(bbox[1] * original_height) + pad_y,  
                    int(bbox[2] * original_width) + pad_x, int(bbox[3] * original_height) + pad_y]
        point_abs = [int(point[0] * original_width) + pad_x, int(point[1] * original_height) + pad_y]

        if (bbox_abs[0] >= crop_x_min and bbox_abs[2] <= crop_x_max and
            bbox_abs[1] >= crop_y_min and bbox_abs[3] <= crop_y_max):
            
            new_bbox = [(bbox_abs[0] - crop_x_min) / crop_width,
                        (bbox_abs[1] - crop_y_min) / crop_height,
                        (bbox_abs[2] - crop_x_min) / crop_width,
                        (bbox_abs[3] - crop_y_min) / crop_height]
            new_point = [(point_abs[0] - crop_x_min) / crop_width,
                         (point_abs[1] - crop_y_min) / crop_height]

            new_element = element.copy()
            new_element['bbox'] = new_bbox
            new_element['point'] = new_point
            new_elements.append(new_element)

    if len(new_elements) == 0:
        return img_copy, metadata

    metadata['element'] = new_elements
    metadata['element_size'] = len(new_elements)
    metadata['img_size'] = cropped_img.size
    return cropped_img, metadata

# support SeeClick
class GroundingDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        dataset,
        base_image_dir,
        processor,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        json_data="hf_train",
        inference=False,
        num_turn=1,
        text2point=1,
        text2bbox=0,
        point2text=0,
        bbox2text=0,
        shuffle_image_token=True,
        uniform_prompt=False,
        crop_min=1,
        crop_max=1,
        random_sample=False,
        merge_patch=0,
        merge_threshold=0,
        merge_inference=False,
        merge_random=None,
        xy_int=False,
        chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ):
        self.processor = processor
        self.samples_per_epoch = samples_per_epoch
    
        self.base_image_dir = os.path.join(base_image_dir, dataset_mapping[dataset])
        self.precision = precision

        META_DIR = os.path.join(self.base_image_dir, "metadata")
        if dataset in ['assistgui', 'omniact', 'osatlas', 'guiexp']:
            self.IMG_DIR = self.base_image_dir
        else:
            self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)
        self.inference = inference
        self.num_turn = num_turn
        
        self.sample_prob = np.array([text2point, text2bbox, point2text, bbox2text])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()
        self.shuffle_image_token = shuffle_image_token

        self.uniform_prompt = uniform_prompt
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.merge_threshold = merge_threshold
        self.merge_inference = merge_inference
        self.merge_random = merge_random

        # self.merge_patch = 0 if self.inference else merge_patch
        if self.inference and not self.merge_inference:
            self.merge_patch = 0
        elif self.inference and self.merge_inference:
            self.merge_patch = 1
        else:
            # training; inference with merge infer.
            self.merge_patch = merge_patch

        self.random_sample = random_sample
        self.chat_template = chat_template

        self.xy_int = xy_int

        print(f"Dataset: {dataset}; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        # return self.samples_per_epoch
        if self.random_sample:
            return self.samples_per_epoch
        else:
            return len(self.json_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        while True:
            try:
                return self.get_qwenvl(idx)
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self.json_data) - 1)

    def get_qwenvl(self, idx):
        if not self.inference and self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)
        
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_phi3v = [Image.open(image_path).convert("RGB")]

            if self.crop_min != 1 or self.crop_max != 1:
                image_phi3v[0], item = random_crop_metadata(image_phi3v[0], item, scale_range=(self.crop_min, self.crop_max))
        else:
            image_path = ""
            image_phi3v = None

        element_list = item['element']
        # print(image_phi3v[0].size)

        k = min(self.num_turn, len(element_list))
        assert k > 0
        element_cand = random.choices(element_list, k=k)

        sample_io = np.random.choice(len(self.sample_prob), p=self.sample_prob)
        # element = random.choices(element_list)[0]
        merge_patch = True if random.random() < self.merge_patch else False

        if len(element_cand) == 1:
            element = element_cand[0]
            element_name = element['instruction']
            answer_xy = element['point'] if sample_io in [0, 2] else element['bbox']
            if self.xy_int:
                # answer_xy = [round(x * 1000, 3) for x in answer_xy]
                answer_xy = [int(x * 1000) for x in answer_xy]
            else:
                answer_xy = [round(x, 2) for x in answer_xy]
            if sample_io in [2, 3]:
                element_name, answer_xy = answer_xy, element_name

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_openai_qwen(element_name, img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt)
            prompt = self.processor.tokenizer.apply_chat_template(source, chat_template=self.chat_template,  tokenize=False, add_generation_prompt=True)
            data_dict_q = self.processor(text=prompt, images=image_phi3v, return_tensors="pt",
                                        merge_patch=merge_patch, merge_threshold=self.merge_threshold, merge_random=self.merge_random,
                                        training=not self.inference)
            data_dict_qa, answer = batch_add_answer(data_dict_q, answer_xy, self.processor)
        else:
            element_name_list = [element['instruction'] for element in element_cand]
            if sample_io in [0, 2]:
                answer_xy_list = [element['point'] for element in element_cand]
            else:
                answer_xy_list = [element['bbox'] for element in element_cand]

            if self.xy_int:
                answer_xy_list = [[int(x * 1000) for x in answer_xy] for answer_xy in answer_xy_list]
            else:
                answer_xy_list = [[round(x, 2) for x in answer_xy] for answer_xy in answer_xy_list]

            if sample_io in [2, 3]:
                element_name_list, answer_xy_list = answer_xy_list, element_name_list
            answer_xy = answer_xy_list[0]

            img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
            source = grounding_to_openai_qwen(element_name_list[0], img_dict, sample_io, self.shuffle_image_token, self.xy_int, self.uniform_prompt)
            prompt = self.processor.tokenizer.apply_chat_template(source, chat_template=self.chat_template, tokenize=False, add_generation_prompt=True)
            data_dict_q = self.processor(text=prompt, images=image_phi3v, return_tensors="pt",
                                        merge_patch=merge_patch, merge_threshold=self.merge_threshold, merge_random=self.merge_random,
                                        training=not self.inference)
            data_dict_qa, answer = batch_add_answer_append(data_dict_q, answer_xy, self.processor,
                                                            append_element=element_name_list[1:], 
                                                            append_answer=answer_xy_list[1:])

        max_seq_len = self.processor.tokenizer.model_max_length
        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
            image_sizes=data_dict_qa["image_grid_thw"],
            pixel_values=data_dict_qa["pixel_values"],
            labels=data_dict_qa["labels"][0],        
        )
        assert data_dict_qa["input_ids"][0].shape[0] <= max_seq_len, f"Input seq. is surpass max. seq len: {data_dict_qa['input_ids'][0].shape[0]} > {max_seq_len}"

        if 'patch_assign' in data_dict_q:
            data_dict['patch_assign'] = data_dict_q['patch_assign']
        if 'patch_assign_len' in data_dict_q:
            data_dict['patch_assign_len'] = data_dict_q['patch_assign_len']
        if 'patch_pos' in data_dict_q:
            data_dict['patch_pos'] = data_dict_q['patch_pos']
        if 'select_mask' in data_dict_q:
            data_dict['select_mask'] = data_dict_q['select_mask']

        return (
            data_dict,
            item,
        )

if __name__ == '__main__':
    from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    processor = Qwen2VLProcessor.from_pretrained(
                                            "Qwen/Qwen2-VL-2B-Instruct", 
                                            min_pixels=1024*28*28, 
                                            max_pixels=1024*28*28,
                                            model_max_length=1280)
    processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    dataset = GroundingDataset(
        "showui",
        "/blob/v-lqinghong/data/GUI_database",
        processor,
        num_turn=100,
        # json_data="hf_train_ground",
        json_data="hf_train_4o",
        text2point=1,
        text2bbox=0,
        point2text=0,
        bbox2text=0,
        shuffle_image_token=False,
        uniform_prompt=True,
        crop_min=0.5,
        crop_max=1.5,
        merge_patch=1,
        merge_threshold=1,
        xy_int=False,
    )
    size_dist = {}

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])