import json
import os
import re
import pdb
import random

import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from IPython.display import display

import sys
sys.path.append('.')
from data.template import screenspot_to_openai, screenspot_to_openai_qwen, batch_add_answer

class ScreenSpotDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        processor,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        json_data="hf_test_full",
        inference=True,
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

        self.base_image_dir = os.path.join(base_image_dir, 'ScreenSpot')
        self.precision = precision

        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)
        self.inference = inference
        assert self.inference == True # only support inference mode

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

        self.chat_template = chat_template
        self.xy_int = xy_int
        print(f"Dataset: Screenspot; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        if 'Qwen2VL' in self.processor.image_processor.image_processor_type:
            return self.get_qwenvl(idx)
        else:
            return self.get_phi3v(idx)

    def get_qwenvl(self, idx):
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_phi3v = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_phi3v = None
        item['img_url_abs'] = image_path
        # item['idx'] = idx

        merge_patch = True if random.random() < self.merge_patch else False

        task = item['task']
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        source = screenspot_to_openai_qwen(task, img_dict, self.xy_int)
        prompt = self.processor.tokenizer.apply_chat_template(source, chat_template=self.chat_template,  tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(text=prompt, images=image_phi3v, return_tensors="pt",
                                        merge_patch=merge_patch, merge_threshold=self.merge_threshold, merge_random=self.merge_random,
                                        training=not self.inference)

        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        # print(prompt)
        # print(merge_patch, data_dict_q["input_ids"][0].shape, data_dict_q["image_grid_thw"], data_dict_q['patch_assign_len'].sum(), [x.size for x in image_phi3v])

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"],
            image_sizes=data_dict_q["image_grid_thw"],
            labels=data_dict_q["labels"][0],
        )

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

    def get_phi3v(self, idx):
        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_phi3v = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_phi3v = None
        item['img_url_abs'] = image_path
        # item['idx'] = idx

        task = item['task']
        source = screenspot_to_openai(task, None)
        prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(prompt, image_phi3v, return_tensors="pt")

        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"][0],
            image_sizes=data_dict_q["image_sizes"][0],
            labels=data_dict_q["labels"][0],
        )
        # print(prompt)
        return (
            data_dict,
            item,
        )

def draw_points(image_input, pred=None, radius=6, scaled=True):
    if isinstance(image_input, str):
        if image_input.startswith("http://") or image_input.startswith("https://"):
            response = requests.get(image_input)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path, URL, or PIL image object")
    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    # font = ImageFont.load_default()

    if pred:
        if len(pred) == 2:
            # pred is a point
            if scaled:
                pred_x, pred_y = int(pred[0] * width), int(pred[1] * height)
            else:
                pred_x, pred_y = int(pred[0]), int(pred[1])
            draw.ellipse((pred_x-radius, pred_y-radius, pred_x+radius, pred_y+radius), fill="red", outline="red")
            # draw.text((pred_x + 10, pred_y - 10), "Pred", f|ill="blue", font=font)
        elif len(pred) == 4:
            # pred is a box
            x1, y1 = int(pred[0] * width), int(pred[1] * height)
            x2, y2 = int(pred[2] * width), int(pred[3] * height)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            # draw.text((x1 + 10, y1 - 10), "Pred", fill="blue", font=font)
    
    # display(img)
    return img

class ScreenSpotZoomIn(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        processor,
        json_data="hf_test_full",
        inference=True,
    ):
        self.processor = processor

        self.base_image_dir = os.path.join(base_image_dir, 'ScreenSpot')

        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)
        self.inference = inference
        assert self.inference == True # only support inference mode

        print(f"Dataset: Screenspot; Split: {json_data}; # samples: {len(self.json_data)}")
    
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, item, click_seq, ratio_seq):
        # click_seq, ratio_seq = item['click_seq'], item['ratio_seq']
        image_crop, crop_size = self.get_zoomin_crop(item, click_seq, ratio_seq)
        image_phi3v = [image_crop]
        item['crop_size'] = crop_size

        task = item['task']
        source = screenspot_to_openai(task, None)
        prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(prompt, image_phi3v, return_tensors="pt")

        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"][0],
            image_sizes=data_dict_q["image_sizes"][0],
            labels=data_dict_q["labels"][0],
        )
        return (
            data_dict,
            item,
        )

    def get_zoomin_crop(self, item, click_seq, ratio_seq, alpha=4):
        """
        'ratio_seq' so far is the h, w of the region before cropping,
        so we set the default alpha=4 to ensure each crop results a half size of the previous area
        """
        img = Image.open(item['img_url_abs']).convert("RGB")

        for click, ratio in zip(click_seq, ratio_seq):
            width, height = img.size

            x = click[0] * width
            y = click[1] * height

            left = max(int(x - ratio[0] // alpha), 0)
            right = min(int(x + ratio[0] // alpha), width)
            upper = max(int(y - ratio[1] // alpha), 0)
            lower = min(int(y + ratio[1] // alpha), height)

            img = img.crop((left, upper, right, lower))
        return img, list(img.size)

    def get_ref_click(self, item, click_seq, ratio_seq, alpha=4):
        """
        'ratio_seq' so far is the h, w of the region before cropping,
        so we set the default alpha=4 to ensure each crop results a half size of the previous area
        """
        img = Image.open(item['img_url_abs']).convert("RGB")
        og_width, og_height = ratio_seq[0]

        ref_seq = []
        ref_left, ref_upper = 0, 0

        for click, ratio in zip(click_seq, ratio_seq):
            width, height = img.size

            x = click[0] * width
            y = click[1] * height

            left = max(int(x - ratio[0] // alpha), 0)
            right = min(int(x + ratio[0] // alpha), width)
            upper = max(int(y - ratio[1] // alpha), 0)
            lower = min(int(y + ratio[1] // alpha), height)

            x_abs = ref_left + x
            y_abs = ref_upper + y

            x_scale = round(x_abs / og_width, 2)
            y_scale = round(y_abs / og_height, 2)
            ref_seq.append([x_scale, y_scale])

            ref_left += left
            ref_upper += upper

            # pdb.set_trace()
            # img_crop = draw_points(img, click)
            # img_crop.save("img_crop.png")
            # img_crop = draw_points(item['img_url_abs'], [x_scale, y_scale])
            # img_crop.save("img_crop.png")

            img = img.crop((left, upper, right, lower))
        # return img, ref_seq
        return ref_seq

if __name__ == '__main__':
    from transformers import AutoProcessor  # do not remove this line
    # from model.phi_3_vision.processing_phi3_v import Phi3VProcessor
    # from model.phi_35_vision.processing_phi3_v import Phi3VProcessor
    # processor = Phi3VProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct",
    # # processor = Phi3VProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct",
    #                                         #    padding_side='right',
    #                                            model_max_length=4096)

    from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    processor = Qwen2VLProcessor.from_pretrained(
                                            "Qwen/Qwen2-VL-2B-Instruct", 
                                            min_pixels=672*28*28, 
                                            max_pixels=3600*28*28,
                                            model_max_length=4096)
    processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


    dataset = ScreenSpotDataset(
        "/blob/v-lqinghong/data/GUI_database",
        processor,
        merge_inference=True,
        merge_patch=1,
        merge_threshold=1,
        xy_int=True,
    )

    size_dist = {}
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        # print(data[0]['input_ids'].shape)
        data_size = str(data[1]['img_size'])
        if data_size not in size_dist:
            size_dist[data_size] = 0
        size_dist[data_size] += 1
        # print(size_dist)