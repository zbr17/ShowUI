import json
import os
import re
import pdb
import random

import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import sys
sys.path.append('.')
from data.template import aitw_to_openai, aitw_to_openai_qwenvl, batch_add_answer
from qwen_vl_utils import process_vision_info

def get_answer(step):
    action_type_id = step['action_type_id']
    action_type_text = step['action_type_text']

    click_point = None
    type_text = None
    if action_type_id == 4:
        if action_type_text == 'click':
            touch = step['touch']
            lift = step['lift']
            click_point = [(touch[0] + lift[0]) / 2, (touch[1] + lift[1]) / 2]
            click_point = [round(item, 2) for item in click_point]
    elif action_type_id == 3:
        type_text = step['type_text']

    answer = {'action': action_type_text.upper(), 'value': type_text, 'position': click_point}
    return answer

# def get_history(sample, num_history):
#     step_history = sample['step_history']
#     action_history = []
#     for i, step in enumerate(step_history[-num_history:], start=1):
#         action = get_answer(step)
#         action_history.append(f'Step{i}: {action}')
#     return '; '.join(action_history)

def get_history(sample, num_history, interleaved_history='tttt'):
    if num_history == 0:
        return None
    step_history = sample['step_history']
    action_history = []
    action_prefix = []
    for i, step in enumerate(step_history[-num_history:], start=1):
        action = get_answer(step)
        if interleaved_history == 'vvtt':
            action_prefix.append(f'Step{i}: <|image_{i+1}|>')
        elif interleaved_history == 'ttvv':
            action_prefix.append(f'Step{i}: {action}')

        if interleaved_history in ['tttt', 'vvtt']:
            action_history.append(f'Step{i}: {action}')
        elif interleaved_history in ['vvvv', 'ttvv']:
            action_history.append(f'Step{i}: <|image_{i+1}|>')
        elif interleaved_history == 'vtvt':
            action_history.append(f'Step{i}: <|image_{i+1}|> {action}')
        elif interleaved_history == 'tvtv':
            action_history.append(f'Step{i}: {action} <|image_{i+1}|>')
        # action_history.append(f'Step{i}: {action}')
    tmp_prev = '; '.join(action_prefix)
    tmp_post = '; '.join(action_history)
    tmp = tmp_prev + '; ' + tmp_post if tmp_prev != '' else tmp_post
    return tmp

def draw_history(sample, image_phi3v, num_history, radius_in=10, radius_out=None):
    if num_history == 0:
        assert len(image_phi3v) == 1
        return image_phi3v
    step_history = sample['step_history']
    for i, step in enumerate(step_history[-num_history:], start=1):
        action = get_answer(step)
    
        image = image_phi3v[i]
        draw = ImageDraw.Draw(image)
        width, height = image.size

        point = action['position']
        if point is not None:
            x, y = point[0] * width, point[1] * height
            draw.ellipse((x - radius_in, y - radius_in, x + radius_in, y + radius_in), fill='red', outline='red')
            if radius_out is not None:
                draw.ellipse((x - radius_out, y - radius_out, x + radius_out, y + radius_out), outline='red', width=2)

        # epid = sample['ep_id']
        # image_phi3v[i].save(f'/home/qinghong/example/history/{epid}_{i}_{action}.png')
    return image_phi3v

class AitwDataset(torch.utils.data.Dataset):
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
        json_data="hf_train",
        num_turn=1,
        num_history=4,
        interleaved_history='tttt',
        inference=False,
        draw_history=0,
        random_sample=False,
        decay_factor=1,
        merge_patch=0,
        merge_threshold=0,
        merge_inference=False,
        merge_random=None,
        skip_readme_train=False,
        skip_readme_test=False,
        chat_template="{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ):
        self.processor = processor
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = os.path.join(base_image_dir, 'AITW')
        self.precision = precision

        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)
        self.num_turn = num_turn
        self.num_history = num_history
        self.interleaved_history = interleaved_history
        # the interleaved history should be one of the following
        assert self.interleaved_history in ['tttt', 'vvvv', 'vtvt', 'tvtv', 'vvtt', 'ttvv']
        self.inference = inference
        self.draw_history = draw_history    # the radius of the circle to draw the history

        self.random_sample = random_sample

        # used by qwenvl
        self.decay_factor = decay_factor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.merge_threshold = merge_threshold
        self.merge_inference = merge_inference
        self.merge_random = merge_random
        # self.merge_patch = 0 if self.inference else merge_patch
        if self.inference and not self.merge_inference:
            self.merge_patch = 0
        else:
            # training; inference with merge infer.
            self.merge_patch = merge_patch

        self.skip_readme_train = skip_readme_train
        self.skip_readme_test = skip_readme_test

        self.chat_template = chat_template

        if 'Qwen2VL' in self.processor.image_processor.image_processor_type:
            self.vis_start = self.processor.tokenizer('<|vision_start|>')['input_ids']
            self.vis_end = self.processor.tokenizer('<|vision_end|>')['input_ids']

        print(f"Dataset: Aitw; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        # inference
        if self.inference:
            return len(self.json_data)
    
        # training
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

    def append_history_image(self, sample, num_history, image_phi3v, url_only=False):
        if num_history == 0:
            return image_phi3v
        step_history = sample['step_history']
        for i, step in enumerate(step_history[-num_history:], start=1):
            image_path = os.path.join(self.IMG_DIR, step["img_filename"]+'.png')
            if url_only:
                image_phi3v.append(image_path)
            else:
                image_phi3v.append(Image.open(image_path).convert("RGB"))
        return image_phi3v

    def __getitem__(self, idx):
        # if self.processor.tokenizer.name_or_path in ['Qwen/Qwen2-VL-2B-Instruct']:
        if 'Qwen2VL' in self.processor.image_processor.image_processor_type:
            return self.get_qwenvl(idx)
        else:
            return self.get_phi3v(idx)

    def get_phi3v(self, idx):
        if not self.inference and self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)

        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"]+'.png')
            image_phi3v = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_phi3v = None
        item['img_url_abs'] = image_path

        if self.interleaved_history in ['vvvv', 'vvtt', 'ttvv', 'vtvt', 'tvtv']:
            image_phi3v = self.append_history_image(item, self.num_history, image_phi3v)

        # kevin: 10/10; put the first image to the end; after history
        image_phi3v.append(image_phi3v.pop(0))

        task = item['task']
        answer_dict = get_answer(item)
        action_history = get_history(item, self.num_history, self.interleaved_history) if item['step_history'] != [] else None

        if self.draw_history > 0:
            image_phi3v = draw_history(item, image_phi3v, self.num_history, radius_in=self.draw_history)

        item['anno_id'] = idx
        item['answer'] = answer_dict

        source = aitw_to_openai(task, action_history, None)

        prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(prompt, image_phi3v, return_tensors="pt")

        if self.inference:
            if 'labels' not in data_dict_q:
                data_dict_q['labels'] = data_dict_q['input_ids']

            data_dict = dict(
                input_ids=data_dict_q["input_ids"][0],
                # pixel_values=data_dict_q["pixel_values"][0],
                # image_sizes=data_dict_q["image_sizes"][0],
                pixel_values=data_dict_q["pixel_values"],
                image_sizes=data_dict_q["image_sizes"],
                labels=data_dict_q["input_ids"][0],
            )
            return (
                data_dict,
                item,
        )

        data_dict_qa, answer = batch_add_answer(data_dict_q, answer_dict, self.processor)

        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
        #     pixel_values=data_dict_qa["pixel_values"][0],
        #     image_sizes=data_dict_qa["image_sizes"][0],
            pixel_values=data_dict_qa["pixel_values"],
            image_sizes=data_dict_qa["image_sizes"],
            labels=data_dict_qa["labels"][0],
        )

        return (
            data_dict,
            item,
        )

    def get_history_qwenvl(self, image_list, sample, num_history, interleaved_history='tttt', decay_factor=1):
        # last one is the current image, past are the history
        curr_image = image_list[-1]
        # curr_dict = [{'type': 'image', 'image': curr_image}]
        curr_dict = [{'type': 'image', 'image': curr_image, 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}]
        if num_history == 0 or sample['step_history'] == []:
            assert len(image_list) == 1
            return curr_dict

        step_history = sample['step_history']
        action_history = []
        action_prefix = []
        for i, step in enumerate(step_history[-num_history:]):
            action = get_answer(step)
            max_pixels = max(self.min_pixels, self.max_pixels * decay_factor ** (num_history - i))
            # print(max_pixels, self.min_pixels, self.max_pixels, decay_factor, num_history, i)
            # pdb.set_trace()
            if interleaved_history == 'vvtt':
                action_prefix.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'ttvv':
                action_prefix.append({"type": "text", "text": f'{action}'})

            if interleaved_history in ['tttt', 'vvtt']:
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history in ['vvvv', 'ttvv']:
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'vtvt':
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history == 'tvtv':
                action_history.append({"type": "text", "text": f'{action}'})
                action_history.append({"type": "image", "image": image_list[i], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            # action_history.append(f'Step{i}: {action}')
        # tmp_prev = '; '.join(action_prefix)
        # tmp_post = '; '.join(action_history)
        # tmp = tmp_prev + '; ' + tmp_post if tmp_prev != '' else tmp_post
        tmp = action_prefix + action_history + curr_dict
        return tmp

    def activate_history_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, vis_end: list, vis_start: list):
        # for streaming training;
        # we replace the labels where satisfy <|vision_end|>xxx<|vision_start|> with the input_ids that satisfy the same position;
        assert input_ids.shape == labels.shape, "input_ids and labels must have the same shape"
        
        L = input_ids.shape[1]
        vis_end_len = len(vis_end)
        vis_start_len = len(vis_start)
        
        i = 0
        while i <= L - (vis_end_len + vis_start_len):
            if torch.equal(input_ids[0, i:i+vis_end_len], torch.tensor(vis_end)):
                for j in range(i + vis_end_len, L - vis_start_len + 1):
                    if torch.equal(input_ids[0, j:j+vis_start_len], torch.tensor(vis_start)):
                        labels[0, i+vis_end_len:j] = input_ids[0, i+vis_end_len:j]
                        i = j + vis_start_len - 1
                        break
            i += 1
        return labels

    def get_qwenvl(self, idx):
        if not self.inference and self.random_sample:
            idx = random.randint(0, len(self.json_data) - 1)
        idx = idx % len(self.json_data)

        item = self.json_data[idx]
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"]+'.png')
            image_phi3v = [image_path]
        else:
            image_path = ""
            image_phi3v = None
        item['img_url_abs'] = image_path

        if self.interleaved_history in ['vvvv', 'vvtt', 'ttvv', 'vtvt', 'tvtv']:
            image_phi3v = self.append_history_image(item, self.num_history, image_phi3v, url_only=True)
        # kevin: 10/10; put the first image to the end; after history
        image_phi3v.append(image_phi3v.pop(0))

        task = item['task']
        answer_dict = get_answer(item)
        action_history = self.get_history_qwenvl(image_phi3v, item, self.num_history, self.interleaved_history, self.decay_factor) # if item['step_history'] != [] else image_phi3v

        if self.draw_history > 0:
            raise NotImplementedError
            # image_phi3v = draw_history(item, image_phi3v, self.num_history, radius_in=self.draw_history)

        item['anno_id'] = idx
        item['answer'] = answer_dict

        if self.inference:
            skip_readme = self.skip_readme_test
        else:
            skip_readme = self.skip_readme_train
        source = aitw_to_openai_qwenvl(task, action_history, None, skip_readme)
        # pdb.set_trace()

        # qwen2-vl ft miss the chat template
        if self.processor.chat_template is None:
            self.processor.chat_template = self.chat_template
        # prompt = self.processor.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        prompt = self.processor.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(source)

        # data_dict_q = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        merge_patch = True if random.random() < self.merge_patch else False

        data_dict_q = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt",
                                        merge_patch=merge_patch, merge_threshold=self.merge_threshold, merge_random=self.merge_random,
                                        training=not self.inference)
        # print(len(image_inputs),  data_dict_q['pixel_values'].shape, data_dict_q['image_grid_thw'])
        if self.inference:
            if 'labels' not in data_dict_q:
                data_dict_q['labels'] = data_dict_q['input_ids']

            # patch_assign = data_dict_q['patch_assign'] if merge_patch else None
            # patch_assign_len = data_dict_q['patch_assign_len'] if merge_patch else None
            data_dict = dict(
                input_ids=data_dict_q["input_ids"][0],
                # pixel_values=data_dict_q["pixel_values"][0],
                image_sizes=data_dict_q["image_grid_thw"],
                pixel_values=data_dict_q["pixel_values"],
                # image_sizes=data_dict_q["image_sizes"],
                labels=data_dict_q["input_ids"][0],
                # patch_assign=patch_assign,
                # patch_assign_len=patch_assign_len,
                # patch_pos=data_dict_q['patch_pos'],
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

        data_dict_qa, answer = batch_add_answer(data_dict_q, answer_dict, self.processor)
        # print(answer)
        # print(merge_patch, data_dict_qa["input_ids"][0].shape, data_dict_qa["image_grid_thw"], data_dict_q['patch_assign_len'].sum(), [x.size for x in image_inputs])
        # # print(data_dict_qa["input_ids"][0].shape)

        if self.num_turn > 1:
            # print(data_dict_qa["labels"].tolist())
            # print((data_dict_qa["labels"]==-100).sum())
            data_dict_qa["labels"] = self.activate_history_labels(data_dict_qa["input_ids"], data_dict_qa["labels"], self.vis_end, self.vis_start)
            # print((data_dict_qa["labels"]==-100).sum())
            # print(data_dict_qa["labels"].tolist())
            # pdb.set_trace()

        data_dict = dict(
            input_ids=data_dict_qa["input_ids"][0],
        #     pixel_values=data_dict_qa["pixel_values"][0],
            image_sizes=data_dict_qa["image_grid_thw"],
            pixel_values=data_dict_qa["pixel_values"],
            # image_sizes=data_dict_qa["image_sizes"],
            labels=data_dict_qa["labels"][0],
            # patch_assign=data_dict_q['patch_assign'],
            # patch_assign_len=data_dict_q['patch_assign_len'],
            # patch_pos=data_dict_q['patch_pos'],
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

if __name__ == '__main__':
    # from transformers import AutoProcessor  # do not remove this line
    # from model.phi_3_vision.processing_phi3_v import Phi3VProcessor
    # processor = Phi3VProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct",
    #                                             num_crops=4)
    from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

    processor = Qwen2VLProcessor.from_pretrained(
                                                "/blob/v-lqinghong/experiments/VideoVLA/aitw_qwen_bs1g2_max1600_hist0_m0.5/2024-10-19_06-18-28/ckpt_model/merged_model",
                                                max_pixels=1000*28*28,
                                                model_max_length=4096)
    # from model.phi_35_vision.processing_phi3_v import Phi3VProcessor
    # processor = Phi3VProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct",
    #                                         #    padding_side='right',
    #                                            model_max_length=4096,
    #                                            num_crops=16)

    # use unk rather than eos token to prevent endless generation
    # processor.tokenizer.pad_token = processor.tokenizer.unk_token
    # processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    # processor.tokenizer.padding_side = 'right'

    dataset = AitwDataset(
        "/blob/v-lqinghong/data/GUI_database",
        processor,
        json_data="hf_train",
        inference=True,
        num_turn=4,
        num_history=4,
        interleaved_history='vtvt',
        draw_history=0,
        random_sample=False,
        # decay_factor=0.5,
        merge_patch=1,
        # decay_factor=1,
        merge_threshold=1,
        merge_inference=True,
        merge_random="shuffle",
        skip_readme_train=True,
        skip_readme_test=False,
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        # print(data[0]['pixel_values'].shape, data[0]['input_ids'].shape)
        # pdb.set_trace()