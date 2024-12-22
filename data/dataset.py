import numpy as np
import torch

import pdb
import sys
sys.path.append(".")
from data.data_utils import IGNORE_INDEX
from data.dset_miniwob import MiniWobDataset
from data.dset_aitw import AitwDataset
from data.dset_aitz import AitzDataset
from data.dset_omniact import OmniActDataset
from data.dset_seeclick import SeeClickDataset
from data.dset_mind2web import Mind2WebDataset
from data.dset_screenspot import ScreenSpotDataset
from data.dset_odyssey import OdysseyDataset
from data.dset_guiworld import GUIWorldDataset
from data.dset_act2cap import Act2CapDataset

from data.dset_shared_grounding import GroundingDataset
from data.dset_shared_onestep import OneStepDataset
from data.dset_shared_captioning import CaptioningDataset
from data.dset_shared_navigation import NavigationDataset
from data.dset_shared_chat import ChatDataset
from data.dset_shared_llava import LLaVADataset
from data.dset_xlam import XLAMDataset

from transformers import AutoProcessor

def collate_fn(batch, processor=None):
    batch_data_phi3v = [x[0] for x in batch]

    input_ids, labels = tuple([instance[key] for instance in batch_data_phi3v]
                              for key in ("input_ids", "labels"))
    
    # print(processor.tokenizer.pad_token_id)
    padding_value=processor.tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=padding_value)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :processor.tokenizer.model_max_length]
    labels = labels[:, :processor.tokenizer.model_max_length]

    # so far, do not imple mixed text, image, video training
    # text
    if batch_data_phi3v[0]['pixel_values'] is None:
        pixel_values = None
        image_sizes = None
    # qwenvl
    elif len(batch_data_phi3v[0]['pixel_values'].shape) == 2:
        pixel_values = [instance["pixel_values"] for instance in batch_data_phi3v]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in batch_data_phi3v]
        image_sizes = torch.cat(image_sizes, dim=0)
    # image 
    elif len(batch_data_phi3v[0]['pixel_values'].shape) == 4:
        pixel_values = [instance["pixel_values"] for instance in batch_data_phi3v]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in batch_data_phi3v]
        image_sizes = torch.stack(image_sizes, dim=0)
        # print(pixel_values.shape, image_sizes.shape)
    # multi-image, video
    elif len(batch_data_phi3v[0]['pixel_values'].shape) == 5:
        pixel_values = [instance["pixel_values"] for instance in batch_data_phi3v]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in batch_data_phi3v]
        image_sizes = torch.cat(image_sizes, dim=0)
    # print(pixel_values.shape, image_sizes.shape)

    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

    # batch_data_sam = [x[1] for x in batch]
    # image_sam_list, masks_list, \
    # seg_label_list, resize_list = tuple([instance[key] for instance in batch_data_sam]
    #                                     for key in ("image_sam", "masks", "seg_labels", "resize"))

    meta_data = [x[-1] for x in batch]

    result = {
        # "images_sam": torch.stack(image_sam_list, dim=0),
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        # "attention_mask": attention_mask,
        # "masks_list": masks_list,
        # "label_list": seg_label_list,
        # "resize_list": resize_list,
        "image_sizes": image_sizes,
        "meta_data": meta_data,
    }

    for key in ['patch_assign', 'patch_assign_len']:
        if key in batch_data_phi3v[0]:
            key_tmp = [instance[key] for instance in batch_data_phi3v]
            key_tmp = torch.cat(key_tmp, dim=0)
            result[key] = key_tmp 

    # if "patch_assign" in batch_data_phi3v[0]:
    #     patch_assign = [instance["patch_assign"] for instance in batch_data_phi3v]
    #     patch_assign = torch.cat(patch_assign, dim=0)
    #     result["patch_assign"] = patch_assign
    #     patch_assign_len = [instance["patch_assign_len"] for instance in batch_data_phi3v]
    #     patch_assign_len = torch.cat(patch_assign_len, dim=0)
    #     result["patch_assign_len"] = patch_assign_len

    for key, pad_val in zip(['patch_pos', 'select_mask'], [-1, True]):
        if key in batch_data_phi3v[0]:
            key_tmp = [instance[key] for instance in batch_data_phi3v]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=pad_val) for pos in key_tmp]
            key_tmp = torch.cat(padded_key, dim=0)
            result[key] = key_tmp

    # pad the answer by -1;
    # if "patch_pos" in batch_data_phi3v[0]:
    #     patch_pos = [instance["patch_pos"] for instance in batch_data_phi3v]
    #     max_length = input_ids.size(1)
    #     if patch_pos[0].dtype == torch.bool:
    #         padded_patch_pos = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in patch_pos]
    #     else:
    #         padded_patch_pos = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in patch_pos]
    #     patch_pos = torch.cat(padded_patch_pos, dim=0)
    #     result["patch_pos"] = patch_pos
    return result

class HybridDataset(torch.utils.data.Dataset):
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
        dataset="mind2web,seeclick",
        sample_rate=[1,1],
        # randomly sample
        uniform_sample=False,
        random_sample=False,
        record_sample=False,
        miniwob_data="hf_train",
        assistgui_data="hf_train_full",
        seeclick_data="hf_train",
        aitw_data="hf_train",
        aitz_data="hf_train",
        mind2web_data="hf_train",
        screenspot_data="hf_test_full",
        rico_data="hf_train_rico",
        ricosca_data="hf_train_ricosca",
        widget_data="hf_train_widget",
        screencap_data="hf_train_screencap",
        guienv_data="hf_train",
        synthesis_data="hf_train_10k",
        guiact_data="hf_train_web-single",
        guiact_g_data="hf_train_web-single_ground",
        guichat_data="hf_train",
        llava_data="llava_v1_5_mix665k",
        guiworld_data="hf_train",
        act2cap_data="hf_train",
        showui_data="hf_train",
        omniact_data="hf_train_ground",
        omniact_nav_data="hf_train_ground",
        amex_data="hf_train_ele",
        amexcap_data="hf_train_cap",
        odyssey_data="hf_train_random",
        guiexp_data="hf_train_ground",
        guiexpweb_data="hf_train_raw",
        osatlas_data="hf_desktop",
        xlam_data="hf_train",
        inference=False,
        uniform_prompt=False,
        num_turn=1,
        text2point=1,
        text2bbox=0,
        point2text=0,
        bbox2text=0,
        shuffle_image_token=False,
        # aitz
        prob_plan=1,
        prob_cap=1,
        prob_res=1,
        prob_think=1,
        # grounding crop
        crop_min=1,
        crop_max=1,
        # vid
        num_frames=4,
        max_frames=16,
        frame_sampling="uniform",
        # interleaved history
        num_history=4,
        interleaved_history='tttt',
        draw_history=0,
        # by qwen
        decay_factor=1,
        merge_patch=0,
        merge_threshold=0,
        merge_inference=False,
        merge_random=None,
        xy_int=False,   # only apply on seeclick for exps;
        # 
        skip_readme_train=False,
        skip_readme_test=False,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        self.uniform_sample = uniform_sample
        self.random_sample = random_sample
        self.record_sample = record_sample
        self.random_len_per_dset = self.random_sample and not self.record_sample,

        self.base_image_dir = base_image_dir
        self.precision = precision

        self.datasets =  [item for item in dataset.split(",") if item]
        assert len(self.datasets) == len(sample_rate)
        self.inference = inference
        self.num_turn = num_turn
        self.uniform_prompt = uniform_prompt

        self.text2point = text2point
        self.text2bbox = text2bbox
        self.point2text = point2text
        self.bbox2text = bbox2text
        self.shuffle_image_token = shuffle_image_token

        # aitz
        self.prob_cap = prob_cap
        self.prob_res = prob_res
        self.prob_plan = prob_plan
        self.prob_think = prob_think

        # vid
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.frame_sampling = frame_sampling

        # 
        self.num_history = num_history
        self.interleaved_history = interleaved_history
        self.draw_history = draw_history

        # by qwen
        self.decay_factor = decay_factor
        self.merge_patch = merge_patch
        self.merge_threshold = merge_threshold
        self.merge_inference = merge_inference
        self.merge_random = merge_random

        # 
        self.skip_readme_train = skip_readme_train
        self.skip_readme_test = skip_readme_test

        self.xy_int = xy_int

        if self.inference:
            self.shuffle_image_token = False
            assert len(self.datasets) == 1

        # allow multiple split for one dataset
        for dataset_ms in ['guienv']:
            if dataset_ms in self.datasets:
                gui_env_list = [item for item in guienv_data.split(",") if item]
                assert len(gui_env_list) == self.datasets.count(dataset_ms)
        for dataset_ms in ['guiact']:
            if dataset_ms in self.datasets:
                gui_act_list = [item for item in guiact_data.split(",") if item]
                assert len(gui_act_list) == self.datasets.count(dataset_ms)
        for dataset_ms in ['amex']:
            if dataset_ms in self.datasets:
                amex_list = [item for item in amex_data.split(",") if item]
                assert len(amex_list) == self.datasets.count(dataset_ms)
        for dataset_ms in ['osatlas']:
            if dataset_ms in self.datasets:
                osatlas_list = [item for item in osatlas_data.split(",") if item]
                assert len(osatlas_list) == self.datasets.count(dataset_ms)

        if self.inference:
            print(f"Loading {len(self.datasets)} Validation Datasets")
        else:
            print(f"Loading {len(self.datasets)} Training Datasets")

        self.all_datasets = []
        dataset_queue = []
        for dataset in self.datasets:
            dataset_queue.append(dataset)
            # Pretraining / SFT
            if dataset == "seeclick":
                self.all_datasets.append(
                    # SeeClickDataset(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        seeclick_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        xy_int=self.xy_int,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "rico":
                self.all_datasets.append(
                    # RicoDataset(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        rico_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        xy_int=self.xy_int,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "assistgui":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        assistgui_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,        
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "synthesis":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        synthesis_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "guiexp":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guiexp_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "guiexpweb":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guiexpweb_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "omniact":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        omniact_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "showui":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        showui_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "guienv":
                guienv_data_tmp = gui_env_list[dataset_queue.count(dataset) - 1]
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guienv_data_tmp,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "osatlas":
                osatlas_data_tmp = osatlas_list[dataset_queue.count(dataset) - 1]
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        osatlas_data_tmp,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "guiact_g":
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guiact_g_data,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "guiact":
                guiact_data_tmp = gui_act_list[dataset_queue.count(dataset) - 1]
                self.all_datasets.append(
                    NavigationDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guiact_data_tmp,
                        num_turn=self.num_turn,
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        decay_factor=self.decay_factor,
                        inference=self.inference,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        skip_readme_train=self.skip_readme_train,
                        skip_readme_test=self.skip_readme_test,
                    )
                )
            elif dataset == "guichat":
                self.all_datasets.append(
                    ChatDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guichat_data,
                        inference=self.inference,
                        shuffle_image_token=self.shuffle_image_token,
                        chat_ground=True,
                        chat_ground_point=0.5,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "llava":
                self.all_datasets.append(
                    LLaVADataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        llava_data,
                        inference=self.inference,
                        shuffle_image_token=self.shuffle_image_token,
                        # chat_ground=True,
                        # chat_ground_point=0.5,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "guiworld":
                self.all_datasets.append(
                    GUIWorldDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        guiworld_data,
                        inference=self.inference,
                        shuffle_image_token=self.shuffle_image_token,
                        num_frames=self.num_frames,
                        max_frames=self.max_frames,
                        frame_sampling=self.frame_sampling,
                        random_sample=False,
                    )
                )
            elif dataset == "act2cap":
                self.all_datasets.append(
                    Act2CapDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        act2cap_data,
                        inference=self.inference,
                        shuffle_image_token=self.shuffle_image_token,
                        num_frames=self.num_frames,
                        max_frames=self.max_frames,
                        frame_sampling=self.frame_sampling,
                        random_sample=False,
                    )
                )
            elif dataset == "amex":
                amex_data_tmp = amex_list[dataset_queue.count(dataset) - 1]
                self.all_datasets.append(
                    GroundingDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        amex_data_tmp,
                        inference=False,
                        num_turn=self.num_turn,
                        text2point=self.text2point,
                        text2bbox=self.text2bbox,
                        point2text=self.point2text,
                        bbox2text=self.bbox2text,
                        shuffle_image_token=self.shuffle_image_token,
                        crop_min=crop_min,
                        crop_max=crop_max,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        uniform_prompt=self.uniform_prompt
                    )
                )
            elif dataset == "ricosca":
                self.all_datasets.append(
                GroundingDataset(
                    dataset,
                    base_image_dir,
                    processor,
                    samples_per_epoch,
                    precision,
                    ricosca_data,
                    inference=False,
                    num_turn=self.num_turn,
                    text2point=self.text2point,
                    text2bbox=self.text2bbox,
                    point2text=self.point2text,
                    bbox2text=self.bbox2text,
                    shuffle_image_token=self.shuffle_image_token,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    random_sample=False,
                    merge_patch=self.merge_patch,
                    merge_threshold=self.merge_threshold,
                    merge_inference=self.merge_inference,
                    merge_random=self.merge_random,
                    uniform_prompt=self.uniform_prompt
                )
            )
            elif dataset == "widget":
                self.all_datasets.append(
                GroundingDataset(
                    dataset,
                    base_image_dir,
                    processor,
                    samples_per_epoch,
                    precision,
                    widget_data,
                    inference=False,
                    num_turn=self.num_turn,
                    text2point=self.text2point,
                    text2bbox=self.text2bbox,
                    point2text=self.point2text,
                    bbox2text=self.bbox2text,
                    shuffle_image_token=self.shuffle_image_token,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    random_sample=False,
                    merge_patch=self.merge_patch,
                    merge_threshold=self.merge_threshold,
                    merge_inference=self.merge_inference,
                    merge_random=self.merge_random,
                    uniform_prompt=self.uniform_prompt
                )
            )
            elif dataset == "screencap":
                self.all_datasets.append(
                    CaptioningDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        json_data=screencap_data,
                        inference=False,
                        shuffle_image_token=self.shuffle_image_token,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "amexcap":
                self.all_datasets.append(
                    CaptioningDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        json_data=amexcap_data,
                        inference=False,
                        shuffle_image_token=self.shuffle_image_token,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            # Downstream task
            elif dataset == "miniwob":
                self.all_datasets.append(
                    MiniWobDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        miniwob_data,
                        inference=False,
                        num_turn=self.num_turn,
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        draw_history=self.draw_history,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "mind2web":
                self.all_datasets.append(
                    Mind2WebDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        mind2web_data,
                        inference=self.inference,
                        num_history=self.num_history,
                        num_turn=self.num_turn,
                        interleaved_history=self.interleaved_history,
                        draw_history=self.draw_history,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "aitw":
                self.all_datasets.append(
                    AitwDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        aitw_data,
                        inference=self.inference,
                        num_turn=self.num_turn,
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        draw_history=self.draw_history,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        skip_readme_train=self.skip_readme_train,
                        skip_readme_test=self.skip_readme_test,
                    )
                )
            elif dataset == "omniact_nav":
                self.all_datasets.append(
                    OmniActDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        omniact_nav_data,
                        inference=self.inference,
                        num_turn=self.num_turn,
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        draw_history=self.draw_history,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "odyssey":
                self.all_datasets.append(
                    OdysseyDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        odyssey_data,
                        inference=self.inference,
                        num_turn=self.num_turn,                        
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        draw_history=self.draw_history,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            elif dataset == "aitz":
                self.all_datasets.append(
                    AitzDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        aitz_data,
                        inference=self.inference,
                        num_turn=self.num_turn,
                        num_history=self.num_history,
                        interleaved_history=self.interleaved_history,
                        prob_cap=self.prob_cap,
                        prob_res=self.prob_res,
                        prob_plan=self.prob_plan,
                        prob_think=self.prob_think,
                        random_sample=False,
                        decay_factor=self.decay_factor,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                    )
                )
            # Zero-shot evaluation
            elif dataset == "screenspot":
                self.all_datasets.append(
                    ScreenSpotDataset(
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        screenspot_data,
                        inference=True,
                        random_sample=False,
                        merge_patch=self.merge_patch,
                        merge_threshold=self.merge_threshold,
                        merge_inference=self.merge_inference,
                        merge_random=self.merge_random,
                        xy_int=self.xy_int
                    )
                )
            elif dataset == "xlam":
                self.all_datasets.append(
                    XLAMDataset(
                        dataset,
                        base_image_dir,
                        processor,
                        samples_per_epoch,
                        precision,
                        xlam_data,
                        inference=False,
                    )
                )

        self.sample_recorder = [set() for _ in range(len(self.all_datasets))]  # 用于记录每个数据集的采样索引

    def __len__(self):
        total_len = sum([len(dataset) for dataset in self.all_datasets])
        if self.inference:
            return total_len

        if self.random_sample:
            return self.samples_per_epoch
        else:
            return total_len

    def refresh_sample_recorder(self, ind):
        print(f"Refresh sample recorder for dataset {ind}")
        self.sample_recorder[ind] = set()

    def __getitem__(self, idx):
        # A: Prob. sample + fixed steps                 -- random_sample
        # B: Prob. sample + fixed steps + unrepeated;   -- random_sample & record_sample
        # C: Concat sample unrepeated once;             -- not random_sample

        if not self.inference:
            if self.uniform_sample:
                # pdb.set_trace()
                sample_rate = np.array([len(x) for x in self.all_datasets])
                self.sample_rate = sample_rate / sample_rate.sum()
                print(f"Using uniform sampling with default ratio: {self.sample_rate}.")

            # training with randomly sampling
            if self.random_sample and not self.record_sample:
                ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
                data = self.all_datasets[ind]

                data_main, data_meta = data[idx]
                return data_main, data_meta

            # training with randomly sampling but not repeatly
            elif self.random_sample and self.record_sample:
                ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
                data = self.all_datasets[ind]
                unseen_indices = set(range(len(data))) - self.sample_recorder[ind]
                if not unseen_indices:
                    self.refresh_sample_recorder(ind)
                    unseen_indices = set(range(len(data)))
                sample_idx = np.random.choice(list(unseen_indices))
                self.sample_recorder[ind].add(sample_idx)
                data_main, data_meta = data[sample_idx]
                # print("Sampled index: ", sample_idx)
                # print("Unseen indices: ", len(unseen_indices))
                return data_main, data_meta

            # training with all datasets concated once
            elif not self.random_sample:
                if idx >= len(self):
                    raise IndexError
                for data in self.all_datasets:
                    if idx < len(data):
                        data_main, data_meta = data[idx]
                        return data_main, data_meta
                    else:
                        idx -= len(data)

        # evaluation with only one dataset;
        elif self.inference:
            assert len(self.all_datasets) == 1
            data = self.all_datasets[0]
            data_main, data_meta = data[idx]
            return data_main, data_meta


if __name__ == '__main__':
    from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
    from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    processor = Qwen2VLProcessor.from_pretrained(
                                            "Qwen/Qwen2-VL-2B-Instruct", 
                                            min_pixels=672*28*28, 
                                            max_pixels=3600*28*28,
                                            model_max_length=4096)

    train_dataset = HybridDataset(
        "/blob/v-lqinghong/data/GUI_database",
        processor,
        samples_per_epoch=10000,
        precision="bf16",
        # dataset="guienv,guienv",
        # dataset="guienv,ricosca",
        # dataset="amex,amex",
        # sample_rate=[1,1],
        dataset="showui,amex,guiexp",
        guiexpweb_data="hf_train_raw",
        sample_rate=[1,1,1],
        uniform_sample=True,     
        random_sample=True,
        record_sample=True,
        # amex_data='hf_train_raw,hf_train_raw',
        # guienv_data="hf_train_guienv_stage1,hf_train_guienv_stage2",
        # guienv_data="hf_train_guienv_stage1",
        # text2point=1,
        # text2bbox=1,
        # point2text=1,
        # bbox2text=1,
        shuffle_image_token=True
    )
    for i in range(len(train_dataset)):
        item = train_dataset.__getitem__(0)
        # import pdb
        # pdb.set_trace()