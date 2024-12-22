# ShowUI Training codebase
## Install

```
conda create -n showui python=3.10
conda activate showui
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --user
pip install -r requirements.txt --user
```

Provide wandb key under `train.py`.

## Setup Datasets
Download grounding training dataset -- [ShowUI-desktop-8K](https://huggingface.co/datasets/showlab/ShowUI-desktop-8K).
Download grounding evaluation dataset -- [ScreenSpot](https://huggingface.co/datasets/KevinQHLin/ScreenSpot)

```
huggingface-cli download showlab/ShowUI-desktop-8K --repo-type dataset --local-dir .
huggingface-cli download KevinQHLin/ScreenSpot --repo-type dataset --local-dir .
```

and organize them as following structure:
```
[any _DIR you want]
    - ScreenSpot
        - images
        - metadata
    - ShowUI-desktop
        - images
        - metadata
```

## Create the metadata
You need to assign the download dataset path in the following py file.
```
python3 hf_screenspot.py
python3 hf_mind2web.py
```

For omniact, we provided the existed files for reference. Please refer the data structure, if you want to customize your own dataset;

## Define Dataloader
You can simply re-use existed implementation of `dset_shared_grounding.py` for UI grounding;
or `dset_shared_navigation.py` for UI navigation;

For grounding, you just need to define the dataset_mapping for path identification such as `"seeclick": "SeeClick"`

Please organize the UI grounding metadata as following:
```
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
}
"""
```

For navigation, you need to define the dataset_mapping as above;
Beside, you need to define the action space in `template/shared_navigation.py` for your customized scenario.

## Start Training
Below are instruction for training on grounding then evaluation on screenspot grounding;

Please keep the bsz as 1, if you want to enlarge the bsz, just increase the grad_accumulation_steps.
```
deepspeed --include localhost:1 --master_port 4221 train.py \
  --model_id='Qwen/Qwen2-VL-2B-Instruct' \
  --version='Qwen/Qwen2-VL-2B-Instruct' \
  --dataset_dir='/blob/v-lqinghong/data/GUI_database' \
  --log_base_dir='/blob/v-lqinghong/experiments/VideoVLA' \
  --epochs=1 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=1 \
  --model_max_length=4096 \
  --val_dataset="screenspot" \
  --val_omniact_nav_data="hf_test" \
  --exp_id="debug" \
  --sample_rates="1"  \
  --dataset="omniact"  \
  --omniact_data="hf_train_showui_desktop"  \
  --amex_data="hf_train_ele,hf_train_func"  \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=4 \
  --lora_r=8 \
  --lora_alpha=64  \
  --min_visual_tokens=256  \
  --max_visual_tokens=1344  \
  --num_history=4 \
  --num_turn=1 \
  --interleaved_history='tttt' \
  --crop_min=0.5 \
  --crop_max=1.5 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt  \
  --debug \
  --ds_zero="zero2" \
  --gradient_checkpointing
```

If you want to evaluate on your own setting, you need to define the evaluation function and place it under `main/eval_X.py`

Then, you should able monitor the training information in wandb panel.

## Save the model checkpoint;
Please refer to the `merge.sh`, which provide the instruction, how can you convert ds weight to pytorch.bin or hf model package.
