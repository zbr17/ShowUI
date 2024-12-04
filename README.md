<img src="examples/showui.png" alt="ShowUI" width="640">

ShowUI is a lightweight vision-language-action model for GUI agents. 

You can easily run ShowUI on [Windows](#usage) and [macOS](#macOS)! For macOS users, please set the torch device to "mps".

[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ShowUI--2B-blue)](https://huggingface.co/showlab/ShowUI-2B)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2411.17465-b31b1b.svg)](https://arxiv.org/abs/2411.17465)&nbsp;
[![demo](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/showlab/ShowUI)&nbsp;
[![dataset](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/showlab/ShowUI-desktop-8K)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/KevinQHLin/status/1861664707654099114)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2FShowUI&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

**If you like our project, please give us a star ⭐ for the latest update.**

### 📢 News
<!--  -->
- [x] [2024.11.16] `showlab/ShowUI-2B` is available at huggingface.
- [x] [2024.11.27] We release the [arXiv paper](https://arxiv.org/abs/2411.17465), [HF Spaces demo](https://huggingface.co/spaces/showlab/ShowUI) and [`ShowUI-desktop-8K`](https://huggingface.co/datasets/showlab/ShowUI-desktop-8K).
- [x] [2024.12.1] We support iterative refinement to improve grounding accuracy. Try it at [HF Spaces demo](https://huggingface.co/spaces/showlab/ShowUI).

### 📝 Todo
- [ ] Support [OOTB](https://github.com/showlab/computer_use_ootb?tab=readme-ov-file) for local run.
- [ ] Support UI-Graph Token selection for Efficient Inference.
- [ ] Release fine-tuned code and instructions.

## ⭐ Quick Start

### 1. Prerequisites
- Install miniconda on your system through this [link](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=topnavalldocs). (**Python Version: >= 3.11**).

#### 1.1. Clone the Repository 📂
Open the Conda Terminal. (After installation Of Miniconda, it will appear in the Start menu.)
Run the following command on **Conda Terminal**.
```bash
git clone https://github.com/showlab/ShowUI.git
cd ShowUI
```

#### 1.2. Install Dependencies 🔧
```bash
pip install -r requirements.txt
```

#### 1.3. Start the Interface ▶️
```bash
python app.py
```
If you successfully start the interface, you will see two URLs in the terminal:
```bash
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live (Do not share this link with others, or they will be able to control your computer.)
```

### 2. <a name="usage"></a>Load model

🔨 <a name="macOS"></a>**Note for macOS users:** Please set the torch device to "mps" if you are using an Apple Silicon Mac. 

```python
import ast
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def draw_point(image_input, point=None, radius=5):
    if isinstance(image_input, str):
        image = Image.open(BytesIO(requests.get(image_input).content)) if image_input.startswith('http') else Image.open(image_input)
    else:
        image = image_input

    if point:
        x, y = point[0] * image.width, point[1] * image.height
        ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
    display(image)
    return

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

min_pixels = 256*28*28
max_pixels = 1344*28*28

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

### 3. **UI Grounding**
```python
img_url = 'examples/web_dbd7514b-9ca3-40cd-b09a-990f7b955da1.png'
query = "Nahant"


_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": _SYSTEM},
            {"type": "image", "image": img_url, "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": query}
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

click_xy = ast.literal_eval(output_text)
# [0.73, 0.21]

draw_point(img_url, click_xy, 10)
```

This will visualize the grounding results like (where the red points are [x,y])

![download](https://github.com/user-attachments/assets/8fe2783d-05b6-44e6-a26c-8718d02b56cb)

### 4. **UI Navigation**
- Set up system prompt.
```python
_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

action_map = {
'web': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",

'phone': """
1. `INPUT`: Type a string into an element, value is not applicable and the position [x,y] is required. 
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.
"""
}
```

```python
img_url = 'examples/chrome.png'
split='web'
system_prompt = _NAV_SYSTEM.format(_APP=split, _ACTION_SPACE=action_map[split])
query = "Search the weather for the New York city."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f'Task: {query}'},
            # {"type": "text", "text": PAST_ACTION},
            {"type": "image", "image": img_url, "min_pixels": min_pixels, "max_pixels": max_pixels},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(output_text)
# {'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]},
# {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]},
# {'action': 'ENTER', 'value': None, 'position': None}
```

![download](https://github.com/user-attachments/assets/624097ea-06f2-4c8f-83f6-b6b9ee439c0c)


If you find our work helpful, please consider citing our paper.

```
@misc{lin2024showui,
      title={ShowUI: One Vision-Language-Action Model for GUI Visual Agent}, 
      author={Kevin Qinghong Lin and Linjie Li and Difei Gao and Zhengyuan Yang and Shiwei Wu and Zechen Bai and Weixian Lei and Lijuan Wang and Mike Zheng Shou},
      year={2024},
      eprint={2411.17465},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17465}, 
}
```

## Join Discussion
Welcome to discuss with us and continuously improve the user experience of Computer Use - OOTB. Reach us using this [**Discord Channel**](https://discord.gg/HnHng5de) or the WeChat QR code below!

<div style="display: flex; flex-direction: row; justify-content: space-around;">

<img src="assets/wechat.jpg" alt="gradio_interface" style="width:30%;">


</div>
