# ShowUI
[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/showlab/ShowUI-2B)&nbsp;

### 📢 News
<!--  -->
- [2024.11.16] Upload the `showlab/ShowUI-2B` at huggingface.

## ⭐ Quick Start

1. Load model
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

2. Load screenshot and query
```python
img_url = 'web_dbd7514b-9ca3-40cd-b09a-990f7b955da1.png'
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
