import torch

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head", "vision_model", "img_projection", "visual_model"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    # lora_namespan_exclude += ["vision_model", "img_projection", "visual_model"]
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names