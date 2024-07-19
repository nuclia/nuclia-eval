from pathlib import Path
from typing import Union

import safetensors.torch
from torch import nn


def load_lora_low_mem(
    model: nn.Module, lora_path: Union[Path, str], scaling: float = 2.0
):  # pragma: no cover
    """Loads LoRA checkpoint with a low memory footprint compared to the official implementation"""
    lora_path = Path(lora_path)
    assert lora_path.is_file(), f"{lora_path} does not exist or is not a file"

    lora_state_dict = safetensors.torch.load_file(lora_path)

    lora_dtypes = set([p.dtype for p in lora_state_dict.values()])
    assert (
        len(lora_dtypes) == 1
    ), f"LoRA weights have multipe different dtypes {lora_dtypes}. All weights need to have the same dtype"
    lora_dtype = lora_dtypes.pop()
    assert (
        lora_dtype == model.dtype
    ), f"LoRA weights dtype differs from model's dtype {lora_dtype} != {model.dtype}"
    assert all("lora" in key for key in lora_state_dict.keys())

    state_dict = model.state_dict()
    # Move this state dict to cpu
    state_dict = {k: v.to("cpu") for k, v in state_dict.items()}

    if model.args.lora is None:
        # move tensors to device
        lora_state_dict = {k: v.to(model.device) for k, v in lora_state_dict.items()}
        # replace every nn.Linear with a LoRALinear with 'meta' device except the output layer
        named_modules = dict(model.named_modules())
        for name, module in named_modules.items():
            if isinstance(module, nn.Linear) and name != "output":
                layer_id = name.split(".")[1]
                if layer_id in model.layers:
                    weight = (
                        module.weight
                        + (
                            lora_state_dict[name + ".lora_B.weight"].to(model.device)
                            @ lora_state_dict[name + ".lora_A.weight"].to(model.device)
                        )
                        * scaling
                    )

                    state_dict[name + ".weight"] = weight.to("cpu")
    else:
        for k, v in lora_state_dict.items():
            state_dict.update(lora_state_dict)

            layer_id = k.split(".")[1]
            if layer_id in model.layers:
                state_dict[k] = v
    # Move model to cpu
    device = model.device
    model = model.to("cpu")
    model.load_state_dict(state_dict, strict=True)
    # Move model back to desired device
    model = model.to(device)


def inherit_docstrings(cls):  # pragma: no cover
    for name, func in vars(cls).items():
        if not func.__doc__ and hasattr(getattr(cls, name), "__doc__"):
            func.__doc__ = getattr(cls, name).__doc__
    return cls
