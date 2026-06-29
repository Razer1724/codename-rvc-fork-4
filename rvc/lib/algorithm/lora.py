import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
from torch.nn.utils.parametrizations import weight_norm


def _has_old_weight_norm(module):
    for hook in module._forward_pre_hooks.values():
        if getattr(hook, "__class__", None).__name__ == "WeightNorm":
            return True
    return False


def _remove_old_weight_norm(module):
    for hook in list(module._forward_pre_hooks.values()):
        if getattr(hook, "__class__", None).__name__ == "WeightNorm":
            torch.nn.utils.remove_weight_norm(module)
            break


class Conv1dLoRA(nn.Module):
    def __init__(self, conv: nn.Conv1d, r: int = 8):
        super().__init__()
        self.was_weight_normed = is_parametrized(conv, "weight") or _has_old_weight_norm(conv)
        if is_parametrized(conv, "weight"):
            remove_parametrizations(conv, "weight")
        if _has_old_weight_norm(conv):
            _remove_old_weight_norm(conv)
        self.conv = conv
        self.conv.weight.requires_grad_(False)

        C_out, C_in, k = conv.weight.shape
        self.r = r

        device = conv.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, C_in * k, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(C_out, r, device=device))

    def forward(self, x):
        out = self.conv(x)
        C_out, C_in, k = self.conv.weight.shape
        delta_w = (self.lora_B @ self.lora_A).view(C_out, C_in, k)
        out += F.conv1d(
            x, delta_w, None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out

    def merge(self):
        C_out, C_in, k = self.conv.weight.shape
        delta_w = (self.lora_B @ self.lora_A).view(C_out, C_in, k)
        with torch.no_grad():
            self.conv.weight.add_(delta_w)
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.lora_A)

    def extra_repr(self):
        return f"r={self.r}"


class ConvTranspose1dLoRA(nn.Module):
    def __init__(self, conv: nn.ConvTranspose1d, r: int = 8):
        super().__init__()
        self.was_weight_normed = is_parametrized(conv, "weight") or _has_old_weight_norm(conv)
        if is_parametrized(conv, "weight"):
            remove_parametrizations(conv, "weight")
        if _has_old_weight_norm(conv):
            _remove_old_weight_norm(conv)
        self.conv = conv
        self.conv.weight.requires_grad_(False)
        C_in, C_out, k = conv.weight.shape
        self.r = r

        device = conv.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, C_in * k, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(C_out, r, device=device))

    def forward(self, x):
        out = self.conv(x)
        C_in, C_out, k = self.conv.weight.shape
        delta_w_flat = self.lora_B @ self.lora_A
        delta_w = delta_w_flat.view(C_out, C_in, k).transpose(0, 1)
        out += F.conv_transpose1d(
            x, delta_w, None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            output_padding=self.conv.output_padding,
            groups=self.conv.groups,
            dilation=self.conv.dilation,
        )
        return out

    def merge(self):
        C_in, C_out, k = self.conv.weight.shape
        delta_w_flat = self.lora_B @ self.lora_A
        delta_w = delta_w_flat.view(C_out, C_in, k).transpose(0, 1)
        with torch.no_grad():
            self.conv.weight.add_(delta_w)
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.lora_A)

    def extra_repr(self):
        return f"r={self.r}"


class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8):
        super().__init__()
        self.was_weight_normed = False
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        C_out, C_in = linear.weight.shape
        self.r = r

        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, C_in, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(C_out, r, device=device))

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        out = self.linear(x)
        delta = self.lora_B @ self.lora_A
        out += F.linear(x, delta)
        return out

    def merge(self):
        delta = self.lora_B @ self.lora_A
        with torch.no_grad():
            self.linear.weight.add_(delta)
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.lora_A)

    def extra_repr(self):
        return f"r={self.r}"


def _is_target_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear))


_WRAPPER_TYPES = (Conv1dLoRA, ConvTranspose1dLoRA, LinearLoRA)


def apply_lora(model: nn.Module, r: int = 8):
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            for i, sub in enumerate(child):
                if isinstance(sub, _WRAPPER_TYPES):
                    continue
                if isinstance(sub, nn.Conv1d):
                    child[i] = Conv1dLoRA(sub, r=r)
                elif isinstance(sub, nn.ConvTranspose1d):
                    child[i] = ConvTranspose1dLoRA(sub, r=r)
                elif isinstance(sub, nn.Linear):
                    child[i] = LinearLoRA(sub, r=r)
                elif not isinstance(sub, nn.ModuleList):
                    apply_lora(sub, r=r)
        elif isinstance(child, _WRAPPER_TYPES):
            continue
        elif isinstance(child, nn.Conv1d):
            setattr(model, name, Conv1dLoRA(child, r=r))
        elif isinstance(child, nn.ConvTranspose1d):
            setattr(model, name, ConvTranspose1dLoRA(child, r=r))
        elif isinstance(child, nn.Linear):
            setattr(model, name, LinearLoRA(child, r=r))
        else:
            apply_lora(child, r=r)


def merge_lora(model: nn.Module, remove_wrappers: bool = True, restore_weight_norm: bool = False):
    for name, child in list(model.named_children()):
        if isinstance(child, _WRAPPER_TYPES):
            child.merge()
            if remove_wrappers:
                inner = child.conv if hasattr(child, "conv") else child.linear
                if restore_weight_norm and child.was_weight_normed:
                    weight_norm(inner)
                setattr(model, name, inner)
        elif isinstance(child, nn.ModuleList):
            for i, sub in enumerate(child):
                if isinstance(sub, _WRAPPER_TYPES):
                    sub.merge()
                    if remove_wrappers:
                        inner = sub.conv if hasattr(sub, "conv") else sub.linear
                        if restore_weight_norm and sub.was_weight_normed:
                            weight_norm(inner)
                        child[i] = inner
                elif not isinstance(sub, nn.ModuleList):
                    merge_lora(sub, remove_wrappers=remove_wrappers, restore_weight_norm=restore_weight_norm)
        else:
            merge_lora(child, remove_wrappers=remove_wrappers, restore_weight_norm=restore_weight_norm)
