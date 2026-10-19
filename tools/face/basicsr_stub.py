"""The two basicsr helpers the vendored GFPGAN architecture needs, so basicsr isn't required."""
import torch
from torch import nn
from torch.nn import init


class _Registry:
    def register(self):
        return lambda cls: cls


ARCH_REGISTRY = _Registry()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
