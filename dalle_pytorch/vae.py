from pathlib import Path
from math import sqrt
from omegaconf import OmegaConf
import importlib

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from dalle_pytorch import distributed_utils

# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()
        model_path = vqgan_model_path
        config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = 2**(len(config.model.params.ddconfig.ch_mult)-1)
        self.fmap_size = config.model.params.ddconfig.resolution // f
        self.num_layers = len(config.model.params.ddconfig.ch_mult)-1
        self.image_size = config.model.params.ddconfig.resolution
        self.num_tokens = config.model.params.n_embed

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, '(b n) 1 -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embedding.weight

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img # 0. ~ 1.

    def forward(self, img):
        raise NotImplemented
