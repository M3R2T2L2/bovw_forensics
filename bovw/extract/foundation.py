"""Foundation-model patch tokens (frozen backbones via Hugging Face).

Returns patch tokens as local descriptors and the CLS token as the global
embedding, so the global-embedding baseline comes for free.

Registered backbones start with DINOv2; add SigLIP2 / PE-Core / CLIP by
adding entries to BACKBONES with the right token layout.
"""

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from . import LocalFeatures


@dataclass(frozen=True)
class Backbone:
    hf_id: str
    n_prefix: int  # tokens before the patch grid (CLS + registers)
    has_cls: bool = True


BACKBONES = {
    "dinov2_s": Backbone("facebook/dinov2-small", n_prefix=1),
    "dinov2_b": Backbone("facebook/dinov2-base", n_prefix=1),
    "dinov2_s_reg": Backbone("facebook/dinov2-with-registers-small", n_prefix=5),
    "dinov2_b_reg": Backbone("facebook/dinov2-with-registers-base", n_prefix=5),
}


class FoundationExtractor:
    def __init__(self, name: str, image_size: int = 224, batch_size: int = 64,
                 device: str | None = None, dtype: str = "float16", layer: int = -1):
        import torch
        from transformers import AutoModel

        if name not in BACKBONES:
            raise KeyError(f"Unknown backbone '{name}'. Known: {sorted(BACKBONES)}")
        self.spec = BACKBONES[name]
        self.image_size, self.batch_size, self.layer = image_size, batch_size, layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        try:
            model = AutoModel.from_pretrained(self.spec.hf_id, dtype=self.dtype)
        except TypeError:  # transformers < 4.56 only knows torch_dtype
            model = AutoModel.from_pretrained(self.spec.hf_id, torch_dtype=self.dtype)
        self.model = model.to(self.device).eval()
        # ImageNet normalisation used by DINOv2
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prep(self, batch):
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.stack(batch)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - self.mean) / self.std
        return x.to(self.dtype)

    def __call__(self, images, progress: bool = True) -> LocalFeatures:
        import torch

        patches, cls = [], []
        n = len(images)
        with torch.inference_mode():
            for s in tqdm(range(0, n, self.batch_size), desc=self.spec.hf_id, disable=not progress):
                x = self._prep([images[i] for i in range(s, min(s + self.batch_size, n))])
                out = self.model(pixel_values=x, output_hidden_states=self.layer != -1)
                # HF Dinov2 applies the final layernorm to last_hidden_state
                h = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
                h = h.float().cpu().numpy()
                cls.append(h[:, 0])
                patches.extend(h[:, self.spec.n_prefix:])  # (P, D) per image
        return LocalFeatures.from_list(patches, global_=np.concatenate(cls).astype(np.float32))
