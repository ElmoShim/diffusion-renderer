# Minimal conditioner for Cosmos Diffusion Renderer inference.
# Simplified from cosmos_predict1/diffusion/conditioner.py — no training logic.

from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch


@dataclass
class VideoDiffusionRendererCondition:
    crossattn_emb: torch.Tensor
    crossattn_mask: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    image_size: Optional[torch.Tensor] = None
    latent_condition: Optional[torch.Tensor] = None
    context_index: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
