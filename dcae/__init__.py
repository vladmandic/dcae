# credits: <https://github.com/mit-han-lab/efficientvit/blob/master/applications/dc_ae/README.md>

from typing import Callable, Optional
import torch
from .models import dc_ae


class DCAE(dc_ae.DCAE):
    models: dict[str, tuple[Callable, Optional[str]]] = { # https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
        'dc-ae-f32c32-in-1.0': (dc_ae.dc_ae_f32c32, 'mit-han-lab/dc-ae-f32c32-in-1.0'),
        'dc-ae-f64c128-in-1.0': (dc_ae.dc_ae_f64c128, 'mit-han-lab/dc-ae-f64c128-in-1.0'),
        'dc-ae-f128c512-in-1.0': (dc_ae.dc_ae_f128c512, 'mit-han-lab/dc-ae-f128c512-in-1.0'),
        'dc-ae-f32c32-mix-1.0': (dc_ae.dc_ae_f32c32, 'mit-han-lab/dc-ae-f32c32-mix-1.0'),
        'dc-ae-f64c128-mix-1.0': (dc_ae.dc_ae_f64c128, 'mit-han-lab/dc-ae-f64c128-mix-1.0'),
        'dc-ae-f128c512-mix-1.0': (dc_ae.dc_ae_f128c512, 'mit-han-lab/dc-ae-f128c512-mix-1.0'),
    }

    def __init__(self, model: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, cache_dir: Optional[str] = None):
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        cls, repo_id = self.models[model]
        cfg = cls(model, None) # bypass built-in weights loader
        dc_ae.DCAE.__init__(self, cfg)
        self.scale = 2 ** (self.decoder.num_stages - 1)
        model_path = hf_hub_download(repo_id, filename="model.safetensors", cache_dir=cache_dir)
        state_dict = safetensors.torch.load_file(model_path, device='cpu')
        self.load_state_dict(state_dict)
        self = self.to(device=device, dtype=dtype).eval()
