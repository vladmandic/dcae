# DCAE

## Credit

- [EfficientViT DC-AE](https://github.com/mit-han-lab/efficientvit)

## What's Changed?

- Removed internal dependencies on anything thats not pure dc-ae inference code  
- Refactored to work as standalone module with relative imports  
- Simplified dependencies  
- Replaced model loader  
- Added single class for anything related to dc-ae  
- Unified device/dtype handling  

## Usage

```py
from dcae import DCAE
ae = DCAE(model='dc-ae-f32c32-mix-1.0', device=torch.device('cuda'), dtype=torch.bfloat16, cache_dir='~/.cache/huggingface')
encoded = ae.encode(tensor)
decoded = ae.decode(encoded)
```

For more detailed example, see `test.py`

## Notes

- 2 variants: `in` and `mix`  
  no documentation on how they differ?  
- each variant in 3 flavors: `f32c32`, `f64c128`, `f128c512`  
  with increasing number of internal stages  
  resulting sizes are same as each stage compresses by 2* and adds 2** channels:  
  example for 1024x1024 input:  
  - `f32c32`:   1.26GB, 5 stages (2**5=32)  encodes to 32x32x32
  - `f64c128`:  2.64GB, 6 stages (2**6=64)  encodes to 128x16x16  
  - `f128c512`: 4.37GB, 7 stages (2**7=128) encodes to 512x8x8  
- notes in paper on FID/PSNR dont make sense  
- despite large size for an autoencoder, its fast and has relative low resource requirements  
  typical encode/decode is ~0.1s for 1k image on RTX4090  
- without any tiling it can do native 4k encode/decode in ~0.5s using 20GB VRAM  

![dcae](https://github.com/user-attachments/assets/37c52565-7bcf-4a36-ae73-2a92e2a7fb94)
