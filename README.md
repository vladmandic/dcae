# AutoEncoders

## DC-AE

### What's Changed?

From original EfficientViT DC-AE code...

- Removed internal dependencies on anything thats not pure dc-ae inference code  
- Refactored to work as standalone module with relative imports  
- Simplified dependencies  
- Replaced model loader  
- Added single class for anything related to dc-ae  
- Unified device/dtype handling  

### Usage

```py
from dcae import DCAE
ae = DCAE(model='dc-ae-f32c32-mix-1.0', device=torch.device('cuda'), dtype=torch.bfloat16, cache_dir='~/.cache/huggingface')
encoded = ae.encode(tensor)
decoded = ae.decode(encoded)
```

### Credit

- [EfficientViT DC-AE](https://github.com/mit-han-lab/efficientvit)

### Notes

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

-----------

## Comparing AutoEncoders

> `python compare.py <image>`

Will run encode/decode on a given image using all available DC-AE models and other known autoencoders and produce an image grid with memory usage and time taken for each:

- `dc-ae-f32c32-in-1.0`, `dc-ae-f64c128-in-1.0`, `dc-ae-f128c512-in-1.0`,
- `dc-ae-f32c32-mix-1.0`, `dc-ae-f64c128-mix-1.0`, `dc-ae-f128c512-mix-1.0`,
- `madebyollin/taesd`, `madebyollin/taesdxl`, `madebyollin/sdxl-vae-fp16-fix`,
- `ostris/vae-kl-f8-d16`, `cross-attention/asymmetric-autoencoder-kl-x-1-5`,

Examples:
