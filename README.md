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

Will run encode/decode on a given image using all available DC-AE models and other known autoencoders and produce an image grid with:

- image after encode/decode
- memory usage and time taken for each
- diff image from original
- diff, mse, ssim and fid scores

Supported models:

- `dc-ae-f32c32-in-1.0`, `dc-ae-f64c128-in-1.0`, `dc-ae-f128c512-in-1.0`,
- `dc-ae-f32c32-mix-1.0`, `dc-ae-f64c128-mix-1.0`, `dc-ae-f128c512-mix-1.0`,
- `madebyollin/taesd`, `madebyollin/taesdxl`, `madebyollin/sdxl-vae-fp16-fix`,
- `ostris/vae-kl-f8-d16`, `cross-attention/asymmetric-autoencoder-kl-x-1-5`,

### Examples

![helmet-vae-grid](https://github.com/user-attachments/assets/5804f703-3954-4b06-9777-89db1f258f77)
![afghan-vae-grid](https://github.com/user-attachments/assets/44a0f8a5-7317-4d6d-8af4-95c32db5aaa8)
![cara-vae-grid](https://github.com/user-attachments/assets/d454c9e0-45ad-468a-a78a-2ea2782c9f4d)
![alla-vae-grid](https://github.com/user-attachments/assets/4d80c63d-946d-4908-8fe0-214174f406fd)
![asian-vae-grid](https://github.com/user-attachments/assets/8886d989-0194-401e-8ed7-eb00150aaa01)
![paint-vae-grid](https://github.com/user-attachments/assets/af8808d0-5340-428a-8672-d03dcbd34246)

### Originals

[Link](https://github.com/vladmandic/dcae/issues/1)
