import os
import sys
import time

import cv2
import torch
import diffusers
import skimage.metrics
from rich import print as rprint
from PIL import Image

from dcae import DCAE
from app.grid import grid
from app.util import get_image, get_tensor, mem
from app.fid import FID


repeat = 3 # run n runs to get time/memory average
clip = True # clip output to 0-1
diff = True # generate diff image
ext = 'jpg' # output image format
models = [ # models to test
    'dc-ae-f32c32-in-1.0', 'dc-ae-f64c128-in-1.0', 'dc-ae-f128c512-in-1.0',
    'dc-ae-f32c32-mix-1.0', 'dc-ae-f64c128-mix-1.0', 'dc-ae-f128c512-mix-1.0',
    'madebyollin/taesd', 'madebyollin/taesdxl', 'madebyollin/sdxl-vae-fp16-fix',
    'ostris/vae-kl-f8-d16', 'cross-attention/asymmetric-autoencoder-kl-x-1-5',
    'openai/consistency-decoder',
]
device = torch.device('cuda')
dtype = torch.bfloat16
cache_dir = '/mnt/models/huggingface'
if not os.path.isdir(cache_dir):
    cache_dir = None
_x = torch.randn(1, device=device) # init torch
torch.cuda.set_per_process_memory_fraction(1.0, 0) # limit torch memory so it doesnt spill into ram
fid = FID()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        rprint(f'usage: {sys.argv[0]} <image>')
        sys.exit(1)
    sys.argv.pop(0)

    rprint(f'start: repeats={repeat} device={device} dtype={dtype} clip={clip} diff={diff} ext={ext}')
    rprint(f'models: {models}')
    for fn in sys.argv:
        base = os.path.splitext(os.path.basename(fn))[0]
        out = f'{base}-vae-grid.{ext}'
        image = get_image(fn)
        rprint(f'image: input="{fn}" shape={image.shape}')

        if diff:
            images = []
            labels = []
        else:
            images = [Image.fromarray(image)]
            labels = [f'original\nshape {image.shape}']

        for model in models:
            m1 = mem()
            t1 = time.time()
            if 'dc-ae' in model:
                ae = DCAE(model=model, device=device, dtype=dtype, cache_dir=cache_dir)
            elif 'taesd' in model:
                ae = diffusers.AutoencoderTiny.from_pretrained(model, torch_dtype=dtype, cache_dir=cache_dir)
                ae = ae.to(device=device)
                ae.scale = 8
            elif 'asymmetric' in model:
                ae = diffusers.AsymmetricAutoencoderKL.from_pretrained(model, torch_dtype=dtype, cache_dir=cache_dir)
                ae = ae.to(device=device)
                ae.scale = 8
            elif 'consistency' in model:
                ae = diffusers.ConsistencyDecoderVAE.from_pretrained(model, torch_dtype=dtype, cache_dir=cache_dir)
                ae = ae.to(device=device)
                ae.scale = 8
            else:
                ae = diffusers.AutoencoderKL.from_pretrained(model, torch_dtype=dtype, cache_dir=cache_dir)
                ae = ae.to(device=device)
                ae.scale = 8
            t2 = time.time()
            m2 = mem()
            rprint(f' load: model={model} scale={ae.scale} time={t2-t1:.3f} mem={m2-m1}')

            tensor, image = get_tensor(image, ae.scale, device, dtype)
            tensor = 0.5 * tensor + 0.5 # normalize

            msg = ''
            try:
                for n in range(repeat):
                    with torch.inference_mode():
                        if 'dc-ae' in model:
                            latent = ae.encode(tensor)
                        else:
                            latent = ae.encode(tensor, return_dict=False)[0]
                            if hasattr(latent, 'sample'):
                                latent = latent.sample()
                    t3 = time.time()
                    m3 = mem(reset=False)
                    size = latent.nelement() * latent.element_size()
                    rprint(f'  encode: shape={latent.shape} size={size} time={t3-t2:.3f} mem={m3-m2} repeat={n}')

                    with torch.inference_mode():
                        if 'dc-ae' in model:
                            tensors = ae.decode(latent)
                        else:
                            tensors = ae.decode(latent, return_dict=False)[0]
                    t4 = time.time()
                    m4 = mem(reset=False)
                    rprint(f'  decode: shape={tensors.shape} time={t4-t3:.3f} mem={m4-m3} repeat={n}')
            except Exception as e:
                t4 = time.time()
                m4 = mem(reset=False)
                tensors = [torch.zeros(image.shape[2], image.shape[0], image.shape[1], device=device, dtype=dtype)]
                msg = e.__class__.__name__
                rprint(f'  error: model={model} {e}')

            output = tensors[0].permute(1, 2, 0).cpu().float().numpy()
            output = 2.0 * output - 1.0 # renormalize
            if clip:
                output = output.clip(0, 1)

            # difference image from original
            output = (255 * output).astype('uint8')
            delta = cv2.absdiff(image, output)

            # calculate stats
            diff_score = delta.sum() / (255 * image.shape[0] * image.shape[1] * image.shape[2])
            ssim_score = skimage.metrics.structural_similarity(image, output, channel_axis=2)
            mse_score = round(skimage.metrics.mean_squared_error(image, output))
            fid_score = fid(image, output)
            stats = f'\ndelta {diff_score:.4f}\nssim {ssim_score:.4f}\nmse {mse_score}\nfid {fid_score:.2f}'
            rprint(f'  stats: delta={diff_score:.4f} ssim={ssim_score:.4f} mse={mse_score} fid={fid_score:.2f}')

            images.append(Image.fromarray(output))
            labels.append(f'{os.path.basename(model)}\n\nmem {m4}\ntime {t4-t2:.3f}{stats if not diff else ""}\n{msg}')
            if diff:
                images.append(Image.fromarray(delta))
                labels.append(f'{os.path.basename(model)}\nabsdiff\n{stats}')
            ae = None

        image = grid(images, labels)
        image.save(out)
        rprint(f'grid: images={len(images)} size={image.size} output={out}')

    print('done')
