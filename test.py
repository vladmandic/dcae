import os
import sys
import time
import cv2
import torch
from dcae import DCAE


if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16
    model = 'dc-ae-f32c32-mix-1.0'
    cache_dir = '/mnt/models/huggingface'
    output_dir = '/tmp/dcae'

    t1 = time.time()
    print(f'starting: model={model} device={device} dtype={dtype} cache_dir={cache_dir} output_dir={output_dir}')
    ae = DCAE(model=model, device=device, dtype=dtype, cache_dir=cache_dir)
    t2 = time.time()
    print(f'dcae loaded: model={model} scale={ae.scale} time={t2-t1:.2f}')

    sys.argv.pop(0)
    i = 0
    total_size = 0
    encode_time = 0
    decode_time = 0
    for fn in sys.argv:
        t3 = time.time()
        if not os.path.isfile(fn):
            continue
        image = cv2.imread(fn)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h, _ = image.shape
        if w < ae.scale or h < ae.scale:
            continue
        image = cv2.resize(image, (ae.scale * (h // ae.scale), ae.scale * (w // ae.scale)))
        tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
        tensor = tensor.to(device, dtype)
        t4 = time.time()
        print(f'image: input={w,h} reshaped={tensor.shape} time={t4-t3:.2f}')

        with torch.no_grad():
            latent = ae.encode(tensor)
        t5 = time.time()
        encode_time += t5-t4
        size = latent.nelement() * latent.element_size()
        print(f'  encoded: shape={latent.shape} size={size} time={t5-t4:.2f}')

        with torch.no_grad():
            tensors = ae.decode(latent)
        t6 = time.time()
        decode_time += t6-t5
        print(f'  decoded: shape={tensors.shape} time={t6-t5:.2f}')

        output = tensors[0].permute(1, 2, 0).cpu().float().numpy() * 255
        fn = os.path.splitext(os.path.basename(fn))[0]
        f1 = os.path.join(output_dir, f'{fn}.png')
        cv2.imwrite(f1, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        f2 = os.path.join(output_dir, f'{fn}-{model}.png')
        ae.encode
        cv2.imwrite(f2, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        print(f'  images: input={f1} output={f2}')
        i += 1
        total_size += size

    t7 = time.time()
    print(f'done: model={model} images={i} time={t7-t2:.2f} encode={encode_time:.2f} decode={decode_time:.2f} size={total_size}')
