import re
import math
from PIL import Image, ImageDraw, ImageFont


def wrap(text: str, font: ImageFont.ImageFont, length: int):
    lines = ['']
    for word in re.split(r'[ ,-]', text):
        line = f'{lines[-1]} {word}'.strip()
        if font.getlength(line) <= length or '\n' in line:
            lines[-1] = line
        else:
            lines.append(word)
    return '\n'.join(lines)


def grid(images: list[Image.Image], labels: list[str] = None, width: int = 0, height: int = 0, border: int = 0, square: bool = False, horizontal: bool = False, vertical: bool = False, font: int = 0): # pylint: disable=redefined-outer-name
    if horizontal:
        rows = 1
    elif vertical:
        rows = len(images)
    elif square:
        rows = round(math.sqrt(len(images)))
    else:
        rows = math.floor(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    size = [0, 0]
    if width == 0:
        w = max([i.size[0] for i in images])
        size[0] = cols * w + cols * border
    else:
        size[0] = width
        w = round(width / cols)
    if height == 0:
        h = max([i.size[1] for i in images])
        size[1] = rows * h + rows * border
    else:
        size[1] = height
        h = round(height / rows)
    size = tuple(size)
    image = Image.new('RGB', size = size, color = 'black') # pylint: disable=redefined-outer-name
    font = ImageFont.truetype('DejaVuSansMono', size=round(w / 20) if font == 0 else font)
    for i, img in enumerate(images): # pylint: disable=redefined-outer-name
        x = (i % cols * w) + (i % cols * border)
        y = (i // cols * h) + (i // cols * border)
        img.thumbnail((w, h), Image.Resampling.HAMMING)
        image.paste(img, box=(x + int(border / 2), y + int(border / 2)))
        if labels is not None and len(images) == len(labels):
            ctx = ImageDraw.Draw(image)
            label = wrap(labels[i], font, w)
            ctx.text((x + 1 + round(w / 200), y + 1 + round(w / 200)), label, font = font, fill = (0, 0, 0))
            ctx.text((x, y), label, font = font, fill = (255, 255, 255))
    return image
