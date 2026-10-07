import io

from PIL import Image


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(
        buf,
        format="JPEG",
        quality=int(quality)
    )
    buf.seek(0)

    return Image.open(buf).convert("RGB")


def resize_down_up(
    img: Image.Image,
    scale: float,
    down=Image.BILINEAR,
    up=Image.BILINEAR
) -> Image.Image:
    w, h = img.size

    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    return img.resize(
        (nw, nh),
        down
    ).resize(
        (w, h),
        up
    )