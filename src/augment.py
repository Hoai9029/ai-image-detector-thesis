import random

from PIL import Image

from src.degrade import jpeg_compress, resize_down_up


class RobustAugment:
    def __init__(
        self,
        jpeg_prob=0.5,
        resize_prob=0.5,
        jpeg_quality_range=(30, 70),
        resize_scale_range=(0.5, 0.75)
    ):
        self.jpeg_prob = jpeg_prob
        self.resize_prob = resize_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.resize_scale_range = resize_scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.jpeg_prob:
            quality = random.randint(
                self.jpeg_quality_range[0],
                self.jpeg_quality_range[1]
            )

            img = jpeg_compress(
                img,
                quality
            )

        if random.random() < self.resize_prob:
            scale = random.uniform(
                self.resize_scale_range[0],
                self.resize_scale_range[1]
            )

            img = resize_down_up(
                img,
                scale
            )

        return img