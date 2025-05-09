import random

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def gaussian_blur(img, strength=1):
    img = preprocess_image_for_pil(img)
    pil_img = Image.fromarray(img)  # Ensure it's a PIL Image
    radius = 1 + strength * 0.5  # more intense blur with higher strength
    blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred_img)


def motion_blur(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    kernel_size = int(5 + strength * 2)
    angle = random.choice([0, 45, 90])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    if angle == 0:
        kernel[int(kernel_size / 2), :] = np.ones(kernel_size)
    elif angle == 90:
        kernel[:, int(kernel_size / 2)] = np.ones(kernel_size)
    else:
        np.fill_diagonal(kernel, 1)

    kernel /= kernel.sum()
    img_np = np.array(img)
    blurred = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(np.uint8(blurred))


def low_resolution(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    w, h = img.size
    factor = int(1 + strength * 1.25)  # more downsampling
    downsample = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return downsample.resize((w, h), Image.BILINEAR)


def rotate_image(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    angle = strength * 10 * random.choice([-1, 1])  # rotate up to ±75°
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)


def affine_transform(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    width, height = img.size
    scale = 0.2 * strength  # stronger distortion
    return img.transform(
        (width, height),
        Image.AFFINE,
        (1, random.uniform(-scale, scale), 0,
         random.uniform(-scale, scale), 1, 0),
        resample=Image.BILINEAR,
        fillcolor=0
    )


def occlude_image(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    box_size = int(min(w, h) * (0.10 + 0.15 * strength))  # larger occlusion
    x1 = random.randint(0, w - box_size)
    y1 = random.randint(0, h - box_size)
    draw.rectangle([x1, y1, x1 + box_size, y1 + box_size], fill=0)  # black box
    return img


# Function to apply random degradation
def apply_random_degradation(img, strength=1):
    img = preprocess_image_for_pil(img)
    img = Image.fromarray(img)
    fn = random.choice(degradation_pool)
    degraded_img = fn(img.copy(), strength=strength)
    return degraded_img, fn.__name__, strength


def preprocess_image_for_pil(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype == np.float32:
        # Convert to uint8, keeping values between 0 and 1 without scaling
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img


degradation_pool = [
    gaussian_blur,
    motion_blur,
    low_resolution,
    rotate_image,
    affine_transform,
    occlude_image,
]
