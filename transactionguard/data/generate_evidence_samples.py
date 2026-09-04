import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
import io

RANDOM_SEED = 11
N_PER_CLASS = 40
IMG_SIZE = 256
OUT_DIR = "data/evidence_samples"


def _draw_synthetic_product_photo(rng: random.Random) -> Image.Image:
    """A cheap stand-in for a real 'item received damaged' photo: a solid
    background plus a handful of random shapes (simulating a product and
    a damage mark), so images differ enough from each other to make
    splicing/copy-move meaningfully detectable."""
    bg = tuple(rng.randint(140, 230) for _ in range(3))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=bg)
    draw = ImageDraw.Draw(img)

    # "product" body
    body_color = tuple(rng.randint(20, 200) for _ in range(3))
    x0, y0 = rng.randint(20, 60), rng.randint(20, 60)
    x1, y1 = x0 + rng.randint(100, 160), y0 + rng.randint(100, 160)
    draw.rectangle([x0, y0, x1, y1], fill=body_color, outline=(0, 0, 0))

    # a "damage" mark (scratch/dent) as a dark irregular blob
    damage_color = tuple(max(0, c - rng.randint(60, 120)) for c in body_color)
    cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
    for _ in range(6):
        r = rng.randint(4, 14)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=damage_color)
        cx += rng.randint(-15, 15)
        cy += rng.randint(-15, 15)

    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _splice(base: Image.Image, donor: Image.Image, rng: random.Random) -> Image.Image:
    """Splices a patch from `donor` onto `base`. Critically, the donor patch
    is first round-tripped through its OWN separate JPEG compression before
    pasting — this bakes in a distinct quantization history for that region,
    which is what actually makes ELA / double-compression detection work.
    Without this step there is nothing for a compression-history forensic
    check to find, no matter how the final composite is saved."""
    donor_quality = rng.choice([95, 92, 88])
    buf = io.BytesIO()
    donor.save(buf, "JPEG", quality=donor_quality)
    buf.seek(0)
    donor_compressed = Image.open(buf).convert("RGB")

    patch_w, patch_h = rng.randint(40, 80), rng.randint(40, 80)
    sx = rng.randint(0, IMG_SIZE - patch_w)
    sy = rng.randint(0, IMG_SIZE - patch_h)
    dx = rng.randint(0, IMG_SIZE - patch_w)
    dy = rng.randint(0, IMG_SIZE - patch_h)
    patch = donor_compressed.crop((sx, sy, sx + patch_w, sy + patch_h))
    tampered = base.copy()
    tampered.paste(patch, (dx, dy))
    return tampered


def _copy_move(base: Image.Image, rng: random.Random) -> Image.Image:
    patch_w, patch_h = rng.randint(30, 60), rng.randint(30, 60)
    sx = rng.randint(0, IMG_SIZE - patch_w)
    sy = rng.randint(0, IMG_SIZE - patch_h)
    dx = rng.randint(0, IMG_SIZE - patch_w)
    dy = rng.randint(0, IMG_SIZE - patch_h)
    while abs(dx - sx) < patch_w and abs(dy - sy) < patch_h:  
        dx = rng.randint(0, IMG_SIZE - patch_w)
        dy = rng.randint(0, IMG_SIZE - patch_h)
    patch = base.crop((sx, sy, sx + patch_w, sy + patch_h))
    tampered = base.copy()
    tampered.paste(patch, (dx, dy))
    return tampered


def generate():
    rng = random.Random(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = {}

    base_images = [_draw_synthetic_product_photo(rng) for _ in range(N_PER_CLASS * 2)]

    idx = 0
    for i in range(N_PER_CLASS):
        img = base_images[idx]; idx += 1
        fname = f"authentic_{i:03d}.jpg"
        img.save(os.path.join(OUT_DIR, fname), "JPEG", quality=rng.choice([85, 90, 95]))
        labels[fname] = 0

    for i in range(N_PER_CLASS):
        base = base_images[idx]; idx += 1
        technique = "splice" if i % 2 == 0 else "copy_move"
        if technique == "splice":
            donor = base_images[rng.randrange(len(base_images))]
            tampered = _splice(base, donor, rng)
        else:
            tampered = _copy_move(base, rng)
        fname = f"tampered_{technique}_{i:03d}.jpg"
        tampered.save(os.path.join(OUT_DIR, fname), "JPEG", quality=rng.choice([60, 70, 75]))
        labels[fname] = 1

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Wrote {len(labels)} images to {OUT_DIR}/ "
          f"({sum(1 for v in labels.values() if v==0)} authentic, "
          f"{sum(1 for v in labels.values() if v==1)} tampered)")


if __name__ == "__main__":
    generate()
