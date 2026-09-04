from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np
from PIL import Image, ImageChops

ELA_RESAVE_QUALITY = 90
BLOCK_SIZE = 16
HASH_SIZE = 8            
CLONE_HAMMING_THRESHOLD = 4     
CLONE_MIN_SPATIAL_DIST = 40     

WEIGHT_ELA = 0.45
WEIGHT_QUALITY_MISMATCH = 0.20
WEIGHT_CLONE = 0.35

MANUAL_REVIEW_THRESHOLD = 0.45   


def error_level_analysis(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=ELA_RESAVE_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(img.convert("RGB"), resaved)
    diff_arr = np.asarray(diff).astype(np.float32)
    error_map = diff_arr.mean(axis=2)  

    h, w = error_map.shape
    block_means = []
    for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            block_means.append(float(error_map[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE].mean()))
    block_means = np.array(block_means)

    median_block = float(np.median(block_means))
    q1, q3 = np.percentile(block_means, [25, 75])
    iqr = float(q3 - q1)
    max_block = float(np.max(block_means))

    robust_z = (max_block - median_block) / (max(iqr, 0.5))

    return {
        "median_block_error": round(median_block, 3),
        "max_block_error": round(max_block, 3),
        "block_iqr": round(iqr, 3),
        "robust_z": round(float(robust_z), 3),
    }


def quality_consistency(img: Image.Image) -> dict:
    """Splits the image into quadrants and re-saves each quadrant alone at a
    fixed quality, comparing how much each quadrant 'moves' under
    recompression. Quadrants from a different generation/edit history than
    the rest of the image tend to shift by a visibly different amount."""
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        arr[0:mid_h, 0:mid_w], arr[0:mid_h, mid_w:w],
        arr[mid_h:h, 0:mid_w], arr[mid_h:h, mid_w:w],
    ]

    shifts = []
    for q in quadrants:
        q_img = Image.fromarray(q)
        buf = io.BytesIO()
        q_img.save(buf, "JPEG", quality=ELA_RESAVE_QUALITY)
        buf.seek(0)
        q_resaved = Image.open(buf)
        d = np.asarray(ImageChops.difference(q_img, q_resaved)).astype(np.float32)
        shifts.append(float(d.mean()))

    shifts = np.array(shifts)
    cv = float(shifts.std() / (shifts.mean() + 1e-6))
    return {"quadrant_shifts": [round(s, 3) for s in shifts], "quadrant_cv": round(cv, 3)}



def _average_hash(block: np.ndarray) -> int:
    gray = block.mean(axis=2) if block.ndim == 3 else block
    small = Image.fromarray(gray.astype(np.uint8)).resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    small_arr = np.asarray(small)
    avg = small_arr.mean()
    bits = (small_arr > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def copy_move_detection(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape
    positions, hashes = [], []

    for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            block = arr[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            if block.std() < 3: 
                continue
            positions.append((x, y))
            hashes.append(_average_hash(block))

    n = len(hashes)
    matches = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = _hamming(hashes[i], hashes[j])
            if dist <= CLONE_HAMMING_THRESHOLD:
                (x1, y1), (x2, y2) = positions[i], positions[j]
                spatial_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if spatial_dist >= CLONE_MIN_SPATIAL_DIST:
                    matches.append({"a": positions[i], "b": positions[j], "hamming": dist})

    return {"n_blocks_compared": n, "n_clone_matches": len(matches), "sample_matches": matches[:5]}


def analyze_image(path: str) -> dict:
    img = Image.open(path)
    ela = error_level_analysis(img)
    quality = quality_consistency(img)
    clone = copy_move_detection(img)

    ela_score = min(1.0, max(0.0, (ela["robust_z"] - 3.0) / 5.0))
    quality_score = min(1.0, max(0.0, (quality["quadrant_cv"] - 0.15) / 0.6))
    clone_score = min(1.0, clone["n_clone_matches"] / 3.0)

    composite = (WEIGHT_ELA * ela_score + WEIGHT_QUALITY_MISMATCH * quality_score
                 + WEIGHT_CLONE * clone_score)

    reasons = []
    if ela_score > 0.3:
        reasons.append(f"ELA block anomaly z-score {ela['robust_z']:.2f} — one region "
                        f"re-compresses very differently from the rest of the image")
    if quality_score > 0.3:
        reasons.append(f"Inconsistent compression across image quadrants "
                        f"(CV={quality['quadrant_cv']:.2f}) — suggests mixed edit history")
    if clone_score > 0.3:
        reasons.append(f"{clone['n_clone_matches']} near-duplicate region(s) found far apart "
                        f"in the image — possible copy-move edit")
    if not reasons:
        reasons.append("No strong tampering signal detected")

    action = "manual_review" if composite >= MANUAL_REVIEW_THRESHOLD else "accept"

    return {
        "path": path,
        "composite_suspicion_score": round(float(composite), 4),
        "recommended_action": action,
        "reasons": reasons,
        "signals": {"ela": ela, "quality_consistency": quality, "copy_move": clone},
    }


def main():
    parser = argparse.ArgumentParser(description="Return-evidence photo integrity checker (defense-only).")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", default="reports/evidence_scored.csv")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.jpg")) +
                        glob.glob(os.path.join(args.input, "*.jpeg")) +
                        glob.glob(os.path.join(args.input, "*.png")))
    else:
        paths = [args.input]

    import pandas as pd
    rows = []
    for p in paths:
        try:
            result = analyze_image(p)
            rows.append({
                "path": p,
                "composite_suspicion_score": result["composite_suspicion_score"],
                "recommended_action": result["recommended_action"],
                "reasons": " | ".join(result["reasons"]),
            })
            print(f"{os.path.basename(p):35s} score={result['composite_suspicion_score']:.3f}  "
                  f"-> {result['recommended_action']:15s} ({result['reasons'][0]})")
        except Exception as e:
            print(f"Failed on {p}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
