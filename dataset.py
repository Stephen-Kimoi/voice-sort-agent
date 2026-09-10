"""Generates a small synthetic "washer" image set for the anomaly classifier.

We deliberately avoid MVTec AD here: anomalib's built-in loader for it downloads
the *entire* 5.2GB multi-category archive just to reach one category folder, and
the dataset itself is licensed for non-commercial research only. A synthetic set
gives full, original, license-free images in seconds and is good enough to prove
the pipeline — swap in your own product photos for a real deployment.

"Good" washers are a clean ring. "Anomalous" washers have one of: a radial crack,
a chipped edge, or a surface scratch. Padim only ever trains on "good" images
(anomaly detection is one-class); the anomalous ones exist purely to test it.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SIZE = 256
RING_OUTER = 90
RING_INNER = 40
CENTER = IMAGE_SIZE // 2


def _draw_good(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    jitter = rng.randint(-3, 3)
    bbox_outer = [CENTER - RING_OUTER + jitter, CENTER - RING_OUTER + jitter,
                  CENTER + RING_OUTER + jitter, CENTER + RING_OUTER + jitter]
    bbox_inner = [CENTER - RING_INNER + jitter, CENTER - RING_INNER + jitter,
                  CENTER + RING_INNER + jitter, CENTER + RING_INNER + jitter]
    metal = (150 + rng.randint(-5, 5),) * 3
    draw.ellipse(bbox_outer, fill=metal, outline=(90, 90, 90), width=2)
    draw.ellipse(bbox_inner, fill=(235, 235, 235))
    return img


def _draw_anomalous(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = _draw_good(seed)
    draw = ImageDraw.Draw(img)
    defect = rng.choice(["crack", "chip", "scratch"])
    if defect == "crack":
        angle = rng.uniform(0, 360)
        import math
        x1 = CENTER + RING_INNER * math.cos(math.radians(angle))
        y1 = CENTER + RING_INNER * math.sin(math.radians(angle))
        x2 = CENTER + RING_OUTER * math.cos(math.radians(angle))
        y2 = CENTER + RING_OUTER * math.sin(math.radians(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=4)
    elif defect == "chip":
        angle = rng.uniform(0, 360)
        import math
        cx = CENTER + (RING_OUTER - 10) * math.cos(math.radians(angle))
        cy = CENTER + (RING_OUTER - 10) * math.sin(math.radians(angle))
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(235, 235, 235))
    else:
        x1, y1 = CENTER - 50, CENTER - 20
        x2, y2 = CENTER + 55, CENTER + 15
        draw.line([(x1, y1), (x2, y2)], fill=(60, 60, 60), width=3)
    return img


def build_dataset(root: Path, n_train_good: int = 40, n_test_good: int = 10, n_test_anomalous: int = 10) -> None:
    train_good = root / "train" / "good"
    test_good = root / "test" / "good"
    test_anomalous = root / "test" / "anomalous"
    for d in (train_good, test_good, test_anomalous):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(n_train_good):
        _draw_good(seed=i).save(train_good / f"good_{i:03d}.png")
    for i in range(n_test_good):
        _draw_good(seed=1000 + i).save(test_good / f"good_{i:03d}.png")
    for i in range(n_test_anomalous):
        _draw_anomalous(seed=2000 + i).save(test_anomalous / f"anomalous_{i:03d}.png")

    print(f"Wrote {n_train_good} train/good, {n_test_good} test/good, "
          f"{n_test_anomalous} test/anomalous images under {root}")


if __name__ == "__main__":
    build_dataset(Path("data"))
