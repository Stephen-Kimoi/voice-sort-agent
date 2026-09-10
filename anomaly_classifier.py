"""Trains a Padim anomaly model with Anomalib and serves inference through OpenVINO.

Padim is a one-class model: it only ever sees "good" images during training and
learns what normal looks like statistically (per-patch feature distributions
from a frozen, pretrained ResNet18). At inference time it scores how far a new
image's features fall from that learned distribution — no anomalous examples are
needed for training, which matches the real factory scenario of having plenty of
good units and few labeled defect examples.

Training happens once in PyTorch (Anomalib's Engine). For inference we export
the trained model to OpenVINO's IR format and run it through OpenVINO's runtime
instead of PyTorch — this is the deployment path Intel's own reference kits use,
and it's what makes this pipeline edge-deployable on CPU-only hardware.
"""

from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

MODEL_DIR = Path("model")
IMAGE_SIZE = 256


def train_and_export(data_root: Path = Path("data"), export_dir: Path = MODEL_DIR) -> Path:
    """Trains Padim on data_root/train/good and exports it to OpenVINO IR.

    Returns the path to the exported model.xml.
    """
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim

    datamodule = Folder(
        name="washers",
        root=data_root,
        normal_dir="train/good",
        abnormal_dir="test/anomalous",
        normal_test_dir="test/good",
    )
    model = Padim()
    engine = Engine(max_epochs=1, accelerator="cpu", devices=1, default_root_dir=str(export_dir / "run"))
    engine.fit(model=model, datamodule=datamodule)

    export_dir.mkdir(parents=True, exist_ok=True)
    exported_path = engine.export(
        model=model,
        export_type="openvino",
        export_root=export_dir,
    )
    return Path(exported_path)


class OpenVINOAnomalyClassifier:
    """Loads an exported Padim OpenVINO model and classifies single images."""

    def __init__(self, model_xml: Path, threshold: float | None = None):
        core = ov.Core()
        compiled = core.read_model(model_xml)
        self.model = core.compile_model(compiled, "CPU")
        self.output = self.model.output(0)
        # Padim's OpenVINO export bakes its own decision threshold into the model
        # metadata when one isn't supplied, but we default to the Anomalib
        # standard of 0.5 on the normalized anomaly map/score if none is found.
        self.threshold = threshold if threshold is not None else 0.5

    def _preprocess(self, image_path: Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        return arr

    def classify(self, image_path: Path) -> dict:
        arr = self._preprocess(image_path)
        result = self.model([arr])[self.output]
        score = float(np.asarray(result).reshape(-1).max())
        is_anomalous = score >= self.threshold
        return {"image": str(image_path), "score": score, "is_anomalous": is_anomalous}


if __name__ == "__main__":
    exported = train_and_export()
    print(f"Exported OpenVINO model to {exported}")
