"""Shared helpers for the NomaApp model notebooks.

Why this module exists
----------------------
Training and serving must agree on *exactly* four things, or the model will
look great in a notebook and behave badly behind the API:

1. input size        -> 224 x 224
2. colour handling   -> RGB (not BGR, not greyscale)
3. pixel scaling     -> float32 in [0, 1] (divide by 255.0)
4. class order       -> the index order used by ``app/services/inference.py``

This file is the single place where those conventions live for the notebook
side. If you change them here, change them in ``app/services/preprocess.py``
and ``app/services/inference.py`` too.

TensorFlow is imported lazily inside the functions that need it, so the data /
EDA helpers work in a lightweight environment with only numpy + Pillow.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Conventions shared with the serving code
# ---------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (224, 224)
RESCALE: float = 255.0
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

#: Must stay in the same order as CLASS_NAMES in app/services/inference.py
CLASS_NAMES: list[str] = [
    "Maize_Blight",
    "Maize_Common_Rust",
    "Maize_Gray_Leaf_Spot",
    "Maize_Healthy",
    "Rice_Bacterial_leaf_blight",
    "Rice_Brown_spot",
    "Rice_Leaf_smut",
    "bean_angular_leaf_spot",
    "bean_healthy",
    "bean_rust",
]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Path = REPO_ROOT / "notebooks" / "data"
DEFAULT_ARTIFACT_DIR: Path = REPO_ROOT / "notebooks" / "artifacts"

_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train", "training"),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}


# ---------------------------------------------------------------------------
# Paths and reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Make a run as repeatable as TensorFlow allows."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - lightweight environments
        pass


def data_dir(path: str | Path | None = None) -> Path:
    """Return the dataset root (``NOMA_DATA_DIR`` env var wins over defaults)."""
    if path is not None:
        return Path(path).expanduser().resolve()
    env = os.environ.get("NOMA_DATA_DIR")
    return Path(env).expanduser().resolve() if env else DEFAULT_DATA_DIR


def artifact_dir(path: str | Path | None = None) -> Path:
    """Return the output directory for trained models and plots."""
    target = Path(path) if path is not None else DEFAULT_ARTIFACT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def find_split_dir(root: str | Path, split: str) -> Path:
    """Locate ``train/``, ``valid/`` or ``test/`` inside ``root``."""
    if split not in _SPLIT_ALIASES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_ALIASES)}, got {split!r}")
    root = Path(root)
    for name in _SPLIT_ALIASES[split]:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    found = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    raise FileNotFoundError(
        f"No {split}/ directory under {root} (found: {found or 'nothing'}). "
        "See notebooks/data/README.md for the expected layout."
    )


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------
def list_class_dirs(split_dir: str | Path) -> list[str]:
    """Sorted class-folder names inside a split directory."""
    return sorted(p.name for p in Path(split_dir).iterdir() if p.is_dir())


def count_images(directory: str | Path) -> int:
    """Number of image files anywhere under ``directory`` (recursive)."""
    directory = Path(directory)
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_images(split_dir: str | Path) -> list[tuple[Path, str]]:
    """Return ``(path, class_name)`` for every image under a split directory."""
    split_dir = Path(split_dir)
    pairs: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((image_path, class_dir.name))
    return pairs


def summarize_dataset(root: str | Path) -> dict[str, dict[str, int]]:
    """Count images per split and class, e.g. ``{"train": {"maize_blight": 120}}``."""
    root = Path(root)
    summary: dict[str, dict[str, int]] = {}
    for split in _SPLIT_ALIASES:
        try:
            split_dir = find_split_dir(root, split)
        except FileNotFoundError:
            continue
        summary[split] = {
            class_dir.name: count_images(class_dir)
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        }
    return summary


def image_stats(paths: list[Path], limit: int | None = 500) -> dict[str, object]:
    """Quick sanity check on image sizes/modes (catches odd exports early)."""
    sample = paths[:limit] if limit else paths
    sizes: dict[tuple[int, int], int] = {}
    modes: dict[str, int] = {}
    for path in sample:
        with Image.open(path) as image:
            sizes[image.size] = sizes.get(image.size, 0) + 1
            modes[image.mode] = modes.get(image.mode, 0) + 1
    return {
        "count": len(sample),
        "sizes": dict(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "modes": modes,
    }


# ---------------------------------------------------------------------------
# Preprocessing (mirrors app/services/preprocess.py)
# ---------------------------------------------------------------------------
def preprocess_image(
    image_path: str | Path, size: tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """Load one image exactly the way the API does: RGB -> resize -> /255.0.

    Returns a batched array shaped ``(1, H, W, 3)`` ready for ``model.predict``.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(size)
        array = np.asarray(image, dtype="float32") / RESCALE
    return np.expand_dims(array, axis=0)


def make_dataset(
    split_dir: str | Path,
    class_names: list[str] | None = None,
    *,
    batch_size: int = 32,
    seed: int = 42,
    shuffle: bool = True,
):
    """Build a batched ``tf.data`` pipeline with an explicit, stable class order."""
    import tensorflow as tf

    split_dir = Path(split_dir)
    if class_names is None:
        class_names = list_class_dirs(split_dir)

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        class_names=class_names,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_train_val_datasets(
    root: str | Path,
    class_names: list[str] | None = None,
    *,
    batch_size: int = 32,
    seed: int = 42,
):
    """Convenience wrapper returning ``(train, valid)`` from ``train/`` + ``valid/``."""
    root = Path(root)
    train_dir = find_split_dir(root, "train")
    valid_dir = find_split_dir(root, "valid")
    if class_names is None:
        class_names = list_class_dirs(train_dir)
    train = make_dataset(
        train_dir, class_names, batch_size=batch_size, seed=seed, shuffle=True
    )
    valid = make_dataset(
        valid_dir, class_names, batch_size=batch_size, seed=seed, shuffle=False
    )
    return train, valid


def augmentation_layer():
    """Light augmentation; keeps leaves recognisable and works in-graph."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------------------
# Class-name bookkeeping
# ---------------------------------------------------------------------------
def save_class_names(path: str | Path, class_names: list[str]) -> Path:
    """Persist the class order next to the model weights — never assume it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(class_names), indent=2) + "\n", encoding="utf-8")
    return path


def load_class_names(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def predict_with_serving_preprocess(
    model, image_path: str | Path, class_names: list[str] | None = None
) -> tuple[str, float]:
    """Predict one image using the *serving* preprocessing path (parity check)."""
    class_names = class_names or CLASS_NAMES
    batch = preprocess_image(image_path)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def plot_training_history(history, *, title: str = "Training history"):
    """Accuracy/loss curves from a Keras ``History`` object."""
    import matplotlib.pyplot as plt

    metrics = history.history
    available = [m for m in ("accuracy", "loss") if m in metrics]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    axes = np.atleast_1d(axes)

    for axis, metric in zip(axes, available):
        axis.plot(metrics[metric], label="train")
        if f"val_{metric}" in metrics:
            axis.plot(metrics[f"val_{metric}"], label="validation")
        axis.set_title(f"{title} — {metric}")
        axis.set_xlabel("epoch")
        axis.set_ylabel(metric)
        axis.legend()
        axis.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def classification_report_arrays(model, dataset, class_names: list[str]) -> str:
    """Human-readable precision/recall/F1 from a batched ``tf.data`` dataset."""
    from sklearn.metrics import classification_report

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_tflite(model, out_path: str | Path, *, quantize: bool = False) -> Path:
    """Convert a Keras model to TFLite (optionally dynamic-range quantized)."""
    import tensorflow as tf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    out_path.write_bytes(converter.convert())
    return out_path


def export_onnx(model, out_path: str | Path, *, opset: int = 17) -> Path:
    """Convert a Keras model to ONNX (requires ``tf2onnx``).

    The target format is the one used by ``model_cache/manifest.json``
    (``router/`` and ``experts/`` ``.onnx`` files), so exports go straight into
    the shape the serving layer expects.
    """
    import tf2onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(model, opset=opset, output_path=str(out_path))
    return out_path


def manifest_entry(
    *,
    kind: str,
    file: str,
    version: str,
    classes: list[str] | None = None,
    input_size: tuple[int, int] = IMG_SIZE,
) -> dict:
    """Build a ``model_cache/manifest.json``-compatible entry.

    ``kind="router"`` produces the top-level router block;
    ``kind="expert"`` produces one entry for the ``experts`` mapping.
    """
    if kind == "router":
        return {
            "file": file,
            "version": version,
            "input_size": list(input_size),
            "classes": list(classes or []),
        }
    if kind == "expert":
        return {"file": file, "version": version, "diseases": list(classes or [])}
    raise ValueError("kind must be 'router' or 'expert'")
