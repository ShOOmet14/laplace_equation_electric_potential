from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
IMAGES_DIR = PROJECT_ROOT / "images"


def get_images_dir(lang: str) -> Path:
    path = IMAGES_DIR / lang
    path.mkdir(parents=True, exist_ok=True)
    return path
