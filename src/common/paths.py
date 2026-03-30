from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
IMAGES_DIR = PROJECT_ROOT / "images"


def get_images_dir(lang: str) -> Path:
    """
    Returns the path to the images directory for the given language.

    If the directory does not exist, it is created along with any mising parent directories.

    Args:
    - lang: Language code used as subdirectory name inside 'IMAGES_DIR'

    Returns:
    - Path to the images directory for the given language
    """

    path = IMAGES_DIR / lang
    path.mkdir(parents=True, exist_ok=True)
    return path
