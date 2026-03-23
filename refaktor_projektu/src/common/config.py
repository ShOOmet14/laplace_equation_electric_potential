from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = PROJECT_ROOT / "images"

SIDE_LENGTH = 1.0
GRID_SIZES = (10, 20, 40, 80)
TOLERANCES = {
    10: 1e-10,
    20: 1e-10,
    40: 1e-11,
    80: 1e-11,
}
MAX_ITERATIONS = 50000

PLOT_GRID_SIZE = 81
PLOT_TOLERANCE = 1e-10
PLOT_MAX_ITERATIONS = 30000

Y_LEVELS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
ITERATIONS_TO_SAVE = (20, 100, 200, 2000)
ERROR_THRESHOLD = 1e-4


def get_images_dir(language_code: str) -> Path:
    images_dir = IMAGES_ROOT / language_code
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir
