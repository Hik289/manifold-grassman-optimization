"""Redraw the paper's method figure using the shared README diagram renderer."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "assets"))
from draw_intuition import draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "paper/figures/fig_method_update_detailed.png",
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    draw(args.output, args.preview)
