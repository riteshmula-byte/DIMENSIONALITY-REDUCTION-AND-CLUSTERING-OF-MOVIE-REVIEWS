"""Re-draw every figure in results/figures from results/ (no recomputation)."""
from pathlib import Path
from usla.plots import make_all

if __name__ == "__main__":
    for p in make_all(Path(__file__).resolve().parent):
        print(p)
