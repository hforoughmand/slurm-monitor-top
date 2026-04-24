from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slurm_top.app import main


if __name__ == "__main__":
    main()