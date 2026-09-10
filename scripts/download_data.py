import shutil #utility functions for copying and archiving files
from pathlib import Path

import kagglehub

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print(f"Downloaded to cache: {path}")
    src = Path(path)
    target = src / "twcs" / "twcs.csv"
    if not target.exists():
        candidates = list(src.rglob("twcs.csv")) #search for twcs.csv in the downloaded dataset
        if not candidates:
            raise SystemExit(f"twcs.csv not found under {src}")
        target = candidates[0]
    dest = RAW_DIR / "twcs.csv"
    shutil.copy2(target, dest)
    print(f"Copied {target} -> {dest}")


if __name__ == "__main__":
    main()
