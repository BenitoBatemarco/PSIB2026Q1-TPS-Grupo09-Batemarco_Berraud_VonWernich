
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _has_subjects(path):
    try:
        import glob
        return bool(glob.glob(os.path.join(path, "sub-*")))
    except Exception:
        return False


def _detect_dataset():
    """Busca la carpeta del dataset cerca del codigo."""
    candidates = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        HERE,                       # el propio codigo (si esta junto a sub-XX)
        os.path.dirname(HERE),      # carpeta que contiene a psib_eeg
        os.getcwd(),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and _has_subjects(c):
            return c
    return None


def main():
    from gui.app import launch
    launch(initial_dir=_detect_dataset())


if __name__ == "__main__":
    main()
