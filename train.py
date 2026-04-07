#!/usr/bin/env python3
"""
Entry shim so you can run from the repo root::

    python train.py --config configs/C1.yaml

Same as ``python -m code_base.train ...`` (useful in Colab after ``%cd`` into the clone).
"""

from code_base.train import main

if __name__ == "__main__":
    main()
