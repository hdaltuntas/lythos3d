#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run Lythos 3D without installing it.

    python main.py info                 versions and the linear solvers available
    python main.py demo -o out          analyse a footing and write it for ParaView
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    from lythos3d.cli import main

    sys.exit(main())
