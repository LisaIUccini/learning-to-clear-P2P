"""Every paper figure and table regenerates identically from the saved results
(tables byte for byte, figures pixel for pixel; PDFs embed a creation date,
so the PNG twin of each figure is compared instead)."""

import filecmp
import os
import subprocess
import sys

import matplotlib.image as mpimg
import numpy as np

from l2p2p import paths


def test_paper_outputs_regenerate_identically(tmp_path):
    env = dict(os.environ, L2P2P_PAPER_OUT=str(tmp_path))
    subprocess.run([sys.executable, str(paths.PAPER / "make_all.py")], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    for tex in sorted(paths.PAPER_TABLES.glob("*.tex")):
        assert filecmp.cmp(tex, tmp_path / "tables" / tex.name, shallow=False), tex.name
    for png in sorted(paths.PAPER_FIGURES.glob("*.png")):
        assert np.array_equal(mpimg.imread(png), mpimg.imread(tmp_path / "figures" / png.name)), png.name
