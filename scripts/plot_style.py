"""Style of the delivery figures: the same as the experiment plots (plot_theme.py), English labels, PNG output.

The same PNG is used in the report and in the slides (only the inclusion width changes).
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_theme import GREY, INK, INK2, PALETTE, RC  # noqa: E402,F401

# One colour per concept, the same in every figure.
CONCEPT = {'baseline': GREY, 'final': PALETTE[0], 'train': PALETTE[1], 'val': PALETTE[0], 'other': '#c9c8c2'}
SIZES = {'full': (6.4, 3.4), 'half': (5.0, 3.8)}   # inches, as the experiment plots (6.4 x 3.8)
DPI = 200


def apply():
    matplotlib.rcParams.update(RC)


def new_figure(layout, height_scale=1.0):
    w, h = SIZES[layout]
    return plt.subplots(figsize=(w, h * height_scale))
