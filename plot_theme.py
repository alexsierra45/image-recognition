"""Single source of the plot style: experiment plots (plot_utils.py) and delivery figures (scripts/plot_style.py).

Convention: one colour per experiment / family, train = dashed line, validation = solid line.
"""
# Categorical slots in fixed order (validated colour-blind-safe reference palette); a 9th series folds into grey.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GREY = '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'

RC = {
    'figure.dpi': 110, 'savefig.dpi': 130, 'savefig.bbox': 'tight', 'font.size': 9,
    'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'axes.titlesize': 10, 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.markersize': 5,
}
