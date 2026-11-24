"""Shared print style for the paper's loss plots.

Colors, typography, and legends follow the figures4papers scientific-figure
style. Styling never changes the observations or the smoothing window.
"""
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from artifact_io import save_figure


COLORS = {
    'GrassRoute': '#0F4D92',
    'AdamW': '#3F454B',
    'PeriodicSVD': '#B64342',
    'PeriodicDiag': '#B64342',
    'PeriodicThinSVD': '#B64342',
    'Fixed': '#8E969E',
    'SGD': '#A0803E',
    'SubTrackOfficial': '#42949E',
}
LINESTYLES = {
    'GrassRoute': '-',
    'AdamW': '-',
    'PeriodicSVD': (0, (5.0, 2.2)),
    'PeriodicDiag': (0, (5.0, 2.2)),
    'PeriodicThinSVD': (0, (5.0, 2.2)),
    'Fixed': (0, (1.3, 1.7)),
    'SGD': (0, (6.5, 2.0, 1.2, 2.0)),
    'SubTrackOfficial': (0, (3.8, 1.5, 1.0, 1.5)),
}
NAMES = {
    'GrassRoute': 'GrassRoute',
    'AdamW': 'AdamW',
    'PeriodicSVD': 'Periodic SVD',
    'PeriodicDiag': 'PeriodicDiag',
    'PeriodicThinSVD': 'Thin SVD',
    'Fixed': 'Fixed',
    'SGD': 'SGD-M',
    'SubTrackOfficial': 'SubTrack++',
}
RCPARAMS = {
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9.5,
    'axes.labelsize': 10,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'axes.labelcolor': '#252A30',
    'text.color': '#252A30',
    'xtick.color': '#4D535B',
    'ytick.color': '#4D535B',
    'axes.edgecolor': '#545C65',
    'axes.linewidth': .85,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'legend.handlelength': 2.6,
    'legend.handletextpad': .55,
    'legend.columnspacing': 1.35,
    'legend.labelspacing': .45,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'lines.dash_capstyle': 'butt',
}


def line_style(method):
    return dict(color=COLORS[method], linestyle=LINESTYLES[method],
                linewidth=2.15 if method == 'GrassRoute' else 1.45,
                zorder=5 if method == 'GrassRoute' else 3)


def legend_handles(methods):
    return [Line2D([], [], label=NAMES[m], **line_style(m)) for m in methods]


def plot_training(ax, epochs, loss, method):
    """Plot every raw loss and the unchanged 25-epoch trailing mean."""
    epochs = np.asarray(epochs)
    loss = np.asarray(loss, dtype=float)
    assert len(epochs) == len(loss) and len(loss) >= 25
    assert np.isfinite(loss).all() and (loss > 0).all()
    ax.plot(epochs, loss, color=COLORS[method], alpha=.18, linewidth=.45,
            rasterized=True, zorder=1)
    mean = np.convolve(loss, np.ones(25) / 25, mode='valid')
    ax.plot(epochs[24:], mean, label=NAMES[method], **line_style(method))
    return mean


def _log_label(value, _position=None):
    exponent = int(math.floor(math.log10(value) + 1e-9))
    mantissa = value / 10 ** exponent
    if abs(mantissa - 1) < 1e-7:
        return rf'$10^{{{exponent}}}$'
    return rf'${mantissa:g}\!\times\!10^{{{exponent}}}$'


def finish_axis(ax, series, epochs, *, logarithmic=True, ylabel='Training objective'):
    """Set limits from all raw values; never crop unfavorable observations."""
    values = np.concatenate([np.asarray(x, dtype=float) for x in series])
    assert np.isfinite(values).all()
    ax.set_xlim(0, epochs)
    ax.set_xticks(np.linspace(0, epochs, 5, dtype=int))
    ax.set_xlabel('Epoch', labelpad=5)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.tick_params(axis='both', which='major', length=3, width=.75, pad=3)
    ax.tick_params(axis='both', which='minor', length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', color='#E7EBF0', linewidth=.65)
    if logarithmic:
        assert (values > 0).all()
        low, high = np.log10(values.min()), np.log10(values.max())
        margin = max(.055, (high - low) * .045)
        limits = (10 ** (low - margin), 10 ** (high + margin))
        ax.set_yscale('log')
        powers = np.arange(math.floor(low - margin), math.ceil(high + margin) + 1)
        ticks = [10. ** k for k in powers if limits[0] <= 10. ** k <= limits[1]]
        if len(ticks) < 3:
            ticks = sorted(t * 10. ** k for k in powers for t in [1., 2., 5.]
                           if limits[0] <= t * 10. ** k <= limits[1])
        if len(ticks) > 8:
            ticks = ticks[::math.ceil(len(ticks) / 8)]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(_log_label))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(limits)
    else:
        span = values.max() - values.min()
        ax.set_ylim(max(0, values.min() - .06 * span), values.max() + .07 * span)


def export_figure(fig, stem):
    """Export a vector PDF and a print-resolution PNG with all labels in view."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, stem.with_suffix('.pdf'), dpi=450, bbox_inches='tight', pad_inches=.07)
    save_figure(fig, stem.with_suffix('.png'), dpi=450, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
