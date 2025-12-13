import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .common import RANDOM_STATE

def setup_environment():
    # Pandas display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 50)
    
    # NumPy reproducibility
    np.random.seed(42)          
    
    # Matplotlib global style
    plt.rcParams.update({
        "figure.figsize": (9, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        # Colorblind-friendly & beautiful default cycle
        "axes.prop_cycle": plt.cycler("color", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ])
    })
    
    # Seaborn style
    sns.set_style("whitegrid")              # clean background with grid
    sns.set_context("notebook", font_scale=1.25)
    sns.set_palette("colorblind")          
    