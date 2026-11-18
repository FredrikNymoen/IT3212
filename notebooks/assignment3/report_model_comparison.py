"""Refresh the section-8 figure from fixed validation exports; no model fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def render(results_dir=None):
    results = Path(results_dir or ROOT / 'results/assignment3/lda_selected')
    data = pd.read_csv(results / 'development_models.csv').set_index('model')
    order = ['Prior baseline', 'Tree bagging control', 'Decision tree', 'Bagged KNN',
             'KNN', 'LDA', 'Bagged trees', 'SVM', 'AdaBoost trees']
    plot = data.loc[order]
    assert np.isfinite(plot[['CV_AP', 'fold_SD']].to_numpy()).all()
    families = plot.family.replace({'linear classifier': 'basic'})
    colors = {'reference': '#72b6a1', 'ensemble control': '#e99675',
              'basic': '#95a3c3', 'bagging': '#db96c0', 'boosting': '#a2c865'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42, 'axes.edgecolor': '#cccccc'})
    fig, ax = plt.subplots(figsize=(11.2, 5.8), layout='constrained')
    positions = np.arange(len(order))
    ax.barh(positions, plot.CV_AP, xerr=plot.fold_SD,
            color=[colors[family] for family in families], height=.72,
            error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1.1})
    ax.set_yticks(positions, order)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.06), xlabel='Cross-Validation Average Precision',
           title='Model Comparison: Average Precision')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#cccccc', linewidth=.8)
    ax.tick_params(axis='both', length=0)
    for i, row in enumerate(plot.itertuples()):
        ax.text(row.CV_AP + row.fold_SD + .006, i, f'{row.CV_AP:.4f}',
                va='center', fontsize=9.5)
    ax.legend(handles=[Patch(facecolor=color, label=family) for family, color in colors.items()],
              title='Model family', loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)
    figure_dir = ROOT / 'figures/assignment3/lda_selected'
    image_dir = ROOT / 'report/Images'
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    for output in [image_dir / '08_model_comparison_AP.pdf',
                   figure_dir / '08_model_comparison_AP.png']:
        fig.savefig(output, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Updated section-8 figure from the nine validation result rows.')


if __name__ == '__main__':
    render()
