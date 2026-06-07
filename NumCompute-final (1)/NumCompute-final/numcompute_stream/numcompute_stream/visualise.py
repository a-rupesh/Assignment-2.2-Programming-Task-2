"""Built-in visualisation module for NumCompute Stream.

All functions use matplotlib only (no seaborn, no pandas, no scikit-learn).
They are designed to render in Jupyter notebooks and save to files.

Functions
---------
plot_metrics
    Line chart of one or more metrics tracked over streaming chunks.
plot_confusion_matrix
    Heatmap-style confusion matrix.
plot_roc_curve
    ROC curve from FPR/TPR arrays.
plot_comparison
    Side-by-side metric comparison for two or more models.
plot_tree_structure
    Simple text-based visualisation of a StreamingDecisionTree.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    '#1B74C4', '#E84C2A', '#2BA84A', '#F0A500',
    '#7E57C2', '#0097A7', '#EF5350', '#8D6E63',
]


def _get_ax(ax, figsize=(8, 4)):
    """Return a (fig, ax) pair; create one if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _style_ax(ax, title='', xlabel='', ylabel='', grid=True):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_metrics(
    metrics: dict[str, list[float]],
    *,
    title: str = 'Streaming metrics over chunks',
    xlabel: str = 'Chunk',
    ylabel: str = 'Score',
    figsize: tuple = (9, 4),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Plot one or more metrics as lines over streaming chunks.

    Parameters
    ----------
    metrics : dict[str, list[float]]
        Mapping from metric name to a list of values (one per chunk).
        Example: ``{'accuracy': [0.7, 0.75, 0.8], 'f1': [0.65, 0.7, 0.78]}``.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple, default=(9, 4)
        Figure size in inches (ignored if ax is supplied).
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on; creates new figure if None.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    tuple[Figure, Axes]

    Raises
    ------
    ValueError
        If metrics is empty or contains non-list values.

    Examples
    --------
    >>> fig, ax = plot_metrics({'accuracy': [0.6, 0.7, 0.8]})
    >>> plt.show()
    """
    if not metrics:
        raise ValueError("metrics must be a non-empty dict.")
    for k, v in metrics.items():
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError(f"metrics['{k}'] must be a list or ndarray.")

    fig, ax = _get_ax(ax, figsize)
    for i, (name, values) in enumerate(metrics.items()):
        xs = np.arange(1, len(values) + 1)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(xs, values, marker='o', markersize=4, linewidth=1.8,
                color=color, label=name)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0.5)
    ax.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str] | None = None,
    title: str = 'Confusion matrix',
    cmap: str = 'Blues',
    figsize: tuple = (5, 4),
    ax=None,
    save_path: str | None = None,
    normalize: bool = False,
) -> tuple:
    """Plot a confusion matrix as a colour-coded grid.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix (rows = true, cols = predicted).
    class_names : list[str] or None
        Class labels; uses '0', '1', … if None.
    title : str
    cmap : str
        Matplotlib colourmap name.
    figsize : tuple
    ax : Axes or None
    save_path : str or None
    normalize : bool
        If True, normalise each row to sum to 1.

    Returns
    -------
    tuple[Figure, Axes]

    Raises
    ------
    ValueError
        If cm is not square.

    Examples
    --------
    >>> from numcompute_stream.visualise import plot_confusion_matrix
    >>> plot_confusion_matrix(np.array([[10, 2], [1, 7]]))
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2D array.")

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    if len(class_names) != n:
        raise ValueError("class_names length must match cm size.")

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums

    fig, ax = _get_ax(ax, figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.8)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    thresh = cm.max() / 2.0
    fmt = '.2f' if normalize else 'd'
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = f'{val:{fmt}}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if val > thresh else 'black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    auc_score: float | None = None,
    label: str = 'ROC',
    title: str = 'ROC curve',
    figsize: tuple = (5, 5),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Plot an ROC curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    auc_score : float or None
        If provided, shown in the legend.
    label : str
        Legend label.
    title : str
    figsize : tuple
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape.")

    fig, ax = _get_ax(ax, figsize)
    leg = label if auc_score is None else f'{label} (AUC={auc_score:.3f})'
    ax.plot(fpr, tpr, linewidth=2, color=_PALETTE[0], label=leg)
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='grey', label='Random')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc='lower right')
    _style_ax(ax, title=title, xlabel='False positive rate', ylabel='True positive rate')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_comparison(
    model_metrics: dict[str, dict[str, list[float]]],
    *,
    metric_name: str = 'accuracy',
    title: str | None = None,
    xlabel: str = 'Chunk',
    ylabel: str | None = None,
    figsize: tuple = (9, 4),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Plot the same metric for multiple models on one chart.

    Parameters
    ----------
    model_metrics : dict[str, dict[str, list[float]]]
        Outer key = model name, inner key = metric name, value = list of scores.
        Example::

            {
                'Decision Tree': {'accuracy': [0.6, 0.7, 0.8]},
                'Random Forest': {'accuracy': [0.65, 0.75, 0.85]},
            }

    metric_name : str
        Which metric to plot (must exist in every inner dict).
    title : str or None
    xlabel : str
    ylabel : str or None
    figsize : tuple
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    tuple[Figure, Axes]

    Raises
    ------
    ValueError
        If model_metrics is empty or metric_name is missing from a model.
    """
    if not model_metrics:
        raise ValueError("model_metrics must be a non-empty dict.")

    fig, ax = _get_ax(ax, figsize)
    for i, (model_name, mdict) in enumerate(model_metrics.items()):
        if metric_name not in mdict:
            raise ValueError(f"Metric '{metric_name}' not found for model '{model_name}'.")
        values = mdict[metric_name]
        xs = np.arange(1, len(values) + 1)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(xs, values, marker='o', markersize=4, linewidth=1.8,
                color=color, label=model_name)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0.5)
    ax.legend(fontsize=9, framealpha=0.7)
    plot_title = title or f'{metric_name.capitalize()} comparison'
    plot_ylabel = ylabel or metric_name.capitalize()
    _style_ax(ax, title=plot_title, xlabel=xlabel, ylabel=plot_ylabel)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_tree_structure(
    tree,
    *,
    max_depth: int = 4,
) -> str:
    """Return an ASCII representation of a StreamingDecisionTree.

    Parameters
    ----------
    tree : StreamingDecisionTree
        Fitted tree to inspect.
    max_depth : int, default=4
        Maximum depth to render.

    Returns
    -------
    str
        Multi-line string showing tree structure.
    """
    if tree.root_ is None:
        return "<empty tree>"

    lines = []

    def _render(node, depth: int, prefix: str) -> None:
        if depth > max_depth:
            lines.append(prefix + "...")
            return
        maj = int(node.class_counts.argmax()) if node.n_samples > 0 else 0
        counts_str = '/'.join(str(int(c)) for c in node.class_counts)
        label = tree.classes_[maj] if tree.classes_ is not None else maj
        if node.is_leaf():
            lines.append(f"{prefix}[LEAF] class={label} counts=[{counts_str}] n={node.n_samples}")
        else:
            lines.append(
                f"{prefix}[SPLIT] f{node.split_feature} <= {node.split_threshold:.4f}"
                f" | class={label} n={node.n_samples}"
            )
            if node.left is not None:
                _render(node.left,  depth + 1, prefix + "  ├─ ")
            if node.right is not None:
                _render(node.right, depth + 1, prefix + "  └─ ")

    _render(tree.root_, 0, "")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spec-required function names
# ---------------------------------------------------------------------------

def plot_metric_over_time(
    metric_values,
    title: str = 'Metric over time',
    ylabel: str = 'Value',
    xlabel: str = 'Chunk',
    figsize: tuple = (9, 4),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Plot a single metric (e.g. accuracy) across streaming chunks.

    This is the spec-required name for plot_metrics() with a single series.

    Parameters
    ----------
    metric_values : list of float
        One value per chunk.
    title : str
    ylabel : str
    xlabel : str
    figsize : tuple
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    tuple[Figure, Axes]

    Examples
    --------
    >>> plot_metric_over_time([0.6, 0.7, 0.8], title='Accuracy', ylabel='Accuracy')
    """
    return plot_metrics(
        {ylabel: metric_values},
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        ax=ax,
        save_path=save_path,
    )


def compare_models(
    metric1,
    metric2,
    labels: list | None = None,
    title: str = 'Model comparison',
    ylabel: str = 'Score',
    xlabel: str = 'Chunk',
    figsize: tuple = (9, 4),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Compare two models on streaming metrics.

    This is the spec-required name; wraps plot_metrics() with two series.

    Parameters
    ----------
    metric1 : list of float
        Metric values for the first model.
    metric2 : list of float
        Metric values for the second model.
    labels : list of str or None
        Names for the two models. Defaults to ['Model 1', 'Model 2'].
    title : str
    ylabel : str
    xlabel : str
    figsize : tuple
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    tuple[Figure, Axes]

    Examples
    --------
    >>> compare_models([0.6, 0.7], [0.65, 0.75], labels=['Tree', 'Forest'])
    """
    if labels is None:
        labels = ['Model 1', 'Model 2']
    if len(labels) < 2:
        raise ValueError("labels must have at least 2 entries.")
    return plot_metrics(
        {labels[0]: metric1, labels[1]: metric2},
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        ax=ax,
        save_path=save_path,
    )


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = 'Predictions vs Ground Truth',
    figsize: tuple = (9, 4),
    ax=None,
    save_path: str | None = None,
) -> tuple:
    """Visualise predictions against actual labels for the latest chunk.

    Plots true labels and predicted labels as step lines so differences
    are immediately visible.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    title : str
    figsize : tuple
    ax : Axes or None
    save_path : str or None

    Returns
    -------
    tuple[Figure, Axes]

    Examples
    --------
    >>> plot_predictions_vs_ground_truth([0,1,1,0], [0,1,0,0])
    """
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    fig, ax_obj = _get_ax(ax, figsize)
    xs = np.arange(len(y_true))

    ax_obj.step(xs, y_true, where='mid', linewidth=1.8,
                color=_PALETTE[0], label='Ground truth')
    ax_obj.step(xs, y_pred, where='mid', linewidth=1.8,
                color=_PALETTE[1], label='Predicted', linestyle='--')

    # Highlight mismatches
    mismatch = y_true != y_pred
    if mismatch.any():
        ax_obj.scatter(xs[mismatch], y_pred[mismatch], color=_PALETTE[2],
                       zorder=5, s=40, label='Mismatch')

    ax_obj.legend(fontsize=9, framealpha=0.7)
    _style_ax(ax_obj, title=title, xlabel='Sample index', ylabel='Class')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax_obj
