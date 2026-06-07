"""NumCompute Stream — streaming decision tree extensions for NumCompute.

New modules
-----------
preprocessing : StreamingStandardScaler, StreamingMinMaxScaler,
                StreamingImputer, StreamingOneHotEncoder
tree          : StreamingDecisionTree (alias: DecisionTreeClassifier)
ensemble      : StreamingBaggingClassifier, StreamingRandomForest
                (alias: EnsembleClassifier)
stream        : StreamTrainer
stats         : ChunkStats (update_stats API), StreamingStats
metrics       : StreamingMetrics (update/reset/result API)
pipeline      : StreamingPipeline (partial_fit support)
visualise     : plot_metric_over_time, compare_models,
                plot_predictions_vs_ground_truth, plot_metrics,
                plot_confusion_matrix, plot_roc_curve, plot_comparison
"""

from .preprocessing import (
    StreamingStandardScaler,
    StreamingMinMaxScaler,
    StreamingImputer,
    StreamingOneHotEncoder,
)
from .tree import StreamingDecisionTree, DecisionTreeClassifier
from .ensemble import (
    StreamingBaggingClassifier,
    StreamingRandomForest,
    EnsembleClassifier,
)
from .stream import StreamTrainer
from .stats import ChunkStats, StreamingStats
from .metrics import StreamingMetrics
from .pipeline import StreamingPipeline
from . import visualise

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Preprocessing
    "StreamingStandardScaler",
    "StreamingMinMaxScaler",
    "StreamingImputer",
    "StreamingOneHotEncoder",
    # Models
    "StreamingDecisionTree",
    "DecisionTreeClassifier",
    "StreamingBaggingClassifier",
    "StreamingRandomForest",
    "EnsembleClassifier",
    # Trainer
    "StreamTrainer",
    # Stats
    "ChunkStats",
    "StreamingStats",
    # Metrics
    "StreamingMetrics",
    # Pipeline
    "StreamingPipeline",
    # Visualisation
    "visualise",
]
