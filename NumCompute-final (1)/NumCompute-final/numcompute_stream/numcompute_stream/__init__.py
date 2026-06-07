"""NumCompute Stream — streaming decision tree extensions for NumCompute.

This package extends the base NumCompute toolkit with:

- Streaming-compatible preprocessing (partial_fit)
- Hoeffding-bound decision trees (StreamingDecisionTree)
- Ensemble methods (StreamingBaggingClassifier, StreamingRandomForest)
- Built-in visualisation (visualise.py)

Usage
-----
    from numcompute_stream.tree import StreamingDecisionTree
    from numcompute_stream.ensemble import StreamingRandomForest
    from numcompute_stream.preprocessing import StreamingStandardScaler
    from numcompute_stream import visualise
"""

from .preprocessing import StreamingStandardScaler, StreamingMinMaxScaler
from .tree import StreamingDecisionTree
from .ensemble import StreamingBaggingClassifier, StreamingRandomForest
from . import visualise

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Preprocessing
    "StreamingStandardScaler",
    "StreamingMinMaxScaler",
    # Models
    "StreamingDecisionTree",
    "StreamingBaggingClassifier",
    "StreamingRandomForest",
    # Visualisation
    "visualise",
]
