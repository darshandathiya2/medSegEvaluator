PerformanceMetrics
=================

`PerformanceMetrics` is a comprehensive evaluation class for medical image segmentation. 
It provides a variety of metrics for quantitative assessment of segmentation performance.

Class
-----

.. autoclass:: medsegevaluator.PerformanceMetrics
    :members:
    :undoc-members:
    :show-inheritance:

Description
-----------

The `PerformanceMetrics` class supports multiple types of metrics:

1. **Region-based metrics**
   - Dice coefficient
   - Jaccard index (IoU)
   - Precision, Recall
   - F1 score

2. **Boundary-based metrics**
   - Hausdorff distance
   - Average surface distance
   - Boundary F1 score

3. **Topology-aware metrics**
   - Connectivity metrics
   - Object-wise metrics for instance segmentation

Usage Example
-------------

```python
import numpy as np
from medsegevaluator import PerformanceMetrics

# Example ground truth and prediction masks
y_true = np.random.randint(0, 2, (256, 256))
y_pred = np.random.randint(0, 2, (256, 256))

# Initialize metrics evaluator
evaluator = PerformanceMetrics()

# Compute Dice coefficient
dice_score = evaluator.dice(y_true, y_pred)
print("Dice Score:", dice_score)
