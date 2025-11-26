Performance Metrics
===================

MedSegEvaluator provides a wide range of metrics for evaluating the quality of 
medical image segmentation. These metrics are grouped into four categories:

- **Region-Level Overlap**
- **Boundary Accuracy**
- **Surface Distances**
- **Volumetric Agreement**

This page describes each metric, its mathematical formulation, and 
how to compute it using MedSegEvaluator.

.. contents::
   :local:
   :depth: 2


Overview
--------

The available segmentation metrics include:

- Dice Similarity Coefficient
- Intersection over Union (IoU)
- Precision & Recall
- F1 Score
- Hausdorff Distance (HD95)
- Average Symmetric Surface Distance (ASSD)
- Volume Similarity (VS)
- Boundary F1 Score
- Surface Dice

Dice Similarity Coefficient
---------------------------

The Dice coefficient measures the overlap between ground truth and 
predicted segmentation.

Formula:

.. math::

    Dice = \frac{2 \cdot |A \cap B|}{|A| + |B|}

Where: 
``A`` is the set of voxels/pixels in the ground truth mask, ``B`` is the set of voxels/pixels in the predicted mask.

Usage::

    from performance_metrics import dice_score
    score = dice_score(gt, pred)
