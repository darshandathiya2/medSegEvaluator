"""
MedicalSegmentationMetrics
==========================

A comprehensive and Sphinx-friendly metrics module for medical image segmentation.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List

__all__ = ["MedicalSegmentationMetrics"]


class MedicalSegmentationMetrics:
    """
    A comprehensive and professional medical image segmentation evaluation class.
    
    This class provides a rich set of region-based, boundary-based, and
    distance-based metrics widely used in biomedical image segmentation tasks.

    **Region-based metrics:**
        - Dice score  
        - Jaccard Index (IoU)  
        - Precision  
        - Recall  
        - Specificity  
        - Accuracy  

    **Distance-based metrics:**
        - Hausdorff Distance  
        - Average Surface Distance (ASD)  
        - 95th Percentile Hausdorff Distance (HD95)

    **Boundary-based metrics:**
        - Surface Dice  
        - Boundary F1 Score (BF-Score)
    """

    # ----------------------------------------------------------------------
    # Core Segmentation Metrics
    # ----------------------------------------------------------------------

    @staticmethod
    def dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Dice score."""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2.0 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-8)

    @staticmethod
    def jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Jaccard Index (IoU)."""
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return intersection / (union + 1e-8)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision = TP / (TP + FP)."""
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        return TP / (TP + FP + 1e-8)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall = TP / (TP + FN)."""
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 1) & (y_true == 0))
        return TP / (TP + FN + 1e-8)

    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Specificity = TN / (TN + FP)."""
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        return TN / (TN + FP + 1e-8)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy = (TP + TN) / Total."""
        return np.mean(y_true == y_pred)

    # ----------------------------------------------------------------------
    # Distance-based Metrics
    # ----------------------------------------------------------------------

    @staticmethod
    def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Hausdorff Distance.
        """
        y_true_pts = np.argwhere(y_true == 1)
        y_pred_pts = np.argwhere(y_pred == 1)
        if len(y_true_pts) == 0 or len(y_pred_pts) == 0:
            return np.nan
        return max(
            directed_hausdorff(y_true_pts, y_pred_pts)[0],
            directed_hausdorff(y_pred_pts, y_true_pts)[0],
        )

    @staticmethod
    def hd95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute 95th percentile Hausdorff Distance.
        """
        y_true_pts = np.argwhere(y_true == 1)
        y_pred_pts = np.argwhere(y_pred == 1)
        if len(y_true_pts) == 0 or len(y_pred_pts) == 0:
            return np.nan
        d1 = np.min(np.linalg.norm(y_true_pts[:, None] - y_pred_pts[None, :], axis=-1), axis=1)
        d2 = np.min(np.linalg.norm(y_pred_pts[:, None] - y_true_pts[None, :], axis=-1), axis=1)
        return np.percentile(np.concatenate([d1, d2]), 95)

    @staticmethod
    def average_surface_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Average Surface Distance (ASD)."""
        y_true_pts = np.argwhere(y_true == 1)
        y_pred_pts = np.argwhere(y_pred == 1)
        if len(y_true_pts) == 0 or len(y_pred_pts) == 0:
            return np.nan
        d1 = np.min(np.linalg.norm(y_true_pts[:, None] - y_pred_pts[None, :], axis=-1), axis=1)
        d2 = np.min(np.linalg.norm(y_pred_pts[:, None] - y_true_pts[None, :], axis=-1), axis=1)
        return (d1.mean() + d2.mean()) / 2.0

    # ----------------------------------------------------------------------
    # Boundary-based Metrics
    # ----------------------------------------------------------------------

    @staticmethod
    def surface_dice(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 1.0) -> float:
        """Compute Surface Dice score."""
        dt_true = distance_transform_edt(1 - y_true)
        dt_pred = distance_transform_edt(1 - y_pred)
        s1 = np.sum((y_pred == 1) & (dt_true <= tolerance))
        s2 = np.sum((y_true == 1) & (dt_pred <= tolerance))
        denom = np.sum(y_true == 1) + np.sum(y_pred == 1)
        return 2 * (s1 + s2) / (denom + 1e-8)

    @staticmethod
    def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 2) -> float:
        """Compute Boundary F1 Score (BF-Score)."""
        se = np.ones((3, 3))
        y_true_b = y_true - binary_erosion(y_true, structure=se)
        y_pred_b = y_pred - binary_erosion(y_pred, structure=se)
        dt_true_b = distance_transform_edt(1 - y_true_b)
        dt_pred_b = distance_transform_edt(1 - y_pred_b)
        precision = np.mean(dt_pred_b[y_pred_b == 1] <= tolerance)
        recall = np.mean(dt_true_b[y_true_b == 1] <= tolerance)
        return 2 * precision * recall / (precision + recall + 1e-8)

    # ----------------------------------------------------------------------
    # Summary + Batch Evaluation (STATIC for Sphinx)
    # ----------------------------------------------------------------------

    @staticmethod
    def print_summary(y_true: np.ndarray, y_pred: np.ndarray):
        """Print all segmentation metrics in a readable summary."""
        metrics = {
            "Dice": MedicalSegmentationMetrics.dice(y_true, y_pred),
            "Jaccard": MedicalSegmentationMetrics.jaccard(y_true, y_pred),
            "Precision": MedicalSegmentationMetrics.precision(y_true, y_pred),
            "Recall": MedicalSegmentationMetrics.recall(y_true, y_pred),
            "Specificity": MedicalSegmentationMetrics.specificity(y_true, y_pred),
            "Accuracy": MedicalSegmentationMetrics.accuracy(y_true, y_pred),
            "Hausdorff Distance": MedicalSegmentationMetrics.hausdorff_distance(y_true, y_pred),
            "HD95": MedicalSegmentationMetrics.hd95(y_true, y_pred),
            "ASD": MedicalSegmentationMetrics.average_surface_distance(y_true, y_pred),
            "Surface Dice": MedicalSegmentationMetrics.surface_dice(y_true, y_pred),
            "Boundary F1": MedicalSegmentationMetrics.boundary_f1(y_true, y_pred),
        }

        print("\n===== Medical Segmentation Metrics Summary =====")
        for k, v in metrics.items():
            print(f"{k:25s}: {v:.4f}")
        print("================================================\n")

    @staticmethod
    def batch_evaluate(
        y_true_list: List[np.ndarray], 
        y_pred_list: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """
        Evaluate metrics for multiple images in a batch.

        Parameters
        ----------
        y_true_list : list of np.ndarray
            List of ground-truth masks.
        y_pred_list : list of np.ndarray
            List of predicted masks.

        Returns
        -------
        list of dict
            Metric dictionary for each image.
        """
        results = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            results.append({
                "Dice": MedicalSegmentationMetrics.dice(y_true, y_pred),
                "Jaccard": MedicalSegmentationMetrics.jaccard(y_true, y_pred),
                "Precision": MedicalSegmentationMetrics.precision(y_true, y_pred),
                "Recall": MedicalSegmentationMetrics.recall(y_true, y_pred),
                "Specificity": MedicalSegmentationMetrics.specificity(y_true, y_pred),
                "Accuracy": MedicalSegmentationMetrics.accuracy(y_true, y_pred),
                "Hausdorff Distance": MedicalSegmentationMetrics.hausdorff_distance(y_true, y_pred),
                "HD95": MedicalSegmentationMetrics.hd95(y_true, y_pred),
                "ASD": MedicalSegmentationMetrics.average_surface_distance(y_true, y_pred),
                "Surface Dice": MedicalSegmentationMetrics.surface_dice(y_true, y_pred),
                "Boundary F1": MedicalSegmentationMetrics.boundary_f1(y_true, y_pred),
            })
        return results
