from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from .logger import setup_logger

logger = setup_logger(__name__)

class ModelEvaluationReport:
    """Generate comprehensive model evaluation reports."""
    
    def __init__(
        self,
        reports_dir: Optional[Path] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            reports_dir: Directory to store reports
            experiment_id: Optional experiment ID to associate with reports
        """
        self.reports_dir = reports_dir or Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.experiment_id = experiment_id
        
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_importance: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate classification model evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            feature_names: Optional list of feature names
            feature_importance: Optional feature importance scores
            class_names: Optional class names
            
        Returns:
            str: Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.reports_dir / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self._plot_confusion_matrix(
            y_true, y_pred,
            class_names,
            report_dir / "confusion_matrix.png"
        )
        
        self._plot_roc_curve(
            y_true, y_prob,
            report_dir / "roc_curve.png"
        )
        
        self._plot_precision_recall_curve(
            y_true, y_prob,
            report_dir / "precision_recall_curve.png"
        )
        
        if feature_importance is not None and feature_names is not None:
            self._plot_feature_importance(
                feature_importance,
                feature_names,
                report_dir / "feature_importance.png"
            )
        
        # Generate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        # Create report metadata
        report = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "metrics": metrics,
            "plots": {
                "confusion_matrix": "confusion_matrix.png",
                "roc_curve": "roc_curve.png",
                "precision_recall_curve": "precision_recall_curve.png"
            }
        }
        
        if feature_importance is not None:
            report["plots"]["feature_importance"] = "feature_importance.png"
        
        # Save report metadata
        with open(report_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report, report_dir)
        
        logger.info(f"Generated report at {report_dir}")
        return str(html_report)
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]],
        output_path: Path
    ) -> None:
        """Generate confusion matrix plot."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: Path
    ) -> None:
        """Generate ROC curve plot."""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: Path
    ) -> None:
        """Generate precision-recall curve plot."""
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        output_path: Path
    ) -> None:
        """Generate feature importance plot."""
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        importance_df = importance_df.sort_values(
            "importance",
            ascending=True
        )
        
        plt.barh(
            importance_df["feature"],
            importance_df["importance"]
        )
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "avg_precision": average_precision_score(y_true, y_prob)
        }
        
        return metrics
    
    def _generate_html_report(
        self,
        report: Dict[str, Any],
        report_dir: Path
    ) -> Path:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px 0; }
                .plot { margin: 20px 0; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <p>Generated: {timestamp}</p>
            
            <h2>Metrics</h2>
            {metrics_html}
            
            <h2>Plots</h2>
            {plots_html}
        </body>
        </html>
        """
        
        # Generate metrics HTML
        metrics_html = ""
        for name, value in report["metrics"].items():
            metrics_html += f"""
            <div class="metric">
                <strong>{name}:</strong> {value:.4f}
            </div>
            """
        
        # Generate plots HTML
        plots_html = ""
        for name, path in report["plots"].items():
            plots_html += f"""
            <div class="plot">
                <h3>{name}</h3>
                <img src="{path}" alt="{name}">
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            timestamp=report["timestamp"],
            metrics_html=metrics_html,
            plots_html=plots_html
        )
        
        # Save HTML report
        html_path = report_dir / "report.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        
        return html_path