from typing import Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    """
    Handles training and evaluation of the logistic regression model.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state)
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        # TODO: Add any additional training parameters or preprocessing steps
        self.model.fit(X_train, y_train)
        
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Tuple containing:
                - Accuracy score
                - Dictionary with detailed metrics
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # TODO: Add additional evaluation metrics as needed
        return accuracy, report
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Predicted classes
        """
        return self.model.predict(X)