import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    @staticmethod
    def evaluate_model(X_test_scaled, y_test, y_pred):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        actual_direction = np.sign(y_test.values[1:] - y_test.values[:-1])
        predicted_direction = np.sign(y_pred[1:] - y_pred[:-1])
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'R² Score': r2,
            'Mean Absolute Error': mae,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Percentage Error': mape,
            'Directional Accuracy': directional_accuracy
        } 