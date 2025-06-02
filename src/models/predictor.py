from datetime import datetime, timedelta
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from ..data.data_processor import DataProcessor
from .evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class BitcoinPredictor:
    def __init__(self):
        self.model_1h_linear = None
        self.model_1h_xgb = None
        self.model_1d_linear = None
        self.model_1d_xgb = None
        self.scaler = StandardScaler()
        self.last_train_time = None
        self.retrain_interval = timedelta(hours=1)
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()

    def train_models(self):
        try:
            # Prepare data for 1-hour predictions
            df_1h = self.data_processor.prepare_data(timeframe='1h')
            
            if len(df_1h) < 50:
                raise ValueError("Insufficient historical data for training")
            
            # Prepare features and target
            X = df_1h.drop(['Close', 'Returns'], axis=1)
            y = df_1h['Close']
            
            # Split the data chronologically
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train Linear Regression model for 1-hour predictions
            self.model_1h_linear = LinearRegression()
            self.model_1h_linear.fit(X_train_scaled, y_train)
            
            # Define XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'random_state': 42
            }
            
            # Train XGBoost model for 1-hour predictions
            self.model_1h_xgb = XGBRegressor(**params)
            self.model_1h_xgb.fit(X_train_scaled, y_train)
            
            # Train 1-day models
            df_1d = self.data_processor.prepare_data(timeframe='1d')
            
            if len(df_1d) < 30:
                raise ValueError("Insufficient historical daily data for training")
                
            X_daily = df_1d.drop(['Close', 'Returns'], axis=1)
            y_daily = df_1d['Close']
            
            split_idx_daily = int(len(X_daily) * 0.8)
            X_train_daily = X_daily[:split_idx_daily]
            y_train_daily = y_daily[:split_idx_daily]
            
            X_train_daily_scaled = self.scaler.fit_transform(X_train_daily)
            
            # Train Linear Regression model for 1-day predictions
            self.model_1d_linear = LinearRegression()
            self.model_1d_linear.fit(X_train_daily_scaled, y_train_daily)
            
            # Train XGBoost model for 1-day predictions
            self.model_1d_xgb = XGBRegressor(**params)
            self.model_1d_xgb.fit(X_train_daily_scaled, y_train_daily)
            
            self.last_train_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in training models: {str(e)}")
            raise

    def test_model(self):
        """Test and compare both models performance on hourly data"""
        try:
            # Prepare data
            df = self.data_processor.prepare_data(timeframe='1h', for_prediction=False)
            
            if len(df) < 100:
                raise ValueError("Insufficient data for testing")
            
            # Prepare features and target
            X = df.drop(['Close', 'Returns'], axis=1)
            y = df['Close'].shift(-1)  # Next period's price
            
            # Remove the last row as it won't have a target value
            X = X[:-1]
            y = y[:-1].dropna()
            
            # Split the data chronologically (last 20% for testing)
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and test Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X_train_scaled, y_train)
            linear_pred = linear_model.predict(X_test_scaled)
            linear_metrics = self.evaluator.evaluate_model(X_test_scaled, y_test, linear_pred)
            
            # Define XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'random_state': 42
            }
            
            # Train and test XGBoost model
            xgb_model = XGBRegressor(**params)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_metrics = self.evaluator.evaluate_model(X_test_scaled, y_test, xgb_pred)
            
            # Add feature importance for XGBoost
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log top 10 most important features
            logger.info("\nTop 10 most important features (XGBoost):")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
            return {
                'linear': linear_metrics,
                'xgboost': xgb_metrics
            }, len(y_train), len(y_test)
            
        except Exception as e:
            logger.error(f"Error in model testing: {str(e)}")
            raise

    def predict(self, timeframe='1h'):
        try:
            # Check if models need retraining
            if (self.last_train_time is None or 
                datetime.now() - self.last_train_time > self.retrain_interval):
                self.train_models()
            
            # Get latest data for prediction
            df = self.data_processor.prepare_data(timeframe=timeframe, for_prediction=True)
            
            if timeframe == '1h':
                linear_model = self.model_1h_linear
                xgb_model = self.model_1h_xgb
            else:
                linear_model = self.model_1d_linear
                xgb_model = self.model_1d_xgb
                
            if linear_model is None or xgb_model is None:
                raise ValueError("Models not trained yet")
                
            # Prepare features for prediction
            X = df.drop(['Close', 'Returns'], axis=1).iloc[-1:]
            X_scaled = self.scaler.transform(X)
            
            # Make predictions with both models
            linear_prediction = linear_model.predict(X_scaled)[0]
            xgb_prediction = xgb_model.predict(X_scaled)[0]
            current_price = df['Close'].iloc[-1]
            
            return current_price, linear_prediction, xgb_prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise 