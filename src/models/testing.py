import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from typing import Dict, List, Tuple
import logging
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

class AdvancedTesting:
    def __init__(self, data_processor=None, evaluator=None):
        """
        Initialize testing module with optional data processor and evaluator
        If data processor is not provided, create one with API credentials
        """
        if data_processor is None:
            from src.data.data_processor import DataProcessor
            data_processor = DataProcessor(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET
            )
        self.data_processor = data_processor
        self.evaluator = evaluator
        self.scaler = StandardScaler()
        
    def prepare_feature_sets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create different feature sets for testing
        Ensures no target variable leakage
        """
        feature_sets = {}
        
        # 1. Price-only features (excluding current Close price)
        price_features = pd.DataFrame()
        price_features['Open'] = df['Open']
        price_features['High'] = df['High']
        price_features['Low'] = df['Low']
        price_features['Range'] = df['Range']
        price_features['High_Low_Ratio'] = df['High_Low_Ratio']
        # Add lagged features (excluding current Returns which depends on Close)
        for i in range(1, 12):  # Changed from 25 to 12 to match data_processor.py
            price_features[f'Close_lag_{i}'] = df[f'Close_lag_{i}']
            price_features[f'Returns_lag_{i}'] = df[f'Returns_lag_{i}']
        
        # 2. Technical indicators
        tech_features = pd.DataFrame()
        # Moving averages and trends
        for window in [5, 8, 13, 21]:  # Removed 34 to match data_processor.py
            tech_features[f'SMA_{window}'] = df[f'SMA_{window}']
            tech_features[f'EMA_{window}'] = df[f'EMA_{window}']
            tech_features[f'Price_to_SMA_{window}'] = df[f'Price_to_SMA_{window}']
        
        # Add other technical indicators
        tech_features['MACD'] = df['MACD']
        tech_features['MACD_Signal'] = df['MACD_Signal']
        tech_features['MACD_Hist'] = df['MACD_Hist']
        tech_features['RSI'] = df['RSI']
        tech_features['Stoch_K'] = df['Stoch_K']
        tech_features['Stoch_D'] = df['Stoch_D']
        tech_features['BB_High'] = df['BB_High']
        tech_features['BB_Low'] = df['BB_Low']
        tech_features['BB_Mid'] = df['BB_Mid']
        tech_features['BB_Width'] = df['BB_Width']
        
        # 3. Volume indicators
        volume_features = pd.DataFrame()
        volume_features['Volume'] = df['Volume']
        volume_features['OBV'] = df['OBV']
        for window in [5, 8, 13, 21]:  # Removed 34 to match data_processor.py
            volume_features[f'Volume_SMA_{window}'] = df[f'Volume_SMA_{window}']
        
        # Create combined feature set (excluding target variables)
        combined_features = pd.DataFrame()
        # Add price features
        for col in price_features.columns:
            combined_features[f'price_{col}'] = price_features[col]
        # Add technical features
        for col in tech_features.columns:
            combined_features[f'tech_{col}'] = tech_features[col]
        # Add volume features
        for col in volume_features.columns:
            combined_features[f'vol_{col}'] = volume_features[col]
        
        # Store all feature sets
        feature_sets['price'] = price_features
        feature_sets['technical'] = tech_features
        feature_sets['volume'] = volume_features
        feature_sets['all'] = combined_features
        
        return feature_sets
        
    def walk_forward_test(self, timeframe: str = '1h', window_months: int = 3, 
                         test_months: int = 1) -> Dict:
        """
        Perform walk-forward testing with expanding window
        Default timeframe changed to daily for longer historical coverage
        """
        try:
            # Get full dataset with increased limit
            df = self.data_processor.prepare_data(timeframe=timeframe, for_prediction=False)
            
            # Convert index to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Calculate available months
            date_range = df.index.max() - df.index.min()
            available_months = date_range.days / 30.44  # Average days in a month
            
            # Adjust window sizes based on available data
            if available_months < (window_months + test_months):
                # If we have less than requested months, adjust window sizes
                total_months = max(1, int(available_months))
                window_months = max(1, int(total_months * 0.8))  # Use 80% for training
                test_months = max(1, total_months - window_months)  # Rest for testing
                logger.info(f"Adjusted window sizes due to limited data: training={window_months}m, test={test_months}m")
            
            # We need at least 2 months total (1 for training, 1 for testing)
            min_test_periods = 1
            min_months_needed = window_months + test_months
            
            if available_months < min_months_needed:
                raise ValueError(
                    f"Insufficient data: need at least {min_months_needed} months, "
                    f"but only have {available_months:.1f} months"
                )
            
            logger.info(f"Available months of data: {available_months:.1f}")
            logger.info(f"Data range: from {df.index.min()} to {df.index.max()}")
            logger.info(f"Data frequency: {timeframe}")
            logger.info(f"Training window: {window_months} months")
            logger.info(f"Test window: {test_months} months")
            
            # Prepare feature sets
            feature_sets = self.prepare_feature_sets(df)
            
            # Initialize results storage
            results = {
                'price': {'linear': [], 'xgb': []},
                'technical': {'linear': [], 'xgb': []},
                'volume': {'linear': [], 'xgb': []},
                'all': {'linear': [], 'xgb': []}
            }
            
            # Get unique months in the dataset
            months = df.index.to_period('M').unique()
            
            # Calculate number of possible test periods
            num_test_periods = len(months) - window_months - test_months + 1
            logger.info(f"Number of possible test periods: {num_test_periods}")
            
            # Minimum required samples based on timeframe
            min_train_samples = 24 if timeframe == '1d' else 168  # 24 days or 1 week hourly
            min_test_samples = 7 if timeframe == '1d' else 24    # 1 week or 1 day hourly
            
            # Perform walk-forward testing
            valid_periods = 0
            
            for i in range(len(months) - window_months - test_months + 1):
                train_start = months[i]
                train_end = months[i + window_months - 1]
                test_start = months[i + window_months]
                test_end = months[i + window_months + test_months - 1]
                
                logger.info(f"\nTesting period {i+1}/{num_test_periods}")
                logger.info(f"Train: {train_start} to {train_end}")
                logger.info(f"Test: {test_start} to {test_end}")
                
                period_valid = False
                
                # Test each feature set
                for feature_set_name, features in feature_sets.items():
                    # Get train/test data
                    train_mask = (features.index.to_period('M') >= train_start) & \
                               (features.index.to_period('M') <= train_end)
                    test_mask = (features.index.to_period('M') >= test_start) & \
                               (features.index.to_period('M') <= test_end)
                    
                    X_train = features[train_mask]
                    X_test = features[test_mask]
                    y_train = df['Close'][train_mask]
                    y_test = df['Close'][test_mask]
                    
                    # Validate data sizes
                    if len(X_train) < min_train_samples or len(X_test) < min_test_samples:
                        logger.warning(
                            f"Skipping {feature_set_name} features for period {i+1} due to insufficient data "
                            f"(train: {len(X_train)}/{min_train_samples}, test: {len(X_test)}/{min_test_samples})"
                        )
                        continue
                    
                    # Remove target-related columns from features to prevent data leakage
                    leakage_columns = [col for col in X_train.columns if 'Close_lag_' in col or 'Returns_lag_' in col]
                    X_train = X_train.drop(columns=leakage_columns)
                    X_test = X_test.drop(columns=leakage_columns)
                    
                    # Scale features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    try:
                        # Train and evaluate Linear Regression
                        linear_model = LinearRegression()
                        linear_model.fit(X_train_scaled, y_train)
                        linear_pred = linear_model.predict(X_test_scaled)
                        linear_metrics = self.evaluator.evaluate_model(X_test_scaled, y_test, linear_pred)
                        
                        # Train and evaluate XGBoost
                        xgb_model = XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            min_child_weight=1,
                            gamma=0,
                            random_state=42
                        )
                        xgb_model.fit(X_train_scaled, y_train)
                        xgb_pred = xgb_model.predict(X_test_scaled)
                        xgb_metrics = self.evaluator.evaluate_model(X_test_scaled, y_test, xgb_pred)
                        
                        # Store results
                        results[feature_set_name]['linear'].append({
                            'period': f"{test_start} - {test_end}",
                            'metrics': linear_metrics,
                            'train_size': len(X_train),
                            'test_size': len(X_test)
                        })
                        results[feature_set_name]['xgb'].append({
                            'period': f"{test_start} - {test_end}",
                            'metrics': xgb_metrics,
                            'train_size': len(X_train),
                            'test_size': len(X_test)
                        })
                        
                        # Log progress
                        logger.info(f"\n{feature_set_name.upper()} Features - Period {i+1}:")
                        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
                        logger.info(f"Linear R² Score: {linear_metrics['R² Score']:.4f}")
                        logger.info(f"XGBoost R² Score: {xgb_metrics['R² Score']:.4f}")
                        
                        period_valid = True
                        
                    except Exception as e:
                        logger.error(
                            f"Error processing {feature_set_name} features for period {i+1}: {str(e)}"
                        )
                        continue
                
                if period_valid:
                    valid_periods += 1
            
            # Validate we have enough valid test periods
            if valid_periods < min_test_periods:
                raise ValueError(
                    f"Insufficient valid test periods: got {valid_periods}, "
                    f"need at least {min_test_periods}"
                )
            
            logger.info(f"\nCompleted walk-forward testing with {valid_periods} valid test periods")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward testing: {str(e)}")
            raise
            
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze and summarize walk-forward testing results
        """
        summary = {}
        
        for feature_set, models in results.items():
            summary[feature_set] = {}
            
            for model_name, periods in models.items():
                # Skip if no results for this combination
                if not periods:
                    continue
                    
                metrics = {
                    'R² Score': [],
                    'Mean Absolute Error': [],
                    'Root Mean Squared Error': [],
                    'Mean Absolute Percentage Error': [],
                    'Directional Accuracy': []
                }
                
                # Collect metrics across all periods
                for period in periods:
                    for metric_name, value in period['metrics'].items():
                        metrics[metric_name].append(value)
                
                # Calculate summary statistics only if we have data
                if any(len(values) > 0 for values in metrics.values()):
                    summary[feature_set][model_name] = {
                        metric: {
                            'mean': np.mean(values) if values else np.nan,
                            'std': np.std(values) if len(values) > 1 else np.nan,
                            'min': np.min(values) if values else np.nan,
                            'max': np.max(values) if values else np.nan,
                            'count': len(values)
                        }
                        for metric, values in metrics.items()
                    }
        
        return summary 

    def get_next_prediction(self, timeframe: str = '1h') -> Dict:
        """
        Get predictions for the next period using both Linear and XGBoost models
        Returns predictions and model metrics
        """
        try:
            # Get data for prediction
            df = self.data_processor.prepare_data(timeframe=timeframe, for_prediction=True)
            
            # Get feature sets
            feature_sets = self.prepare_feature_sets(df)
            
            # Initialize results
            predictions = {}
            
            # Get the last known close price for reference
            last_close = df['Close'].iloc[-1]
            current_time = df.index[-1]
            
            logger.info(f"Making predictions for period after {current_time}")
            logger.info(f"Last known close price: ${last_close:.2f}")
            
            # Make predictions for each feature set
            for feature_set_name, features in feature_sets.items():
                # Remove target-related columns to prevent data leakage
                leakage_columns = [col for col in features.columns if 'Close_lag_' in col or 'Returns_lag_' in col]
                X = features.drop(columns=leakage_columns)
                
                # Use only the last row for prediction
                X = X.iloc[-1:].copy()
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                try:
                    # Train Linear model on all data except last point
                    linear_model = LinearRegression()
                    train_X = self.scaler.fit_transform(features.iloc[:-1].drop(columns=leakage_columns))
                    train_y = df['Close'].iloc[:-1]
                    linear_model.fit(train_X, train_y)
                    linear_pred = linear_model.predict(X_scaled)[0]
                    
                    # Train XGBoost model
                    xgb_model = XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=1,
                        gamma=0,
                        random_state=42
                    )
                    xgb_model.fit(train_X, train_y)
                    xgb_pred = xgb_model.predict(X_scaled)[0]
                    
                    # Calculate predicted changes
                    linear_change = ((linear_pred - last_close) / last_close) * 100
                    xgb_change = ((xgb_pred - last_close) / last_close) * 100
                    
                    predictions[feature_set_name] = {
                        'linear': {
                            'price': linear_pred,
                            'change': linear_change
                        },
                        'xgb': {
                            'price': xgb_pred,
                            'change': xgb_change
                        }
                    }
                    
                    logger.info(f"\n{feature_set_name.upper()} Features:")
                    logger.info(f"Linear Model: ${linear_pred:.2f} ({linear_change:+.2f}%)")
                    logger.info(f"XGBoost Model: ${xgb_pred:.2f} ({xgb_change:+.2f}%)")
                    
                except Exception as e:
                    logger.error(f"Error making predictions with {feature_set_name} features: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("No valid predictions generated")
            
            # Calculate average predictions across all feature sets
            linear_preds = [pred['linear']['price'] for pred in predictions.values()]
            xgb_preds = [pred['xgb']['price'] for pred in predictions.values()]
            
            avg_linear = sum(linear_preds) / len(linear_preds)
            avg_xgb = sum(xgb_preds) / len(xgb_preds)
            
            avg_linear_change = ((avg_linear - last_close) / last_close) * 100
            avg_xgb_change = ((avg_xgb - last_close) / last_close) * 100
            
            logger.info(f"\nAVERAGE PREDICTIONS:")
            logger.info(f"Linear Model: ${avg_linear:.2f} ({avg_linear_change:+.2f}%)")
            logger.info(f"XGBoost Model: ${avg_xgb:.2f} ({avg_xgb_change:+.2f}%)")
            
            predictions['average'] = {
                'linear': {
                    'price': avg_linear,
                    'change': avg_linear_change
                },
                'xgb': {
                    'price': avg_xgb,
                    'change': avg_xgb_change
                },
                'last_close': last_close,
                'prediction_time': current_time
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise 