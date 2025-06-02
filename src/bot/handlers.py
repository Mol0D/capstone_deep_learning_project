import logging
from telegram import Update
from telegram.ext import ContextTypes
from ..models.predictor import BitcoinPredictor
from ..models.testing import AdvancedTesting
from ..data.data_processor import DataProcessor
from ..models.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class BotHandlers:
    def __init__(self):
        self.predictor = BitcoinPredictor()
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.advanced_testing = AdvancedTesting(self.data_processor, self.evaluator)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_message = (
            "👋 Welcome to the Bitcoin Price Prediction Bot!\n\n"
            "Available commands:\n"
            "/predict_1h - Get 1-hour price prediction\n"
            "/predict_1d - Get 1-day price prediction\n"
            "/test - Run model performance test\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("🔄 Running comprehensive model performance test...")
            
            # Run walk-forward testing
            results = self.advanced_testing.walk_forward_test(
                timeframe='1h',
                window_months=3,
                test_months=1
            )
            
            # Analyze results
            summary = self.advanced_testing.analyze_results(results)
            
            # Prepare the message
            message_parts = []
            message_parts.append("📊 Model Performance Test Results\n")
            
            for feature_set in ['price', 'technical', 'volume', 'all']:
                if feature_set not in summary:
                    continue
                    
                message_parts.append(f"\n🔍 Feature Set: {feature_set.upper()}\n")
                
                for model in ['linear', 'xgb']:
                    if model not in summary[feature_set]:
                        continue
                        
                    message_parts.append(f"\n{model.upper()} Model:\n")
                    model_metrics = summary[feature_set][model]
                    
                    for metric, stats in model_metrics.items():
                        # Skip if no data for this metric
                        if stats['count'] == 0:
                            continue
                            
                        if metric == 'Mean Absolute Error' or metric == 'Root Mean Squared Error':
                            message_parts.append(
                                f"- {metric} (n={stats['count']}):\n"
                                f"  Mean: ${stats['mean']:,.2f}\n"
                                f"  Std: ${stats['std']:,.2f}\n"
                                f"  Min: ${stats['min']:,.2f}\n"
                                f"  Max: ${stats['max']:,.2f}\n"
                            )
                        elif metric == 'Mean Absolute Percentage Error' or metric == 'Directional Accuracy':
                            message_parts.append(
                                f"- {metric} (n={stats['count']}):\n"
                                f"  Mean: {stats['mean']:.2f}%\n"
                                f"  Std: {stats['std']:.2f}%\n"
                                f"  Min: {stats['min']:.2f}%\n"
                                f"  Max: {stats['max']:.2f}%\n"
                            )
                        else:
                            message_parts.append(
                                f"- {metric} (n={stats['count']}):\n"
                                f"  Mean: {stats['mean']:.4f}\n"
                                f"  Std: {stats['std']:.4f}\n"
                                f"  Min: {stats['min']:.4f}\n"
                                f"  Max: {stats['max']:.4f}\n"
                            )
            
            if not message_parts or len(message_parts) <= 1:
                await update.message.reply_text(
                    "⚠️ No valid test results were generated. This might be due to:\n"
                    "- Insufficient historical data\n"
                    "- Data quality issues (missing values)\n"
                    "- Too few samples in training/test periods\n\n"
                    "Please try again later when more data is available."
                )
                return
            
            # Split message into chunks if needed (Telegram has a message length limit)
            message = "".join(message_parts)
            if len(message) > 4000:
                chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(message)
            
        except ValueError as ve:
            error_message = f"⚠️ {str(ve)}"
            logger.error(f"Validation error in test command: {str(ve)}")
            await update.message.reply_text(error_message)
        except Exception as e:
            error_message = (
                "❌ Sorry, there was an error running the performance test. "
                "This might be due to insufficient data or temporary API issues. "
                "Please try again in a few minutes."
            )
            logger.error(f"Error in test command: {str(e)}")
            await update.message.reply_text(error_message)

    async def predict_1h(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("🔄 Fetching data and making prediction...")
            
            # Get predictions using advanced testing
            predictions = self.advanced_testing.get_next_prediction(timeframe='1h')
            
            # Get the average predictions and current price
            current_price = predictions['average']['last_close']
            prediction_time = predictions['average']['prediction_time']
            
            # Prepare the message
            message_parts = []
            message_parts.append(f"🤖 Bitcoin Price Prediction (1 hour)\n")
            message_parts.append(f"Current Time: {prediction_time}\n")
            message_parts.append(f"Current Price: ${current_price:,.2f}\n\n")
            
            # Add predictions for each feature set
            for feature_set, models in predictions.items():
                if feature_set != 'average':
                    message_parts.append(f"\n{feature_set.upper()} Features:")
                    message_parts.append(
                        f"\nLinear Model: ${models['linear']['price']:,.2f} ({models['linear']['change']:+.2f}%)"
                    )
                    message_parts.append(
                        f"\nXGBoost Model: ${models['xgb']['price']:,.2f} ({models['xgb']['change']:+.2f}%)"
                    )
            
            # Add average predictions
            message_parts.append(f"\n\n📊 AVERAGE PREDICTIONS:")
            message_parts.append(
                f"\nLinear Model: ${predictions['average']['linear']['price']:,.2f} "
                f"({predictions['average']['linear']['change']:+.2f}%)"
            )
            message_parts.append(
                f"\nXGBoost Model: ${predictions['average']['xgb']['price']:,.2f} "
                f"({predictions['average']['xgb']['change']:+.2f}%)"
            )
            
            # Send the message
            await update.message.reply_text("".join(message_parts))
            
        except Exception as e:
            error_message = (
                "❌ Sorry, there was an error making the prediction. "
                "This might be due to temporary data availability issues. "
                "Please try again in a few minutes."
            )
            logger.error(f"Error in 1h prediction: {str(e)}")
            await update.message.reply_text(error_message)

    async def predict_1d(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("🔄 Fetching data and making prediction...")
            current_price, linear_prediction, xgb_prediction = self.predictor.predict(timeframe='1d')
            change_pct_linear = ((linear_prediction - current_price) / current_price) * 100
            change_pct_xgb = ((xgb_prediction - current_price) / current_price) * 100
            
            message = (
                f"🤖 Bitcoin Price Prediction (1 day)\n\n"
                f"Current Price: ${current_price:,.2f}\n"
                f"Linear Regression Prediction: ${linear_prediction:,.2f}\n"
                f"XGBoost Prediction: ${xgb_prediction:,.2f}\n"
                f"Expected Change (Linear Regression): {change_pct_linear:+.2f}%\n"
                f"Expected Change (XGBoost): {change_pct_xgb:+.2f}%"
            )
            await update.message.reply_text(message)
        except Exception as e:
            error_message = (
                "Sorry, there was an error making the prediction. "
                "This might be due to temporary data availability issues. "
                "Please try again in a few minutes."
            )
            logger.error(f"Error in 1d prediction: {str(e)}")
            await update.message.reply_text(error_message) 