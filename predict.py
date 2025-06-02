from src.models.testing import AdvancedTesting
from src.data.data_processor import DataProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    try:
        # Initialize testing module
        tester = AdvancedTesting()
        
        # Get predictions for next hour
        predictions = tester.get_next_prediction(timeframe='1h')
        
        # Print summary
        print("\n=== BITCOIN PRICE PREDICTIONS ===")
        print(f"Last Close: ${predictions['average']['last_close']:.2f}")
        print(f"Time: {predictions['average']['prediction_time']}")
        print("\nPredictions by Feature Set:")
        
        for feature_set, models in predictions.items():
            if feature_set != 'average':
                print(f"\n{feature_set.upper()} Features:")
                print(f"Linear: ${models['linear']['price']:.2f} ({models['linear']['change']:+.2f}%)")
                print(f"XGBoost: ${models['xgb']['price']:.2f} ({models['xgb']['change']:+.2f}%)")
        
        print("\nAVERAGE PREDICTIONS:")
        print(f"Linear: ${predictions['average']['linear']['price']:.2f} ({predictions['average']['linear']['change']:+.2f}%)")
        print(f"XGBoost: ${predictions['average']['xgb']['price']:.2f} ({predictions['average']['xgb']['change']:+.2f}%)")
        
    except Exception as e:
        logging.error(f"Error in prediction script: {str(e)}")
        raise

if __name__ == "__main__":
    main() 