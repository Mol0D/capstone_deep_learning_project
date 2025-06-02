# Bitcoin Price Prediction Bot

A Telegram bot that predicts Bitcoin price movements using machine learning models and technical analysis.

## Features

- Real-time Bitcoin price predictions (1-hour and 1-day forecasts)
- Multiple prediction models:
  - Linear Regression
  - XGBoost
- Different feature sets:
  - Price-based features
  - Technical indicators
  - Volume indicators
  - Combined features
- Comprehensive model testing and evaluation
- Telegram bot interface for easy access

## Technical Details

- Uses Binance API for real-time and historical data
- Technical indicators include:
  - Moving Averages (SMA, EMA)
  - MACD
  - RSI
  - Bollinger Bands
  - Stochastic Oscillator
  - On-Balance Volume
- Walk-forward testing with expanding window
- Data leakage prevention in feature engineering

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd dl_capstone
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

## Usage

1. Start the Telegram bot:
```bash
python bot.py
```

2. Available commands in Telegram:
- `/start` - Show welcome message and available commands
- `/predict_1h` - Get 1-hour price prediction
- `/predict_1d` - Get 1-day price prediction
- `/test` - Run model performance test
- `/help` - Show help message

## Project Structure

```
dl_capstone/
├── src/
│   ├── bot/
│   │   ├── handlers.py    # Telegram bot command handlers
│   │   └── __init__.py
│   ├── data/
│   │   ├── data_processor.py    # Data fetching and preprocessing
│   │   └── __init__.py
│   ├── models/
│   │   ├── predictor.py    # Price prediction models
│   │   ├── testing.py      # Model testing functionality
│   │   ├── evaluator.py    # Model evaluation metrics
│   │   └── __init__.py
├── bot.py                  # Main bot entry point
├── config.py              # Configuration and environment variables
├── requirements.txt       # Project dependencies
└── README.md
```

## Dependencies

- python-telegram-bot
- pandas
- numpy
- scikit-learn
- xgboost
- python-binance
- ta (Technical Analysis library)
- python-dotenv

## License

MIT License 