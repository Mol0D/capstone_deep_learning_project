import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Ensure API credentials are available
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError(
        "Binance API credentials not found. Please set BINANCE_API_KEY and "
        "BINANCE_API_SECRET in your .env file"
    ) 