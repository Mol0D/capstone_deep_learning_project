import logging
import pandas as pd
import numpy as np
import ccxt
import time
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize DataProcessor with Binance API credentials
        """
        # Binance API configuration
        self.api_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        }
        
        try:
            self.exchange = ccxt.binance(self.api_config)
            self.exchange.load_markets()
            logger.info("Successfully initialized Binance exchange connection")
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {str(e)}")
            raise

    def get_bitcoin_data(self, timeframe='1d', limit=1000):
        """
        Fetch Bitcoin price data from Binance
        timeframe options: 1h, 1d
        Default changed to 1d to get longer historical coverage
        """
        try:
            symbol = 'BTC/USDT'
            
            if symbol not in self.exchange.markets:
                raise ValueError(f"{symbol} not available on Binance")

            # Initialize variables for pagination
            all_data = []
            chunk_size = 1000  # Maximum allowed by Binance
            remaining = limit
            
            # Get the current timestamp
            now = self.exchange.milliseconds()
            
            # For the first request, start from current time and go backwards
            end_time = now
            
            logger.info(f"Starting data collection from Binance")
            logger.info(f"Target amount: {limit} candles")
            logger.info(f"Timeframe: {timeframe}")

            while remaining > 0:
                try:
                    # Calculate the start time for this chunk
                    timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
                    start_time = end_time - (chunk_size * timeframe_ms)
                    
                    logger.info(f"Fetching chunk from {self.exchange.iso8601(start_time)} to {self.exchange.iso8601(end_time)}")
                    
                    # Fetch chunk of data
                    chunk = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=chunk_size,
                        since=start_time,
                        params={'endTime': end_time}
                    )
                    
                    if not chunk or len(chunk) == 0:
                        logger.info("No more data available")
                        break
                        
                    logger.info(f"Received chunk of {len(chunk)} candles")
                    
                    # Add the chunk to our results
                    all_data.extend(chunk)
                    
                    # Update the end time to be the start time of the current chunk
                    end_time = start_time - 1
                    
                    # Update remaining count
                    remaining -= len(chunk)
                    
                    # If we got less than chunk_size, there's no more historical data
                    if len(chunk) < chunk_size:
                        logger.info("Received less data than chunk size, assuming no more historical data available")
                        break
                        
                    # Add delay to respect rate limits
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    logger.info(f"Total data points collected so far: {len(all_data)}")
                    
                except Exception as e:
                    logger.error(f"Error during pagination: {str(e)}")
                    break

            if not all_data:
                raise Exception("Failed to fetch Bitcoin data from Binance")
                
            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Calculate the date range
            date_range = df.index.max() - df.index.min()
            months = date_range.days / 30.44  # Average days in a month
            
            logger.info(f"Successfully collected {len(df)} unique candles")
            logger.info(f"Data range: from {df.index.min()} to {df.index.max()}")
            logger.info(f"Total months of data: {months:.1f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data from Binance: {str(e)}")
            raise

    def prepare_data(self, timeframe='1d', for_prediction=False):
        """
        Prepare data for model training/testing with optimized feature set
        to minimize data loss while maintaining predictive power
        """
        # Download Bitcoin price data with adjusted limits
        if timeframe == '1h':
            df = self.get_bitcoin_data(timeframe='1h', limit=2500)
        else:
            df = self.get_bitcoin_data(timeframe='1d', limit=1000)
        
        if df.empty:
            raise ValueError("No data received from exchange")
            
        # Calculate available months of data
        date_range = df.index.max() - df.index.min()
        months = date_range.days / 30.44
        
        # Adjust minimum months requirement based on timeframe
        min_months = 3 if timeframe == '1h' else 15
        if months < min_months:
            raise ValueError(f"Insufficient data: only {months:.1f} months available, need at least {min_months} months")
            
        logger.info(f"Initial data points: {len(df)}")
        logger.info(f"Data timeframe: {timeframe}")
        logger.info(f"Available months of data: {months:.1f}")

        # Create features DataFrame with basic price data
        features = pd.DataFrame()
        features['Close'] = df['close']
        features['Open'] = df['open']
        features['High'] = df['high']
        features['Low'] = df['low']
        features['Volume'] = df['volume']

        # Essential price features (no shifting for prediction)
        if for_prediction and timeframe == '1h':
            features['Returns'] = features['Close'].pct_change()
            features['Range'] = features['High'] - features['Low']
            features['High_Low_Ratio'] = features['High'] / features['Low']
            
            # Use current price for technical indicators in prediction
            price_for_indicators = features['Close']
            
            # Add lagged prices for target generation (future values)
            for i in range(1, 12):  # Generate 11 future lags to match testing code
                features[f'Close_lag_{i}'] = features['Close'].shift(-i)
                features[f'Returns_lag_{i}'] = features['Returns'].shift(-i)
        else:
            # Create lagged features for training
            features['Close_Lag1'] = features['Close'].shift(1)
            features['Returns'] = features['Close'].pct_change()
            features['Range'] = features['High'] - features['Low']
            features['High_Low_Ratio'] = features['High'] / features['Low']
            
            # Add target variables for training (past values)
            for i in range(1, 12):  # Generate 11 past lags to match testing code
                features[f'Close_lag_{i}'] = features['Close'].shift(i)
                features[f'Returns_lag_{i}'] = features['Returns'].shift(i)
            
            price_for_indicators = features['Close_Lag1']

        logger.info(f"Data points after basic price features: {len(features.dropna())}")
        
        # Technical indicators with windows matching testing code
        sma_windows = [5, 8, 13, 21]  # Match testing code windows
        for window in sma_windows:
            features[f'SMA_{window}'] = SMAIndicator(price_for_indicators, window=window).sma_indicator()
            features[f'EMA_{window}'] = EMAIndicator(price_for_indicators, window=window).ema_indicator()
            # Add Price to SMA ratio as in testing code
            features[f'Price_to_SMA_{window}'] = price_for_indicators / features[f'SMA_{window}']
            # Add Volume SMA as in testing code
            features[f'Volume_SMA_{window}'] = SMAIndicator(features['Volume'], window=window).sma_indicator()
        
        # MACD parameters adjusted for consistency
        macd = MACD(
            price_for_indicators,
            window_slow=21,  # Match testing code
            window_fast=8,
            window_sign=5
        )
        features['MACD'] = macd.macd()
        features['MACD_Signal'] = macd.macd_signal()
        features['MACD_Hist'] = macd.macd_diff()  # Add histogram as in testing
        
        # RSI with standard window
        rsi = RSIIndicator(price_for_indicators, window=14)
        features['RSI'] = rsi.rsi()
        
        # Stochastic Oscillator as in testing code
        stoch = StochasticOscillator(
            features['High'],
            features['Low'],
            price_for_indicators,
            window=14,
            smooth_window=3
        )
        features['Stoch_K'] = stoch.stoch()
        features['Stoch_D'] = stoch.stoch_signal()
        
        # Bollinger Bands with standard window
        bb = BollingerBands(price_for_indicators, window=20)
        features['BB_High'] = bb.bollinger_hband()
        features['BB_Low'] = bb.bollinger_lband()
        features['BB_Mid'] = bb.bollinger_mavg()
        features['BB_Width'] = (features['BB_High'] - features['BB_Low']) / features['BB_Mid']
        
        # Volume indicators as in testing code
        obv = OnBalanceVolumeIndicator(price_for_indicators, features['Volume'])
        features['OBV'] = obv.on_balance_volume()
        
        logger.info(f"Data points after technical indicators: {len(features.dropna())}")

        # Remove temporary columns
        if not for_prediction or timeframe != '1h':
            features = features.drop(['Close_Lag1'], axis=1)

        # Data quality checks with detailed logging
        features = features.replace([np.inf, -np.inf], np.nan)
        initial_rows = len(features)
        
        if for_prediction:
            if timeframe == '1h':
                # For hourly predictions, keep more recent points
                features = features.iloc[-25:].copy()  # Increased to account for all indicators
            else:
                features = features.iloc[-20:].copy()  # Increased from 15
        else:
            min_points = 21 if timeframe == '1h' else 21  # Match the largest window size
            features = features.iloc[min_points:].copy()
            
        features = features.dropna()
        final_rows = len(features)
        
        logger.info(f"Data points lost in preprocessing: {initial_rows - final_rows}")
        logger.info(f"Final processed data points: {final_rows}")
        
        min_required = 3 if timeframe == '1h' else 5
        if features.empty or len(features) < min_required:
            raise ValueError(f"Insufficient data points after preprocessing: {len(features)} points remaining (minimum required: {min_required})")
            
        return features 