import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from config import Config
import time
from functools import wraps

# Retry decorator for network requests
def retry_request(max_retries=3, delay=1, backoff=2):
    """Decorator to retry network requests with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (Exception,) as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"❌ Failed after {max_retries} retries: {e}")
                        raise
                    
                    wait_time = delay * (backoff ** (retries - 1))
                    print(f"⚠️ Request failed, retrying in {wait_time}s... (attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class TechnicalAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index with proper handling of edge cases"""
        try:
            if not isinstance(prices, pd.Series):
                prices = pd.Series(prices)
                
            if len(prices) < period + 1:
                # Not enough data for RSI calculation
                return pd.Series([50] * len(prices), index=prices.index)
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate rolling averages
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            
            # Handle division by zero with small epsilon
            epsilon = 1e-10
            avg_loss_safe = avg_loss.replace(0, epsilon)
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss_safe
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with neutral RSI (50)
            rsi = rsi.fillna(50)
            
            # Ensure RSI is within valid range [0, 100]
            rsi = rsi.clip(0, 100)
            
            return rsi
            
        except Exception as e:
            print(f"❌ Error calculating RSI: {e}")
            # Return neutral RSI series in case of error
            if isinstance(prices, pd.Series):
                return pd.Series([50] * len(prices), index=prices.index)
            else:
                return pd.Series([50])
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence) with validation"""
        try:
            if not isinstance(prices, pd.Series):
                prices = pd.Series(prices)
                
            if len(prices) < slow + signal:
                # Not enough data for MACD calculation
                return {
                    'macd': pd.Series([0] * len(prices), index=prices.index),
                    'signal': pd.Series([0] * len(prices), index=prices.index),
                    'histogram': pd.Series([0] * len(prices), index=prices.index)
                }
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
            ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Fill NaN values with zeros
            macd_line = macd_line.fillna(0)
            signal_line = signal_line.fillna(0)
            histogram = histogram.fillna(0)
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            print(f"❌ Error calculating MACD: {e}")
            # Return zero series in case of error
            zero_series = pd.Series([0] * len(prices), index=prices.index if isinstance(prices, pd.Series) else None)
            return {
                'macd': zero_series,
                'signal': zero_series,
                'histogram': zero_series
            }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands with validation"""
        try:
            if not isinstance(prices, pd.Series):
                prices = pd.Series(prices)
                
            if len(prices) < period:
                # Not enough data for Bollinger Bands
                return {
                    'upper': prices,
                    'middle': prices,
                    'lower': prices
                }
            
            # Calculate moving average and standard deviation
            sma = prices.rolling(window=period, min_periods=period).mean()
            std = prices.rolling(window=period, min_periods=period).std()
            
            # Calculate bands
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Fill NaN values with price values
            upper_band = upper_band.fillna(prices)
            lower_band = lower_band.fillna(prices)
            sma = sma.fillna(prices)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
            
        except Exception as e:
            print(f"❌ Error calculating Bollinger Bands: {e}")
            # Return price series as fallback
            return {
                'upper': prices,
                'middle': prices,
                'lower': prices
            }
    
    @retry_request(max_retries=3, delay=1, backoff=1.5)
    def fetch_symbol_data(self, symbol, period="3mo"):
        """Fetch data for a single symbol with retry logic"""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol: {symbol}")
            
            symbol = symbol.strip().upper()
            
            # Create ticker and fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                print(f"⚠️ No data available for {symbol}")
                return None
            
            # Validate data quality
            if len(data) < 10:
                print(f"⚠️ Insufficient data for {symbol}: {len(data)} days")
                return None
            
            # Check for missing or invalid prices
            if data['Close'].isna().all() or (data['Close'] <= 0).any():
                print(f"⚠️ Invalid price data for {symbol}")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            raise  # Re-raise to trigger retry
    
    def analyze_symbol(self, symbol):
        """Perform complete technical analysis for a single symbol"""
        try:
            # Fetch data
            data = self.fetch_symbol_data(symbol)
            if data is None:
                return {}
            
            prices = data['Close']
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            bb_data = self.calculate_bollinger_bands(prices)
            
            # Get current values
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            current_macd = macd_data['macd'].iloc[-1] if len(macd_data['macd']) > 0 else 0
            current_signal = macd_data['signal'].iloc[-1] if len(macd_data['signal']) > 0 else 0
            current_price = prices.iloc[-1]
            bb_upper = bb_data['upper'].iloc[-1]
            bb_lower = bb_data['lower'].iloc[-1]
            bb_middle = bb_data['middle'].iloc[-1]
            
            # Generate signals
            signals = self.generate_signals(current_rsi, current_macd, current_signal, 
                                          current_price, bb_upper, bb_lower, bb_middle)
            
            return {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'signals': signals,
                'analysis_time': datetime.now().replace(microsecond=0)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return {}
    
    def generate_signals(self, rsi, macd, macd_signal, price, bb_upper, bb_lower, bb_middle):
        """Generate trading signals based on technical indicators"""
        signals = []
        
        try:
            # RSI signals
            if rsi < 30:
                signals.append(('BUY', 'RSI Oversold', 70))
            elif rsi > 70:
                signals.append(('SELL', 'RSI Overbought', 70))
            
            # MACD signals
            if macd > macd_signal:
                signals.append(('BUY', 'MACD Bullish Crossover', 65))
            elif macd < macd_signal:
                signals.append(('SELL', 'MACD Bearish Crossover', 65))
            
            # Bollinger Bands signals
            if price <= bb_lower:
                signals.append(('BUY', 'Price at Lower Bollinger Band', 60))
            elif price >= bb_upper:
                signals.append(('SELL', 'Price at Upper Bollinger Band', 60))
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return []
    
    def analyze_all_symbols(self):
        """Analyze all configured symbols"""
        try:
            all_symbols = (self.config.STOCK_SYMBOLS + 
                          self.config.CRYPTO_SYMBOLS + 
                          self.config.FOREX_SYMBOLS)
            
            results = {}
            
            for symbol in all_symbols:
                try:
                    analysis = self.analyze_symbol(symbol)
                    if analysis:  # Only add if analysis was successful
                        results[symbol] = analysis
                    
                    # Add small delay to avoid overwhelming APIs
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️ Skipping {symbol} due to error: {e}")
                    continue
            
            print(f"✅ Technical analysis completed for {len(results)} symbols")
            return results
            
        except Exception as e:
            print(f"❌ Error in bulk analysis: {e}")
            return {}

# Test the analyzer
if __name__ == "__main__":
    analyzer = TechnicalAnalyzer()
    
    # Test single symbol analysis
    test_symbol = 'AAPL'
    print(f"Testing technical analysis for {test_symbol}...")
    
    result = analyzer.analyze_symbol(test_symbol)
    if result:
        print(f"✅ Analysis successful for {test_symbol}")
        print(f"RSI: {result.get('rsi', 'N/A'):.2f}")
        print(f"MACD: {result.get('macd', 'N/A'):.4f}")
        print(f"Signals: {len(result.get('signals', []))}")
    else:
        print(f"❌ Analysis failed for {test_symbol}")
