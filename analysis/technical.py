import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from config import Config

class TechnicalAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            if len(prices) < period:
                return [50] * len(prices)
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).tolist()
        except:
            return [50] * len(prices)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            
            return macd.iloc[-1], macd_signal.iloc[-1]
        except:
            return 0, 0
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
        except:
            current_price = prices.iloc[-1] if len(prices) > 0 else 100
            return current_price * 1.02, current_price, current_price * 0.98
    
    def analyze_symbol(self, symbol):
        """Analyze single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                return self._get_default_analysis(symbol)
            
            prices = hist['Close']
            
            # Calculate indicators
            rsi_values = self.calculate_rsi(prices)
            current_rsi = rsi_values[-1] if rsi_values else 50
            
            macd, macd_signal = self.calculate_macd(prices)
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices)
            
            current_price = prices.iloc[-1]
            
            # Generate signals
            signals = []
            
            # RSI signals
            if current_rsi < 30:
                signals.append(('BUY', 'RSI Oversold', 75))
            elif current_rsi > 70:
                signals.append(('SELL', 'RSI Overbought', 75))
            
            # MACD signals
            if macd > macd_signal:
                signals.append(('BUY', 'MACD Bullish', 70))
            else:
                signals.append(('SELL', 'MACD Bearish', 65))
            
            # Bollinger Band signals
            if current_price < lower_bb:
                signals.append(('BUY', 'Below Lower BB', 70))
            elif current_price > upper_bb:
                signals.append(('SELL', 'Above Upper BB', 70))
            
            return {
                'symbol': symbol,
                'rsi': current_rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'bollinger_upper': upper_bb,
                'bollinger_middle': middle_bb,
                'bollinger_lower': lower_bb,
                'current_price': current_price,
                'signals': signals,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return self._get_default_analysis(symbol)
    
    def _get_default_analysis(self, symbol):
        """Default analysis when data unavailable"""
        return {
            'symbol': symbol,
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'bollinger_upper': 100,
            'bollinger_middle': 95,
            'bollinger_lower': 90,
            'current_price': 95,
            'signals': [('HOLD', 'Insufficient data', 50)],
            'analysis_time': datetime.now()
        }
    
    def analyze_all_symbols(self):
        """Analyze all configured symbols"""
        results = {}
        
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        for symbol in all_symbols[:10]:  # Limit to prevent timeouts
            try:
                results[symbol] = self.analyze_symbol(symbol)
            except:
                results[symbol] = self._get_default_analysis(symbol)
        
        return results

if __name__ == "__main__":
    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze_symbol('AAPL')
    print(f"RSI: {result['rsi']:.2f}")
    print(f"Signals: {result['signals']}")
