import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from config import Config

class TechnicalAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd.fillna(0),
                'signal': signal_line.fillna(0),
                'histogram': histogram.fillna(0)
            }
        except:
            return {
                'macd': pd.Series([0] * len(prices), index=prices.index),
                'signal': pd.Series([0] * len(prices), index=prices.index),
                'histogram': pd.Series([0] * len(prices), index=prices.index)
            }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band.fillna(prices.mean()),
                'middle': sma.fillna(prices.mean()),
                'lower': lower_band.fillna(prices.mean())
            }
        except:
            mean_price = prices.mean()
            return {
                'upper': pd.Series([mean_price * 1.02] * len(prices), index=prices.index),
                'middle': pd.Series([mean_price] * len(prices), index=prices.index),
                'lower': pd.Series([mean_price * 0.98] * len(prices), index=prices.index)
            }
    
    def calculate_volume_indicators(self, prices, volumes, period=20):
        """Calculate volume-based indicators"""
        try:
            # Volume SMA
            volume_sma = volumes.rolling(window=period).mean()
            
            # On-Balance Volume (OBV)
            obv = []
            obv_value = 0
            for i in range(len(prices)):
                if i > 0:
                    if prices.iloc[i] > prices.iloc[i-1]:
                        obv_value += volumes.iloc[i]
                    elif prices.iloc[i] < prices.iloc[i-1]:
                        obv_value -= volumes.iloc[i]
                obv.append(obv_value)
            
            obv_series = pd.Series(obv, index=prices.index)
            
            # Volume Rate of Change
            volume_roc = volumes.pct_change(periods=period) * 100
            
            return {
                'volume_sma': volume_sma.fillna(volumes.mean()),
                'obv': obv_series,
                'volume_roc': volume_roc.fillna(0)
            }
        except:
            return {
                'volume_sma': pd.Series([volumes.mean()] * len(volumes), index=volumes.index),
                'obv': pd.Series([0] * len(volumes), index=volumes.index),
                'volume_roc': pd.Series([0] * len(volumes), index=volumes.index)
            }
    
    def detect_support_resistance(self, prices, window=20):
        """Detect support and resistance levels"""
        try:
            # Find local minima and maxima
            highs = prices.rolling(window=window, center=True).max()
            lows = prices.rolling(window=window, center=True).min()
            
            # Support levels (local minima)
            support_levels = []
            for i in range(len(prices)):
                if prices.iloc[i] == lows.iloc[i]:
                    support_levels.append(prices.iloc[i])
            
            # Resistance levels (local maxima)
            resistance_levels = []
            for i in range(len(prices)):
                if prices.iloc[i] == highs.iloc[i]:
                    resistance_levels.append(prices.iloc[i])
            
            # Get most significant levels
            support_levels = sorted(set(support_levels))[-3:] if support_levels else [prices.min()]
            resistance_levels = sorted(set(resistance_levels))[-3:] if resistance_levels else [prices.max()]
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'current_support': support_levels[-1] if support_levels else prices.min(),
                'current_resistance': resistance_levels[-1] if resistance_levels else prices.max()
            }
        except:
            return {
                'support': [prices.min()],
                'resistance': [prices.max()],
                'current_support': prices.min(),
                'current_resistance': prices.max()
            }
    
    def calculate_momentum_indicators(self, prices, period=14):
        """Calculate momentum indicators"""
        try:
            # Rate of Change
            roc = prices.pct_change(periods=period) * 100
            
            # Stochastic Oscillator
            high_max = prices.rolling(window=period).max()
            low_min = prices.rolling(window=period).min()
            stoch_k = ((prices - low_min) / (high_max - low_min)) * 100
            stoch_d = stoch_k.rolling(window=3).mean()
            
            # Williams %R
            williams_r = ((high_max - prices) / (high_max - low_min)) * -100
            
            return {
                'roc': roc.fillna(0),
                'stoch_k': stoch_k.fillna(50),
                'stoch_d': stoch_d.fillna(50),
                'williams_r': williams_r.fillna(-50)
            }
        except:
            return {
                'roc': pd.Series([0] * len(prices), index=prices.index),
                'stoch_k': pd.Series([50] * len(prices), index=prices.index),
                'stoch_d': pd.Series([50] * len(prices), index=prices.index),
                'williams_r': pd.Series([-50] * len(prices), index=prices.index)
            }
    
    def analyze_symbol(self, symbol, period="3mo", interval="1h"):
        """Comprehensive technical analysis for a symbol"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return self.get_default_analysis(symbol)
            
            prices = data['Close']
            volumes = data['Volume']
            highs = data['High']
            lows = data['Low']
            
            # Calculate all indicators
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            bb_data = self.calculate_bollinger_bands(prices)
            volume_data = self.calculate_volume_indicators(prices, volumes)
            sr_levels = self.detect_support_resistance(prices)
            momentum_data = self.calculate_momentum_indicators(prices)
            
            # Current values
            current_price = prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            
            # Generate signals
            signals = self.generate_technical_signals(
                current_price, current_rsi, current_macd, current_signal,
                bb_data, sr_levels, momentum_data
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_histogram': macd_data['histogram'].iloc[-1],
                'bb_upper': bb_data['upper'].iloc[-1],
                'bb_middle': bb_data['middle'].iloc[-1],
                'bb_lower': bb_data['lower'].iloc[-1],
                'support_level': sr_levels['current_support'],
                'resistance_level': sr_levels['current_resistance'],
                'volume_sma': volume_data['volume_sma'].iloc[-1],
                'obv': volume_data['obv'].iloc[-1],
                'stoch_k': momentum_data['stoch_k'].iloc[-1],
                'stoch_d': momentum_data['stoch_d'].iloc[-1],
                'williams_r': momentum_data['williams_r'].iloc[-1],
                'roc': momentum_data['roc'].iloc[-1],
                'signals': signals,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return self.get_default_analysis(symbol)
    
    def generate_technical_signals(self, price, rsi, macd, signal, bb_data, sr_levels, momentum_data):
        """Generate buy/sell signals from technical indicators"""
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append(('BUY', 'RSI Oversold', 75))
        elif rsi > 70:
            signals.append(('SELL', 'RSI Overbought', 75))
        
        # MACD signals
        if macd > signal and macd > 0:
            signals.append(('BUY', 'MACD Bullish', 70))
        elif macd < signal and macd < 0:
            signals.append(('SELL', 'MACD Bearish', 70))
        
        # Bollinger Bands signals
        bb_upper = bb_data['upper'].iloc[-1]
        bb_lower = bb_data['lower'].iloc[-1]
        
        if price <= bb_lower:
            signals.append(('BUY', 'BB Oversold', 65))
        elif price >= bb_upper:
            signals.append(('SELL', 'BB Overbought', 65))
        
        # Support/Resistance signals
        if abs(price - sr_levels['current_support']) / price < 0.02:
            signals.append(('BUY', 'Near Support', 60))
        elif abs(price - sr_levels['current_resistance']) / price < 0.02:
            signals.append(('SELL', 'Near Resistance', 60))
        
        # Stochastic signals
        stoch_k = momentum_data['stoch_k'].iloc[-1]
        if stoch_k < 20:
            signals.append(('BUY', 'Stochastic Oversold', 65))
        elif stoch_k > 80:
            signals.append(('SELL', 'Stochastic Overbought', 65))
        
        return signals
    
    def get_default_analysis(self, symbol):
        """Return default analysis when data is unavailable"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'bb_upper': 0,
            'bb_middle': 0,
            'bb_lower': 0,
            'support_level': 0,
            'resistance_level': 0,
            'volume_sma': 0,
            'obv': 0,
            'stoch_k': 50,
            'stoch_d': 50,
            'williams_r': -50,
            'roc': 0,
            'signals': [],
            'analysis_time': datetime.now()
        }
    
    def analyze_all_symbols(self):
        """Analyze all configured symbols"""
        print("🔍 Starting comprehensive technical analysis...")
        
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        results = {}
        
        for i, symbol in enumerate(all_symbols):
            print(f"📊 Analyzing {symbol} ({i+1}/{len(all_symbols)})")
            results[symbol] = self.analyze_symbol(symbol)
        
        print("✅ Technical analysis complete!")
        return results

if __name__ == "__main__":
    # Test the technical analyzer
    analyzer = TechnicalAnalyzer()
    
    # Test single symbol
    print("Testing AAPL analysis...")
    aapl_analysis = analyzer.analyze_symbol('AAPL')
    
    print(f"AAPL RSI: {aapl_analysis['rsi']:.2f}")
    print(f"AAPL MACD: {aapl_analysis['macd']:.4f}")
    print(f"AAPL Signals: {aapl_analysis['signals']}")
