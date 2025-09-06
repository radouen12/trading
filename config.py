# Trading System Configuration
import os
from datetime import datetime

class Config:
    # Capital Management
    TOTAL_CAPITAL = 10000  # Change this to your actual capital
    MAX_POSITION_SIZE = 0.05  # 5% per position
    MIN_POSITION_SIZE = 0.01  # 1% per position
    RESERVE_CASH_RATIO = 0.2  # Keep 20% cash
    MAX_DAILY_LOSS = 0.03  # Stop trading if down 3% in a day
    
    # Risk Management
    MAX_CORRELATION_POSITIONS = 3
    MAX_SECTOR_WEIGHT = 0.4
    MIN_CONFIDENCE_SCORE = 70
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = "your_api_key_here"  # Get free at alphavantage.co
    NEWS_API_KEY = "your_news_api_key_here"  # Get free at newsapi.org
    
    # Data Sources
    STOCK_SYMBOLS = [
        # Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        # Indices
        'SPY', 'QQQ', 'IWM', 'VIX',
        # Sectors
        'XLF', 'XLE', 'XLK', 'XLV', 'XRT'
    ]
    
    CRYPTO_SYMBOLS = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD',
        'DOT-USD', 'AVAX-USD', 'MATIC-USD'
    ]
    
    FOREX_SYMBOLS = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 
        'USDCAD=X', 'USDCHF=X', 'DX-Y.NYB'  # Dollar Index
    ]
    
    # Update Intervals
    REAL_TIME_INTERVAL = 60  # seconds
    ANALYSIS_INTERVAL = 600  # 10 minutes
    
    # Database
    DB_PATH = "data/trading_data.db"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/trading.log"
    
    # Trading Hours (ET)
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    
    @classmethod
    def get_available_capital(cls):
        """Calculate available capital for new positions"""
        return cls.TOTAL_CAPITAL * (1 - cls.RESERVE_CASH_RATIO)
    
    @classmethod
    def get_position_size(cls, risk_level='medium'):
        """Calculate position size based on risk level"""
        available = cls.get_available_capital()
        
        if risk_level == 'low':
            return available * cls.MIN_POSITION_SIZE
        elif risk_level == 'high':
            return available * cls.MAX_POSITION_SIZE
        else:  # medium
            return available * ((cls.MIN_POSITION_SIZE + cls.MAX_POSITION_SIZE) / 2)
