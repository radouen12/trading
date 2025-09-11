# Trading System Configuration
import os
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except:
    print("⚠️ python-dotenv not installed or .env file not found")

# Get project root directory at module level
PROJECT_ROOT = Path(__file__).parent

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
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'demo')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_SENDER = os.getenv('EMAIL_SENDER', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', '')
    
    @classmethod
    def validate_api_keys(cls):
        """Validate API key configuration"""
        issues = []
        
        if cls.ALPHA_VANTAGE_API_KEY == "demo" or not cls.ALPHA_VANTAGE_API_KEY:
            issues.append("Alpha Vantage API key not configured (using demo mode)")
            
        if cls.NEWS_API_KEY == "demo" or not cls.NEWS_API_KEY:
            issues.append("News API key not configured (using demo mode)")
            
        if not cls.EMAIL_SENDER or not cls.EMAIL_PASSWORD:
            issues.append("Email alerts not configured")
            
        return issues
    
    # Data Sources
    STOCK_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        'SPY', 'QQQ', 'IWM', 'VIX',
        'XLF', 'XLE', 'XLK', 'XLV', 'XRT'
    ]
    
    CRYPTO_SYMBOLS = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD',
        'DOT-USD', 'AVAX-USD', 'MATIC-USD'
    ]
    
    FOREX_SYMBOLS = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 
        'USDCAD=X', 'USDCHF=X', 'DX-Y.NYB'
    ]
    
    # Update Intervals
    REAL_TIME_INTERVAL = 60  # seconds
    ANALYSIS_INTERVAL = 600  # 10 minutes
    
    # Database and Logging - Fixed paths using module-level PROJECT_ROOT
    DB_PATH = str(PROJECT_ROOT / "data" / "trading_data.db")
    LOG_FILE = str(PROJECT_ROOT / "logs" / "trading.log")
    LOG_LEVEL = "INFO"
    
    @classmethod
    def get_project_root(cls):
        """Get project root directory"""
        return PROJECT_ROOT
    
    @classmethod
    def initialize_directories(cls):
        """Initialize required directories"""
        # Create directories
        data_dir = PROJECT_ROOT / "data"
        logs_dir = PROJECT_ROOT / "logs"
        
        data_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        # Update class paths
        cls.DB_PATH = str(data_dir / "trading_data.db")
        cls.LOG_FILE = str(logs_dir / "trading.log")
        
        return {
            'data_directory': str(data_dir),
            'logs_directory': str(logs_dir),
            'data_success': True,
            'logs_success': True,
            'db_path': cls.DB_PATH,
            'log_file': cls.LOG_FILE
        }
    
    # Trading Hours
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    
    # Currency
    DEFAULT_CURRENCY = "USD"
    
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

# Initialize directories when module is imported
try:
    Config.initialize_directories()
    print(f"✅ Configuration initialized - DB: {Config.DB_PATH}")
except Exception as e:
    print(f"⚠️ Configuration initialization warning: {e}")
