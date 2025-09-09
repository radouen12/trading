# Trading System Configuration
import os
from datetime import datetime

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

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
    
    # API Configuration - Using environment variables for security
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')  # Get free at alphavantage.co
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'demo')  # Get free at newsapi.org
    
    # Email Configuration for Alerts
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
        elif len(cls.ALPHA_VANTAGE_API_KEY) < 10:  # Basic validation
            issues.append("Alpha Vantage API key appears invalid (too short)")
            
        if cls.NEWS_API_KEY == "demo" or not cls.NEWS_API_KEY:
            issues.append("News API key not configured (using demo mode)")
        elif len(cls.NEWS_API_KEY) < 10:  # Basic validation
            issues.append("News API key appears invalid (too short)")
            
        # Email validation
        if not cls.EMAIL_SENDER or not cls.EMAIL_PASSWORD:
            issues.append("Email alerts not configured (missing sender/password)")
        if not cls.EMAIL_RECIPIENT:
            issues.append("Email alerts not configured (missing recipient)")
            
        return issues
    
    @classmethod
    def is_alpha_vantage_available(cls):
        """Check if Alpha Vantage API is properly configured"""
        return (cls.ALPHA_VANTAGE_API_KEY and 
                cls.ALPHA_VANTAGE_API_KEY != "demo" and 
                len(cls.ALPHA_VANTAGE_API_KEY) >= 10)
    
    @classmethod
    def is_news_api_available(cls):
        """Check if News API is properly configured"""
        return (cls.NEWS_API_KEY and 
                cls.NEWS_API_KEY != "demo" and 
                len(cls.NEWS_API_KEY) >= 10)
    
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
    
    # Database - Using absolute path with fallback
    DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'trading_data.db'))
    
    @classmethod
    def ensure_data_directory(cls):
        """Ensure data directory exists with proper error handling"""
        from pathlib import Path
        import tempfile
        
        try:
            # First try the configured path
            data_dir = Path(cls.DB_PATH).parent
            data_dir.mkdir(exist_ok=True, parents=True)
            
            # Test write permissions
            test_file = data_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
            print(f"✅ Data directory ready: {data_dir}")
            return True, str(data_dir)
            
        except PermissionError:
            print(f"⚠️ Permission denied for {data_dir}. Using temp directory.")
            temp_dir = Path(tempfile.gettempdir()) / "trading_data"
            temp_dir.mkdir(exist_ok=True, parents=True)
            cls.DB_PATH = str(temp_dir / "trading_data.db")
            print(f"✅ Using temp directory: {temp_dir}")
            return True, str(temp_dir)
            
        except Exception as e:
            print(f"❌ Failed to create data directory: {e}")
            # Last resort: use current directory
            current_dir = Path.cwd() / "temp_trading_data"
            current_dir.mkdir(exist_ok=True)
            cls.DB_PATH = str(current_dir / "trading_data.db")
            print(f"⚠️ Using current directory fallback: {current_dir}")
            return True, str(current_dir)
    
    @classmethod
    def initialize_directories(cls):
        """Initialize all required directories"""
        data_success, data_dir = cls.ensure_data_directory()
        logs_success, logs_dir = cls.ensure_logs_directory()
        
        return {
            'data_directory': data_dir,
            'logs_directory': logs_dir,
            'data_success': data_success,
            'logs_success': logs_success,
            'db_path': cls.DB_PATH,
            'log_file': cls.LOG_FILE
        }
    
    # Logging - Using absolute path with fallback
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'trading.log'))
    
    @classmethod
    def ensure_logs_directory(cls):
        """Ensure logs directory exists"""
        from pathlib import Path
        try:
            log_dir = Path(cls.LOG_FILE).parent
            log_dir.mkdir(exist_ok=True, parents=True)
            print(f"✅ Logs directory ready: {log_dir}")
            return True, str(log_dir)
        except Exception as e:
            print(f"⚠️ Could not create logs directory: {e}")
            # Fallback to current directory
            cls.LOG_FILE = "trading.log"
            return True, "."
    
    # Trading Hours (ET)
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    
    # Internationalization
    CURRENCY_SYMBOLS = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "CHF": "CHF",
        "CNY": "¥"
    }
    
    DEFAULT_CURRENCY = "USD"
    DEFAULT_LOCALE = "US"  # US, EU, etc.
    
    @classmethod
    def get_currency_symbol(cls, currency_code: str = None) -> str:
        """Get currency symbol for the specified currency"""
        if currency_code is None:
            currency_code = cls.DEFAULT_CURRENCY
        return cls.CURRENCY_SYMBOLS.get(currency_code.upper(), "$")
    
    @classmethod
    def get_supported_currencies(cls) -> list:
        """Get list of supported currencies"""
        return list(cls.CURRENCY_SYMBOLS.keys())
    
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
