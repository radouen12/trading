import logging
import os
from datetime import datetime
from pathlib import Path

class TradingLogger:
    def __init__(self, log_level="INFO"):
        self.setup_logger(log_level)
    
    def setup_logger(self, log_level):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_trade(self, action, symbol, price, quantity, reasoning):
        """Log trading actions"""
        message = f"TRADE: {action} {quantity} {symbol} @ ${price:.2f} - {reasoning}"
        self.logger.info(message)
    
    def log_error(self, message):
        """Log errors"""
        self.logger.error(f"ERROR: {message}")
    
    def log_data_fetch(self, asset_count, success=True):
        """Log data fetching"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"DATA_FETCH: {status} - {asset_count} assets")
    
    def log_suggestion(self, symbol, action, confidence, timeframe):
        """Log trading suggestions"""
        message = f"SUGGESTION: {action} {symbol} - Confidence: {confidence}% - Timeframe: {timeframe}"
        self.logger.info(message)

# Global logger instance
trading_logger = TradingLogger()
