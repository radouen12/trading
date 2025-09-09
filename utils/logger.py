import logging
import os
from datetime import datetime
from pathlib import Path
import gzip
import shutil
from logging.handlers import RotatingFileHandler

class TradingLogger:
    def __init__(self, log_level="INFO", max_log_size_mb=10, backup_count=5):
        self.max_log_size = max_log_size_mb * 1024 * 1024  # Convert to bytes
        self.backup_count = backup_count
        self.setup_logger(log_level)
    
    def setup_logger(self, log_level):
        """Setup logging configuration with rotation and error handling"""
        try:
            # Create logs directory with proper error handling
            log_dir = Path("logs")
            
            try:
                log_dir.mkdir(exist_ok=True, parents=True)
            except PermissionError:
                # Try alternative directory if permission denied
                import tempfile
                log_dir = Path(tempfile.gettempdir()) / "trading_logs"
                log_dir.mkdir(exist_ok=True, parents=True)
                print(f"⚠️ Using alternative log directory: {log_dir}")
            except Exception as e:
                print(f"❌ Failed to create log directory: {e}")
                # Fall back to current directory
                log_dir = Path(".")
                print("Using current directory for logs")
            
            # Setup log file with rotation
            log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            
            # Create console handler
            console_handler = logging.StreamHandler()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Setup root logger
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                handlers=[file_handler, console_handler],
                force=True  # Override existing handlers
            )
            
            self.logger = logging.getLogger(__name__)
            self.log_dir = log_dir
            
            # Log successful initialization
            self.logger.info(f"Trading logger initialized. Log directory: {log_dir}")
            
        except Exception as e:
            print(f"❌ Critical error setting up logger: {e}")
            # Setup minimal console-only logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.log_dir = None
    
    def compress_old_logs(self):
        """Compress old log files to save space"""
        if not self.log_dir:
            return
            
        try:
            # Find log files older than 7 days
            cutoff_date = datetime.now().timestamp() - (7 * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("trading_*.log.*"):
                if log_file.stat().st_mtime < cutoff_date and not log_file.suffix == '.gz':
                    # Compress the file
                    gz_file = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(gz_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    log_file.unlink()
                    self.logger.info(f"Compressed old log file: {log_file} -> {gz_file}")
                    
        except Exception as e:
            self.logger.error(f"Error compressing old logs: {e}")
    
    def cleanup_old_logs(self, days_to_keep=30):
        """Remove very old log files"""
        if not self.log_dir:
            return
            
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("trading_*.log*"):
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
                    self.logger.info(f"Removed old log file: {log_file}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
    
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
