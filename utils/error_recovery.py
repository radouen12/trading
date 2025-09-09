"""
Error Recovery System - Graceful Degradation
Handles failures and provides fallback mechanisms
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path

class ErrorRecoveryManager:
    """
    Centralized error handling and recovery system
    """
    
    def __init__(self, log_file: str = "logs/errors.log"):
        self.log_file = log_file
        self.error_counts = {}
        self.fallback_data = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup error logging"""
        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger('error_recovery')
        self.logger.setLevel(logging.ERROR)
        
        # File handler for errors
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def handle_error(self, 
                    error: Exception, 
                    context: str, 
                    fallback_value: Any = None,
                    log_level: str = "ERROR") -> Any:
        """
        Handle an error with logging and recovery
        
        Args:
            error: The exception that occurred
            context: Description of what was being attempted
            fallback_value: Value to return if recovery fails
            log_level: Logging level (ERROR, WARNING, INFO)
            
        Returns:
            fallback_value or appropriate recovery value
        """
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        error_msg = f"{context} failed: {str(error)}"
        if log_level == "ERROR":
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        elif log_level == "WARNING":
            self.logger.warning(error_msg)
        
        # Print user-friendly message
        if self.error_counts[error_key] == 1:
            print(f"⚠️ {context} encountered an issue: {str(error)}")
            if fallback_value is not None:
                print(f"🔄 Using fallback mechanism...")
        elif self.error_counts[error_key] <= 3:
            print(f"⚠️ {context} still having issues (attempt #{self.error_counts[error_key]})")
        
        return fallback_value
    
    def safe_execute(self, 
                    func: Callable, 
                    context: str,
                    fallback_value: Any = None,
                    max_retries: int = 3,
                    *args, **kwargs) -> Any:
        """
        Safely execute a function with retries and error handling
        
        Args:
            func: Function to execute
            context: Description of the operation
            fallback_value: Value to return on failure
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Function result or fallback_value
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"🔄 Retrying {context} (attempt {attempt + 2}/{max_retries + 1})")
                    continue
                else:
                    return self.handle_error(e, context, fallback_value)
        
        return fallback_value
    
    def get_fallback_data(self, data_type: str) -> Dict[str, Any]:
        """
        Get cached fallback data for when APIs fail
        """
        if data_type not in self.fallback_data:
            self.fallback_data[data_type] = self._generate_fallback_data(data_type)
        
        return self.fallback_data[data_type]
    
    def _generate_fallback_data(self, data_type: str) -> Dict[str, Any]:
        """Generate synthetic fallback data"""
        current_time = datetime.now()
        
        if data_type == "stock_data":
            # Return basic stock data with neutral values
            return {
                'AAPL': {
                    'symbol': 'AAPL',
                    'price': 150.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'volume': 1000000,
                    'high': 151.0,
                    'low': 149.0,
                    'timestamp': current_time,
                    'asset_type': 'stock',
                    '_fallback': True
                },
                'MSFT': {
                    'symbol': 'MSFT',
                    'price': 300.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'volume': 800000,
                    'high': 302.0,
                    'low': 298.0,
                    'timestamp': current_time,
                    'asset_type': 'stock',
                    '_fallback': True
                }
            }
        
        elif data_type == "crypto_data":
            return {
                'BTC-USD': {
                    'symbol': 'BTC-USD',
                    'price': 45000.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'volume': 100000,
                    'high': 46000.0,
                    'low': 44000.0,
                    'timestamp': current_time,
                    'asset_type': 'crypto',
                    '_fallback': True
                }
            }
        
        elif data_type == "forex_data":
            return {
                'EURUSD=X': {
                    'symbol': 'EURUSD=X',
                    'price': 1.1000,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'volume': 0,
                    'high': 1.1020,
                    'low': 1.0980,
                    'timestamp': current_time,
                    'asset_type': 'forex',
                    '_fallback': True
                }
            }
        
        return {}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'unique_errors': len(self.error_counts),
            'error_breakdown': self.error_counts.copy(),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }
    
    def reset_error_counts(self):
        """Reset error tracking"""
        self.error_counts.clear()
        print("✅ Error tracking reset")

# Global error recovery instance
error_recovery = ErrorRecoveryManager()

def safe_data_fetch(fetch_function, data_type: str, context: str = "Data fetch"):
    """
    Wrapper for safe data fetching with fallback
    """
    def fetch_with_fallback():
        try:
            data = fetch_function()
            if data and len(data) > 0:
                return data
            else:
                raise Exception("Empty data returned")
        except Exception as e:
            fallback_data = error_recovery.get_fallback_data(data_type)
            print(f"⚠️ Using fallback data for {data_type}")
            return fallback_data
    
    return error_recovery.safe_execute(
        fetch_with_fallback,
        context,
        fallback_value=error_recovery.get_fallback_data(data_type)
    )

def with_error_recovery(context: str, fallback_value: Any = None):
    """
    Decorator for automatic error recovery
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return error_recovery.safe_execute(
                func, context, fallback_value, *args, **kwargs
            )
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Test error recovery
    @with_error_recovery("Test function", fallback_value="fallback")
    def test_function():
        raise Exception("Test error")
    
    result = test_function()
    print(f"Result: {result}")
    
    # Print error summary
    summary = error_recovery.get_error_summary()
    print(f"Error summary: {summary}")
