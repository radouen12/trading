"""
Alert Manager - Phase 3
Centralized alert management and coordination
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from .email_alerts import EmailAlerts
import logging

class AlertManager:
    """
    Centralized alert management system
    """
    
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        self.email_alerts = EmailAlerts()
        
        # Alert settings
        self.alert_settings = {
            'email_enabled': False,
            'sms_enabled': False,
            'min_confidence_for_alerts': 75,
            'max_alerts_per_hour': 10,
            'daily_report_time': '17:00',
            'risk_alert_enabled': True
        }
        
        # Alert tracking
        self.recent_alerts = []
        self.alert_count_tracker = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize alert tracking database"""
        
        try:
            Path(self.db_path).parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        alert_type TEXT,
                        symbol TEXT,
                        message TEXT,
                        confidence REAL,
                        sent_email BOOLEAN,
                        sent_sms BOOLEAN,
                        details TEXT
                    )
                """)
                
                # Create alert settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alert_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                conn.commit()
                
            self.logger.info("Alert database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize alert database: {e}")
    
    def configure_email(self, sender_email: str, sender_password: str, recipient_email: str):
        """Configure email alerts"""
        
        self.email_alerts.configure_smtp(sender_email, sender_password, recipient_email)
        self.alert_settings['email_enabled'] = True
        self._save_alert_settings()
        
        # Test email connection
        if self.email_alerts.test_connection():
            self.logger.info("Email alerts configured and tested successfully")
            return True
        else:
            self.logger.error("Email configuration test failed")
            self.alert_settings['email_enabled'] = False
            return False
    
    def update_alert_settings(self, settings: Dict):
        """Update alert configuration settings"""
        
        self.alert_settings.update(settings)
        self._save_alert_settings()
        self.logger.info("Alert settings updated")
    
    def process_trading_signals(self, signals: List[Dict]) -> bool:
        """Process and send alerts for trading signals"""
        
        if not signals:
            return True
        
        # Filter signals by confidence threshold
        high_confidence_signals = [
            s for s in signals 
            if s.get('confidence', 0) >= self.alert_settings['min_confidence_for_alerts']
        ]
        
        if not high_confidence_signals:
            self.logger.info("No signals meet confidence threshold for alerts")
            return True
        
        # Check rate limiting
        if not self._check_rate_limit('trading_signals'):
            self.logger.warning("Rate limit exceeded for trading signal alerts")
            return False
        
        success = True
        
        # Send email alert
        if self.alert_settings['email_enabled']:
            email_success = self.email_alerts.send_trading_signal_alert(high_confidence_signals)
            if email_success:
                self._log_alert('trading_signals', '', 'Trading signals sent via email', 
                              max([s.get('confidence', 0) for s in high_confidence_signals]), 
                              sent_email=True)
            success = success and email_success
        
        # Update rate limiting
        self._update_rate_limit('trading_signals')
        
        return success
    
    def send_position_alert(self, alert_type: str, symbol: str, details: Dict) -> bool:
        """Send position-related alerts"""
        
        # Check rate limiting
        if not self._check_rate_limit('position_alerts'):
            return False
        
        success = True
        
        # Send email alert
        if self.alert_settings['email_enabled']:
            email_success = self.email_alerts.send_position_alert(alert_type, symbol, details)
            if email_success:
                self._log_alert('position_alert', symbol, f"{alert_type} alert for {symbol}", 
                              details.get('confidence', 0), sent_email=True)
            success = success and email_success
        
        self._update_rate_limit('position_alerts')
        
        return success
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if alert type is within rate limits"""
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        key = f"{alert_type}_{current_hour}"
        
        current_count = self.alert_count_tracker.get(key, 0)
        max_per_hour = self.alert_settings['max_alerts_per_hour']
        
        return current_count < max_per_hour
    
    def _update_rate_limit(self, alert_type: str):
        """Update rate limit tracking"""
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        key = f"{alert_type}_{current_hour}"
        
        self.alert_count_tracker[key] = self.alert_count_tracker.get(key, 0) + 1
        
        # Clean up old entries (keep only current hour)
        keys_to_remove = [k for k in self.alert_count_tracker.keys() 
                         if not k.endswith(str(current_hour))]
        for key in keys_to_remove:
            del self.alert_count_tracker[key]
    
    def _log_alert(self, alert_type: str, symbol: str, message: str, confidence: float,
                   sent_email: bool = False, sent_sms: bool = False, details: Dict = None):
        """Log alert to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO alerts (timestamp, alert_type, symbol, message, confidence,
                                      sent_email, sent_sms, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    alert_type,
                    symbol,
                    message,
                    confidence,
                    sent_email,
                    sent_sms,
                    json.dumps(details) if details else None
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
    
    def _save_alert_settings(self):
        """Save alert settings to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for key, value in self.alert_settings.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO alert_settings (key, value)
                        VALUES (?, ?)
                    """, (key, json.dumps(value)))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save alert settings: {e}")
    
    def test_all_alerts(self) -> Dict[str, bool]:
        """Test all configured alert methods"""
        
        results = {}
        
        # Test email
        if self.alert_settings['email_enabled']:
            results['email'] = self.email_alerts.test_connection()
        else:
            results['email'] = False
        
        return results
