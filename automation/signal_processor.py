"""
Signal Processor - Phase 4
Advanced signal processing and validation system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import warnings

warnings.filterwarnings('ignore')

class SignalProcessor:
    """Advanced signal processing system for automated trading"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.min_confidence = 0.6
        self.max_signals_per_hour = 10
        self._init_signal_db()
    
    def _init_signal_db(self):
        """Initialize signal processing database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                status TEXT DEFAULT 'PENDING',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing signal database: {e}")
    
    def process_signal(self, signal_data: Dict) -> Dict:
        """Process and validate incoming trading signals"""
        try:
            # Validate signal structure
            if not self._validate_signal_structure(signal_data):
                return {
                    'status': 'REJECTED',
                    'reason': 'Invalid signal structure'
                }
            
            symbol = signal_data['symbol']
            action = signal_data['action']
            confidence = signal_data.get('confidence', 0.5)
            source = signal_data.get('source', 'UNKNOWN')
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                return {
                    'status': 'REJECTED',
                    'reason': f'Confidence {confidence:.2f} below minimum {self.min_confidence}'
                }
            
            # Check rate limits
            if not self._check_rate_limits(symbol, source):
                return {
                    'status': 'RATE_LIMITED',
                    'reason': 'Signal rate limit exceeded'
                }
            
            # Store processed signal
            signal_id = self._store_processed_signal(signal_data)
            
            return {
                'status': 'ACCEPTED',
                'signal_id': signal_id,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'reason': f'Signal processing error: {str(e)}'
            }
    
    def _validate_signal_structure(self, signal_data: Dict) -> bool:
        """Validate basic signal structure"""
        required_fields = ['symbol', 'action']
        for field in required_fields:
            if field not in signal_data:
                return False
        
        valid_actions = ['BUY', 'SELL', 'HOLD']
        if signal_data.get('action') not in valid_actions:
            return False
        
        confidence = signal_data.get('confidence', 0.5)
        if not 0 <= confidence <= 1:
            return False
        
        return True
    
    def _check_rate_limits(self, symbol: str, source: str) -> bool:
        """Check if signal rate limits are exceeded"""
        try:
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT COUNT(*) FROM processed_signals
            WHERE symbol = ? AND source = ? AND created_at > ?
            ''', (symbol, source, one_hour_ago))
            
            recent_count = cursor.fetchone()[0]
            conn.close()
            
            return recent_count < self.max_signals_per_hour
            
        except Exception as e:
            return True  # Default to allow on error
    
    def _store_processed_signal(self, signal_data: Dict) -> int:
        """Store processed signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO processed_signals 
            (timestamp, symbol, action, confidence, source, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                signal_data['symbol'],
                signal_data['action'],
                signal_data.get('confidence', 0.5),
                signal_data.get('source', 'UNKNOWN'),
                'PROCESSED'
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return signal_id
            
        except Exception as e:
            print(f"Error storing processed signal: {e}")
            return 0
    
    def get_signal_analytics(self) -> Dict:
        """Get signal processing analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Recent signal counts
            cursor.execute('''
            SELECT status, COUNT(*) FROM processed_signals
            WHERE created_at > ?
            GROUP BY status
            ''', ((datetime.now() - timedelta(hours=24)).isoformat(),))
            
            daily_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'daily_signal_counts': daily_counts,
                'analytics_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
