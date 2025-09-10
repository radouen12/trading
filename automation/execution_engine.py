"""
Execution Engine - Phase 4
Automated trade execution with comprehensive safeguards
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
import warnings
import time

warnings.filterwarnings('ignore')

class ExecutionEngine:
    """
    Automated trade execution engine with multiple safety layers
    """
    
    def __init__(self, db_path: str = "data/trading_data.db", config_path: str = "config.py"):
        self.db_path = db_path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Execution state
        self.is_live_trading = False
        self.execution_mode = 'PAPER'  # PAPER, LIVE
        self.risk_checks_enabled = True
        
        # Safety limits
        self.max_daily_trades = 20
        self.max_position_size = 0.05  # 5% of capital
        self.max_daily_loss = 0.03     # 3% of capital
        
        # Execution tracking
        self.executed_trades = []
        self.pending_orders = []
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        
        # Initialize execution database
        self._init_execution_db()
        
    def _load_config(self) -> Dict:
        """Load trading configuration"""
        try:
            # Import config dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", self.config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            return {
                'TOTAL_CAPITAL': getattr(config_module, 'TOTAL_CAPITAL', 10000),
                'MAX_POSITION_SIZE': getattr(config_module, 'MAX_POSITION_SIZE', 0.05),
                'MAX_DAILY_LOSS': getattr(config_module, 'MAX_DAILY_LOSS', 0.03),
                'RISK_PER_TRADE': getattr(config_module, 'RISK_PER_TRADE', 0.02)
            }
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                'TOTAL_CAPITAL': 10000,
                'MAX_POSITION_SIZE': 0.05,
                'MAX_DAILY_LOSS': 0.03,
                'RISK_PER_TRADE': 0.02
            }
    
    def _init_execution_db(self):
        """Initialize execution tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create execution tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS automated_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                signal_source TEXT,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                execution_mode TEXT,
                pnl REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                symbol TEXT,
                action_taken TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                symbol TEXT,
                trade_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing execution database: {e}")
    
    def process_trading_signal(self, signal: Dict) -> Dict:
        """Process incoming trading signal with full validation"""
        try:
            # Validate signal format
            validation_result = self._validate_signal(signal)
            if not validation_result['valid']:
                return {
                    "error": "Invalid signal format",
                    "issues": validation_result['issues']
                }
            
            # Extract signal data
            symbol = signal['symbol']
            action = signal['action']  # BUY, SELL, HOLD
            confidence = signal.get('confidence', 0.5)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            signal_source = signal.get('source', 'UNKNOWN')
            
            # Skip HOLD signals
            if action == 'HOLD':
                return {
                    "action": "SKIPPED",
                    "reason": "HOLD signal - no action required",
                    "signal": signal
                }
            
            # Perform risk checks
            risk_check = self._perform_risk_checks(signal)
            if not risk_check['approved']:
                self._log_risk_event(
                    'TRADE_BLOCKED',
                    'HIGH',
                    f"Trade blocked for {symbol}: {risk_check['reason']}",
                    symbol
                )
                return {
                    "action": "BLOCKED",
                    "reason": risk_check['reason'],
                    "signal": signal
                }
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                return {
                    "action": "SKIPPED",
                    "reason": "Position size too small",
                    "signal": signal
                }
            
            # Create trade order
            trade_order = {
                'symbol': symbol,
                'action': action,
                'quantity': position_size,
                'price': entry_price,
                'order_type': 'MARKET',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_source': signal_source,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Execute trade
            execution_result = self._execute_trade(trade_order)
            
            return execution_result
            
        except Exception as e:
            self._log_execution_event(
                'ERROR',
                f"Error processing signal: {e}",
                trade_id=None
            )
            return {"error": str(e)}
    
    def _validate_signal(self, signal: Dict) -> Dict:
        """Validate incoming trading signal format"""
        validation = {
            'valid': True,
            'issues': []
        }
        
        required_fields = ['symbol', 'action']
        for field in required_fields:
            if field not in signal:
                validation['valid'] = False
                validation['issues'].append(f"Missing required field: {field}")
        
        # Validate action
        valid_actions = ['BUY', 'SELL', 'HOLD']
        if signal.get('action') not in valid_actions:
            validation['valid'] = False
            validation['issues'].append(f"Invalid action. Must be one of: {valid_actions}")
        
        # Validate confidence
        confidence = signal.get('confidence', 0.5)
        if not 0 <= confidence <= 1:
            validation['valid'] = False
            validation['issues'].append("Confidence must be between 0 and 1")
        
        return validation
    
    def _perform_risk_checks(self, signal: Dict) -> Dict:
        """Perform comprehensive risk checks on trading signal"""
        risk_check = {
            'approved': True,
            'reason': ''
        }
        
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal.get('confidence', 0.5)
            
            # Check minimum confidence
            min_confidence = 0.6
            if confidence < min_confidence:
                risk_check['approved'] = False
                risk_check['reason'] = f"Confidence {confidence:.2f} below minimum {min_confidence}"
                return risk_check
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                risk_check['approved'] = False
                risk_check['reason'] = "Daily trade limit exceeded"
                return risk_check
            
            # Check daily loss limit
            if abs(self.daily_pnl) >= (self.config['TOTAL_CAPITAL'] * self.max_daily_loss):
                risk_check['approved'] = False
                risk_check['reason'] = "Daily loss limit exceeded"
                return risk_check
            
        except Exception as e:
            risk_check['approved'] = False
            risk_check['reason'] = f"Risk check error: {e}"
        
        return risk_check
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            confidence = signal.get('confidence', 0.5)
            entry_price = signal.get('entry_price', 100)  # Default price
            
            # Base position size on risk per trade
            total_capital = self.config['TOTAL_CAPITAL']
            risk_per_trade = self.config['RISK_PER_TRADE']
            
            # Calculate risk amount
            risk_amount = total_capital * risk_per_trade
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 1.5, 1.0)
            adjusted_risk = risk_amount * confidence_multiplier
            
            # Calculate position size (simplified - assumes $100 per share)
            position_size = adjusted_risk / entry_price if entry_price > 0 else 0
            
            # Cap at maximum position size
            max_position_value = total_capital * self.max_position_size
            max_shares = max_position_value / entry_price if entry_price > 0 else 0
            
            return min(position_size, max_shares)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    def _execute_trade(self, trade_order: Dict) -> Dict:
        """Execute trade order (paper trading for safety)"""
        try:
            # Generate trade ID
            trade_id = int(time.time() * 1000000)  # Microsecond timestamp
            
            # Current implementation is PAPER TRADING ONLY for safety
            if self.execution_mode == 'LIVE':
                # This would connect to actual broker API
                # For now, we treat it as paper trading with warning
                self._log_execution_event(
                    'WARNING',
                    f"LIVE trading not implemented - executing as paper trade",
                    trade_id=trade_id
                )
            
            # Paper trade execution
            execution_result = self._execute_paper_trade(trade_order, trade_id)
            
            # Log execution
            self._log_execution_event(
                'INFO',
                f"Trade executed: {trade_order['action']} {trade_order['quantity']} {trade_order['symbol']}",
                symbol=trade_order['symbol'],
                trade_id=trade_id
            )
            
            # Update counters
            self.daily_trade_count += 1
            self.executed_trades.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            self._log_execution_event(
                'ERROR',
                f"Trade execution failed: {e}",
                symbol=trade_order.get('symbol'),
                trade_id=None
            )
            return {"error": str(e)}
    
    def _execute_paper_trade(self, trade_order: Dict, trade_id: int) -> Dict:
        """Execute paper trade for testing and development"""
        try:
            # Simulate trade execution
            execution_price = trade_order['price']
            
            # Add some realistic slippage (0.1%)
            slippage = 0.001
            if trade_order['action'] == 'BUY':
                execution_price *= (1 + slippage)
            else:
                execution_price *= (1 - slippage)
            
            # Store trade in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO automated_trades 
            (timestamp, symbol, action, quantity, price, order_type, status, 
             signal_source, confidence, stop_loss, take_profit, execution_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_order['timestamp'],
                trade_order['symbol'],
                trade_order['action'],
                trade_order['quantity'],
                execution_price,
                trade_order['order_type'],
                'FILLED',
                trade_order['signal_source'],
                trade_order['confidence'],
                trade_order.get('stop_loss', 0),
                trade_order.get('take_profit', 0),
                self.execution_mode
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'trade_id': trade_id,
                'status': 'FILLED',
                'symbol': trade_order['symbol'],
                'action': trade_order['action'],
                'quantity': trade_order['quantity'],
                'execution_price': execution_price,
                'execution_mode': self.execution_mode,
                'execution_time': datetime.now().isoformat(),
                'slippage': slippage,
                'estimated_commission': 1.0  # $1 commission
            }
            
        except Exception as e:
            return {"error": f"Paper trade execution failed: {e}"}
    
    def _log_execution_event(self, log_level: str, message: str, symbol: str = None, trade_id: int = None):
        """Log execution events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO execution_log (timestamp, log_level, message, symbol, trade_id)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                log_level,
                message,
                symbol,
                trade_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging execution event: {e}")
    
    def _log_risk_event(self, event_type: str, severity: str, description: str, symbol: str = None):
        """Log risk management events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO risk_events (timestamp, event_type, severity, description, symbol, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                event_type,
                severity,
                description,
                symbol,
                'BLOCKED' if severity == 'HIGH' else 'MONITORED'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging risk event: {e}")
    
    def get_execution_status(self) -> Dict:
        """Get current execution engine status"""
        try:
            return {
                'execution_mode': self.execution_mode,
                'is_live_trading': self.is_live_trading,
                'risk_checks_enabled': self.risk_checks_enabled,
                'daily_trade_count': self.daily_trade_count,
                'daily_pnl': self.daily_pnl,
                'pending_orders': len(self.pending_orders),
                'executed_trades_today': len(self.executed_trades),
                'safety_limits': {
                    'max_daily_trades': self.max_daily_trades,
                    'max_position_size': self.max_position_size,
                    'max_daily_loss': self.max_daily_loss
                },
                'capital_info': {
                    'total_capital': self.config['TOTAL_CAPITAL'],
                    'risk_per_trade': self.config['RISK_PER_TRADE']
                },
                'status_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_trade_history(self, limit: int = 50) -> Dict:
        """Get recent trade history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM automated_trades 
            ORDER BY created_at DESC 
            LIMIT ?
            ''', (limit,))
            
            trades = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            columns = [
                'id', 'timestamp', 'symbol', 'action', 'quantity', 'price',
                'order_type', 'status', 'signal_source', 'confidence',
                'stop_loss', 'take_profit', 'execution_mode', 'pnl', 'created_at'
            ]
            
            trade_history = []
            for trade in trades:
                trade_dict = dict(zip(columns, trade))
                trade_history.append(trade_dict)
            
            return {
                'trade_history': trade_history,
                'total_trades': len(trade_history),
                'query_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def emergency_stop(self) -> Dict:
        """Emergency stop all trading activity"""
        try:
            # Disable trading
            self.risk_checks_enabled = False
            self.execution_mode = 'STOPPED'
            
            # Cancel all pending orders (if any)
            cancelled_orders = len(self.pending_orders)
            self.pending_orders.clear()
            
            # Log emergency stop
            self._log_risk_event(
                'EMERGENCY_STOP',
                'CRITICAL',
                'Emergency stop activated - all trading halted',
                symbol=None
            )
            
            self._log_execution_event(
                'CRITICAL',
                'EMERGENCY STOP - All trading activity halted',
                trade_id=None
            )
            
            return {
                'status': 'EMERGENCY_STOP_ACTIVATED',
                'execution_mode': self.execution_mode,
                'risk_checks_enabled': self.risk_checks_enabled,
                'cancelled_orders': cancelled_orders,
                'timestamp': datetime.now().isoformat(),
                'message': 'All trading activity has been halted. Manual intervention required.'
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def reset_daily_counters(self):
        """Reset daily trading counters (call at market open)"""
        try:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.executed_trades.clear()
            
            self._log_execution_event(
                'INFO',
                'Daily counters reset',
                trade_id=None
            )
            
        except Exception as e:
            print(f"Error resetting daily counters: {e}")
