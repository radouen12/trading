"""
Risk Monitor - Phase 4
Advanced risk monitoring and alert system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import warnings

warnings.filterwarnings('ignore')

class RiskMonitor:
    """Advanced risk monitoring system with real-time alerts"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_daily_loss': 0.03,
            'max_position_size': 0.05,
            'max_portfolio_var': 0.02,
            'max_volatility': 0.05,
            'max_drawdown': 0.10,
            'min_cash_reserve': 0.20
        }
        
        self.portfolio_value = 0
        self.cash_balance = 0
        self._init_risk_db()
    
    def _init_risk_db(self):
        """Initialize risk monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                symbol TEXT,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                volatility REAL,
                var_1day REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing risk database: {e}")
    
    def monitor_portfolio_risk(self, portfolio_data: Dict) -> Dict:
        """Monitor portfolio-level risk metrics"""
        try:
            risk_metrics = self._calculate_portfolio_risk_metrics(portfolio_data)
            alerts = self._check_portfolio_risk_alerts(risk_metrics)
            self._store_portfolio_snapshot(portfolio_data, risk_metrics)
            
            return {
                'risk_metrics': risk_metrics,
                'risk_alerts': alerts,
                'risk_score': self._calculate_overall_risk_score(risk_metrics),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_portfolio_risk_metrics(self, portfolio_data: Dict) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            risk_metrics = {}
            
            total_value = portfolio_data.get('total_value', 0)
            cash_balance = portfolio_data.get('cash_balance', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            
            risk_metrics['total_value'] = total_value
            risk_metrics['cash_ratio'] = cash_balance / total_value if total_value > 0 else 1
            risk_metrics['daily_pnl_pct'] = daily_pnl / total_value if total_value > 0 else 0
            
            # Get historical returns
            returns = self._get_portfolio_returns()
            
            if len(returns) > 1:
                daily_vol = np.std(returns)
                risk_metrics['daily_volatility'] = daily_vol
                risk_metrics['annualized_volatility'] = daily_vol * np.sqrt(252)
                
                var_95 = np.percentile(returns, 5)
                risk_metrics['var_1day'] = abs(var_95)
                
                cumulative_returns = np.cumprod(1 + np.array(returns))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                risk_metrics['max_drawdown'] = abs(np.min(drawdowns))
                
                avg_return = np.mean(returns)
                risk_metrics['sharpe_ratio'] = (avg_return / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
            
            return risk_metrics
            
        except Exception as e:
            return {}
    
    def _get_portfolio_returns(self, periods: int = 30) -> List[float]:
        """Get historical portfolio returns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT total_value FROM portfolio_snapshots 
            ORDER BY created_at DESC 
            LIMIT ?
            ''', (periods + 1,))
            
            values = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if len(values) < 2:
                return []
            
            values.reverse()
            returns = []
            for i in range(1, len(values)):
                ret = (values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0
                returns.append(ret)
            
            return returns
            
        except Exception as e:
            return []
    
    def _check_portfolio_risk_alerts(self, risk_metrics: Dict) -> List[Dict]:
        """Check for portfolio risk threshold breaches"""
        alerts = []
        
        try:
            daily_loss = abs(risk_metrics.get('daily_pnl_pct', 0))
            if daily_loss > self.risk_thresholds['max_daily_loss']:
                alerts.append({
                    'type': 'DAILY_LOSS_LIMIT',
                    'severity': 'HIGH',
                    'current_value': daily_loss,
                    'threshold': self.risk_thresholds['max_daily_loss'],
                    'message': f"Daily loss {daily_loss:.2%} exceeds limit"
                })
            
            var_1day = risk_metrics.get('var_1day', 0)
            if var_1day > self.risk_thresholds['max_portfolio_var']:
                alerts.append({
                    'type': 'VAR_BREACH',
                    'severity': 'MEDIUM',
                    'current_value': var_1day,
                    'threshold': self.risk_thresholds['max_portfolio_var'],
                    'message': f"1-day VaR {var_1day:.2%} exceeds limit"
                })
            
            volatility = risk_metrics.get('daily_volatility', 0)
            if volatility > self.risk_thresholds['max_volatility']:
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'severity': 'MEDIUM',
                    'current_value': volatility,
                    'threshold': self.risk_thresholds['max_volatility'],
                    'message': f"Portfolio volatility {volatility:.2%} is high"
                })
            
            for alert in alerts:
                self._store_alert(alert)
            
            return alerts
            
        except Exception as e:
            return []
    
    def _store_alert(self, alert: Dict):
        """Store risk alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO risk_alerts 
            (timestamp, alert_type, severity, metric_name, current_value, 
             threshold_value, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                alert['type'],
                alert['severity'],
                alert['type'].lower(),
                alert['current_value'],
                alert['threshold'],
                alert['message']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing alert: {e}")
    
    def _store_portfolio_snapshot(self, portfolio_data: Dict, risk_metrics: Dict):
        """Store portfolio snapshot for historical analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO portfolio_snapshots 
            (timestamp, total_value, cash_balance, daily_pnl, volatility, 
             var_1day, max_drawdown, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                portfolio_data.get('total_value', 0),
                portfolio_data.get('cash_balance', 0),
                portfolio_data.get('daily_pnl', 0),
                risk_metrics.get('daily_volatility', 0),
                risk_metrics.get('var_1day', 0),
                risk_metrics.get('max_drawdown', 0),
                risk_metrics.get('sharpe_ratio', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing portfolio snapshot: {e}")
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict) -> Dict:
        """Calculate overall risk score (0-100, lower is better)"""
        try:
            volatility_score = min(risk_metrics.get('daily_volatility', 0) * 2000, 100)
            var_score = min(risk_metrics.get('var_1day', 0) * 5000, 100)
            drawdown_score = min(risk_metrics.get('max_drawdown', 0) * 1000, 100)
            cash_score = max(0, 100 - risk_metrics.get('cash_ratio', 0.2) * 500)
            
            overall_score = (
                volatility_score * 0.3 +
                var_score * 0.25 +
                drawdown_score * 0.25 +
                cash_score * 0.2
            )
            
            if overall_score < 30:
                risk_level = 'LOW'
            elif overall_score < 60:
                risk_level = 'MEDIUM'
            elif overall_score < 80:
                risk_level = 'HIGH'
            else:
                risk_level = 'VERY_HIGH'
            
            return {
                'overall_risk_score': overall_score,
                'risk_level': risk_level,
                'volatility_score': volatility_score,
                'var_score': var_score,
                'drawdown_score': drawdown_score,
                'liquidity_score': cash_score
            }
            
        except Exception as e:
            return {'overall_risk_score': 50, 'risk_level': 'UNKNOWN'}
    
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard"""
        try:
            active_alerts = self._get_active_alerts()
            risk_history = self._get_risk_history()
            latest_snapshot = self._get_latest_portfolio_snapshot()
            current_risk_scores = self._calculate_overall_risk_score(latest_snapshot)
            
            return {
                'active_alerts': active_alerts,
                'risk_history': risk_history,
                'latest_metrics': latest_snapshot,
                'risk_scores': current_risk_scores,
                'risk_thresholds': self.risk_thresholds,
                'dashboard_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get currently active risk alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT alert_type, severity, current_value, threshold_value, description, timestamp
            FROM risk_alerts 
            WHERE status = 'ACTIVE'
            ORDER BY timestamp DESC
            LIMIT 20
            ''')
            
            alerts = cursor.fetchall()
            conn.close()
            
            return [{
                'type': alert[0],
                'severity': alert[1],
                'current_value': alert[2],
                'threshold': alert[3],
                'description': alert[4],
                'timestamp': alert[5]
            } for alert in alerts]
            
        except Exception as e:
            return []
    
    def _get_risk_history(self, days: int = 7) -> List[Dict]:
        """Get risk metrics history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
            SELECT timestamp, volatility, var_1day, max_drawdown, sharpe_ratio
            FROM portfolio_snapshots 
            WHERE timestamp > ? 
            ORDER BY timestamp
            ''', (start_date,))
            
            history = cursor.fetchall()
            conn.close()
            
            return [{
                'timestamp': row[0],
                'volatility': row[1],
                'var_1day': row[2],
                'max_drawdown': row[3],
                'sharpe_ratio': row[4]
            } for row in history]
            
        except Exception as e:
            return []
    
    def _get_latest_portfolio_snapshot(self) -> Dict:
        """Get most recent portfolio metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT total_value, cash_balance, daily_pnl, volatility, var_1day, max_drawdown, sharpe_ratio
            FROM portfolio_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 1
            ''')
            
            snapshot = cursor.fetchone()
            conn.close()
            
            if snapshot:
                return {
                    'total_value': snapshot[0],
                    'cash_balance': snapshot[1],
                    'daily_pnl': snapshot[2],
                    'daily_volatility': snapshot[3],
                    'var_1day': snapshot[4],
                    'max_drawdown': snapshot[5],
                    'sharpe_ratio': snapshot[6],
                    'cash_ratio': snapshot[1] / snapshot[0] if snapshot[0] > 0 else 0,
                    'daily_pnl_pct': snapshot[2] / snapshot[0] if snapshot[0] > 0 else 0
                }
            
            return {}
            
        except Exception as e:
            return {}
