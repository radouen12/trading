import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import uuid

class PortfolioManager:
    def __init__(self):
        self.config = Config()
        self.positions = {}
        self.cash = self.config.TOTAL_CAPITAL
        self.daily_pnl = 0
        self.total_pnl = 0
        self._session_id = str(uuid.uuid4())
        
    def get_portfolio_stats(self):
        """Get current portfolio statistics"""
        total_value = self.cash
        
        for symbol, pos in self.positions.items():
            if pos and 'current_value' in pos:
                total_value += pos['current_value']
        
        available_cash = self.cash
        position_count = len(self.positions)
        
        utilization = 0
        if self.config.TOTAL_CAPITAL > 0:
            utilization = (total_value - available_cash) / self.config.TOTAL_CAPITAL
        
        return {
            'total_value': total_value,
            'available_cash': available_cash,
            'position_count': position_count,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'utilization': max(0, min(1, utilization)),
            'max_position_size': self.get_max_position_size(),
            'risk_capacity': self.get_risk_capacity(),
            'session_id': self._session_id,
            'last_update': datetime.now()
        }
    
    def get_max_position_size(self):
        """Calculate maximum position size"""
        if self.cash <= 0:
            return 0
        
        available = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
        return available * self.config.MAX_POSITION_SIZE
    
    def get_risk_capacity(self):
        """Calculate remaining risk capacity"""
        max_daily_loss = self.config.TOTAL_CAPITAL * self.config.MAX_DAILY_LOSS
        current_loss = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        return max(0, max_daily_loss - current_loss)
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, confidence_score, timeframe='daily'):
        """Calculate optimal position size"""
        try:
            entry_price = float(entry_price)
            stop_loss_price = float(stop_loss_price)
            confidence_score = max(0, min(100, float(confidence_score)))
            
            if self.cash <= 0:
                return self._get_zero_position_result("No cash available")
            
            available_capital = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
            
            if available_capital <= 0:
                return self._get_zero_position_result("No available capital")
            
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share <= 0.001:
                return self._get_zero_position_result("Risk too small")
            
            # Base risk 2%
            risk_percentage = 0.02
            
            # Adjust for confidence
            confidence_multiplier = confidence_score / 100
            adjusted_risk = risk_percentage * confidence_multiplier
            
            # Adjust for timeframe
            timeframe_multipliers = {
                'daily': 1.0,
                'weekly': 1.2,
                'monthly': 1.5
            }
            timeframe_multiplier = timeframe_multipliers.get(timeframe.lower(), 1.0)
            adjusted_risk *= timeframe_multiplier
            
            # Cap at 5%
            adjusted_risk = min(adjusted_risk, 0.05)
            
            risk_amount = available_capital * adjusted_risk
            shares = int(risk_amount / risk_per_share)
            
            if shares <= 0:
                return self._get_zero_position_result("Calculated shares is zero")
            
            position_value = shares * entry_price
            
            # Apply position limits
            max_position = available_capital * self.config.MAX_POSITION_SIZE
            min_position = available_capital * self.config.MIN_POSITION_SIZE
            
            if position_value > max_position:
                shares = int(max_position / entry_price)
                position_value = shares * entry_price
            elif position_value < min_position:
                shares = int(min_position / entry_price)
                position_value = shares * entry_price
            
            if position_value > self.cash:
                shares = int(self.cash * 0.9 / entry_price)
                position_value = shares * entry_price
            
            if shares <= 0:
                return self._get_zero_position_result("Final validation failed")
            
            actual_risk_amount = shares * risk_per_share
            actual_risk_percentage = (actual_risk_amount / available_capital * 100) if available_capital > 0 else 0
            
            return {
                'shares': shares,
                'position_value': round(position_value, 2),
                'risk_amount': round(actual_risk_amount, 2),
                'risk_percentage': round(actual_risk_percentage, 2),
                'confidence_adjusted': round(confidence_multiplier, 2),
                'timeframe_adjusted': timeframe_multiplier,
                'validation_passed': True,
                'available_capital': round(available_capital, 2)
            }
            
        except Exception as e:
            return self._get_zero_position_result(f"Error: {e}")
    
    def _get_zero_position_result(self, reason="Unknown"):
        """Return zero position result"""
        return {
            'shares': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_percentage': 0,
            'confidence_adjusted': 0,
            'timeframe_adjusted': 0,
            'validation_passed': False,
            'reason': reason
        }
    
    def can_open_position(self, position_value, symbol_correlation_group=None):
        """Check if position can be opened"""
        try:
            position_value = float(position_value)
            
            if position_value > self.cash:
                return False, f"Insufficient cash: need ${position_value:,.2f}, have ${self.cash:,.2f}"
            
            risk_capacity = self.get_risk_capacity()
            position_risk = position_value * 0.05
            if risk_capacity < position_risk:
                return False, f"Daily loss limit exceeded"
            
            if len(self.positions) >= 10:
                return False, f"Maximum positions reached"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def add_position(self, symbol, shares, entry_price, stop_loss, target_price, 
                    confidence, timeframe, reasoning):
        """Add new position"""
        try:
            shares = int(shares)
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            target_price = float(target_price)
            confidence = float(confidence)
            
            if shares <= 0 or entry_price <= 0:
                return False, "Invalid parameters"
            
            symbol = symbol.strip().upper()
            position_value = shares * entry_price
            
            if symbol in self.positions:
                return False, f"Position for {symbol} already exists"
            
            can_open, reason = self.can_open_position(position_value)
            if not can_open:
                return False, reason
            
            if position_value > self.cash:
                return False, f"Insufficient cash"
            
            # Deduct cash
            self.cash -= position_value
            
            # Create position
            self.positions[symbol] = {
                'symbol': symbol,
                'shares': shares,
                'entry_price': entry_price,
                'current_price': entry_price,
                'current_value': position_value,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'confidence': confidence,
                'timeframe': timeframe,
                'reasoning': reasoning,
                'entry_time': datetime.now(),
                'unrealized_pnl': 0.0,
                'position_id': str(uuid.uuid4())
            }
            
            return True, f"Position opened: {shares} shares of {symbol}"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def update_position_prices(self, market_data):
        """Update position prices"""
        try:
            if not market_data:
                return
            
            for symbol in list(self.positions.keys()):
                if symbol not in self.positions:
                    continue
                
                position = self.positions[symbol]
                symbol_data = market_data.get(symbol)
                
                if not symbol_data:
                    continue
                
                current_price = symbol_data.get('price', 0)
                if current_price <= 0:
                    continue
                
                shares = position.get('shares', 0)
                entry_price = position.get('entry_price', current_price)
                
                new_current_value = shares * current_price
                new_unrealized_pnl = (current_price - entry_price) * shares
                
                position['current_price'] = current_price
                position['current_value'] = new_current_value
                position['unrealized_pnl'] = new_unrealized_pnl
                position['last_updated'] = datetime.now()
                
        except Exception as e:
            print(f"Error updating prices: {e}")
    
    def get_position_alerts(self):
        """Get position alerts"""
        alerts = []
        
        for symbol, pos in self.positions.items():
            current_price = pos['current_price']
            
            # Stop loss alert
            if current_price <= pos['stop_loss']:
                alerts.append({
                    'type': 'STOP_LOSS',
                    'symbol': symbol,
                    'message': f"Stop loss triggered for {symbol} at ${current_price:.2f}",
                    'urgency': 'HIGH'
                })
            
            # Target reached
            if current_price >= pos['target_price']:
                alerts.append({
                    'type': 'TARGET_REACHED',
                    'symbol': symbol,
                    'message': f"Target reached for {symbol} at ${current_price:.2f}",
                    'urgency': 'MEDIUM'
                })
        
        return alerts
    
    def get_suggestions_based_on_capital(self, market_data, analysis_results):
        """Generate basic suggestions"""
        suggestions = []
        
        if self.cash <= 0:
            return suggestions
        
        available_capital = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
        
        if available_capital < 100:
            return suggestions
        
        timeframes = ['daily', 'weekly', 'monthly']
        
        for timeframe in timeframes:
            max_suggestions = 3 if timeframe == 'daily' else 2
            timeframe_suggestions = []
            
            for symbol, data in market_data.items():
                if len(timeframe_suggestions) >= max_suggestions:
                    break
                
                if symbol in self.positions:
                    continue
                
                price_change = data.get('change_pct', 0)
                volume = data.get('volume', 0)
                
                score = 50
                
                if price_change > 2:
                    score += 15
                elif price_change > 1:
                    score += 10
                
                if volume > 1000000:
                    score += 10
                
                if score >= 70:
                    entry_price = data['price']
                    stop_loss = entry_price * 0.95
                    target = entry_price * 1.08
                    
                    position_info = self.calculate_position_size(
                        symbol, entry_price, stop_loss, score, timeframe
                    )
                    
                    if position_info['shares'] > 0:
                        timeframe_suggestions.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'entry_price': entry_price,
                            'target_price': target,
                            'stop_loss': stop_loss,
                            'position_size': position_info['position_value'],
                            'shares': position_info['shares'],
                            'confidence': score,
                            'timeframe': timeframe,
                            'reasoning': f"Strong momentum ({price_change:+.1f}%)",
                            'risk_reward': (target - entry_price) / (entry_price - stop_loss)
                        })
            
            suggestions.extend(timeframe_suggestions)
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
