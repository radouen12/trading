import pandas as pd
from datetime import datetime, timedelta
from config import Config
import math

class PortfolioManager:
    def __init__(self):
        self.config = Config()
        self.positions = {}  # Active positions
        self.cash = self.config.TOTAL_CAPITAL
        self.daily_pnl = 0
        self.total_pnl = 0
        
    def get_portfolio_stats(self):
        """Get current portfolio statistics"""
        total_value = self.cash
        position_count = len(self.positions)
        
        # Calculate total position value
        for symbol, pos in self.positions.items():
            total_value += pos['current_value']
        
        available_cash = self.cash
        portfolio_utilization = (total_value - available_cash) / self.config.TOTAL_CAPITAL
        
        return {
            'total_value': total_value,
            'available_cash': available_cash,
            'position_count': position_count,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'utilization': portfolio_utilization,
            'max_position_size': self.get_max_position_size(),
            'risk_capacity': self.get_risk_capacity()
        }
    
    def get_max_position_size(self):
        """Calculate maximum position size based on available capital"""
        available = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
        return available * self.config.MAX_POSITION_SIZE
    
    def get_risk_capacity(self):
        """Calculate how much risk we can take today"""
        max_daily_loss = self.config.TOTAL_CAPITAL * self.config.MAX_DAILY_LOSS
        current_loss = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        remaining_risk = max_daily_loss - current_loss
        return max(0, remaining_risk)
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, confidence_score, timeframe='daily'):
        """
        Calculate optimal position size based on:
        - Available capital
        - Risk per trade
        - Confidence score
        - Timeframe
        """
        try:
            # Base position size calculation
            available_capital = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
            
            # Risk-based sizing
            risk_per_share = abs(entry_price - stop_loss_price)
            risk_percentage = 0.02  # 2% base risk
            
            # Adjust risk based on confidence
            confidence_multiplier = confidence_score / 100
            adjusted_risk = risk_percentage * confidence_multiplier
            
            # Adjust risk based on timeframe
            timeframe_multipliers = {
                'daily': 1.0,
                'weekly': 1.5,
                'monthly': 2.0
            }
            adjusted_risk *= timeframe_multipliers.get(timeframe, 1.0)
            
            # Calculate position size
            risk_amount = available_capital * adjusted_risk
            shares = int(risk_amount / risk_per_share)
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
            
            # Final validation
            if position_value > self.cash:
                shares = int(self.cash * 0.9 / entry_price)  # Use 90% of available cash
                position_value = shares * entry_price
            
            return {
                'shares': shares,
                'position_value': position_value,
                'risk_amount': shares * risk_per_share,
                'risk_percentage': (shares * risk_per_share) / available_capital * 100,
                'confidence_adjusted': confidence_multiplier,
                'timeframe_adjusted': timeframe_multipliers.get(timeframe, 1.0)
            }
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return {
                'shares': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'confidence_adjusted': 1.0,
                'timeframe_adjusted': 1.0
            }
    
    def can_open_position(self, position_value, symbol_correlation_group=None):
        """Check if we can open a new position"""
        # Check cash availability
        if position_value > self.cash:
            return False, "Insufficient cash"
        
        # Check daily loss limit
        if self.get_risk_capacity() < position_value * 0.05:  # 5% of position as potential loss
            return False, "Daily loss limit approached"
        
        # Check position count
        if len(self.positions) >= 10:  # Max 10 positions
            return False, "Maximum positions reached"
        
        # Check correlation limits (simplified)
        correlation_count = sum(1 for pos in self.positions.values() 
                              if pos.get('correlation_group') == symbol_correlation_group)
        if correlation_count >= self.config.MAX_CORRELATION_POSITIONS:
            return False, "Too many correlated positions"
        
        return True, "OK"
    
    def add_position(self, symbol, shares, entry_price, stop_loss, target_price, 
                    confidence, timeframe, reasoning):
        """Add a new position to the portfolio"""
        position_value = shares * entry_price
        
        can_open, reason = self.can_open_position(position_value)
        if not can_open:
            return False, reason
        
        # Deduct cash
        self.cash -= position_value
        
        # Add position
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
            'unrealized_pnl': 0,
            'correlation_group': self.get_correlation_group(symbol)
        }
        
        return True, f"Position opened: {shares} shares of {symbol}"
    
    def update_position_prices(self, market_data):
        """Update current prices for all positions"""
        for symbol in self.positions:
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                position = self.positions[symbol]
                
                position['current_price'] = current_price
                position['current_value'] = position['shares'] * current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
    
    def get_correlation_group(self, symbol):
        """Simple correlation grouping"""
        if symbol in ['SPY', 'QQQ', 'IWM']:
            return 'indices'
        elif symbol in ['XLF', 'XLE', 'XLK', 'XLV']:
            return 'sectors'
        elif 'USD' in symbol:
            return 'crypto'
        elif '=X' in symbol:
            return 'forex'
        else:
            return 'stocks'
    
    def get_position_alerts(self):
        """Check for positions that need attention"""
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
            
            # Target reached alert
            if current_price >= pos['target_price']:
                alerts.append({
                    'type': 'TARGET_REACHED',
                    'symbol': symbol,
                    'message': f"Target reached for {symbol} at ${current_price:.2f}",
                    'urgency': 'MEDIUM'
                })
            
            # Large unrealized loss
            loss_pct = (pos['unrealized_pnl'] / (pos['shares'] * pos['entry_price'])) * 100
            if loss_pct < -10:
                alerts.append({
                    'type': 'LARGE_LOSS',
                    'symbol': symbol,
                    'message': f"Large loss ({loss_pct:.1f}%) on {symbol}",
                    'urgency': 'HIGH'
                })
        
        return alerts
    
    def get_suggestions_based_on_capital(self, market_data, analysis_results):
        """Generate trading suggestions based on available capital and risk"""
        suggestions = []
        
        available_capital = self.cash * (1 - self.config.RESERVE_CASH_RATIO)
        if available_capital < 100:  # Need at least $100 to trade
            return suggestions
        
        # Generate suggestions for different timeframes
        timeframes = ['daily', 'weekly', 'monthly']
        
        for timeframe in timeframes:
            max_suggestions = 3 if timeframe == 'daily' else 2
            timeframe_suggestions = []
            
            for symbol, data in market_data.items():
                if len(timeframe_suggestions) >= max_suggestions:
                    break
                
                # Skip if we already have this position
                if symbol in self.positions:
                    continue
                
                # Simple scoring based on price movement and volume
                price_change = data.get('change_pct', 0)
                volume = data.get('volume', 0)
                
                # Basic scoring algorithm (will be enhanced in Phase 2)
                score = 50  # Base score
                
                if price_change > 2:
                    score += 15
                elif price_change > 1:
                    score += 10
                
                if volume > 1000000:  # High volume
                    score += 10
                
                if score >= 70:  # Only high confidence suggestions
                    entry_price = data['price']
                    stop_loss = entry_price * 0.95  # 5% stop loss
                    target = entry_price * 1.08   # 8% target
                    
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
                            'reasoning': f"Strong momentum ({price_change:+.1f}%) with high volume",
                            'risk_reward': (target - entry_price) / (entry_price - stop_loss)
                        })
            
            suggestions.extend(timeframe_suggestions)
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)

# Example usage and testing
if __name__ == "__main__":
    portfolio = PortfolioManager()
    
    # Test position sizing
    test_entry = 150.0
    test_stop = 145.0
    test_confidence = 75
    
    size_info = portfolio.calculate_position_size(
        'AAPL', test_entry, test_stop, test_confidence, 'daily'
    )
    
    print("📊 Position Sizing Test:")
    print(f"Shares: {size_info['shares']}")
    print(f"Position Value: ${size_info['position_value']:.2f}")
    print(f"Risk Amount: ${size_info['risk_amount']:.2f}")
    print(f"Risk %: {size_info['risk_percentage']:.2f}%")
    
    print(f"\n💰 Portfolio Stats:")
    stats = portfolio.get_portfolio_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
