import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import math
import threading
from copy import deepcopy
import uuid

class PortfolioManager:
    def __init__(self):
        self.config = Config()
        self.positions = {}  # Active positions
        self.cash = self.config.TOTAL_CAPITAL
        self.daily_pnl = 0
        self.total_pnl = 0
        
        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._session_id = str(uuid.uuid4())  # Unique session identifier
        
        # State validation
        self._last_update = datetime.now()
        self._update_counter = 0
        
    def get_portfolio_stats(self):
        """Get current portfolio statistics with thread safety"""
        with self._lock:
            try:
                # Validate cash value
                if self.cash is None:
                    self.cash = self.config.TOTAL_CAPITAL
                    print("⚠️ Cash value was None, reset to initial capital")
                
                total_value = max(0, self.cash)  # Ensure non-negative
                position_count = len(self.positions)
                
                # Calculate total position value with validation
                for symbol, pos in self.positions.items():
                    if pos and isinstance(pos, dict) and 'current_value' in pos:
                        current_value = pos.get('current_value', 0)
                        if isinstance(current_value, (int, float)) and current_value >= 0:
                            total_value += current_value
                        else:
                            print(f"⚠️ Invalid current_value for {symbol}: {current_value}")
                
                available_cash = max(0, self.cash)
                portfolio_utilization = 0
                if self.config.TOTAL_CAPITAL > 0:
                    portfolio_utilization = min(1.0, (total_value - available_cash) / self.config.TOTAL_CAPITAL)
                
                # Update state tracking
                self._update_counter += 1
                self._last_update = datetime.now()
                
                return {
                    'total_value': total_value,
                    'available_cash': available_cash,
                    'position_count': position_count,
                    'daily_pnl': self.daily_pnl,
                    'total_pnl': self.total_pnl,
                    'utilization': max(0, min(1, portfolio_utilization)),  # Clamp to 0-1
                    'max_position_size': self.get_max_position_size(),
                    'risk_capacity': self.get_risk_capacity(),
                    'session_id': self._session_id,
                    'last_update': self._last_update,
                    'update_counter': self._update_counter
                }
            except Exception as e:
                print(f"❌ Error getting portfolio stats: {e}")
                return {
                    'total_value': 0,
                    'available_cash': 0,
                    'position_count': 0,
                    'daily_pnl': 0,
                    'total_pnl': 0,
                    'utilization': 0,
                    'max_position_size': 0,
                    'risk_capacity': 0,
                    'session_id': self._session_id,
                    'last_update': self._last_update,
                    'update_counter': self._update_counter,
                    'error': str(e)
                }
    
    def get_max_position_size(self):
        """Calculate maximum position size based on available capital"""
        with self._lock:
            if self.cash is None or self.cash <= 0:
                return 0
            
            # Validate configuration values
            reserve_ratio = getattr(self.config, 'RESERVE_CASH_RATIO', 0.2)
            max_position_ratio = getattr(self.config, 'MAX_POSITION_SIZE', 0.05)
            
            # Ensure ratios are valid (0-1 range)
            reserve_ratio = max(0, min(1, reserve_ratio))
            max_position_ratio = max(0, min(1, max_position_ratio))
            
            available = self.cash * (1 - reserve_ratio)
            return max(0, available * max_position_ratio)
    
    def get_risk_capacity(self):
        """Calculate how much risk we can take today"""
        with self._lock:
            # Validate configuration
            total_capital = getattr(self.config, 'TOTAL_CAPITAL', 10000)
            max_daily_loss_ratio = getattr(self.config, 'MAX_DAILY_LOSS', 0.03)
            
            # Ensure values are positive and ratios are valid
            total_capital = max(0, total_capital)
            max_daily_loss_ratio = max(0, min(1, max_daily_loss_ratio))
            
            max_daily_loss = total_capital * max_daily_loss_ratio
            
            # Validate daily_pnl
            daily_pnl = self.daily_pnl if self.daily_pnl is not None else 0
            current_loss = abs(daily_pnl) if daily_pnl < 0 else 0
            
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
        with self._lock:
            try:
                # Input validation
                if not self._validate_position_inputs(entry_price, stop_loss_price, confidence_score):
                    return self._get_zero_position_result("Invalid input parameters")
                
                # Normalize confidence score
                confidence_score = max(0, min(100, float(confidence_score)))  # Ensure float and clamp
                
                # Validate cash before proceeding
                if self.cash is None or self.cash <= 0:
                    return self._get_zero_position_result("No available cash")
                
                # Base position size calculation with safe config access
                reserve_ratio = getattr(self.config, 'RESERVE_CASH_RATIO', 0.2)
                reserve_ratio = max(0, min(1, reserve_ratio))  # Validate ratio
                available_capital = self.cash * (1 - reserve_ratio)
                
                if available_capital <= 0:
                    return self._get_zero_position_result("No available capital")
                
                # Risk-based sizing with validation
                risk_per_share = abs(float(entry_price) - float(stop_loss_price))
                if risk_per_share <= 0.001:  # Minimum meaningful risk
                    return self._get_zero_position_result("Risk per share too small")
                
                risk_percentage = 0.02  # 2% base risk
                
                # Adjust risk based on confidence (with bounds checking)
                confidence_multiplier = max(0.1, min(2.0, confidence_score / 100))  # 10%-200% range
                adjusted_risk = risk_percentage * confidence_multiplier
                
                # Adjust risk based on timeframe (with validation)
                timeframe_multipliers = {
                    'daily': 1.0,
                    'weekly': 1.2,  # Reduced from 1.5 for safety
                    'monthly': 1.5   # Reduced from 2.0 for safety
                }
                timeframe_multiplier = timeframe_multipliers.get(str(timeframe).lower(), 1.0)
                adjusted_risk *= timeframe_multiplier
                
                # Ensure risk doesn't exceed maximum
                max_risk = 0.05  # 5% maximum risk per trade
                adjusted_risk = min(adjusted_risk, max_risk)
                
                # Calculate position size
                risk_amount = available_capital * adjusted_risk
                shares = max(0, int(risk_amount / risk_per_share))
                
                if shares <= 0:
                    return self._get_zero_position_result("Calculated shares is zero")
                
                position_value = shares * float(entry_price)
                
                # Apply position limits with validation
                max_position_ratio = getattr(self.config, 'MAX_POSITION_SIZE', 0.05)
                min_position_ratio = getattr(self.config, 'MIN_POSITION_SIZE', 0.01)
                
                max_position = available_capital * max(0, min(1, max_position_ratio))
                min_position = available_capital * max(0, min(1, min_position_ratio))
                
                # Bounds checking
                if position_value > max_position and max_position > 0:
                    shares = max(0, int(max_position / float(entry_price)))
                    position_value = shares * float(entry_price)
                elif position_value < min_position and min_position <= available_capital:
                    shares = max(0, int(min_position / float(entry_price)))
                    position_value = shares * float(entry_price)
                
                # Final cash validation with safety margin
                if position_value > self.cash * 0.95:  # Keep 5% buffer
                    shares = max(0, int(self.cash * 0.9 / float(entry_price)))
                    position_value = shares * float(entry_price)
                
                # Final validation
                if shares <= 0 or position_value <= 0:
                    return self._get_zero_position_result("Final validation failed")
                
                # Calculate actual risk metrics
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
                
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"❌ Error calculating position size: {e}")
                return self._get_zero_position_result(f"Calculation error: {e}")
            except Exception as e:
                print(f"❌ Unexpected error calculating position size: {e}")
                return self._get_zero_position_result(f"Unexpected error: {e}")
    
    def _validate_position_inputs(self, entry_price, stop_loss_price, confidence_score):
        """Validate position calculation inputs with comprehensive checks"""
        try:
            # Convert to float and validate entry price
            try:
                entry_price = float(entry_price)
                if entry_price <= 0 or not np.isfinite(entry_price):
                    print(f"⚠️ Invalid entry price: {entry_price}")
                    return False
            except (ValueError, TypeError):
                print(f"⚠️ Cannot convert entry price to float: {entry_price}")
                return False
            
            # Convert to float and validate stop loss price
            try:
                stop_loss_price = float(stop_loss_price)
                if stop_loss_price <= 0 or not np.isfinite(stop_loss_price):
                    print(f"⚠️ Invalid stop loss price: {stop_loss_price}")
                    return False
            except (ValueError, TypeError):
                print(f"⚠️ Cannot convert stop loss price to float: {stop_loss_price}")
                return False
            
            # Convert to float and validate confidence score
            try:
                confidence_score = float(confidence_score)
                if not np.isfinite(confidence_score):
                    print(f"⚠️ Invalid confidence score: {confidence_score}")
                    return False
            except (ValueError, TypeError):
                print(f"⚠️ Cannot convert confidence score to float: {confidence_score}")
                return False
            
            # Price range validation
            if entry_price > 1000000:  # Unreasonably high price
                print(f"⚠️ Entry price too high: ${entry_price:,.2f}")
                return False
            
            if stop_loss_price > 1000000:  # Unreasonably high price
                print(f"⚠️ Stop loss price too high: ${stop_loss_price:,.2f}")
                return False
            
            # Logical validation with safer division
            try:
                price_diff_pct = abs(entry_price - stop_loss_price) / entry_price * 100
                
                if price_diff_pct > 50:  # Stop loss more than 50% away
                    print(f"⚠️ Stop loss too far from entry price: {price_diff_pct:.1f}%")
                    return False
                
                if price_diff_pct < 0.05:  # Stop loss too close (less than 0.05%)
                    print(f"⚠️ Stop loss too close to entry price: {price_diff_pct:.1f}%")
                    return False
                    
            except ZeroDivisionError:
                print("⚠️ Cannot calculate price difference percentage: entry price is zero")
                return False
            
            # Additional sanity checks
            if abs(entry_price - stop_loss_price) < 0.001:  # Very small absolute difference
                print(f"⚠️ Price difference too small: ${abs(entry_price - stop_loss_price):.4f}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating inputs: {e}")
            return False
    
    def _get_zero_position_result(self, reason="Unknown"):
        """Return zero position result with reason"""
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
        """Check if we can open a new position with thread safety"""
        with self._lock:
            try:
                # Validate inputs
                if position_value is None or position_value <= 0:
                    return False, "Invalid position value"
                
                # Convert to float for consistency
                try:
                    position_value = float(position_value)
                except (ValueError, TypeError):
                    return False, "Position value must be numeric"
                
                if self.cash is None:
                    self.cash = self.config.TOTAL_CAPITAL
                    print("⚠️ Cash was None, reset to initial capital")
                
                # Check cash availability with buffer
                if position_value > self.cash * 0.95:  # Keep 5% buffer
                    return False, f"Insufficient cash: need ${position_value:,.2f}, have ${self.cash:,.2f}"
                
                # Check daily loss limit
                risk_capacity = self.get_risk_capacity()
                position_risk = position_value * 0.05  # 5% of position as potential loss
                if risk_capacity < position_risk:
                    return False, f"Daily loss limit: risk capacity ${risk_capacity:,.2f} < position risk ${position_risk:,.2f}"
                
                # Check position count with configurable limit
                max_positions = getattr(self.config, 'MAX_POSITIONS', 10)
                if len(self.positions) >= max_positions:
                    return False, f"Maximum positions reached: {len(self.positions)}/{max_positions}"
                
                # Check correlation limits
                max_correlation = getattr(self.config, 'MAX_CORRELATION_POSITIONS', 3)
                if symbol_correlation_group:
                    correlation_count = sum(
                        1 for pos in self.positions.values() 
                        if isinstance(pos, dict) and pos.get('correlation_group') == symbol_correlation_group
                    )
                    if correlation_count >= max_correlation:
                        return False, f"Too many correlated positions: {correlation_count}/{max_correlation} in {symbol_correlation_group}"
                
                return True, "OK"
                
            except Exception as e:
                print(f"❌ Error checking position eligibility: {e}")
                return False, f"Error: {e}"
    
    def add_position(self, symbol, shares, entry_price, stop_loss, target_price, 
                    confidence, timeframe, reasoning):
        """Add a new position to the portfolio with thread safety"""
        with self._lock:
            try:
                # Input validation with type conversion
                try:
                    shares = int(shares)
                    entry_price = float(entry_price)
                    stop_loss = float(stop_loss)
                    target_price = float(target_price)
                    confidence = float(confidence)
                except (ValueError, TypeError) as e:
                    return False, f"Invalid numeric input: {e}"
                
                # Validate position data
                if shares <= 0 or entry_price <= 0:
                    return False, "Invalid position parameters: shares and entry_price must be positive"
                
                if stop_loss <= 0 or target_price <= 0:
                    return False, "Invalid price parameters: stop_loss and target_price must be positive"
                
                # Validate symbol
                if not symbol or not isinstance(symbol, str) or len(symbol.strip()) == 0:
                    return False, "Invalid symbol"
                
                symbol = symbol.strip().upper()  # Normalize symbol
                
                position_value = shares * entry_price
                
                # Check for duplicate positions
                if symbol in self.positions:
                    existing_pos = self.positions[symbol]
                    return False, f"Position for {symbol} already exists with {existing_pos.get('shares', 0)} shares"
                
                # Check position eligibility
                correlation_group = self.get_correlation_group(symbol)
                can_open, reason = self.can_open_position(position_value, correlation_group)
                if not can_open:
                    return False, reason
                
                # Validate cash availability one more time
                if self.cash is None:
                    self.cash = self.config.TOTAL_CAPITAL
                
                if position_value > self.cash:
                    return False, f"Insufficient cash: need ${position_value:,.2f}, have ${self.cash:,.2f}"
                
                # Deduct cash
                self.cash -= position_value
                
                # Validate reasoning and timeframe
                reasoning = str(reasoning).strip() if reasoning else "No reasoning provided"
                timeframe = str(timeframe).strip() if timeframe else "daily"
                
                # Create position with comprehensive data
                current_time = datetime.now()
                position_id = str(uuid.uuid4())
                
                self.positions[symbol] = {
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': entry_price,
                    'current_price': entry_price,  # Initialize with entry price
                    'current_value': position_value,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'confidence': max(0, min(100, confidence)),  # Clamp confidence
                    'timeframe': timeframe,
                    'reasoning': reasoning,
                    'entry_time': current_time,
                    'unrealized_pnl': 0.0,
                    'correlation_group': correlation_group,
                    'position_id': position_id,
                    'last_updated': current_time,
                    'entry_session': self._session_id
                }
                
                # Update state tracking
                self._update_counter += 1
                self._last_update = current_time
                
                print(f"✓ Position added: {shares} shares of {symbol} at ${entry_price:.2f}")
                return True, f"Position opened: {shares} shares of {symbol} (${position_value:,.2f})"
                
            except Exception as e:
                print(f"❌ Error adding position: {e}")
                # Rollback cash if we deducted it
                if 'position_value' in locals() and hasattr(self, 'cash'):
                    try:
                        self.cash += position_value
                        print(f"♾️ Rolled back cash deduction of ${position_value:,.2f}")
                    except:
                        pass
                return False, f"Error: {e}"
    
    def update_position_prices(self, market_data):
        """Update current prices for all positions with thread safety"""
        with self._lock:
            try:
                if not market_data or not isinstance(market_data, dict):
                    print("⚠️ No valid market data provided for price update")
                    return
                
                updated_count = 0
                errors = []
                
                # Use list() to avoid modification during iteration
                for symbol in list(self.positions.keys()):
                    try:
                        if symbol not in self.positions:
                            continue  # Position may have been removed
                        
                        position = self.positions[symbol]
                        if not isinstance(position, dict):
                            errors.append(f"Invalid position data for {symbol}")
                            continue
                        
                        # Get market data for this symbol
                        symbol_data = market_data.get(symbol)
                        if not symbol_data or not isinstance(symbol_data, dict):
                            continue  # No data available for this symbol
                        
                        # Extract and validate current price
                        current_price = symbol_data.get('price', 0)
                        
                        # Validate price data
                        try:
                            current_price = float(current_price)
                            if current_price <= 0 or not np.isfinite(current_price):
                                continue  # Invalid price, skip update
                        except (ValueError, TypeError):
                            continue  # Cannot convert to float
                        
                        # Sanity check: price shouldn't change more than 50% in one update
                        old_price = position.get('current_price', current_price)
                        if old_price > 0:
                            price_change_pct = abs(current_price - old_price) / old_price * 100
                            if price_change_pct > 50:
                                print(f"⚠️ Suspicious price change for {symbol}: {price_change_pct:.1f}%")
                                continue  # Skip suspicious price updates
                        
                        # Update position with new price data
                        shares = position.get('shares', 0)
                        entry_price = position.get('entry_price', current_price)
                        
                        # Validate shares and entry_price
                        try:
                            shares = int(shares)
                            entry_price = float(entry_price)
                            if shares <= 0 or entry_price <= 0:
                                errors.append(f"Invalid position data for {symbol}: shares={shares}, entry={entry_price}")
                                continue
                        except (ValueError, TypeError):
                            errors.append(f"Cannot convert position data for {symbol}")
                            continue
                        
                        # Calculate new values
                        new_current_value = shares * current_price
                        new_unrealized_pnl = (current_price - entry_price) * shares
                        
                        # Update position data
                        position['current_price'] = current_price
                        position['current_value'] = new_current_value
                        position['unrealized_pnl'] = new_unrealized_pnl
                        position['last_updated'] = datetime.now()
                        
                        # Add additional market data if available
                        if 'volume' in symbol_data:
                            position['last_volume'] = symbol_data['volume']
                        if 'change_pct' in symbol_data:
                            position['daily_change_pct'] = symbol_data['change_pct']
                        
                        updated_count += 1
                        
                    except Exception as e:
                        errors.append(f"Error updating {symbol}: {e}")
                        continue
                
                # Update state tracking
                if updated_count > 0:
                    self._update_counter += 1
                    self._last_update = datetime.now()
                
                # Log results
                if updated_count > 0:
                    print(f"✓ Updated prices for {updated_count} positions")
                
                if errors:
                    print(f"⚠️ Price update errors: {len(errors)}")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"  - {error}")
                
            except Exception as e:
                print(f"❌ Error updating position prices: {e}")
    
    def get_correlation_group(self, symbol):
        """Determine correlation group for a symbol with validation"""
        try:
            if not symbol or not isinstance(symbol, str):
                return 'unknown'
            
            symbol = symbol.upper().strip()
            
            # Define correlation groups
            if symbol in ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']:
                return 'indices'
            elif symbol.startswith('XL'):  # Sector ETFs
                return 'sectors'
            elif 'USD' in symbol or symbol.endswith('-USD'):
                return 'crypto'
            elif '=X' in symbol or symbol.endswith('=X'):
                return 'forex'
            elif symbol in ['GLD', 'SLV', 'USO', 'UNG']:  # Commodities
                return 'commodities'
            elif len(symbol) <= 5 and symbol.isalpha():  # Likely stock
                return 'stocks'
            else:
                return 'other'
                
        except Exception as e:
            print(f"⚠️ Error determining correlation group for {symbol}: {e}")
            return 'unknown'
    
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
            entry_value = pos['shares'] * pos['entry_price']
            if entry_value > 0:  # Prevent division by zero
                loss_pct = (pos['unrealized_pnl'] / entry_value) * 100
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
        
        # Validate cash and config values
        if self.cash is None or self.cash <= 0:
            return suggestions
            
        reserve_ratio = getattr(self.config, 'RESERVE_CASH_RATIO', 0.2)  # Default to 20% if not set
        if not (0 <= reserve_ratio <= 1):
            reserve_ratio = 0.2  # Fallback to safe default
            
        available_capital = self.cash * (1 - reserve_ratio)
        min_trade_amount = getattr(self.config, 'MIN_TRADE_AMOUNT', 100)  # Configurable minimum
        
        if available_capital < min_trade_amount:
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
