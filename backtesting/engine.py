"""
Backtesting Engine - Phase 3
Core backtesting functionality for strategy testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from dataclasses import dataclass

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    trade_type: str  # 'BUY' or 'SELL'
    pnl: float
    pnl_percent: float
    strategy: str
    confidence: float

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    trade_type: str
    strategy: str
    confidence: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

class BacktestEngine:
    """
    Advanced Backtesting Engine for strategy validation
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
    def run_backtest(self, strategy_signals: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """
        Run comprehensive backtest on strategy signals
        
        Args:
            strategy_signals: DataFrame with columns ['symbol', 'date', 'action', 'price', 'confidence']
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dict with backtest results and performance metrics
        """
        
        # Validate date inputs
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            return {"error": f"Invalid date format: {e}"}
            
        if start_dt >= end_dt:
            return {"error": "Start date must be before end date"}
            
        if strategy_signals.empty:
            return {"error": "No signals provided for backtesting"}
        
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Ensure date column is datetime and timezone-aware
        signals = strategy_signals.copy()
        signals['date'] = pd.to_datetime(signals['date'])
        
        # Filter signals within date range
        signals = signals[(signals['date'] >= start_dt) & (signals['date'] <= end_dt)]
        
        if signals.empty:
            return {"error": "No signals found in the specified date range"}
        
        # Sort signals by date
        signals = signals.sort_values('date')
        
        print(f"🔄 Running backtest from {start_date} to {end_date}")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"📊 Processing {len(signals)} signals...")
        print(f"📅 Date range: {signals['date'].min()} to {signals['date'].max()}")
        
        # Process each signal
        for idx, signal in signals.iterrows():
            self._process_signal(signal)
            
            # Update equity curve daily
            signal_date = pd.to_datetime(signal['date']).normalize()  # Normalize to date only
            
            if (len(self.equity_curve) == 0 or 
                signal_date != pd.to_datetime(self.equity_curve[-1][0]).normalize()):
                
                total_equity = self._calculate_total_equity(signal['date'])
                self.equity_curve.append((signal['date'], total_equity))
                
                # Calculate daily return
                if len(self.equity_curve) > 1:
                    prev_equity = self.equity_curve[-2][1]
                    if prev_equity > 0:  # Prevent division by zero
                        daily_return = (total_equity - prev_equity) / prev_equity
                        self.daily_returns.append(daily_return)
        
        # Close all remaining positions at end date
        self._close_all_positions(pd.to_datetime(end_date))
        
        # Calculate final metrics
        results = self._calculate_performance_metrics()
        
        print(f"✅ Backtest complete!")
        print(f"📈 Total trades: {len(self.trades)}")
        print(f"💰 Final capital: ${self.current_capital:,.2f}")
        print(f"📊 Total return: {((self.current_capital / self.initial_capital) - 1) * 100:.2f}%")
        
        return results
    
    def _process_signal(self, signal: pd.Series):
        """Process a single trading signal"""
        
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        date = signal['date']
        confidence = signal.get('confidence', 50)
        
        if action == 'BUY':
            self._open_position(symbol, price, date, 'BUY', confidence)
        elif action == 'SELL':
            self._close_position(symbol, price, date)
    
    def _open_position(self, symbol: str, price: float, date: datetime, trade_type: str, confidence: float):
        """Open a new position"""
        
        # Position sizing based on confidence and available capital
        position_size = self._calculate_position_size(confidence)
        
        if position_size > self.current_capital * 0.95:  # Leave some cash for fees
            return  # Not enough capital
            
        quantity = int(position_size / price)
        if quantity == 0:
            return
            
        actual_cost = quantity * price * (1 + self.commission)
        
        if actual_cost > self.current_capital:
            return  # Not enough capital
            
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            trade_type=trade_type,
            strategy="AI_Enhanced",
            confidence=confidence,
            current_price=price
        )
        
        self.positions.append(position)
        self.current_capital -= actual_cost
        
    def _close_position(self, symbol: str, price: float, date: datetime):
        """Close position for given symbol"""
        
        positions_to_close = [p for p in self.positions if p.symbol == symbol]
        
        for position in positions_to_close:
            # Calculate P&L
            if position.trade_type == 'BUY':
                pnl = (price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - price) * position.quantity
                
            # Account for commission
            pnl -= (position.quantity * position.entry_price * self.commission)
            pnl -= (position.quantity * price * self.commission)
            
            pnl_percent = pnl / (position.quantity * position.entry_price) * 100
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=price,
                quantity=position.quantity,
                trade_type=position.trade_type,
                pnl=pnl,
                pnl_percent=pnl_percent,
                strategy=position.strategy,
                confidence=position.confidence
            )
            
            self.trades.append(trade)
            self.current_capital += (position.quantity * price * (1 - self.commission))
            
            # Remove position
            self.positions.remove(position)
    
    def _close_all_positions(self, end_date: datetime):
        """Close all remaining positions at end of backtest"""
        
        for position in self.positions.copy():
            # Get final price for symbol
            try:
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(start=end_date - timedelta(days=5), end=end_date + timedelta(days=1))
                if not hist.empty:
                    final_price = hist['Close'].iloc[-1]
                    self._close_position(position.symbol, final_price, end_date)
            except:
                # If can't get price, close at entry price (no gain/loss)
                self._close_position(position.symbol, position.entry_price, end_date)
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and capital"""
        
        # Base allocation: 2-8% of capital based on confidence
        min_allocation = 0.02  # 2%
        max_allocation = 0.08  # 8%
        
        # Scale allocation by confidence (50-100 -> 0-1)
        confidence_factor = max(0, min(1, (confidence - 50) / 50))
        allocation = min_allocation + (max_allocation - min_allocation) * confidence_factor
        
        return self.current_capital * allocation
    
    def _calculate_total_equity(self, current_date: datetime) -> float:
        """Calculate total portfolio equity including unrealized P&L"""
        
        total_equity = self.current_capital
        
        # Add unrealized P&L from open positions
        for position in self.positions:
            try:
                # Get current price (simplified - in real implementation would cache)
                ticker = yf.Ticker(position.symbol)
                hist = ticker.history(start=current_date - timedelta(days=2), end=current_date + timedelta(days=1))
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    if position.trade_type == 'BUY':
                        unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    total_equity += unrealized_pnl
            except:
                pass  # Skip if can't get price
                
        return total_equity
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {"error": "No trades executed"}
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        # Return metrics
        total_return = ((self.current_capital / self.initial_capital) - 1) * 100
        
        # Risk metrics
        if len(self.daily_returns) > 1:
            daily_returns_array = np.array(self.daily_returns)
            volatility = np.std(daily_returns_array) * np.sqrt(252) * 100  # Annualized
            sharpe_ratio = np.mean(daily_returns_array) / np.std(daily_returns_array) * np.sqrt(252) if np.std(daily_returns_array) > 0 else 0
            
            # Calculate max drawdown
            equity_values = [eq[1] for eq in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            max_drawdown *= 100
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            # Trade Statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            
            # P&L Statistics
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf'),
            
            # Return Statistics
            'initial_capital': self.initial_capital,
            'final_capital': round(self.current_capital, 2),
            'total_return_pct': round(total_return, 2),
            
            # Risk Statistics
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            
            # Additional Data
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """Return trade log as DataFrame"""
        
        if not self.trades:
            return pd.DataFrame()
            
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'Symbol': trade.symbol,
                'Entry Date': trade.entry_date,
                'Exit Date': trade.exit_date,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'Quantity': trade.quantity,
                'Type': trade.trade_type,
                'P&L': trade.pnl,
                'P&L %': trade.pnl_percent,
                'Strategy': trade.strategy,
                'Confidence': trade.confidence,
                'Days Held': (trade.exit_date - trade.entry_date).days
            })
            
        return pd.DataFrame(trade_data)
