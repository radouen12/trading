"""
Performance Metrics Module - Phase 3
Advanced performance analytics and risk metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PerformanceMetrics:
    """
    Advanced performance analytics for trading strategies
    """
    
    def __init__(self, trades_df: pd.DataFrame, equity_curve: List[Tuple], initial_capital: float):
        self.trades_df = trades_df
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        
        # Convert equity curve to DataFrame
        self.equity_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity'])
        self.equity_df['Date'] = pd.to_datetime(self.equity_df['Date'])
        self.equity_df.set_index('Date', inplace=True)
        
    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if self.trades_df.empty:
            return {"error": "No trades to analyze"}
            
        # Basic trade metrics
        basic_metrics = self._calculate_basic_metrics()
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Time-based metrics
        time_metrics = self._calculate_time_metrics()
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics()
        
        # Combined metrics
        combined_metrics = {
            **basic_metrics,
            **risk_metrics,
            **time_metrics,
            **drawdown_metrics
        }
        
        return combined_metrics
    
    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic trading metrics"""
        
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['P&L'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['P&L'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = self.trades_df['P&L'].sum()
        avg_win = self.trades_df[self.trades_df['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades_df[self.trades_df['P&L'] < 0]['P&L'].mean() if losing_trades > 0 else 0
        
        largest_win = self.trades_df['P&L'].max() if total_trades > 0 else 0
        largest_loss = self.trades_df['P&L'].min() if total_trades > 0 else 0
        
        # Safe profit factor calculation
        if avg_loss == 0:
            if avg_win > 0:
                profit_factor = float('inf')  # Perfect trades (no losses)
            else:
                profit_factor = 1.0  # No wins, no losses
        else:
            profit_factor = abs(avg_win / avg_loss)
        
        # Safe return calculations
        if self.initial_capital == 0:
            total_return = 0
            final_capital = 0
        else:
            final_capital = self.equity_df['Equity'].iloc[-1] if not self.equity_df.empty else self.initial_capital
            total_return = ((final_capital / self.initial_capital) - 1) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_return_pct': round(total_return, 2),
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2)
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk and volatility metrics"""
        
        if self.equity_df.empty or len(self.equity_df) < 2:
            return {}
            
        # Calculate daily returns
        self.equity_df['Daily_Return'] = self.equity_df['Equity'].pct_change()
        daily_returns = self.equity_df['Daily_Return'].dropna()
        
        if len(daily_returns) < 2:
            return {}
            
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate
        
        if daily_returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        max_dd = self._calculate_max_drawdown()
        if max_dd > 0:
            calmar_ratio = (daily_returns.mean() * 252 * 100) / max_dd
        else:
            calmar_ratio = float('inf') if daily_returns.mean() > 0 else 0
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(daily_returns, 5) * 100
        
        return {
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'var_95_pct': round(var_95, 2)
        }
    
    def _calculate_time_metrics(self) -> Dict:
        """Calculate time-based performance metrics"""
        
        if self.trades_df.empty:
            return {}
            
        # Average holding period
        self.trades_df['Entry Date'] = pd.to_datetime(self.trades_df['Entry Date'])
        self.trades_df['Exit Date'] = pd.to_datetime(self.trades_df['Exit Date'])
        
        avg_holding_days = (self.trades_df['Exit Date'] - self.trades_df['Entry Date']).dt.days.mean()
        
        # Trades per month
        date_range = (self.trades_df['Exit Date'].max() - self.trades_df['Entry Date'].min()).days
        trades_per_month = len(self.trades_df) / (date_range / 30) if date_range > 0 else 0
        
        # Monthly returns
        if not self.equity_df.empty:
            monthly_equity = self.equity_df.resample('M')['Equity'].last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            
            best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
            worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
            avg_monthly_return = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        else:
            best_month = worst_month = avg_monthly_return = 0
        
        return {
            'avg_holding_days': round(avg_holding_days, 1),
            'trades_per_month': round(trades_per_month, 1),
            'best_month_pct': round(best_month, 2),
            'worst_month_pct': round(worst_month, 2),
            'avg_monthly_return_pct': round(avg_monthly_return, 2)
        }
    
    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown analysis"""
        
        max_dd = self._calculate_max_drawdown()
        
        # Calculate drawdown periods
        if not self.equity_df.empty:
            equity_values = self.equity_df['Equity'].values
            peak = equity_values[0]
            drawdown_periods = []
            current_dd_length = 0
            
            for value in equity_values:
                if value >= peak:
                    if current_dd_length > 0:
                        drawdown_periods.append(current_dd_length)
                        current_dd_length = 0
                    peak = value
                else:
                    current_dd_length += 1
            
            # Add current drawdown if still in one
            if current_dd_length > 0:
                drawdown_periods.append(current_dd_length)
            
            avg_dd_length = np.mean(drawdown_periods) if drawdown_periods else 0
            max_dd_length = max(drawdown_periods) if drawdown_periods else 0
            
        else:
            avg_dd_length = max_dd_length = 0
        
        return {
            'max_drawdown_pct': round(max_dd, 2),
            'avg_drawdown_length_days': round(avg_dd_length, 1),
            'max_drawdown_length_days': max_dd_length
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        
        if self.equity_df.empty:
            return 0
            
        equity_values = self.equity_df['Equity'].values
        peak = equity_values[0]
        max_drawdown = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100
    
    def create_performance_charts(self) -> Dict[str, go.Figure]:
        """Create comprehensive performance visualization charts"""
        
        charts = {}
        
        # 1. Equity Curve
        charts['equity_curve'] = self._create_equity_curve_chart()
        
        # 2. Drawdown Chart
        charts['drawdown'] = self._create_drawdown_chart()
        
        # 3. Trade Distribution
        charts['trade_distribution'] = self._create_trade_distribution_chart()
        
        return charts
    
    def _create_equity_curve_chart(self) -> go.Figure:
        """Create equity curve chart"""
        
        fig = go.Figure()
        
        if not self.equity_df.empty:
            fig.add_trace(go.Scatter(
                x=self.equity_df.index,
                y=self.equity_df['Equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00d4ff', width=2)
            ))
            
            # Add benchmark line (initial capital)
            fig.add_hline(
                y=self.initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def _create_drawdown_chart(self) -> go.Figure:
        """Create drawdown chart"""
        
        fig = go.Figure()
        
        if not self.equity_df.empty:
            # Calculate drawdown series
            equity_values = self.equity_df['Equity']
            peak = equity_values.cummax()
            drawdown = (equity_values - peak) / peak * 100
            
            fig.add_trace(go.Scatter(
                x=self.equity_df.index,
                y=drawdown,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=2),
                fill='tonexty'
            ))
        
        fig.update_layout(
            title='Portfolio Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def _create_trade_distribution_chart(self) -> go.Figure:
        """Create trade P&L distribution chart"""
        
        fig = go.Figure()
        
        if not self.trades_df.empty:
            fig.add_trace(go.Histogram(
                x=self.trades_df['P&L %'],
                nbinsx=20,
                name='Trade Returns',
                marker_color='skyblue'
            ))
        
        fig.update_layout(
            title='Trade Return Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400
        )
        
        return fig
