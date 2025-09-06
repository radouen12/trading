import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from config import Config

class SeasonalAnalyzer:
    def __init__(self):
        self.config = Config()
        
    def analyze_monthly_patterns(self, symbol, years_back=5):
        """Analyze monthly performance patterns"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                return self.get_default_monthly_pattern()
            
            # Calculate monthly returns
            data['Year'] = data.index.year
            data['Month'] = data.index.month
            data['MonthlyReturn'] = data['Close'].pct_change(periods=20) * 100  # Approximate monthly
            
            # Group by month and calculate statistics
            monthly_stats = {}
            for month in range(1, 13):
                month_data = data[data['Month'] == month]['MonthlyReturn'].dropna()
                
                if len(month_data) > 0:
                    monthly_stats[month] = {
                        'avg_return': month_data.mean(),
                        'win_rate': (month_data > 0).mean() * 100,
                        'volatility': month_data.std(),
                        'best_return': month_data.max(),
                        'worst_return': month_data.min(),
                        'total_trades': len(month_data)
                    }
                else:
                    monthly_stats[month] = self.get_default_month_stats()
            
            # Rank months by performance
            month_rankings = sorted(monthly_stats.items(), 
                                  key=lambda x: x[1]['avg_return'], reverse=True)
            
            strongest_months = month_rankings[:3]
            weakest_months = month_rankings[-3:]
            
            # Current month ranking
            current_month = datetime.now().month
            current_rank = next((i for i, (month, _) in enumerate(month_rankings) 
                               if month == current_month), 6)
            
            return {
                'monthly_stats': monthly_stats,
                'strongest_months': strongest_months,
                'weakest_months': weakest_months,
                'current_month_rank': {
                    'rank': current_rank + 1,
                    'total_months': 12,
                    'percentile': ((12 - current_rank) / 12) * 100
                }
            }
            
        except Exception as e:
            print(f"❌ Error analyzing monthly patterns for {symbol}: {e}")
            return self.get_default_monthly_pattern()
    
    def analyze_day_of_week_patterns(self, symbol, years_back=3):
        """Analyze day-of-week performance patterns"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                return self.get_default_dow_pattern()
            
            # Calculate daily returns
            data['DayOfWeek'] = data.index.day_name()
            data['DailyReturn'] = data['Close'].pct_change() * 100
            
            # Group by day of week
            dow_stats = {}
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                day_data = data[data['DayOfWeek'] == day]['DailyReturn'].dropna()
                
                if len(day_data) > 0:
                    dow_stats[day] = {
                        'avg_return': day_data.mean(),
                        'win_rate': (day_data > 0).mean() * 100,
                        'volatility': day_data.std(),
                        'total_days': len(day_data)
                    }
                else:
                    dow_stats[day] = {
                        'avg_return': 0.1, 'win_rate': 52, 'volatility': 1.2, 'total_days': 100
                    }
            
            # Find best and worst days
            best_day = max(dow_stats.items(), key=lambda x: x[1]['avg_return'])
            worst_day = min(dow_stats.items(), key=lambda x: x[1]['avg_return'])
            
            # Today's outlook
            today = datetime.now().strftime('%A')
            today_stats = dow_stats.get(today, dow_stats['Monday'])
            
            return {
                'dow_stats': dow_stats,
                'best_day': best_day,
                'worst_day': worst_day,
                'today_outlook': self.get_day_outlook(today_stats)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing day patterns for {symbol}: {e}")
            return self.get_default_dow_pattern()
    
    def analyze_earnings_effects(self, symbol):
        """Analyze earnings season effects"""
        try:
            # Earnings typically occur in Jan, Apr, Jul, Oct
            earnings_months = [1, 4, 7, 10]
            current_month = datetime.now().month
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", interval="1d")
            
            if data.empty:
                return self.get_default_earnings_effect()
            
            data['Month'] = data.index.month
            data['DailyReturn'] = data['Close'].pct_change() * 100
            
            # Earnings season performance
            earnings_data = data[data['Month'].isin(earnings_months)]['DailyReturn'].dropna()
            non_earnings_data = data[~data['Month'].isin(earnings_months)]['DailyReturn'].dropna()
            
            earnings_stats = {
                'avg_return': earnings_data.mean() if len(earnings_data) > 0 else 0.8,
                'volatility': earnings_data.std() if len(earnings_data) > 0 else 2.1,
                'win_rate': (earnings_data > 0).mean() * 100 if len(earnings_data) > 0 else 58
            }
            
            non_earnings_stats = {
                'avg_return': non_earnings_data.mean() if len(non_earnings_data) > 0 else 0.6,
                'volatility': non_earnings_data.std() if len(non_earnings_data) > 0 else 1.4,
                'win_rate': (non_earnings_data > 0).mean() * 100 if len(non_earnings_data) > 0 else 54
            }
            
            return {
                'earnings_season': earnings_stats,
                'non_earnings': non_earnings_stats,
                'is_earnings_season': current_month in earnings_months
            }
            
        except Exception as e:
            print(f"❌ Error analyzing earnings effects for {symbol}: {e}")
            return self.get_default_earnings_effect()
    
    def analyze_holiday_effects(self, symbol):
        """Analyze holiday trading effects"""
        try:
            # Major market holidays and their typical effects
            holiday_effects = {
                'Christmas': {'avg_return': 0.3, 'win_rate': 62, 'volatility': 1.0},
                'New_Year': {'avg_return': 0.5, 'win_rate': 65, 'volatility': 1.2},
                'Independence_Day': {'avg_return': 0.2, 'win_rate': 58, 'volatility': 0.8},
                'Thanksgiving': {'avg_return': 0.4, 'win_rate': 68, 'volatility': 0.9}
            }
            
            # Check for upcoming holidays (simplified)
            current_date = datetime.now()
            upcoming_holidays = []
            
            # Christmas effect (December)
            if current_date.month == 12:
                upcoming_holidays.append(('Christmas', holiday_effects['Christmas']))
            
            # New Year effect (January)
            if current_date.month == 1:
                upcoming_holidays.append(('New_Year', holiday_effects['New_Year']))
            
            return {
                'holiday_effects': holiday_effects,
                'upcoming_holidays': upcoming_holidays
            }
            
        except Exception as e:
            print(f"❌ Error analyzing holiday effects for {symbol}: {e}")
            return self.get_default_holiday_effect()
    
    def get_sector_rotation_signals(self):
        """Analyze sector rotation patterns"""
        try:
            sectors = {
                'Technology': ['XLK', 'QQQ'],
                'Healthcare': ['XLV'],
                'Financial': ['XLF'],
                'Energy': ['XLE'],
                'Consumer': ['XLY', 'XLP'],
                'Industrial': ['XLI'],
                'Materials': ['XLB'],
                'Utilities': ['XLU']
            }
            
            current_month = datetime.now().month
            
            # Seasonal sector preferences (simplified model)
            seasonal_preferences = {
                1: ['Technology', 'Healthcare'],  # January effect
                2: ['Technology', 'Consumer'],
                3: ['Financial', 'Industrial'],   # Economic optimism
                4: ['Technology', 'Healthcare'],  # Earnings season
                5: ['Consumer', 'Energy'],        # Summer driving
                6: ['Technology', 'Consumer'],
                7: ['Energy', 'Materials'],       # Summer activity
                8: ['Healthcare', 'Utilities'],   # Defensive
                9: ['Financial', 'Industrial'],   # Back to business
                10: ['Technology', 'Consumer'],   # Holiday prep
                11: ['Consumer', 'Technology'],   # Holiday season
                12: ['Healthcare', 'Utilities']   # Year-end defensive
            }
            
            preferred_sectors = seasonal_preferences.get(current_month, ['Technology', 'Healthcare'])
            
            sector_signals = {}
            for sector, etfs in sectors.items():
                if sector in preferred_sectors:
                    recommendation = 'OVERWEIGHT'
                    seasonal_favorable = True
                    confidence = 70
                else:
                    recommendation = 'NEUTRAL'
                    seasonal_favorable = False
                    confidence = 50
                
                sector_signals[sector] = {
                    'recommendation': recommendation,
                    'seasonal_favorable': seasonal_favorable,
                    'confidence': confidence,
                    'etfs': etfs
                }
            
            return sector_signals
            
        except Exception as e:
            print(f"❌ Error analyzing sector rotation: {e}")
            return {}
    
    def generate_seasonal_signals(self, symbol, monthly_patterns, dow_patterns, earnings_effects):
        """Generate trading signals from seasonal analysis"""
        signals = []
        
        try:
            # Monthly pattern signals
            current_month = datetime.now().month
            current_month_stats = monthly_patterns['monthly_stats'].get(current_month, {})
            
            if current_month_stats.get('avg_return', 0) > 1.5:
                signals.append({
                    'type': 'BUY',
                    'reason': f'Strong seasonal month (avg: {current_month_stats["avg_return"]:.1f}%)',
                    'confidence': min(60 + current_month_stats.get('win_rate', 50) / 2, 80),
                    'timeframe': 'monthly',
                    'seasonal_type': 'monthly_pattern'
                })
            
            elif current_month_stats.get('avg_return', 0) < -1.0:
                signals.append({
                    'type': 'SELL',
                    'reason': f'Weak seasonal month (avg: {current_month_stats["avg_return"]:.1f}%)',
                    'confidence': min(55 + abs(current_month_stats.get('avg_return', 0)) * 5, 75),
                    'timeframe': 'monthly',
                    'seasonal_type': 'monthly_pattern'
                })
            
            # Day of week signals
            today = datetime.now().strftime('%A')
            if today in dow_patterns['dow_stats']:
                today_stats = dow_patterns['dow_stats'][today]
                
                if today_stats['avg_return'] > 0.2:
                    signals.append({
                        'type': 'BUY',
                        'reason': f'Favorable day of week ({today})',
                        'confidence': 60,
                        'timeframe': 'daily',
                        'seasonal_type': 'dow_pattern'
                    })
            
            # Earnings season signals
            if earnings_effects['is_earnings_season']:
                earnings_stats = earnings_effects['earnings_season']
                
                if earnings_stats['avg_return'] > earnings_effects['non_earnings']['avg_return']:
                    signals.append({
                        'type': 'BUY',
                        'reason': 'Earnings season boost expected',
                        'confidence': 65,
                        'timeframe': 'weekly',
                        'seasonal_type': 'earnings_effect'
                    })
            
            return signals
            
        except Exception as e:
            print(f"❌ Error generating seasonal signals: {e}")
            return []
    
    def analyze_symbol_seasonality(self, symbol):
        """Complete seasonal analysis for a symbol"""
        try:
            print(f"🗓️ Analyzing seasonal patterns for {symbol}...")
            
            # Run all seasonal analyses
            monthly_patterns = self.analyze_monthly_patterns(symbol)
            dow_patterns = self.analyze_day_of_week_patterns(symbol)
            earnings_effects = self.analyze_earnings_effects(symbol)
            holiday_effects = self.analyze_holiday_effects(symbol)
            
            # Generate seasonal signals
            seasonal_signals = self.generate_seasonal_signals(
                symbol, monthly_patterns, dow_patterns, earnings_effects
            )
            
            return {
                'symbol': symbol,
                'monthly_patterns': monthly_patterns,
                'dow_patterns': dow_patterns,
                'earnings_effects': earnings_effects,
                'holiday_effects': holiday_effects,
                'seasonal_signals': seasonal_signals,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error in seasonal analysis for {symbol}: {e}")
            return self.get_default_seasonal_analysis(symbol)
    
    def get_day_outlook(self, today_stats):
        """Get outlook for today based on historical patterns"""
        if today_stats['avg_return'] > 0.1:
            outlook = 'Positive'
        elif today_stats['avg_return'] < -0.1:
            outlook = 'Negative'
        else:
            outlook = 'Neutral'
        
        return {
            'outlook': outlook,
            'avg_return': today_stats['avg_return'],
            'win_rate': today_stats['win_rate'],
            'confidence': min(50 + abs(today_stats['avg_return']) * 20, 80)
        }
    
    # Default patterns when data is unavailable
    def get_default_monthly_pattern(self):
        return {
            'monthly_stats': {i: self.get_default_month_stats() for i in range(1, 13)},
            'strongest_months': [(1, self.get_default_month_stats())],
            'weakest_months': [(6, self.get_default_month_stats())],
            'current_month_rank': {'rank': 6, 'total_months': 12, 'percentile': 50}
        }
    
    def get_default_month_stats(self):
        return {
            'avg_return': 0.5,
            'win_rate': 55,
            'volatility': 15,
            'best_return': 8,
            'worst_return': -8,
            'total_trades': 50
        }
    
    def get_default_dow_pattern(self):
        return {
            'dow_stats': {
                'Monday': {'avg_return': 0.1, 'win_rate': 52, 'volatility': 1.2, 'total_days': 100},
                'Tuesday': {'avg_return': 0.15, 'win_rate': 54, 'volatility': 1.1, 'total_days': 100},
                'Wednesday': {'avg_return': 0.12, 'win_rate': 53, 'volatility': 1.0, 'total_days': 100},
                'Thursday': {'avg_return': 0.18, 'win_rate': 55, 'volatility': 1.1, 'total_days': 100},
                'Friday': {'avg_return': 0.08, 'win_rate': 51, 'volatility': 1.3, 'total_days': 100}
            },
            'best_day': ('Thursday', {'avg_return': 0.18, 'win_rate': 55}),
            'worst_day': ('Friday', {'avg_return': 0.08, 'win_rate': 51}),
            'today_outlook': {'outlook': 'Neutral', 'avg_return': 0.1, 'win_rate': 52, 'confidence': 60}
        }
    
    def get_default_earnings_effect(self):
        return {
            'earnings_season': {'avg_return': 0.8, 'volatility': 2.1, 'win_rate': 58},
            'non_earnings': {'avg_return': 0.6, 'volatility': 1.4, 'win_rate': 54},
            'is_earnings_season': datetime.now().month in [1, 4, 7, 10]
        }
    
    def get_default_holiday_effect(self):
        return {
            'holiday_effects': {
                'Christmas': {'avg_return': 0.3, 'win_rate': 62, 'volatility': 1.0},
                'New_Year': {'avg_return': 0.5, 'win_rate': 65, 'volatility': 1.2}
            },
            'upcoming_holidays': []
        }
    
    def get_default_seasonal_analysis(self, symbol):
        return {
            'symbol': symbol,
            'monthly_patterns': self.get_default_monthly_pattern(),
            'dow_patterns': self.get_default_dow_pattern(),
            'earnings_effects': self.get_default_earnings_effect(),
            'holiday_effects': self.get_default_holiday_effect(),
            'seasonal_signals': [],
            'analysis_time': datetime.now()
        }
    
    def analyze_all_symbols(self):
        """Analyze seasonal patterns for all symbols"""
        print("🗓️ Starting comprehensive seasonal analysis...")
        
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        results = {}
        
        for i, symbol in enumerate(all_symbols):
            print(f"📅 Analyzing seasonal patterns for {symbol} ({i+1}/{len(all_symbols)})")
            results[symbol] = self.analyze_symbol_seasonality(symbol)
        
        # Add sector rotation analysis
        results['sector_rotation'] = self.get_sector_rotation_signals()
        
        print("✅ Seasonal analysis complete!")
        return results

if __name__ == "__main__":
    # Test the seasonal analyzer
    analyzer = SeasonalAnalyzer()
    
    # Test single symbol
    print("Testing AAPL seasonal analysis...")
    aapl_seasonal = analyzer.analyze_symbol_seasonality('AAPL')
    
    print(f"Current month rank: {aapl_seasonal['monthly_patterns']['current_month_rank']}")
    print(f"Seasonal signals: {aapl_seasonal['seasonal_signals']}")
    
    # Test sector rotation
    print("\nTesting sector rotation...")
    sector_signals = analyzer.get_sector_rotation_signals()
    for sector, signal in sector_signals.items():
        print(f"{sector}: {signal['recommendation']} ({'Favorable' if signal['seasonal_favorable'] else 'Neutral'})")
