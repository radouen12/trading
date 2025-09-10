import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

class SeasonalAnalyzer:
    def __init__(self):
        self.config = Config()
        
        # Monthly performance patterns (simplified)
        self.monthly_patterns = {
            1: {'score': 65, 'trend': 'BULLISH'},    # January
            2: {'score': 55, 'trend': 'NEUTRAL'},   # February
            3: {'score': 70, 'trend': 'BULLISH'},   # March
            4: {'score': 75, 'trend': 'BULLISH'},   # April
            5: {'score': 50, 'trend': 'NEUTRAL'},   # May
            6: {'score': 45, 'trend': 'BEARISH'},   # June
            7: {'score': 60, 'trend': 'NEUTRAL'},   # July
            8: {'score': 40, 'trend': 'BEARISH'},   # August
            9: {'score': 45, 'trend': 'BEARISH'},   # September
            10: {'score': 65, 'trend': 'BULLISH'},  # October
            11: {'score': 80, 'trend': 'BULLISH'},  # November
            12: {'score': 75, 'trend': 'BULLISH'}   # December
        }
        
        # Sector seasonal patterns
        self.sector_patterns = {
            'XLF': {'strong_months': [1, 11, 12], 'weak_months': [6, 8, 9]},
            'XLE': {'strong_months': [10, 11, 12], 'weak_months': [4, 5, 6]},
            'XLK': {'strong_months': [1, 3, 4], 'weak_months': [8, 9]},
            'XLV': {'strong_months': [2, 3, 11], 'weak_months': [7, 8]},
            'XRT': {'strong_months': [11, 12, 1], 'weak_months': [6, 7, 8]}
        }
    
    def analyze_symbol(self, symbol):
        """Analyze seasonal patterns for a symbol"""
        try:
            current_month = datetime.now().month
            
            # Get monthly pattern
            monthly_data = self.monthly_patterns.get(current_month, {'score': 50, 'trend': 'NEUTRAL'})
            
            # Check if it's a sector ETF
            sector_data = self.sector_patterns.get(symbol, {})
            
            signals = []
            seasonal_score = monthly_data['score']
            
            # Monthly signal
            if monthly_data['trend'] == 'BULLISH':
                signals.append({
                    'type': 'BUY',
                    'reason': f"Strong seasonal month ({datetime.now().strftime('%B')})",
                    'confidence': seasonal_score,
                    'timeframe': 'monthly'
                })
            elif monthly_data['trend'] == 'BEARISH':
                signals.append({
                    'type': 'SELL',
                    'reason': f"Weak seasonal month ({datetime.now().strftime('%B')})",
                    'confidence': 100 - seasonal_score,
                    'timeframe': 'monthly'
                })
            
            # Sector-specific signals
            if sector_data:
                strong_months = sector_data.get('strong_months', [])
                weak_months = sector_data.get('weak_months', [])
                
                if current_month in strong_months:
                    signals.append({
                        'type': 'BUY',
                        'reason': f"Strong seasonal sector month",
                        'confidence': 75,
                        'timeframe': 'monthly'
                    })
                elif current_month in weak_months:
                    signals.append({
                        'type': 'SELL',
                        'reason': f"Weak seasonal sector month",
                        'confidence': 70,
                        'timeframe': 'monthly'
                    })
            
            # Day of week effect (simplified)
            day_of_week = datetime.now().weekday()  # 0=Monday, 6=Sunday
            
            if day_of_week == 0:  # Monday
                signals.append({
                    'type': 'BUY',
                    'reason': 'Monday effect - often positive',
                    'confidence': 60,
                    'timeframe': 'daily'
                })
            elif day_of_week == 4:  # Friday
                signals.append({
                    'type': 'SELL',
                    'reason': 'Friday effect - profit taking',
                    'confidence': 55,
                    'timeframe': 'daily'
                })
            
            return {
                'symbol': symbol,
                'seasonal_signals': signals,
                'monthly_score': seasonal_score,
                'monthly_trend': monthly_data['trend'],
                'current_month': current_month,
                'analysis_time': datetime.now(),
                'monthly_patterns': {
                    'current_month_rank': {
                        'score': seasonal_score,
                        'percentile': (seasonal_score / 100) * 100
                    }
                }
            }
            
        except Exception as e:
            return self._get_default_seasonal_analysis(symbol)
    
    def _get_default_seasonal_analysis(self, symbol):
        """Default seasonal analysis"""
        return {
            'symbol': symbol,
            'seasonal_signals': [{
                'type': 'NEUTRAL',
                'reason': 'No seasonal pattern identified',
                'confidence': 50,
                'timeframe': 'daily'
            }],
            'monthly_score': 50,
            'monthly_trend': 'NEUTRAL',
            'current_month': datetime.now().month,
            'analysis_time': datetime.now(),
            'monthly_patterns': {
                'current_month_rank': {
                    'score': 50,
                    'percentile': 50
                }
            }
        }
    
    def analyze_all_symbols(self):
        """Analyze seasonal patterns for all symbols"""
        results = {}
        
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        for symbol in all_symbols[:10]:  # Limit for performance
            try:
                results[symbol] = self.analyze_symbol(symbol)
            except:
                results[symbol] = self._get_default_seasonal_analysis(symbol)
        
        return results
    
    def get_sector_rotation_signals(self):
        """Get sector rotation signals"""
        try:
            current_month = datetime.now().month
            signals = []
            
            for sector, patterns in self.sector_patterns.items():
                strong_months = patterns.get('strong_months', [])
                weak_months = patterns.get('weak_months', [])
                
                if current_month in strong_months:
                    signals.append({
                        'sector': sector,
                        'signal': 'OVERWEIGHT',
                        'reason': 'Seasonal strength',
                        'confidence': 75
                    })
                elif current_month in weak_months:
                    signals.append({
                        'sector': sector,
                        'signal': 'UNDERWEIGHT',
                        'reason': 'Seasonal weakness',
                        'confidence': 70
                    })
            
            return signals
            
        except Exception as e:
            return []
    
    def get_monthly_outlook(self):
        """Get outlook for current and next month"""
        try:
            current_month = datetime.now().month
            next_month = (current_month % 12) + 1
            
            current_data = self.monthly_patterns.get(current_month, {'score': 50, 'trend': 'NEUTRAL'})
            next_data = self.monthly_patterns.get(next_month, {'score': 50, 'trend': 'NEUTRAL'})
            
            return {
                'current_month': {
                    'month': current_month,
                    'score': current_data['score'],
                    'trend': current_data['trend']
                },
                'next_month': {
                    'month': next_month,
                    'score': next_data['score'],
                    'trend': next_data['trend']
                },
                'transition': 'IMPROVING' if next_data['score'] > current_data['score'] else 'DECLINING'
            }
            
        except Exception as e:
            return {
                'current_month': {'month': datetime.now().month, 'score': 50, 'trend': 'NEUTRAL'},
                'next_month': {'month': (datetime.now().month % 12) + 1, 'score': 50, 'trend': 'NEUTRAL'},
                'transition': 'STABLE'
            }

if __name__ == "__main__":
    analyzer = SeasonalAnalyzer()
    result = analyzer.analyze_symbol('AAPL')
    print(f"Monthly score: {result['monthly_score']}")
    print(f"Seasonal signals: {result['seasonal_signals']}")
    
    sector_signals = analyzer.get_sector_rotation_signals()
    print(f"Sector signals: {sector_signals}")
