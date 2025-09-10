"""
Arbitrage Detector - Phase 4
Cross-asset arbitrage opportunity detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import warnings

warnings.filterwarnings('ignore')

class ArbitrageDetector:
    """Cross-asset arbitrage opportunity detector"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.min_spread_threshold = 0.005
        
        self.asset_pairs = {
            'ETF_BASKET': [('SPY', 'QQQ'), ('IWM', 'SPY')],
            'SECTOR_ROTATION': [('XLF', 'XLK'), ('XLE', 'XLV')],
            'CRYPTO_FIAT': [('BTC-USD', 'ETH-USD')],
        }
    
    def detect_arbitrage_opportunities(self) -> Dict:
        """Scan for arbitrage opportunities"""
        try:
            opportunities = {
                'statistical_arbitrage': self._detect_statistical_arbitrage(),
                'pair_trading': self._detect_pair_trading_opportunities(),
                'mean_reversion': self._detect_mean_reversion_opportunities()
            }
            
            return {
                'arbitrage_opportunities': opportunities,
                'opportunity_count': sum(len(ops) for ops in opportunities.values()),
                'detection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_statistical_arbitrage(self) -> List[Dict]:
        """Detect statistical arbitrage opportunities"""
        opportunities = []
        
        try:
            for category, pairs in self.asset_pairs.items():
                for asset1, asset2 in pairs:
                    data1 = self._get_recent_prices(asset1, periods=30)
                    data2 = self._get_recent_prices(asset2, periods=30)
                    
                    if len(data1) < 20 or len(data2) < 20:
                        continue
                    
                    ratio_analysis = self._calculate_price_ratio_analysis(data1, data2)
                    
                    if ratio_analysis.get('z_score') and abs(ratio_analysis['z_score']) > 2:
                        opportunity = {
                            'type': 'statistical_arbitrage',
                            'category': category,
                            'asset1': asset1,
                            'asset2': asset2,
                            'current_ratio': ratio_analysis['current_ratio'],
                            'z_score': ratio_analysis['z_score'],
                            'confidence': min(abs(ratio_analysis['z_score']) / 3, 0.90),
                            'direction': 'LONG_A_SHORT_B' if ratio_analysis['z_score'] < -2 else 'SHORT_A_LONG_B'
                        }
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            return []
    
    def _detect_pair_trading_opportunities(self) -> List[Dict]:
        """Detect pair trading opportunities"""
        opportunities = []
        
        try:
            pairs = [('SPY', 'QQQ'), ('XLF', 'XLK')]
            
            for asset1, asset2 in pairs:
                data1 = self._get_recent_prices(asset1, periods=30)
                data2 = self._get_recent_prices(asset2, periods=30)
                
                if len(data1) < 20 or len(data2) < 20:
                    continue
                
                spread_analysis = self._analyze_pair_spread(data1, data2)
                
                if spread_analysis.get('is_opportunity'):
                    opportunity = {
                        'type': 'pair_trading',
                        'asset1': asset1,
                        'asset2': asset2,
                        'spread_zscore': spread_analysis['spread_zscore'],
                        'correlation': spread_analysis['correlation'],
                        'confidence': spread_analysis['confidence'],
                        'entry_signal': spread_analysis['entry_signal']
                    }
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            return []
    
    def _detect_mean_reversion_opportunities(self) -> List[Dict]:
        """Detect mean reversion opportunities"""
        opportunities = []
        
        try:
            symbols = self._get_all_symbols()[:10]
            
            for symbol in symbols:
                prices = self._get_recent_prices(symbol, periods=25)
                
                if len(prices) < 20:
                    continue
                
                reversion_signal = self._calculate_mean_reversion_signal(prices)
                
                if reversion_signal.get('is_opportunity'):
                    opportunity = {
                        'type': 'mean_reversion',
                        'symbol': symbol,
                        'current_price': reversion_signal['current_price'],
                        'deviation_pct': reversion_signal['deviation_pct'],
                        'confidence': reversion_signal['confidence'],
                        'direction': reversion_signal['direction'],
                        'target_price': reversion_signal['target_price']
                    }
                    opportunities.append(opportunity)
            
            return opportunities[:5]
            
        except Exception as e:
            return []
    
    def _get_recent_prices(self, symbol: str, periods: int = 30) -> List[float]:
        """Get recent price data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT close FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (symbol, periods))
            
            prices = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return list(reversed(prices))
            
        except Exception as e:
            return []
    
    def _calculate_price_ratio_analysis(self, prices1: List[float], prices2: List[float]) -> Dict:
        """Calculate price ratio analysis"""
        try:
            if len(prices1) != len(prices2) or len(prices1) < 10:
                return {}
            
            ratios = [p1 / p2 for p1, p2 in zip(prices1, prices2) if p2 != 0]
            
            if len(ratios) < 10:
                return {}
            
            current_ratio = ratios[-1]
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            z_score = (current_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
            
            return {
                'current_ratio': current_ratio,
                'mean_ratio': mean_ratio,
                'z_score': z_score
            }
            
        except Exception as e:
            return {}
    
    def _analyze_pair_spread(self, prices1: List[float], prices2: List[float]) -> Dict:
        """Analyze spread between two assets"""
        try:
            if len(prices1) != len(prices2) or len(prices1) < 15:
                return {'is_opportunity': False}
            
            # Normalize prices
            norm_prices1 = np.array(prices1) / prices1[0]
            norm_prices2 = np.array(prices2) / prices2[0]
            
            # Calculate spread
            spread = norm_prices1 - norm_prices2
            current_spread = spread[-1]
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            
            spread_zscore = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
            correlation = np.corrcoef(prices1, prices2)[0, 1] if len(prices1) > 1 else 0
            
            is_opportunity = (abs(spread_zscore) > 1.5 and correlation > 0.6)
            confidence = min(0.85, abs(spread_zscore) / 3 * correlation) if is_opportunity else 0
            
            return {
                'is_opportunity': is_opportunity,
                'current_spread': current_spread,
                'mean_spread': mean_spread,
                'spread_zscore': spread_zscore,
                'correlation': correlation,
                'confidence': confidence,
                'entry_signal': 'LONG_SPREAD' if spread_zscore < -1.5 else 'SHORT_SPREAD'
            }
            
        except Exception as e:
            return {'is_opportunity': False}
    
    def _calculate_mean_reversion_signal(self, prices: List[float]) -> Dict:
        """Calculate mean reversion signal"""
        try:
            if len(prices) < 15:
                return {'is_opportunity': False}
            
            current_price = prices[-1]
            mean_price = np.mean(prices[-15:])
            deviation_pct = (current_price - mean_price) / mean_price
            
            # Simple RSI
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [max(0, change) for change in price_changes[-10:]]
            losses = [max(0, -change) for change in price_changes[-10:]]
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.001
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            is_opportunity = (abs(deviation_pct) > 0.04 and (rsi > 70 or rsi < 30))
            
            confidence = 0
            direction = 'NEUTRAL'
            target_price = current_price
            
            if is_opportunity:
                if deviation_pct > 0.04 and rsi > 70:
                    direction = 'SHORT'
                    target_price = mean_price
                elif deviation_pct < -0.04 and rsi < 30:
                    direction = 'LONG'
                    target_price = mean_price
                
                confidence = min(0.8, abs(deviation_pct) * 10 + (abs(rsi - 50) - 20) / 30)
            
            return {
                'is_opportunity': is_opportunity,
                'current_price': current_price,
                'mean_price': mean_price,
                'deviation_pct': deviation_pct,
                'confidence': max(0, confidence),
                'direction': direction,
                'target_price': target_price
            }
            
        except Exception as e:
            return {'is_opportunity': False}
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM market_data LIMIT 20')
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return symbols
            
        except Exception as e:
            return []
    
    def generate_arbitrage_report(self) -> Dict:
        """Generate arbitrage opportunity report"""
        try:
            opportunities = self.detect_arbitrage_opportunities()
            
            all_opps = []
            for opp_list in opportunities['arbitrage_opportunities'].values():
                all_opps.extend(opp_list)
            
            summary_stats = {
                'total_opportunities': len(all_opps),
                'high_confidence': len([o for o in all_opps if o.get('confidence', 0) > 0.7]),
                'avg_confidence': np.mean([o.get('confidence', 0) for o in all_opps]) if all_opps else 0
            }
            
            recommendations = self._generate_recommendations(all_opps[:3])
            
            return {
                'arbitrage_report': {
                    'summary_statistics': summary_stats,
                    'opportunities': opportunities['arbitrage_opportunities'],
                    'recommendations': recommendations,
                    'report_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, top_opportunities: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            for i, opp in enumerate(top_opportunities, 1):
                opp_type = opp.get('type', 'unknown')
                confidence = opp.get('confidence', 0)
                
                if opp_type == 'statistical_arbitrage':
                    asset1 = opp.get('asset1', '')
                    asset2 = opp.get('asset2', '')
                    direction = opp.get('direction', '')
                    
                    action = f"long {asset1} / short {asset2}" if direction == 'LONG_A_SHORT_B' else f"short {asset1} / long {asset2}"
                    recommendations.append(f"{i}. Consider {action} position (Confidence: {confidence:.1%})")
                
                elif opp_type == 'pair_trading':
                    asset1 = opp.get('asset1', '')
                    asset2 = opp.get('asset2', '')
                    entry_signal = opp.get('entry_signal', '')
                    
                    recommendations.append(f"{i}. Pair trade: {asset1} vs {asset2} - {entry_signal} (Confidence: {confidence:.1%})")
                
                elif opp_type == 'mean_reversion':
                    symbol = opp.get('symbol', '')
                    direction = opp.get('direction', '')
                    
                    recommendations.append(f"{i}. Mean reversion: {direction} {symbol} (Confidence: {confidence:.1%})")
            
            if not recommendations:
                recommendations.append("No high-confidence arbitrage opportunities detected currently.")
                recommendations.append("Continue monitoring for market volatility increases.")
            
            return recommendations
            
        except Exception as e:
            return ["Error generating recommendations"]
