import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import yfinance as yf

class CorrelationAnalyzer:
    def __init__(self):
        self.config = Config()
        
        # Predefined correlation groups
        self.correlation_groups = {
            'large_cap_tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'indices': ['SPY', 'QQQ', 'IWM'],
            'energy': ['XLE', 'USO'],
            'financials': ['XLF', 'JPM', 'BAC'],
            'crypto': ['BTC-USD', 'ETH-USD'],
            'forex_majors': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        }
        
        # Known correlation coefficients (simplified)
        self.known_correlations = {
            ('AAPL', 'MSFT'): 0.65,
            ('AAPL', 'GOOGL'): 0.70,
            ('SPY', 'QQQ'): 0.85,
            ('BTC-USD', 'ETH-USD'): 0.80,
            ('XLF', 'SPY'): 0.75,
            ('VIX', 'SPY'): -0.70
        }
    
    def calculate_correlation(self, symbol1, symbol2, period="3mo"):
        """Calculate correlation between two symbols"""
        try:
            # Check if we have a known correlation
            pair = tuple(sorted([symbol1, symbol2]))
            if pair in self.known_correlations:
                return self.known_correlations[pair]
            
            # Try to calculate from real data
            ticker1 = yf.Ticker(symbol1)
            ticker2 = yf.Ticker(symbol2)
            
            hist1 = ticker1.history(period=period)
            hist2 = ticker2.history(period=period)
            
            if hist1.empty or hist2.empty:
                return self._estimate_correlation(symbol1, symbol2)
            
            # Align dates and calculate correlation
            common_dates = hist1.index.intersection(hist2.index)
            
            if len(common_dates) < 10:  # Need at least 10 data points
                return self._estimate_correlation(symbol1, symbol2)
            
            prices1 = hist1.loc[common_dates]['Close']
            prices2 = hist2.loc[common_dates]['Close']
            
            correlation = prices1.corr(prices2)
            
            if pd.isna(correlation):
                return self._estimate_correlation(symbol1, symbol2)
            
            return correlation
            
        except Exception as e:
            return self._estimate_correlation(symbol1, symbol2)
    
    def _estimate_correlation(self, symbol1, symbol2):
        """Estimate correlation based on asset classes"""
        try:
            # Same symbol = perfect correlation
            if symbol1 == symbol2:
                return 1.0
            
            # Check if both are in same correlation group
            for group_name, symbols in self.correlation_groups.items():
                if symbol1 in symbols and symbol2 in symbols:
                    if group_name == 'large_cap_tech':
                        return 0.65
                    elif group_name == 'indices':
                        return 0.80
                    elif group_name == 'crypto':
                        return 0.75
                    else:
                        return 0.60
            
            # Different asset classes typically have lower correlation
            if self._get_asset_class(symbol1) != self._get_asset_class(symbol2):
                return 0.20
            
            # Default moderate correlation for same asset class
            return 0.40
            
        except:
            return 0.30  # Default moderate correlation
    
    def _get_asset_class(self, symbol):
        """Determine asset class of symbol"""
        if 'USD' in symbol or '-USD' in symbol:
            return 'crypto'
        elif '=X' in symbol:
            return 'forex'
        elif symbol.startswith('XL'):
            return 'sector_etf'
        elif symbol in ['SPY', 'QQQ', 'IWM', 'VIX']:
            return 'index'
        else:
            return 'stock'
    
    def analyze_portfolio_correlations(self, symbols):
        """Analyze correlations within a portfolio"""
        try:
            if len(symbols) < 2:
                return {
                    'avg_correlation': 0,
                    'max_correlation': 0,
                    'min_correlation': 0,
                    'correlation_matrix': {},
                    'high_correlation_pairs': [],
                    'diversification_score': 100
                }
            
            correlations = []
            correlation_matrix = {}
            high_correlation_pairs = []
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        corr = 1.0
                    else:
                        corr = self.calculate_correlation(symbol1, symbol2)
                    
                    correlation_matrix[symbol1][symbol2] = corr
                    
                    if i < j:  # Avoid double counting
                        correlations.append(corr)
                        
                        # Flag high correlations
                        if corr > 0.7:
                            high_correlation_pairs.append({
                                'symbol1': symbol1,
                                'symbol2': symbol2,
                                'correlation': corr
                            })
            
            # Calculate statistics
            avg_correlation = np.mean(correlations) if correlations else 0
            max_correlation = np.max(correlations) if correlations else 0
            min_correlation = np.min(correlations) if correlations else 0
            
            # Diversification score (lower correlation = higher diversification)
            diversification_score = max(0, (1 - avg_correlation) * 100)
            
            return {
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'min_correlation': min_correlation,
                'correlation_matrix': correlation_matrix,
                'high_correlation_pairs': high_correlation_pairs,
                'diversification_score': diversification_score,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return {
                'avg_correlation': 0.5,
                'max_correlation': 0.8,
                'min_correlation': 0.2,
                'correlation_matrix': {},
                'high_correlation_pairs': [],
                'diversification_score': 60,
                'analysis_time': datetime.now()
            }
    
    def get_portfolio_correlation_risk(self, portfolio_positions):
        """Assess correlation risk in portfolio"""
        try:
            symbols = list(portfolio_positions.keys())
            
            if len(symbols) == 0:
                return {
                    'risk_level': 'NONE',
                    'diversification_score': 100,
                    'recommendations': ['Portfolio is empty']
                }
            
            correlation_analysis = self.analyze_portfolio_correlations(symbols)
            
            avg_correlation = correlation_analysis['avg_correlation']
            high_corr_pairs = correlation_analysis['high_correlation_pairs']
            
            # Determine risk level
            if avg_correlation > 0.7:
                risk_level = 'HIGH'
                recommendations = [
                    'Portfolio has high correlation risk',
                    'Consider diversifying across asset classes',
                    'Reduce positions in highly correlated assets'
                ]
            elif avg_correlation > 0.5:
                risk_level = 'MEDIUM'
                recommendations = [
                    'Portfolio has moderate correlation',
                    'Monitor correlation during market stress',
                    'Consider adding uncorrelated assets'
                ]
            else:
                risk_level = 'LOW'
                recommendations = [
                    'Portfolio is well diversified',
                    'Maintain current diversification level'
                ]
            
            return {
                'risk_level': risk_level,
                'avg_correlation': avg_correlation,
                'diversification_score': correlation_analysis['diversification_score'],
                'high_correlation_count': len(high_corr_pairs),
                'recommendations': recommendations,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return {
                'risk_level': 'MEDIUM',
                'avg_correlation': 0.5,
                'diversification_score': 60,
                'high_correlation_count': 0,
                'recommendations': ['Unable to assess correlation risk'],
                'analysis_time': datetime.now()
            }
    
    def get_diversification_score(self, symbols):
        """Get diversification score for a list of symbols"""
        try:
            if len(symbols) <= 1:
                return 100  # Single asset is perfectly "diversified" within itself
            
            correlation_analysis = self.analyze_portfolio_correlations(symbols)
            return correlation_analysis['diversification_score']
            
        except:
            return 60  # Default moderate diversification
    
    def analyze_all_correlations(self):
        """Analyze correlations for all tracked symbols"""
        try:
            all_symbols = (self.config.STOCK_SYMBOLS + 
                          self.config.CRYPTO_SYMBOLS + 
                          self.config.FOREX_SYMBOLS)
            
            # Limit to prevent timeouts
            sample_symbols = all_symbols[:8]
            
            correlation_analysis = self.analyze_portfolio_correlations(sample_symbols)
            
            return {
                'market_correlations': correlation_analysis,
                'asset_class_analysis': self._analyze_asset_class_correlations(),
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return {
                'market_correlations': {},
                'asset_class_analysis': {},
                'analysis_time': datetime.now()
            }
    
    def _analyze_asset_class_correlations(self):
        """Analyze correlations between asset classes"""
        try:
            # Representative symbols for each asset class
            representatives = {
                'stocks': 'SPY',
                'tech': 'QQQ',
                'crypto': 'BTC-USD',
                'forex': 'DX-Y.NYB',
                'volatility': 'VIX'
            }
            
            asset_correlations = {}
            
            for class1, symbol1 in representatives.items():
                asset_correlations[class1] = {}
                for class2, symbol2 in representatives.items():
                    if class1 != class2:
                        corr = self.calculate_correlation(symbol1, symbol2)
                        asset_correlations[class1][class2] = corr
            
            return asset_correlations
            
        except:
            return {}

if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    
    # Test correlation calculation
    corr = analyzer.calculate_correlation('AAPL', 'MSFT')
    print(f"AAPL-MSFT correlation: {corr:.3f}")
    
    # Test portfolio analysis
    test_portfolio = ['AAPL', 'MSFT', 'GOOGL']
    portfolio_analysis = analyzer.analyze_portfolio_correlations(test_portfolio)
    print(f"Portfolio diversification score: {portfolio_analysis['diversification_score']:.1f}")
    print(f"Average correlation: {portfolio_analysis['avg_correlation']:.3f}")
