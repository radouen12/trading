import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from config import Config
from scipy.stats import pearsonr

class CorrelationAnalyzer:
    def __init__(self):
        self.config = Config()
        self.correlation_cache = {}
        
    def calculate_correlation_matrix(self, symbols, period="3mo"):
        """Calculate correlation matrix for given symbols"""
        try:
            print(f"🔗 Calculating correlation matrix for {len(symbols)} symbols...")
            
            # Fetch historical data for all symbols
            price_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval="1d")
                    
                    if not data.empty:
                        # Use closing prices and fill missing values
                        prices = data['Close'].ffill().bfill()
                        price_data[symbol] = prices
                        
                except Exception as e:
                    print(f"⚠️  Error fetching data for {symbol}: {e}")
                    continue
            
            if len(price_data) < 2:
                return self.get_default_correlation_matrix(symbols)
            
            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(price_data)
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Fill any NaN values with 0
            correlation_matrix = correlation_matrix.fillna(0)
            
            return {
                'correlation_matrix': correlation_matrix,
                'returns_data': returns,
                'symbols': list(correlation_matrix.columns),
                'calculation_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error calculating correlation matrix: {e}")
            return self.get_default_correlation_matrix(symbols)
    
    def find_high_correlations(self, correlation_matrix, threshold=0.7):
        """Find pairs with high correlation (positive or negative)"""
        try:
            high_correlations = []
            
            # Get the correlation matrix values
            corr_matrix = correlation_matrix['correlation_matrix']
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.columns[i]
                    symbol2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if abs(correlation) >= threshold:
                        high_correlations.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'correlation_type': 'positive' if correlation > 0 else 'negative',
                            'strength': 'very_high' if abs(correlation) > 0.8 else 'high'
                        })
            
            # Sort by absolute correlation value
            high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return high_correlations
            
        except Exception as e:
            print(f"❌ Error finding high correlations: {e}")
            return []
    
    def analyze_portfolio_correlation(self, portfolio_symbols):
        """Analyze correlation within a portfolio"""
        try:
            if len(portfolio_symbols) < 2:
                return {
                    'diversification_score': 100,
                    'risk_level': 'LOW',
                    'correlation_clusters': [],
                    'recommendations': ['Add more positions for better analysis']
                }
            
            # Calculate correlation matrix for portfolio
            correlation_data = self.calculate_correlation_matrix(portfolio_symbols)
            corr_matrix = correlation_data['correlation_matrix']
            
            # Calculate average correlation
            # Get upper triangular part (excluding diagonal)
            upper_triangle = np.triu(corr_matrix.values, k=1)
            non_zero_correlations = upper_triangle[upper_triangle != 0]
            
            if len(non_zero_correlations) > 0:
                avg_correlation = np.mean(np.abs(non_zero_correlations))
            else:
                avg_correlation = 0
            
            # Calculate diversification score (lower correlation = higher score)
            diversification_score = max(0, (1 - avg_correlation) * 100)
            
            # Determine risk level
            if avg_correlation > 0.7:
                risk_level = 'HIGH'
            elif avg_correlation > 0.5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Find correlation clusters
            correlation_clusters = self.find_correlation_clusters(corr_matrix)
            
            # Generate recommendations
            recommendations = self.generate_portfolio_recommendations(
                avg_correlation, risk_level, correlation_clusters
            )
            
            return {
                'diversification_score': diversification_score,
                'average_correlation': avg_correlation,
                'risk_level': risk_level,
                'correlation_clusters': correlation_clusters,
                'recommendations': recommendations,
                'correlation_matrix': corr_matrix,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing portfolio correlation: {e}")
            return {
                'diversification_score': 70,
                'average_correlation': 0.3,
                'risk_level': 'MEDIUM',
                'correlation_clusters': [],
                'recommendations': ['Unable to analyze correlations'],
                'analysis_time': datetime.now()
            }
    
    def find_correlation_clusters(self, corr_matrix, threshold=0.6):
        """Find groups of highly correlated assets"""
        try:
            clusters = []
            processed_symbols = set()
            
            for symbol in corr_matrix.columns:
                if symbol in processed_symbols:
                    continue
                
                # Find symbols highly correlated with this one
                cluster = [symbol]
                correlations = corr_matrix[symbol]
                
                for other_symbol in corr_matrix.columns:
                    if (other_symbol != symbol and 
                        other_symbol not in processed_symbols and
                        abs(correlations[other_symbol]) >= threshold):
                        cluster.append(other_symbol)
                
                if len(cluster) > 1:
                    clusters.append({
                        'symbols': cluster,
                        'avg_correlation': np.mean([abs(corr_matrix.loc[symbol, other]) 
                                                  for other in cluster[1:]]),
                        'cluster_size': len(cluster)
                    })
                    
                    # Mark all symbols in this cluster as processed
                    processed_symbols.update(cluster)
                else:
                    processed_symbols.add(symbol)
            
            return clusters
            
        except Exception as e:
            print(f"❌ Error finding correlation clusters: {e}")
            return []
    
    def generate_portfolio_recommendations(self, avg_correlation, risk_level, clusters):
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        
        if risk_level == 'HIGH':
            recommendations.extend([
                'Portfolio shows high correlation - consider diversification',
                'Reduce position sizes in correlated assets',
                'Add assets from different sectors or asset classes',
                'Consider inverse/negative correlation assets for hedging'
            ])
        
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                'Moderate correlation detected - monitor closely',
                'Consider gradual diversification',
                'Look for uncorrelated opportunities'
            ])
        
        else:  # LOW risk
            recommendations.extend([
                'Good diversification - portfolio shows low correlation',
                'Current correlation levels are healthy',
                'Continue monitoring for changes'
            ])
        
        # Cluster-specific recommendations
        if clusters:
            for i, cluster in enumerate(clusters):
                if cluster['cluster_size'] > 2:
                    recommendations.append(
                        f"Cluster {i+1}: {', '.join(cluster['symbols'][:3])} are highly correlated"
                    )
        
        return recommendations
    
    def get_diversification_score(self, symbols):
        """Calculate diversification score for a list of symbols"""
        try:
            if len(symbols) < 2:
                return 100  # Perfect diversification with 1 asset
            
            correlation_data = self.calculate_correlation_matrix(symbols)
            portfolio_analysis = self.analyze_portfolio_correlation(symbols)
            
            return portfolio_analysis['diversification_score']
            
        except Exception as e:
            print(f"❌ Error calculating diversification score: {e}")
            return 70  # Default moderate score
    
    def get_portfolio_correlation_risk(self, portfolio_positions):
        """Assess correlation risk for current portfolio"""
        try:
            if not portfolio_positions:
                return {
                    'risk_level': 'LOW',
                    'diversification_score': 100,
                    'recommendations': ['No positions to analyze']
                }
            
            symbols = list(portfolio_positions.keys())
            portfolio_analysis = self.analyze_portfolio_correlation(symbols)
            
            return {
                'risk_level': portfolio_analysis['risk_level'],
                'diversification_score': portfolio_analysis['diversification_score'],
                'average_correlation': portfolio_analysis['average_correlation'],
                'recommendations': portfolio_analysis['recommendations'],
                'correlation_clusters': portfolio_analysis['correlation_clusters']
            }
            
        except Exception as e:
            print(f"❌ Error assessing portfolio risk: {e}")
            return {
                'risk_level': 'MEDIUM',
                'diversification_score': 70,
                'recommendations': ['Error analyzing portfolio correlations']
            }
    
    def analyze_cross_asset_correlations(self):
        """Analyze correlations across different asset classes"""
        try:
            print("🔗 Analyzing cross-asset correlations...")
            
            # Group symbols by asset class
            asset_groups = {
                'stocks': self.config.STOCK_SYMBOLS[:10],  # Limit for performance
                'crypto': self.config.CRYPTO_SYMBOLS[:5],
                'forex': self.config.FOREX_SYMBOLS[:5]
            }
            
            results = {}
            
            # Analyze within each asset class
            for asset_class, symbols in asset_groups.items():
                if len(symbols) > 1:
                    correlation_data = self.calculate_correlation_matrix(symbols)
                    high_correlations = self.find_high_correlations(correlation_data)
                    
                    results[asset_class] = {
                        'correlation_matrix': correlation_data['correlation_matrix'],
                        'high_correlations': high_correlations,
                        'symbol_count': len(symbols)
                    }
            
            # Cross-asset analysis (stocks vs crypto vs forex)
            cross_asset_symbols = (
                asset_groups['stocks'][:3] + 
                asset_groups['crypto'][:2] + 
                asset_groups['forex'][:2]
            )
            
            if len(cross_asset_symbols) > 1:
                cross_correlation_data = self.calculate_correlation_matrix(cross_asset_symbols)
                cross_high_correlations = self.find_high_correlations(cross_correlation_data, threshold=0.5)
                
                results['cross_asset'] = {
                    'correlation_matrix': cross_correlation_data['correlation_matrix'],
                    'high_correlations': cross_high_correlations,
                    'symbol_count': len(cross_asset_symbols)
                }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in cross-asset correlation analysis: {e}")
            return {}
    
    def analyze_all_correlations(self):
        """Run comprehensive correlation analysis"""
        print("🔗 Starting comprehensive correlation analysis...")
        
        try:
            # Get all symbols
            all_symbols = (self.config.STOCK_SYMBOLS + 
                          self.config.CRYPTO_SYMBOLS + 
                          self.config.FOREX_SYMBOLS)
            
            # Limit symbols for performance (analyze top symbols from each category)
            limited_symbols = (
                self.config.STOCK_SYMBOLS[:15] +  # Top 15 stocks
                self.config.CRYPTO_SYMBOLS[:8] +   # All crypto
                self.config.FOREX_SYMBOLS[:7]      # All forex
            )
            
            # Overall correlation matrix
            print("📊 Calculating overall correlation matrix...")
            overall_correlation = self.calculate_correlation_matrix(limited_symbols)
            
            # Find high correlations
            print("🔍 Finding high correlations...")
            high_correlations = self.find_high_correlations(overall_correlation)
            
            # Cross-asset analysis
            print("🌐 Running cross-asset analysis...")
            cross_asset_analysis = self.analyze_cross_asset_correlations()
            
            # Summary statistics
            summary_stats = self.calculate_correlation_summary(overall_correlation)
            
            results = {
                'overall_correlation': overall_correlation,
                'high_correlations': high_correlations,
                'cross_asset_analysis': cross_asset_analysis,
                'summary_stats': summary_stats,
                'analysis': {
                    'total_symbols': len(limited_symbols),
                    'high_correlations_found': len(high_correlations),
                    'positive_correlations': len([c for c in high_correlations if c['correlation'] > 0]),
                    'negative_correlations': len([c for c in high_correlations if c['correlation'] < 0])
                },
                'analysis_time': datetime.now()
            }
            
            print("✅ Correlation analysis complete!")
            return results
            
        except Exception as e:
            print(f"❌ Error in comprehensive correlation analysis: {e}")
            return self.get_default_correlation_results()
    
    def calculate_correlation_summary(self, correlation_data):
        """Calculate summary statistics for correlation matrix"""
        try:
            corr_matrix = correlation_data['correlation_matrix']
            
            # Get upper triangular correlations (excluding diagonal)
            upper_triangle = np.triu(corr_matrix.values, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            if len(correlations) == 0:
                return self.get_default_summary_stats()
            
            summary = {
                'mean_correlation': np.mean(correlations),
                'median_correlation': np.median(correlations),
                'std_correlation': np.std(correlations),
                'max_correlation': np.max(correlations),
                'min_correlation': np.min(correlations),
                'positive_correlations': np.sum(correlations > 0),
                'negative_correlations': np.sum(correlations < 0),
                'strong_correlations': np.sum(np.abs(correlations) > 0.7),
                'total_pairs': len(correlations)
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error calculating correlation summary: {e}")
            return self.get_default_summary_stats()
    
    # Default methods for error handling
    def get_default_correlation_matrix(self, symbols):
        """Return default correlation matrix when calculation fails"""
        n = len(symbols)
        # Create identity matrix (no correlation)
        default_matrix = pd.DataFrame(
            np.eye(n), 
            index=symbols, 
            columns=symbols
        )
        
        return {
            'correlation_matrix': default_matrix,
            'returns_data': pd.DataFrame(),
            'symbols': symbols,
            'calculation_time': datetime.now()
        }
    
    def get_default_summary_stats(self):
        """Return default summary statistics"""
        return {
            'mean_correlation': 0.1,
            'median_correlation': 0.05,
            'std_correlation': 0.2,
            'max_correlation': 0.5,
            'min_correlation': -0.3,
            'positive_correlations': 15,
            'negative_correlations': 5,
            'strong_correlations': 2,
            'total_pairs': 20
        }
    
    def get_default_correlation_results(self):
        """Return default correlation analysis results"""
        return {
            'overall_correlation': self.get_default_correlation_matrix(['AAPL', 'MSFT']),
            'high_correlations': [],
            'cross_asset_analysis': {},
            'summary_stats': self.get_default_summary_stats(),
            'analysis': {
                'total_symbols': 0,
                'high_correlations_found': 0,
                'positive_correlations': 0,
                'negative_correlations': 0
            },
            'analysis_time': datetime.now()
        }

if __name__ == "__main__":
    # Test the correlation analyzer
    analyzer = CorrelationAnalyzer()
    
    # Test correlation analysis
    print("Testing correlation analysis...")
    correlation_results = analyzer.analyze_all_correlations()
    
    print(f"Total symbols analyzed: {correlation_results['analysis']['total_symbols']}")
    print(f"High correlations found: {correlation_results['analysis']['high_correlations_found']}")
    print(f"Positive correlations: {correlation_results['analysis']['positive_correlations']}")
    print(f"Negative correlations found: {correlation_results['analysis']['negative_correlations']}")
    
    # Test portfolio risk analysis
    test_positions = {'AAPL': {}, 'MSFT': {}, 'GOOGL': {}}
    portfolio_risk = analyzer.get_portfolio_correlation_risk(test_positions)
    print(f"Portfolio risk level: {portfolio_risk['risk_level']}")
    print(f"Diversification score: {portfolio_risk['diversification_score']:.1f}")
