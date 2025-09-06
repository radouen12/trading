import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
from analysis.technical import TechnicalAnalyzer
from analysis.seasonal import SeasonalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.correlation import CorrelationAnalyzer

class EnhancedSuggestionEngine:
    def __init__(self):
        self.config = Config()
        self.technical_analyzer = TechnicalAnalyzer()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Weights for different analysis types
        self.analysis_weights = {
            'technical': 0.4,
            'seasonal': 0.25,
            'sentiment': 0.25,
            'correlation': 0.1
        }
    
    def generate_comprehensive_suggestions(self, market_data, portfolio_positions):
        """Generate enhanced trading suggestions using all analysis engines"""
        try:
            print("🧠 Generating comprehensive trading suggestions...")
            
            # Run all analyses
            print("📊 Running technical analysis...")
            technical_results = self.technical_analyzer.analyze_all_symbols()
            
            print("🗓️ Running seasonal analysis...")
            seasonal_results = self.seasonal_analyzer.analyze_all_symbols()
            
            print("📰 Running sentiment analysis...")
            sentiment_results = self.sentiment_analyzer.analyze_all_symbols()
            
            print("🔗 Running correlation analysis...")
            correlation_results = self.correlation_analyzer.analyze_all_correlations()
            
            # Generate enhanced suggestions
            suggestions = self.combine_all_analyses(
                market_data, 
                technical_results,
                seasonal_results,
                sentiment_results,
                correlation_results,
                portfolio_positions
            )
            
            # Risk adjustment based on portfolio correlation
            risk_adjusted_suggestions = self.apply_risk_adjustments(
                suggestions, 
                correlation_results,
                portfolio_positions
            )
            
            print(f"✅ Generated {len(risk_adjusted_suggestions)} enhanced suggestions")
            return risk_adjusted_suggestions
            
        except Exception as e:
            print(f"❌ Error generating suggestions: {e}")
            return []
    
    def combine_all_analyses(self, market_data, technical_results, seasonal_results, 
                           sentiment_results, correlation_results, portfolio_positions):
        """Combine all analysis results into comprehensive suggestions"""
        try:
            suggestions = []
            
            # Get all available symbols
            all_symbols = list(market_data.keys())
            
            for symbol in all_symbols:
                # Skip if already in portfolio (for new suggestions)
                if symbol in portfolio_positions:
                    continue
                
                # Get analysis results for this symbol
                technical_analysis = technical_results.get(symbol, {})
                seasonal_analysis = seasonal_results.get(symbol, {})
                sentiment_analysis = sentiment_results.get(symbol, {})
                
                # Calculate composite score
                composite_score = self.calculate_composite_score(
                    symbol,
                    technical_analysis,
                    seasonal_analysis, 
                    sentiment_analysis,
                    market_data.get(symbol, {})
                )
                
                if composite_score['overall_confidence'] >= self.config.MIN_CONFIDENCE_SCORE:
                    # Generate suggestion
                    suggestion = self.create_enhanced_suggestion(
                        symbol,
                        market_data.get(symbol, {}),
                        composite_score,
                        technical_analysis,
                        seasonal_analysis,
                        sentiment_analysis
                    )
                    
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Sort by confidence score
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit suggestions per timeframe
            filtered_suggestions = self.filter_suggestions_by_timeframe(suggestions)
            
            return filtered_suggestions
            
        except Exception as e:
            print(f"❌ Error combining analyses: {e}")
            return []
    
    def calculate_composite_score(self, symbol, technical_analysis, seasonal_analysis, 
                                sentiment_analysis, market_data):
        """Calculate composite confidence score from all analyses"""
        try:
            scores = {
                'technical_score': 50,
                'seasonal_score': 50,
                'sentiment_score': 50,
                'volume_score': 50,
                'momentum_score': 50
            }
            
            signals = {
                'technical_signals': [],
                'seasonal_signals': [],
                'sentiment_signals': []
            }
            
            # Technical analysis score
            if technical_analysis and 'signals' in technical_analysis:
                tech_signals = technical_analysis['signals']
                signals['technical_signals'] = tech_signals
                
                if tech_signals:
                    # Average confidence from technical signals
                    tech_confidences = [signal[2] for signal in tech_signals if len(signal) > 2]
                    if tech_confidences:
                        scores['technical_score'] = sum(tech_confidences) / len(tech_confidences)
                
                # RSI contribution
                rsi = technical_analysis.get('rsi', 50)
                if rsi < 30:
                    scores['technical_score'] += 15  # Oversold bonus
                elif rsi > 70:
                    scores['technical_score'] -= 15  # Overbought penalty
                
                # MACD contribution
                macd = technical_analysis.get('macd', 0)
                macd_signal = technical_analysis.get('macd_signal', 0)
                if macd > macd_signal:
                    scores['technical_score'] += 10  # Bullish MACD
                else:
                    scores['technical_score'] -= 5   # Bearish MACD
            
            # Seasonal analysis score
            if seasonal_analysis and 'seasonal_signals' in seasonal_analysis:
                seasonal_signals = seasonal_analysis['seasonal_signals']
                signals['seasonal_signals'] = seasonal_signals
                
                if seasonal_signals:
                    seasonal_confidences = [s.get('confidence', 50) for s in seasonal_signals]
                    if seasonal_confidences:
                        scores['seasonal_score'] = sum(seasonal_confidences) / len(seasonal_confidences)
                
                # Monthly pattern bonus
                monthly_patterns = seasonal_analysis.get('monthly_patterns', {})
                current_month_rank = monthly_patterns.get('current_month_rank', {})
                percentile = current_month_rank.get('percentile', 50)
                
                if percentile > 70:
                    scores['seasonal_score'] += 15  # Strong seasonal month
                elif percentile < 30:
                    scores['seasonal_score'] -= 10  # Weak seasonal month
            
            # Sentiment analysis score
            if sentiment_analysis and 'sentiment_score' in sentiment_analysis:
                scores['sentiment_score'] = sentiment_analysis['sentiment_score']
                
                if 'signals' in sentiment_analysis:
                    signals['sentiment_signals'] = sentiment_analysis['signals']
            
            # Volume and momentum scores
            if market_data:
                change_pct = market_data.get('change_pct', 0)
                volume = market_data.get('volume', 0)
                
                # Momentum score
                if abs(change_pct) > 3:
                    scores['momentum_score'] = 70 + min(abs(change_pct) * 2, 25)
                elif abs(change_pct) > 1:
                    scores['momentum_score'] = 60 + abs(change_pct) * 5
                
                # Volume score (simplified)
                if volume > 1000000:  # High volume
                    scores['volume_score'] = 70
                elif volume > 500000:
                    scores['volume_score'] = 60
            
            # Calculate weighted overall score
            overall_confidence = (
                scores['technical_score'] * self.analysis_weights['technical'] +
                scores['seasonal_score'] * self.analysis_weights['seasonal'] +
                scores['sentiment_score'] * self.analysis_weights['sentiment'] +
                scores['volume_score'] * 0.05 +
                scores['momentum_score'] * 0.05
            )
            
            # Determine overall signal direction
            buy_signals = 0
            sell_signals = 0
            
            for signal_type in signals.values():
                for signal in signal_type:
                    if isinstance(signal, tuple) and len(signal) > 0:
                        if signal[0] == 'BUY':
                            buy_signals += 1
                        elif signal[0] == 'SELL':
                            sell_signals += 1
                    elif isinstance(signal, dict):
                        if signal.get('type') == 'BUY':
                            buy_signals += 1
                        elif signal.get('type') == 'SELL':
                            sell_signals += 1
            
            overall_signal = 'BUY' if buy_signals > sell_signals else 'SELL' if sell_signals > buy_signals else 'NEUTRAL'
            
            return {
                'overall_confidence': min(overall_confidence, 95),
                'overall_signal': overall_signal,
                'component_scores': scores,
                'all_signals': signals,
                'signal_agreement': abs(buy_signals - sell_signals),
                'total_signals': buy_signals + sell_signals
            }
            
        except Exception as e:
            print(f"❌ Error calculating composite score: {e}")
            return {
                'overall_confidence': 50,
                'overall_signal': 'NEUTRAL',
                'component_scores': {},
                'all_signals': {},
                'signal_agreement': 0,
                'total_signals': 0
            }
    
    def create_enhanced_suggestion(self, symbol, market_data, composite_score, 
                                 technical_analysis, seasonal_analysis, sentiment_analysis):
        """Create detailed trading suggestion with all analysis inputs"""
        try:
            if composite_score['overall_signal'] == 'NEUTRAL':
                return None
            
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None
            
            # Determine timeframe based on signal sources
            timeframes = []
            for signals in composite_score['all_signals'].values():
                for signal in signals:
                    if isinstance(signal, dict) and 'timeframe' in signal:
                        timeframes.append(signal['timeframe'])
            
            # Default timeframe logic
            if composite_score['component_scores'].get('momentum_score', 50) > 70:
                primary_timeframe = 'daily'
            elif composite_score['component_scores'].get('seasonal_score', 50) > 65:
                primary_timeframe = 'weekly'
            else:
                primary_timeframe = 'daily'
            
            # Calculate entry, target, and stop loss
            signal_type = composite_score['overall_signal']
            
            if signal_type == 'BUY':
                entry_price = current_price
                
                # Dynamic target based on confidence and volatility
                target_multiplier = 1.02 + (composite_score['overall_confidence'] - 50) / 1000
                target_price = entry_price * target_multiplier
                
                # Dynamic stop loss
                stop_multiplier = 0.98 - (composite_score['overall_confidence'] - 50) / 2000
                stop_loss = entry_price * stop_multiplier
                
            else:  # SELL
                entry_price = current_price
                target_multiplier = 0.98 - (composite_score['overall_confidence'] - 50) / 1000
                target_price = entry_price * target_multiplier
                
                stop_multiplier = 1.02 + (composite_score['overall_confidence'] - 50) / 2000
                stop_loss = entry_price * stop_multiplier
            
            # Create comprehensive reasoning
            reasoning_parts = []
            
            # Technical reasoning
            if composite_score['component_scores'].get('technical_score', 50) > 60:
                tech_signals = composite_score['all_signals'].get('technical_signals', [])
                if tech_signals:
                    reasoning_parts.append(f"Technical: {tech_signals[0][1] if len(tech_signals[0]) > 1 else 'Strong indicators'}")
            
            # Seasonal reasoning
            if composite_score['component_scores'].get('seasonal_score', 50) > 60:
                seasonal_signals = composite_score['all_signals'].get('seasonal_signals', [])
                if seasonal_signals:
                    reasoning_parts.append(f"Seasonal: {seasonal_signals[0].get('reason', 'Favorable timing')}")
            
            # Sentiment reasoning
            if abs(composite_score['component_scores'].get('sentiment_score', 50) - 50) > 15:
                sentiment_score = composite_score['component_scores']['sentiment_score']
                if sentiment_score > 65:
                    reasoning_parts.append("Sentiment: Positive news flow")
                elif sentiment_score < 35:
                    reasoning_parts.append("Sentiment: Negative sentiment (contrarian)")
            
            # Volume/momentum reasoning
            if composite_score['component_scores'].get('momentum_score', 50) > 65:
                change_pct = market_data.get('change_pct', 0)
                reasoning_parts.append(f"Momentum: Strong movement ({change_pct:+.1f}%)")
            
            if not reasoning_parts:
                reasoning_parts.append("Multi-factor analysis")
            
            reasoning = " | ".join(reasoning_parts)
            
            # Risk/reward calculation
            if signal_type == 'BUY':
                risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
            else:
                risk_reward = (entry_price - target_price) / (stop_loss - entry_price)
            
            suggestion = {
                'symbol': symbol,
                'action': signal_type,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': composite_score['overall_confidence'],
                'timeframe': primary_timeframe,
                'reasoning': reasoning,
                'risk_reward': max(risk_reward, 0.5),  # Minimum 1:0.5 ratio
                'analysis_breakdown': {
                    'technical_score': composite_score['component_scores'].get('technical_score', 50),
                    'seasonal_score': composite_score['component_scores'].get('seasonal_score', 50),
                    'sentiment_score': composite_score['component_scores'].get('sentiment_score', 50),
                    'momentum_score': composite_score['component_scores'].get('momentum_score', 50)
                },
                'signal_sources': len(composite_score['all_signals']),
                'signal_agreement': composite_score['signal_agreement'],
                'enhanced': True,
                'analysis_timestamp': datetime.now()
            }
            
            return suggestion
            
        except Exception as e:
            print(f"❌ Error creating suggestion for {symbol}: {e}")
            return None
    
    def filter_suggestions_by_timeframe(self, suggestions):
        """Filter and limit suggestions by timeframe"""
        try:
            filtered = {
                'daily': [],
                'weekly': [],
                'monthly': []
            }
            
            # Group by timeframe
            for suggestion in suggestions:
                timeframe = suggestion.get('timeframe', 'daily')
                if timeframe in filtered:
                    filtered[timeframe].append(suggestion)
            
            # Limit per timeframe and combine
            final_suggestions = []
            final_suggestions.extend(filtered['daily'][:4])   # Max 4 daily
            final_suggestions.extend(filtered['weekly'][:3])  # Max 3 weekly  
            final_suggestions.extend(filtered['monthly'][:2]) # Max 2 monthly
            
            # Sort by confidence
            final_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return final_suggestions[:8]  # Max 8 total suggestions
            
        except Exception as e:
            print(f"❌ Error filtering suggestions: {e}")
            return suggestions[:8]
    
    def apply_risk_adjustments(self, suggestions, correlation_results, portfolio_positions):
        """Apply risk adjustments based on correlation analysis"""
        try:
            if not suggestions:
                return suggestions
            
            # Get portfolio correlation risk
            correlation_analyzer = self.correlation_analyzer
            portfolio_risk = correlation_analyzer.get_portfolio_correlation_risk(portfolio_positions)
            
            risk_adjusted = []
            
            for suggestion in suggestions:
                symbol = suggestion['symbol']
                
                # Adjust confidence based on portfolio correlation risk
                if portfolio_risk['risk_level'] == 'HIGH':
                    # Check if this symbol would increase correlation
                    portfolio_symbols = list(portfolio_positions.keys())
                    test_symbols = portfolio_symbols + [symbol]
                    
                    diversification_score = correlation_analyzer.get_diversification_score(test_symbols)
                    
                    if diversification_score < 50:  # Would decrease diversification
                        suggestion['confidence'] *= 0.85  # Reduce confidence by 15%
                        suggestion['reasoning'] += " | Correlation risk considered"
                    else:
                        suggestion['confidence'] *= 1.05  # Small boost for diversification
                        suggestion['reasoning'] += " | Diversification benefit"
                
                # Adjust position size based on correlation
                suggestion['correlation_adjusted'] = True
                suggestion['portfolio_risk_level'] = portfolio_risk['risk_level']
                
                risk_adjusted.append(suggestion)
            
            return risk_adjusted
            
        except Exception as e:
            print(f"❌ Error applying risk adjustments: {e}")
            return suggestions
    
    def get_market_regime_analysis(self):
        """Analyze current market regime"""
        try:
            # Simple market regime detection based on major indices
            regime_indicators = {}
            
            # VIX level (fear gauge)
            try:
                import yfinance as yf
                vix = yf.Ticker('VIX')
                vix_data = vix.history(period="5d")
                
                if not vix_data.empty:
                    current_vix = vix_data['Close'].iloc[-1]
                    
                    if current_vix > 30:
                        regime_indicators['volatility'] = 'HIGH'
                    elif current_vix > 20:
                        regime_indicators['volatility'] = 'MEDIUM'
                    else:
                        regime_indicators['volatility'] = 'LOW'
                        
                    regime_indicators['vix_level'] = current_vix
            except:
                regime_indicators['volatility'] = 'MEDIUM'
                regime_indicators['vix_level'] = 20
            
            # Market trend (SPY)
            try:
                spy = yf.Ticker('SPY')
                spy_data = spy.history(period="1mo")
                
                if not spy_data.empty and len(spy_data) > 10:
                    # Simple trend detection
                    sma_10 = spy_data['Close'].rolling(10).mean().iloc[-1]
                    sma_20 = spy_data['Close'].rolling(20).mean().iloc[-1]
                    current_price = spy_data['Close'].iloc[-1]
                    
                    if current_price > sma_10 > sma_20:
                        regime_indicators['trend'] = 'BULLISH'
                    elif current_price < sma_10 < sma_20:
                        regime_indicators['trend'] = 'BEARISH'
                    else:
                        regime_indicators['trend'] = 'SIDEWAYS'
            except:
                regime_indicators['trend'] = 'SIDEWAYS'
            
            # Overall regime
            if (regime_indicators.get('volatility') == 'LOW' and 
                regime_indicators.get('trend') == 'BULLISH'):
                overall_regime = 'BULL_MARKET'
            elif (regime_indicators.get('volatility') == 'HIGH' and 
                  regime_indicators.get('trend') == 'BEARISH'):
                overall_regime = 'BEAR_MARKET'
            elif regime_indicators.get('volatility') == 'HIGH':
                overall_regime = 'HIGH_VOLATILITY'
            else:
                overall_regime = 'NEUTRAL'
            
            regime_indicators['overall_regime'] = overall_regime
            regime_indicators['analysis_time'] = datetime.now()
            
            return regime_indicators
            
        except Exception as e:
            print(f"❌ Error analyzing market regime: {e}")
            return {
                'volatility': 'MEDIUM',
                'trend': 'SIDEWAYS',
                'overall_regime': 'NEUTRAL',
                'vix_level': 20,
                'analysis_time': datetime.now()
            }
    
    def generate_market_summary(self):
        """Generate comprehensive market summary"""
        try:
            # Get market regime
            market_regime = self.get_market_regime_analysis()
            
            # Get sector rotation signals
            sector_signals = self.seasonal_analyzer.get_sector_rotation_signals()
            
            # Get overall market sentiment
            market_sentiment = self.sentiment_analyzer.get_market_sentiment()
            
            summary = {
                'market_regime': market_regime,
                'sector_rotation': sector_signals,
                'market_sentiment': market_sentiment,
                'trading_recommendations': self.generate_regime_based_recommendations(market_regime),
                'summary_timestamp': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating market summary: {e}")
            return {}
    
    def generate_regime_based_recommendations(self, market_regime):
        """Generate trading recommendations based on market regime"""
        recommendations = []
        
        regime = market_regime.get('overall_regime', 'NEUTRAL')
        volatility = market_regime.get('volatility', 'MEDIUM')
        
        if regime == 'BULL_MARKET':
            recommendations.extend([
                "Focus on growth stocks and momentum plays",
                "Consider leveraged ETFs for amplified gains",
                "Reduce hedge positions",
                "Look for breakout patterns"
            ])
        
        elif regime == 'BEAR_MARKET':
            recommendations.extend([
                "Emphasize defensive positions",
                "Consider short positions or inverse ETFs",
                "Focus on high-quality dividend stocks",
                "Increase cash reserves"
            ])
        
        elif regime == 'HIGH_VOLATILITY':
            recommendations.extend([
                "Reduce position sizes",
                "Use tighter stop losses",
                "Focus on range-trading strategies",
                "Avoid overnight positions"
            ])
        
        else:  # NEUTRAL
            recommendations.extend([
                "Balanced approach across sectors",
                "Focus on technical analysis",
                "Look for seasonal opportunities",
                "Maintain normal risk levels"
            ])
        
        if volatility == 'HIGH':
            recommendations.append("Consider volatility as an opportunity with proper risk management")
        
        return recommendations

if __name__ == "__main__":
    # Test the enhanced suggestion engine
    engine = EnhancedSuggestionEngine()
    
    print("Testing enhanced suggestion engine...")
    
    # Test market regime analysis
    market_regime = engine.get_market_regime_analysis()
    print(f"Market regime: {market_regime.get('overall_regime', 'Unknown')}")
    print(f"Volatility: {market_regime.get('volatility', 'Unknown')}")
    
    # Test market summary
    market_summary = engine.generate_market_summary()
    print(f"Market summary generated at: {market_summary.get('summary_timestamp', 'Unknown')}")
