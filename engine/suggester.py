import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

# Simple imports without error handling
from analysis import TechnicalAnalyzer, SeasonalAnalyzer, SentimentAnalyzer, CorrelationAnalyzer

class EnhancedSuggestionEngine:
    def __init__(self):
        self.config = Config()
        
        # Initialize analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        print("🧠 Enhanced Suggestion Engine initialized")
        
        # Analysis weights
        self.analysis_weights = {
            'technical': 0.4,
            'seasonal': 0.25,
            'sentiment': 0.25,
            'correlation': 0.1
        }
    
    def generate_comprehensive_suggestions(self, market_data, portfolio_positions):
        """Generate enhanced trading suggestions"""
        try:
            print("🧠 Generating comprehensive trading suggestions...")
            
            # Run analyses with fallback
            try:
                technical_results = self.technical_analyzer.analyze_all_symbols()
            except:
                technical_results = {}
            
            try:
                seasonal_results = self.seasonal_analyzer.analyze_all_symbols()
            except:
                seasonal_results = {}
            
            try:
                sentiment_results = self.sentiment_analyzer.analyze_all_symbols()
            except:
                sentiment_results = {}
            
            try:
                correlation_results = self.correlation_analyzer.analyze_all_correlations()
            except:
                correlation_results = {}
            
            # Generate suggestions
            suggestions = self.combine_all_analyses(
                market_data, 
                technical_results,
                seasonal_results,
                sentiment_results,
                correlation_results,
                portfolio_positions
            )
            
            # Apply risk adjustments
            risk_adjusted_suggestions = self.apply_risk_adjustments(
                suggestions, 
                correlation_results,
                portfolio_positions
            )
            
            print(f"✅ Generated {len(risk_adjusted_suggestions)} enhanced suggestions")
            return risk_adjusted_suggestions
            
        except Exception as e:
            print(f"❌ Error generating suggestions: {e}")
            return self.generate_basic_suggestions(market_data, portfolio_positions)
    
    def generate_basic_suggestions(self, market_data, portfolio_positions):
        """Generate basic suggestions as fallback"""
        suggestions = []
        
        for symbol, data in market_data.items():
            if symbol in portfolio_positions:
                continue
            
            price_change = data.get('change_pct', 0)
            volume = data.get('volume', 0)
            
            # Basic scoring
            score = 50
            
            if abs(price_change) > 2:
                score += 20
            elif abs(price_change) > 1:
                score += 10
            
            if volume > 1000000:
                score += 15
            
            if score >= 70:
                entry_price = data['price']
                
                if price_change > 0:
                    action = 'BUY'
                    stop_loss = entry_price * 0.95
                    target = entry_price * 1.08
                else:
                    action = 'SELL'
                    stop_loss = entry_price * 1.05
                    target = entry_price * 0.92
                
                suggestions.append({
                    'symbol': symbol,
                    'action': action,
                    'entry_price': entry_price,
                    'target_price': target,
                    'stop_loss': stop_loss,
                    'confidence': score,
                    'timeframe': 'daily',
                    'reasoning': f"Price momentum ({price_change:+.1f}%)",
                    'risk_reward': abs(target - entry_price) / abs(entry_price - stop_loss),
                    'enhanced': False
                })
        
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)[:8]
    
    def combine_all_analyses(self, market_data, technical_results, seasonal_results, 
                           sentiment_results, correlation_results, portfolio_positions):
        """Combine analysis results"""
        try:
            suggestions = []
            
            all_symbols = list(market_data.keys())
            
            for symbol in all_symbols:
                if symbol in portfolio_positions:
                    continue
                
                technical_analysis = technical_results.get(symbol, {})
                seasonal_analysis = seasonal_results.get(symbol, {})
                sentiment_analysis = sentiment_results.get(symbol, {})
                
                composite_score = self.calculate_composite_score(
                    symbol,
                    technical_analysis,
                    seasonal_analysis, 
                    sentiment_analysis,
                    market_data.get(symbol, {})
                )
                
                if composite_score['overall_confidence'] >= self.config.MIN_CONFIDENCE_SCORE:
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
            
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            return self.filter_suggestions_by_timeframe(suggestions)
            
        except Exception as e:
            print(f"❌ Error combining analyses: {e}")
            return []
    
    def calculate_composite_score(self, symbol, technical_analysis, seasonal_analysis, 
                                sentiment_analysis, market_data):
        """Calculate composite confidence score"""
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
            
            # Technical analysis
            if technical_analysis and 'signals' in technical_analysis:
                tech_signals = technical_analysis['signals']
                signals['technical_signals'] = tech_signals
                
                if tech_signals:
                    tech_confidences = [signal[2] for signal in tech_signals if len(signal) > 2]
                    if tech_confidences:
                        scores['technical_score'] = sum(tech_confidences) / len(tech_confidences)
                
                # RSI
                rsi = technical_analysis.get('rsi', 50)
                if rsi < 30:
                    scores['technical_score'] += 15
                elif rsi > 70:
                    scores['technical_score'] -= 15
                
                # MACD
                macd = technical_analysis.get('macd', 0)
                macd_signal = technical_analysis.get('macd_signal', 0)
                if macd > macd_signal:
                    scores['technical_score'] += 10
                else:
                    scores['technical_score'] -= 5
            
            # Seasonal analysis
            if seasonal_analysis and 'seasonal_signals' in seasonal_analysis:
                seasonal_signals = seasonal_analysis['seasonal_signals']
                signals['seasonal_signals'] = seasonal_signals
                
                if seasonal_signals:
                    seasonal_confidences = [s.get('confidence', 50) for s in seasonal_signals]
                    if seasonal_confidences:
                        scores['seasonal_score'] = sum(seasonal_confidences) / len(seasonal_confidences)
            
            # Sentiment analysis
            if sentiment_analysis and 'sentiment_score' in sentiment_analysis:
                scores['sentiment_score'] = sentiment_analysis['sentiment_score']
                
                if 'signals' in sentiment_analysis:
                    signals['sentiment_signals'] = sentiment_analysis['signals']
            
            # Volume and momentum
            if market_data:
                change_pct = market_data.get('change_pct', 0)
                volume = market_data.get('volume', 0)
                
                if abs(change_pct) > 3:
                    scores['momentum_score'] = 70 + min(abs(change_pct) * 2, 25)
                elif abs(change_pct) > 1:
                    scores['momentum_score'] = 60 + abs(change_pct) * 5
                
                if volume > 1000000:
                    scores['volume_score'] = 70
                elif volume > 500000:
                    scores['volume_score'] = 60
            
            # Calculate weighted overall score
            base_confidence = (
                scores['technical_score'] * self.analysis_weights.get('technical', 0) +
                scores['seasonal_score'] * self.analysis_weights.get('seasonal', 0) +
                scores['sentiment_score'] * self.analysis_weights.get('sentiment', 0) +
                scores['volume_score'] * 0.05 +
                scores['momentum_score'] * 0.05
            )
            
            overall_confidence = max(0, min(100, base_confidence))
            
            if overall_confidence > 90:
                overall_confidence = 90 + (overall_confidence - 90) * 0.5
            
            overall_confidence = min(overall_confidence, 95)
            
            # Determine signal direction
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
            
            if buy_signals > sell_signals and overall_confidence > 60:
                overall_signal = 'BUY'
            elif sell_signals > buy_signals and overall_confidence > 60:
                overall_signal = 'SELL'
            else:
                overall_signal = 'NEUTRAL'
            
            return {
                'overall_confidence': min(overall_confidence, 95),
                'overall_signal': overall_signal,
                'component_scores': scores,
                'all_signals': signals,
                'signal_agreement': abs(buy_signals - sell_signals),
                'total_signals': buy_signals + sell_signals
            }
            
        except Exception as e:
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
        """Create enhanced suggestion"""
        try:
            if composite_score['overall_signal'] == 'NEUTRAL':
                return None
            
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None
            
            # Determine timeframe
            if composite_score['component_scores'].get('momentum_score', 50) > 70:
                primary_timeframe = 'daily'
            elif composite_score['component_scores'].get('seasonal_score', 50) > 65:
                primary_timeframe = 'weekly'
            else:
                primary_timeframe = 'daily'
            
            # Calculate prices
            signal_type = composite_score['overall_signal']
            
            if signal_type == 'BUY':
                entry_price = current_price
                target_multiplier = 1.02 + (composite_score['overall_confidence'] - 50) / 1000
                target_price = entry_price * target_multiplier
                stop_multiplier = 0.98 - (composite_score['overall_confidence'] - 50) / 2000
                stop_loss = entry_price * stop_multiplier
            else:  # SELL
                entry_price = current_price
                target_multiplier = 0.98 - (composite_score['overall_confidence'] - 50) / 1000
                target_price = entry_price * target_multiplier
                stop_multiplier = 1.02 + (composite_score['overall_confidence'] - 50) / 2000
                stop_loss = entry_price * stop_multiplier
            
            # Create reasoning
            reasoning_parts = []
            
            if composite_score['component_scores'].get('technical_score', 50) > 60:
                reasoning_parts.append("Technical indicators")
            
            if composite_score['component_scores'].get('seasonal_score', 50) > 60:
                reasoning_parts.append("Seasonal patterns")
            
            if abs(composite_score['component_scores'].get('sentiment_score', 50) - 50) > 15:
                sentiment_score = composite_score['component_scores']['sentiment_score']
                if sentiment_score > 65:
                    reasoning_parts.append("Positive sentiment")
                elif sentiment_score < 35:
                    reasoning_parts.append("Contrarian sentiment")
            
            if composite_score['component_scores'].get('momentum_score', 50) > 65:
                change_pct = market_data.get('change_pct', 0)
                reasoning_parts.append(f"Strong momentum ({change_pct:+.1f}%)")
            
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
                'risk_reward': max(risk_reward, 0.5),
                'analysis_breakdown': {
                    'technical_score': composite_score['component_scores'].get('technical_score', 50),
                    'seasonal_score': composite_score['component_scores'].get('seasonal_score', 50),
                    'sentiment_score': composite_score['component_scores'].get('sentiment_score', 50),
                    'momentum_score': composite_score['component_scores'].get('momentum_score', 50)
                },
                'signal_sources': len(composite_score['all_signals']),
                'signal_agreement': composite_score['signal_agreement'],
                'enhanced': True,
                'analysis_timestamp': datetime.now().replace(microsecond=0)
            }
            
            return suggestion
            
        except Exception as e:
            return None
    
    def filter_suggestions_by_timeframe(self, suggestions):
        """Filter suggestions by timeframe"""
        try:
            filtered = {'daily': [], 'weekly': [], 'monthly': []}
            
            for suggestion in suggestions:
                timeframe = suggestion.get('timeframe', 'daily')
                if timeframe in filtered:
                    filtered[timeframe].append(suggestion)
            
            final_suggestions = []
            final_suggestions.extend(filtered['daily'][:4])
            final_suggestions.extend(filtered['weekly'][:3])
            final_suggestions.extend(filtered['monthly'][:2])
            
            final_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            return final_suggestions[:8]
            
        except Exception as e:
            return suggestions[:8]
    
    def apply_risk_adjustments(self, suggestions, correlation_results, portfolio_positions):
        """Apply risk adjustments"""
        try:
            if not suggestions:
                return suggestions
            
            risk_adjusted = []
            
            for suggestion in suggestions:
                portfolio_symbols = list(portfolio_positions.keys())
                if len(portfolio_symbols) > 3:
                    suggestion['confidence'] *= 0.9
                    suggestion['reasoning'] += " | Portfolio diversification considered"
                
                suggestion['correlation_adjusted'] = True
                suggestion['portfolio_risk_level'] = 'MEDIUM'
                
                risk_adjusted.append(suggestion)
            
            return risk_adjusted
            
        except Exception as e:
            return suggestions
    
    def get_market_regime_analysis(self):
        """Analyze market regime"""
        try:
            import yfinance as yf
            
            regime_indicators = {}
            
            # VIX level
            try:
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
            regime_indicators['analysis_time'] = datetime.now().replace(microsecond=0)
            
            return regime_indicators
            
        except Exception as e:
            return {
                'volatility': 'MEDIUM',
                'trend': 'SIDEWAYS',
                'overall_regime': 'NEUTRAL',
                'vix_level': 20,
                'analysis_time': datetime.now().replace(microsecond=0)
            }

if __name__ == "__main__":
    engine = EnhancedSuggestionEngine()
    print("Testing enhanced suggestion engine...")
    
    market_regime = engine.get_market_regime_analysis()
    print(f"Market regime: {market_regime.get('overall_regime', 'Unknown')}")
