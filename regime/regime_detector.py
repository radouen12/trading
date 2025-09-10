"""
Market Regime Detection Module - Phase 4
Advanced market regime classification using ML
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Advanced market regime detection using machine learning
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.regime_models = {}
        self.scaler = StandardScaler()
        self.current_regimes = {}
        
        # Regime definitions
        self.regime_types = {
            0: 'BULL_MARKET',
            1: 'BEAR_MARKET', 
            2: 'SIDEWAYS',
            3: 'HIGH_VOLATILITY',
            4: 'LOW_VOLATILITY',
            5: 'TRENDING',
            6: 'MEAN_REVERTING'
        }
        
        # Feature parameters
        self.lookback_periods = [5, 10, 20, 50, 100]

    def extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        try:
            features = pd.DataFrame(index=data.index)
            features['returns'] = data['close'].pct_change()
            
            # Trend features
            for period in [10, 20, 50]:
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
                features[f'price_sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
                
            # Volatility features
            for period in [10, 20]:
                features[f'volatility_{period}'] = features['returns'].rolling(period).std()
                
            # Technical indicators
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            return features
            
        except Exception as e:
            print(f"Error extracting regime features: {e}")
            return pd.DataFrame()

    def classify_regime_manually(self, data: pd.DataFrame) -> pd.Series:
        """Create manual regime labels for training data"""
        try:
            regimes = pd.Series(index=data.index, dtype=int)
            returns_20 = data['close'].pct_change(20)
            volatility_20 = data['close'].pct_change().rolling(20).std()
            
            high_vol_threshold = 0.02
            trend_threshold = 0.1
            
            for i in range(len(data)):
                current_vol = volatility_20.iloc[i] if not pd.isna(volatility_20.iloc[i]) else 0
                current_return = returns_20.iloc[i] if not pd.isna(returns_20.iloc[i]) else 0
                
                if current_vol > high_vol_threshold:
                    regimes.iloc[i] = 3  # HIGH_VOLATILITY
                elif current_vol < high_vol_threshold * 0.5:
                    regimes.iloc[i] = 4  # LOW_VOLATILITY
                elif current_return > trend_threshold:
                    regimes.iloc[i] = 0  # BULL_MARKET
                elif current_return < -trend_threshold:
                    regimes.iloc[i] = 1  # BEAR_MARKET
                else:
                    regimes.iloc[i] = 2  # SIDEWAYS
            
            return regimes
            
        except Exception as e:
            print(f"Error classifying regimes manually: {e}")
            return pd.Series(dtype=int)

    def train_regime_classifier(self, symbol: str) -> Dict:
        """Train ML classifier for regime detection"""
        try:
            print(f"Training regime classifier for {symbol}...")
            
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            
            data = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
            conn.close()
            
            if len(data) < 200:
                return {"error": f"Insufficient data for {symbol}"}
            
            features = self.extract_regime_features(data)
            regime_labels = self.classify_regime_manually(data)
            
            feature_cols = [col for col in features.columns if features[col].dtype in ['float64', 'int64']]
            X = features[feature_cols].copy()
            y = regime_labels.copy()
            
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                return {"error": f"Insufficient clean data for {symbol}"}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            classifier.fit(X_train_scaled, y_train)
            
            train_accuracy = classifier.score(X_train_scaled, y_train)
            test_accuracy = classifier.score(X_test_scaled, y_test)
            
            self.regime_models[symbol] = {
                'classifier': classifier,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'training_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
            
            return {
                'symbol': symbol,
                'training_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            print(f"Error training regime classifier for {symbol}: {e}")
            return {"error": str(e)}

    def detect_current_regime(self, symbol: str) -> Dict:
        """Detect current market regime for a symbol"""
        try:
            if symbol not in self.regime_models:
                train_result = self.train_regime_classifier(symbol)
                if 'error' in train_result:
                    return train_result
            
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT 100
            """
            
            data = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
            conn.close()
            
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            data = data.sort_index()
            features = self.extract_regime_features(data)
            
            model_data = self.regime_models[symbol]
            classifier = model_data['classifier']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']
            
            latest_features = features[feature_cols].iloc[-1:]
            
            if latest_features.isna().any().any():
                return {"error": f"Missing feature values for {symbol}"}
            
            latest_scaled = scaler.transform(latest_features)
            regime_prob = classifier.predict_proba(latest_scaled)[0]
            regime_prediction = classifier.predict(latest_scaled)[0]
            
            regime_name = self.regime_types[regime_prediction]
            confidence = max(regime_prob)
            
            self.current_regimes[symbol] = {
                'regime': regime_name,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'symbol': symbol,
                'current_regime': regime_name,
                'regime_id': regime_prediction,
                'confidence': confidence,
                'regime_probabilities': dict(zip(
                    [self.regime_types[i] for i in range(len(regime_prob))],
                    regime_prob
                )),
                'current_price': data['close'].iloc[-1],
                'detection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error detecting regime for {symbol}: {e}")
            return {"error": str(e)}

    def get_regime_insights(self, symbol: str) -> Dict:
        """Get comprehensive regime insights for trading strategy"""
        try:
            current_regime = self.detect_current_regime(symbol)
            if 'error' in current_regime:
                return current_regime
            
            regime_name = current_regime['current_regime']
            confidence = current_regime['confidence']
            
            trading_insights = self._generate_regime_trading_insights(regime_name, confidence)
            
            return {
                'symbol': symbol,
                'current_regime_analysis': current_regime,
                'trading_insights': trading_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting regime insights for {symbol}: {e}")
            return {"error": str(e)}

    def _generate_regime_trading_insights(self, regime: str, confidence: float) -> Dict:
        """Generate trading insights based on current regime"""
        insights = {
            'regime': regime,
            'confidence': confidence
        }
        
        if regime == 'BULL_MARKET':
            insights.update({
                'strategy': 'TREND_FOLLOWING',
                'position_bias': 'LONG',
                'risk_level': 'MODERATE',
                'recommended_timeframe': 'MEDIUM_TO_LONG',
                'key_signals': ['Breakouts', 'Momentum', 'Pullback entries'],
                'avoid': ['Counter-trend trades', 'Short positions']
            })
            
        elif regime == 'BEAR_MARKET':
            insights.update({
                'strategy': 'DEFENSIVE',
                'position_bias': 'SHORT_OR_CASH',
                'risk_level': 'HIGH',
                'recommended_timeframe': 'SHORT_TO_MEDIUM',
                'key_signals': ['Breakdown confirmations', 'Bear flag patterns'],
                'avoid': ['Catching falling knives', 'Heavy long exposure']
            })
            
        elif regime == 'SIDEWAYS':
            insights.update({
                'strategy': 'RANGE_TRADING',
                'position_bias': 'NEUTRAL',
                'risk_level': 'MODERATE',
                'recommended_timeframe': 'SHORT',
                'key_signals': ['Support/resistance bounces', 'Mean reversion'],
                'avoid': ['Trend following', 'Breakout trades']
            })
            
        elif regime == 'HIGH_VOLATILITY':
            insights.update({
                'strategy': 'VOLATILITY_BASED',
                'position_bias': 'REDUCED_SIZE',
                'risk_level': 'VERY_HIGH',
                'recommended_timeframe': 'VERY_SHORT',
                'key_signals': ['Volatility spikes', 'Quick reversals'],
                'avoid': ['Large positions', 'Holding overnight']
            })
            
        elif regime == 'LOW_VOLATILITY':
            insights.update({
                'strategy': 'MOMENTUM',
                'position_bias': 'MODERATE_SIZE',
                'risk_level': 'LOW',
                'recommended_timeframe': 'MEDIUM',
                'key_signals': ['Gradual trends', 'Low-risk breakouts'],
                'avoid': ['Expecting large moves', 'High volatility strategies']
            })
            
        else:  # TRENDING or MEAN_REVERTING
            insights.update({
                'strategy': 'ADAPTIVE',
                'position_bias': 'DIRECTION_DEPENDENT',
                'risk_level': 'MODERATE',
                'recommended_timeframe': 'VARIABLE',
                'key_signals': ['Regime-appropriate patterns'],
                'avoid': ['One-size-fits-all approaches']
            })
        
        if confidence < 0.6:
            insights['risk_level'] = 'HIGHER_THAN_NORMAL'
            insights['recommended_position_size'] = 'REDUCED'
        
        return insights

    def batch_regime_analysis(self, symbols: List[str]) -> Dict:
        """Analyze regimes for multiple symbols"""
        results = {
            'regime_analysis': {},
            'market_overview': {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        regime_counts = {regime: 0 for regime in self.regime_types.values()}
        successful_analyses = 0
        
        for symbol in symbols:
            try:
                print(f"Analyzing regime for {symbol}...")
                regime_analysis = self.get_regime_insights(symbol)
                
                if 'error' not in regime_analysis:
                    results['regime_analysis'][symbol] = regime_analysis
                    current_regime = regime_analysis['current_regime_analysis']['current_regime']
                    regime_counts[current_regime] += 1
                    successful_analyses += 1
                else:
                    results['regime_analysis'][symbol] = regime_analysis
                    
            except Exception as e:
                results['regime_analysis'][symbol] = {"error": str(e)}
        
        if successful_analyses > 0:
            results['market_overview'] = {
                'dominant_regime': max(regime_counts, key=regime_counts.get),
                'regime_distribution': regime_counts,
                'market_sentiment': self._assess_market_sentiment(regime_counts, successful_analyses)
            }
        
        return results

    def _assess_market_sentiment(self, regime_counts: Dict, total_symbols: int) -> str:
        """Assess overall market sentiment based on regime distribution"""
        if total_symbols == 0:
            return 'UNKNOWN'
        
        bull_ratio = regime_counts.get('BULL_MARKET', 0) / total_symbols
        bear_ratio = regime_counts.get('BEAR_MARKET', 0) / total_symbols
        vol_ratio = regime_counts.get('HIGH_VOLATILITY', 0) / total_symbols
        
        if bull_ratio > 0.4:
            return 'BULLISH'
        elif bear_ratio > 0.4:
            return 'BEARISH'
        elif vol_ratio > 0.3:
            return 'VOLATILE'
        else:
            return 'MIXED'
