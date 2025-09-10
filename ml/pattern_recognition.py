"""
Pattern Recognition Module - Phase 4 (Continued)
Advanced pattern detection using machine learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class PatternRecognizer:
    """
    Advanced pattern recognition for trading signals using ML clustering
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.pattern_models = {}
        self.pattern_library = {}
        self.scaler = StandardScaler()
        
        # Pattern parameters
        self.pattern_length = 20  # Number of candles to analyze
        self.similarity_threshold = 0.8
    
    def _detect_flags(self, prices: np.ndarray) -> List[Dict]:
        """
        Detect Flag patterns
        """
        patterns = []
        
        try:
            for i in range(30, len(prices) - 10):
                # Look for strong trend before flag
                trend_window = prices[i-30:i-10]
                flag_window = prices[i-10:i]
                
                # Calculate trend strength
                trend_slope = np.polyfit(range(len(trend_window)), trend_window, 1)[0]
                flag_slope = np.polyfit(range(len(flag_window)), flag_window, 1)[0]
                
                # Strong trend followed by consolidation
                if abs(trend_slope) > 0.5 and abs(flag_slope) < 0.1:
                    flag_type = 'bull_flag' if trend_slope > 0 else 'bear_flag'
                    
                    patterns.append({
                        'type': flag_type,
                        'trend_start': i - 30,
                        'flag_start': i - 10,
                        'flag_end': i,
                        'trend_slope': trend_slope,
                        'flag_slope': flag_slope,
                        'confidence': min(abs(trend_slope) / 2, 0.9),
                        'breakout_level': flag_window[-1]
                    })
        
        except Exception as e:
            print(f"Error in flag detection: {e}")
        
        return patterns
    
    def _detect_wedges(self, highs: np.ndarray, lows: np.ndarray) -> List[Dict]:
        """
        Detect Wedge patterns (Rising/Falling)
        """
        patterns = []
        window_size = 25
        
        try:
            for i in range(window_size, len(highs) - 5):
                window_highs = highs[i-window_size:i]
                window_lows = lows[i-window_size:i]
                
                # Calculate trend lines
                high_trend = np.polyfit(range(len(window_highs)), window_highs, 1)
                low_trend = np.polyfit(range(len(window_lows)), window_lows, 1)
                
                high_slope = high_trend[0]
                low_slope = low_trend[0]
                
                # Both lines converging
                if (high_slope > 0 and low_slope > 0 and high_slope < low_slope):  # Rising wedge
                    wedge_type = 'rising_wedge'
                    confidence = 0.6
                elif (high_slope < 0 and low_slope < 0 and abs(high_slope) < abs(low_slope)):  # Falling wedge
                    wedge_type = 'falling_wedge'
                    confidence = 0.6
                else:
                    continue
                
                patterns.append({
                    'type': wedge_type,
                    'start_index': i - window_size,
                    'end_index': i,
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'confidence': confidence,
                    'breakout_level': (window_highs[-1] + window_lows[-1]) / 2
                })
        
        except Exception as e:
            print(f"Error in wedge detection: {e}")
        
        return patterns
    
    def _calculate_pattern_confidence(self, pattern_type: str, key_points: List) -> float:
        """
        Calculate confidence score for detected patterns
        """
        try:
            base_confidence = {
                'head_shoulders': 0.75,
                'double_top': 0.70,
                'double_bottom': 0.70,
                'triangle': 0.60,
                'flag': 0.65,
                'wedge': 0.55
            }
            
            confidence = base_confidence.get(pattern_type, 0.5)
            
            # Adjust based on pattern quality
            if len(key_points) >= 3:
                # More defined patterns get higher confidence
                confidence += 0.1
            
            return min(confidence, 0.95)
            
        except Exception:
            return 0.5
    
    def cluster_patterns(self, symbol: str, n_clusters: int = 8) -> Dict:
        """
        Use ML clustering to find similar patterns
        """
        try:
            # Get historical data
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            
            data = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
            conn.close()
            
            if len(data) < 100:
                return {"error": f"Insufficient data for {symbol}"}
            
            # Extract patterns
            patterns = self.extract_price_patterns(data)
            
            if len(patterns) == 0:
                return {"error": f"No patterns extracted for {symbol}"}
            
            # Remove any NaN values
            patterns = patterns[~np.isnan(patterns).any(axis=1)]
            
            if len(patterns) < n_clusters:
                n_clusters = max(2, len(patterns) // 5)
            
            # Scale patterns
            patterns_scaled = self.scaler.fit_transform(patterns)
            
            # Reduce dimensionality
            pca = PCA(n_components=min(10, patterns.shape[1]))
            patterns_pca = pca.fit_transform(patterns_scaled)
            
            # Cluster patterns
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(patterns_pca)
            
            # Calculate silhouette score
            silhouette = silhouette_score(patterns_pca, cluster_labels)
            
            # Analyze clusters
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_patterns = patterns[cluster_labels == i]
                cluster_size = len(cluster_patterns)
                
                if cluster_size > 0:
                    # Calculate cluster characteristics
                    cluster_center = kmeans.cluster_centers_[i]
                    
                    # Find patterns closest to center
                    distances = np.linalg.norm(patterns_pca[cluster_labels == i] - cluster_center, axis=1)
                    representative_idx = np.argmin(distances)
                    representative_pattern = cluster_patterns[representative_idx]
                    
                    cluster_analysis[f'cluster_{i}'] = {
                        'size': cluster_size,
                        'percentage': cluster_size / len(patterns) * 100,
                        'representative_pattern': representative_pattern.tolist(),
                        'avg_volatility': np.std(representative_pattern),
                        'pattern_strength': np.mean(np.abs(representative_pattern))
                    }
            
            # Store model for future use
            self.pattern_models[symbol] = {
                'kmeans': kmeans,
                'scaler': self.scaler,
                'pca': pca,
                'cluster_labels': cluster_labels,
                'patterns': patterns
            }
            
            return {
                'symbol': symbol,
                'n_patterns': len(patterns),
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'cluster_analysis': cluster_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error clustering patterns for {symbol}: {e}")
            return {"error": str(e)}
    
    def find_similar_patterns(self, symbol: str, current_pattern: np.ndarray, top_k: int = 5) -> Dict:
        """
        Find historical patterns similar to current market state
        """
        try:
            if symbol not in self.pattern_models:
                # Run clustering first
                self.cluster_patterns(symbol)
            
            if symbol not in self.pattern_models:
                return {"error": f"No pattern model available for {symbol}"}
            
            model_data = self.pattern_models[symbol]
            historical_patterns = model_data['patterns']
            scaler = model_data['scaler']
            pca = model_data['pca']
            
            # Scale and transform current pattern
            current_scaled = scaler.transform(current_pattern.reshape(1, -1))
            current_pca = pca.transform(current_scaled)
            
            # Scale and transform historical patterns
            historical_scaled = scaler.transform(historical_patterns)
            historical_pca = pca.transform(historical_scaled)
            
            # Calculate similarities
            similarities = []
            for i, hist_pattern in enumerate(historical_pca):
                # Cosine similarity
                cosine_sim = np.dot(current_pca[0], hist_pattern) / (
                    np.linalg.norm(current_pca[0]) * np.linalg.norm(hist_pattern)
                )
                
                # Euclidean distance (inverted for similarity)
                euclidean_dist = np.linalg.norm(current_pca[0] - hist_pattern)
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # Combined similarity
                combined_sim = (cosine_sim + euclidean_sim) / 2
                
                similarities.append({
                    'index': i,
                    'cosine_similarity': cosine_sim,
                    'euclidean_similarity': euclidean_sim,
                    'combined_similarity': combined_sim,
                    'pattern': historical_patterns[i].tolist()
                })
            
            # Sort by combined similarity
            similarities.sort(key=lambda x: x['combined_similarity'], reverse=True)
            
            # Return top-k similar patterns
            top_similar = similarities[:top_k]
            
            return {
                'symbol': symbol,
                'current_pattern_length': len(current_pattern),
                'total_historical_patterns': len(similarities),
                'top_similar_patterns': top_similar,
                'avg_similarity': np.mean([s['combined_similarity'] for s in top_similar]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error finding similar patterns for {symbol}: {e}")
            return {"error": str(e)}
    
    def predict_from_patterns(self, symbol: str) -> Dict:
        """
        Generate predictions based on pattern analysis
        """
        try:
            # Get recent data for current pattern
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT {self.pattern_length + 10}
            """
            
            data = pd.read_sql(query, conn, index_col='timestamp', parse_dates=['timestamp'])
            conn.close()
            
            if len(data) < self.pattern_length:
                return {"error": f"Insufficient recent data for {symbol}"}
            
            # Sort chronologically and get current pattern
            data = data.sort_index()
            current_data = data.iloc[-self.pattern_length:]
            
            # Extract current pattern
            current_patterns = self.extract_price_patterns(pd.concat([data.iloc[:-1], current_data]))
            if len(current_patterns) == 0:
                return {"error": f"Could not extract current pattern for {symbol}"}
            
            current_pattern = current_patterns[-1]
            
            # Detect chart patterns in recent data
            chart_patterns = self.detect_chart_patterns(data.iloc[-50:])  # Last 50 candles
            
            # Find similar historical patterns
            similar_patterns = self.find_similar_patterns(symbol, current_pattern)
            
            # Generate prediction based on patterns
            predictions = []
            
            # Chart pattern predictions
            for pattern_type, patterns in chart_patterns.items():
                for pattern in patterns:
                    if pattern.get('confidence', 0) > 0.6:
                        predictions.append({
                            'source': 'chart_pattern',
                            'pattern_type': pattern_type,
                            'confidence': pattern['confidence'],
                            'target': pattern.get('target'),
                            'breakout_point': pattern.get('breakout_point'),
                            'direction': 'bullish' if pattern.get('target', 0) > data['close'].iloc[-1] else 'bearish'
                        })
            
            # Similar pattern predictions
            if 'top_similar_patterns' in similar_patterns:
                avg_similarity = similar_patterns.get('avg_similarity', 0)
                if avg_similarity > self.similarity_threshold:
                    # Analyze outcomes of similar patterns
                    historical_outcomes = []
                    for similar in similar_patterns['top_similar_patterns'][:3]:
                        # This would require storing outcome data - simplified for now
                        outcome_confidence = similar['combined_similarity'] * 0.7
                        historical_outcomes.append(outcome_confidence)
                    
                    if historical_outcomes:
                        predictions.append({
                            'source': 'pattern_similarity',
                            'confidence': np.mean(historical_outcomes),
                            'avg_similarity': avg_similarity,
                            'n_similar_patterns': len(similar_patterns['top_similar_patterns']),
                            'direction': 'neutral'  # Would need outcome analysis
                        })
            
            # Combine predictions
            if predictions:
                combined_confidence = np.mean([p['confidence'] for p in predictions])
                dominant_direction = max(set([p.get('direction', 'neutral') for p in predictions]), 
                                       key=[p.get('direction', 'neutral') for p in predictions].count)
            else:
                combined_confidence = 0.3
                dominant_direction = 'neutral'
            
            return {
                'symbol': symbol,
                'current_price': data['close'].iloc[-1],
                'pattern_predictions': predictions,
                'combined_confidence': combined_confidence,
                'predicted_direction': dominant_direction,
                'chart_patterns_detected': len([p for patterns in chart_patterns.values() for p in patterns]),
                'pattern_analysis': {
                    'chart_patterns': chart_patterns,
                    'similar_patterns_summary': {
                        'avg_similarity': similar_patterns.get('avg_similarity', 0),
                        'n_patterns_analyzed': similar_patterns.get('total_historical_patterns', 0)
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error predicting from patterns for {symbol}: {e}")
            return {"error": str(e)}
    
    def analyze_symbol_patterns(self, symbol: str) -> Dict:
        """
        Comprehensive pattern analysis for a symbol
        """
        try:
            print(f"Analyzing patterns for {symbol}...")
            
            # Run all pattern analysis
            results = {
                'symbol': symbol,
                'clustering_analysis': self.cluster_patterns(symbol),
                'pattern_predictions': self.predict_from_patterns(symbol),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing patterns for {symbol}: {e}")
            return {"error": str(e)}
