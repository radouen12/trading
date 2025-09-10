"""
Machine Learning Price Predictor - Phase 4
Advanced ML models for directional price prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import sqlite3

warnings.filterwarnings('ignore')

class PricePredictor:
    """
    Advanced ML-based price prediction system using ensemble methods
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'scaler': StandardScaler()
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'scaler': StandardScaler()
            },
            'ridge': {
                'model': Ridge(alpha=1.0),
                'scaler': StandardScaler()
            }
        }
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.prediction_horizons = [1, 3, 5, 10]  # days ahead
        
    def create_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML training
        """
        try:
            features = data.copy()
            
            # Price-based features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            
            # Technical indicators
            for period in self.lookback_periods:
                # Moving averages
                features[f'sma_{period}'] = features['close'].rolling(period).mean()
                features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
                
                # Price ratios
                features[f'price_sma_ratio_{period}'] = features['close'] / features[f'sma_{period}']
                
                # Volatility measures
                features[f'volatility_{period}'] = features['returns'].rolling(period).std()
                
                # RSI
                delta = features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                bb_middle = features[f'sma_{period}']
                bb_std = features['close'].rolling(period).std()
                features[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                features[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                features[f'bb_position_{period}'] = (features['close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
            
            # Volume features (if available)
            if 'volume' in features.columns:
                features['volume_ma_20'] = features['volume'].rolling(20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_ma_20']
                features['price_volume'] = features['close'] * features['volume']
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = features['close'].shift(lag)
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume'].shift(lag) if 'volume' in features.columns else 0
            
            # Time-based features
            features['day_of_week'] = pd.to_datetime(features.index).dayofweek
            features['month'] = pd.to_datetime(features.index).month
            features['quarter'] = pd.to_datetime(features.index).quarter
            
            # Market microstructure features
            features['high_low_ratio'] = features['high'] / features['low']
            features['open_close_ratio'] = features['open'] / features['close']
            
            # Target variables (future returns)
            for horizon in self.prediction_horizons:
                features[f'target_{horizon}d'] = features['close'].shift(-horizon) / features['close'] - 1
                features[f'target_direction_{horizon}d'] = (features[f'target_{horizon}d'] > 0).astype(int)
            
            return features
            
        except Exception as e:
            print(f"Error creating features for {symbol}: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, symbol: str, prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for a specific symbol and prediction horizon
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
            
            if data.empty:
                return np.array([]), np.array([])
            
            # Create features
            features = self.create_features(data, symbol)
            
            # Select feature columns (exclude targets and non-numeric)
            feature_cols = [col for col in features.columns 
                          if not col.startswith('target_') and 
                          col not in ['open', 'high', 'low', 'close', 'volume'] and
                          features[col].dtype in ['float64', 'int64']]
            
            X = features[feature_cols].copy()
            y = features[f'target_{prediction_horizon}d'].copy()
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            return X.values, y.values
            
        except Exception as e:
            print(f"Error preparing training data for {symbol}: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, symbol: str, prediction_horizon: int = 5) -> Dict:
        """
        Train multiple ML models for a specific symbol
        """
        try:
            print(f"Training ML models for {symbol} (horizon: {prediction_horizon} days)...")
            
            # Prepare data
            X, y = self.prepare_training_data(symbol, prediction_horizon)
            
            if len(X) < 100:  # Minimum data requirement
                return {"error": f"Insufficient data for {symbol} (need at least 100 samples, got {len(X)})"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            model_key = f"{symbol}_{prediction_horizon}d"
            self.models[model_key] = {}
            self.scalers[model_key] = {}
            self.model_performance[model_key] = {}
            
            # Train each model
            for model_name, config in self.model_configs.items():
                print(f"  Training {model_name}...")
                
                # Scale features
                scaler = config['scaler']
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = config['model']
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                
                performance = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'directional_accuracy': np.mean(np.sign(y_test) == np.sign(y_pred)),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                # Store model and performance
                self.models[model_key][model_name] = model
                self.scalers[model_key][model_name] = scaler
                self.model_performance[model_key][model_name] = performance
                
                print(f"    MAE: {performance['mae']:.4f}")
                print(f"    Directional Accuracy: {performance['directional_accuracy']:.2%}")
            
            return self.model_performance[model_key]
            
        except Exception as e:
            print(f"Error training models for {symbol}: {e}")
            return {"error": str(e)}
    
    def predict_price(self, symbol: str, prediction_horizon: int = 5) -> Dict:
        """
        Generate price predictions using ensemble of trained models
        """
        try:
            model_key = f"{symbol}_{prediction_horizon}d"
            
            if model_key not in self.models:
                # Train models if not available
                self.train_models(symbol, prediction_horizon)
            
            if model_key not in self.models or not self.models[model_key]:
                return {"error": f"No trained models available for {symbol}"}
            
            # Get latest data for prediction
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
            
            # Sort data chronologically for feature creation
            data = data.sort_index()
            
            # Create features
            features = self.create_features(data, symbol)
            
            # Get latest feature vector
            feature_cols = [col for col in features.columns 
                          if not col.startswith('target_') and 
                          col not in ['open', 'high', 'low', 'close', 'volume'] and
                          features[col].dtype in ['float64', 'int64']]
            
            latest_features = features[feature_cols].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return {"error": f"Missing feature values for {symbol}"}
            
            # Generate predictions from all models
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models[model_key].items():
                scaler = self.scalers[model_key][model_name]
                
                # Scale features
                latest_scaled = scaler.transform(latest_features)
                
                # Predict
                pred = model.predict(latest_scaled)[0]
                
                # Calculate confidence based on model performance
                performance = self.model_performance[model_key][model_name]
                confidence = performance['directional_accuracy'] * (1 - performance['mae'])
                
                predictions[model_name] = pred
                confidences[model_name] = confidence
            
            # Ensemble prediction (weighted average)
            total_confidence = sum(confidences.values())
            if total_confidence > 0:
                ensemble_prediction = sum(
                    pred * (conf / total_confidence) 
                    for pred, conf in zip(predictions.values(), confidences.values())
                )
            else:
                ensemble_prediction = np.mean(list(predictions.values()))
            
            # Current price
            current_price = data['close'].iloc[-1]
            
            # Predicted price
            predicted_price = current_price * (1 + ensemble_prediction)
            
            # Direction and confidence
            direction = "UP" if ensemble_prediction > 0 else "DOWN"
            avg_confidence = np.mean(list(confidences.values()))
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "predicted_return": ensemble_prediction,
                "direction": direction,
                "confidence": avg_confidence,
                "horizon_days": prediction_horizon,
                "individual_predictions": predictions,
                "model_confidences": confidences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error predicting price for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self, symbol: str, prediction_horizon: int = 5) -> Dict:
        """
        Get feature importance from trained models
        """
        try:
            model_key = f"{symbol}_{prediction_horizon}d"
            
            if model_key not in self.models:
                return {"error": f"No trained models for {symbol}"}
            
            importance_dict = {}
            
            for model_name, model in self.models[model_key].items():
                if hasattr(model, 'feature_importances_'):
                    # Get feature names
                    X, y = self.prepare_training_data(symbol, prediction_horizon)
                    features = self.create_features(pd.DataFrame(), symbol)
                    feature_cols = [col for col in features.columns 
                                  if not col.startswith('target_') and 
                                  col not in ['open', 'high', 'low', 'close', 'volume'] and
                                  features[col].dtype in ['float64', 'int64']]
                    
                    if len(feature_cols) == len(model.feature_importances_):
                        importance_dict[model_name] = dict(zip(feature_cols, model.feature_importances_))
            
            return importance_dict
            
        except Exception as e:
            print(f"Error getting feature importance for {symbol}: {e}")
            return {"error": str(e)}
    
    def batch_predict(self, symbols: List[str], prediction_horizon: int = 5) -> Dict:
        """
        Generate predictions for multiple symbols
        """
        predictions = {}
        
        for symbol in symbols:
            print(f"Generating ML prediction for {symbol}...")
            predictions[symbol] = self.predict_price(symbol, prediction_horizon)
        
        return predictions
    
    def save_models(self, filepath: str = "ml/trained_models.joblib"):
        """
        Save trained models to disk
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            print(f"Models saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, filepath: str = "ml/trained_models.joblib"):
        """
        Load trained models from disk
        """
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.model_performance = model_data.get('performance', {})
                
                print(f"Models loaded from {filepath}")
                print(f"Available models: {list(self.models.keys())}")
                
                return True
            else:
                print(f"Model file not found: {filepath}")
                return False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
