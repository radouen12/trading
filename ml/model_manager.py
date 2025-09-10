"""
Model Manager - Phase 4
Central management system for ML models and training pipelines
"""

import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
from .price_predictor import PricePredictor
from .pattern_recognition import PatternRecognizer

warnings.filterwarnings('ignore')

class ModelManager:
    """
    Centralized manager for all ML models in the trading system
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.price_predictor = PricePredictor(db_path)
        self.pattern_recognizer = PatternRecognizer(db_path)
        
        # Model storage paths
        self.model_dir = "ml/models"
        self.performance_dir = "ml/performance"
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        
        # Model metadata
        self.model_registry = {}
        self.training_history = {}
        self.performance_metrics = {}
        
    def initialize_models(self, symbols: List[str]) -> Dict:
        """
        Initialize and train models for all symbols
        """
        try:
            print("🤖 Initializing ML Models for Phase 4...")
            
            results = {
                'initialized_symbols': [],
                'failed_symbols': [],
                'total_models_trained': 0,
                'performance_summary': {}
            }
            
            for symbol in symbols:
                try:
                    print(f"\n📊 Processing {symbol}...")
                    
                    # Train price prediction models
                    print(f"  🎯 Training price prediction models...")
                    price_results = {}
                    for horizon in [1, 3, 5, 10]:
                        horizon_result = self.price_predictor.train_models(symbol, horizon)
                        if 'error' not in horizon_result:
                            price_results[f'{horizon}d'] = horizon_result
                            results['total_models_trained'] += len(horizon_result)
                    
                    # Train pattern recognition
                    print(f"  🔍 Training pattern recognition...")
                    pattern_result = self.pattern_recognizer.cluster_patterns(symbol)
                    
                    # Store results
                    if price_results or 'error' not in pattern_result:
                        results['initialized_symbols'].append(symbol)
                        results['performance_summary'][symbol] = {
                            'price_prediction': price_results,
                            'pattern_recognition': pattern_result
                        }
                    else:
                        results['failed_symbols'].append(symbol)
                    
                except Exception as e:
                    print(f"  ❌ Error processing {symbol}: {e}")
                    results['failed_symbols'].append(symbol)
            
            # Save models
            self.save_all_models()
            
            print(f"\n✅ Model initialization complete!")
            print(f"   📈 Symbols processed: {len(results['initialized_symbols'])}")
            print(f"   🎯 Total models trained: {results['total_models_trained']}")
            print(f"   ❌ Failed symbols: {len(results['failed_symbols'])}")
            
            return results
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            return {"error": str(e)}
    
    def generate_ml_predictions(self, symbols: List[str]) -> Dict:
        """
        Generate ML-based predictions for all symbols
        """
        try:
            print("🔮 Generating ML Predictions...")
            
            predictions = {
                'price_predictions': {},
                'pattern_predictions': {},
                'ensemble_predictions': {},
                'generation_timestamp': datetime.now().isoformat()
            }
            
            for symbol in symbols:
                try:
                    print(f"  📊 Analyzing {symbol}...")
                    
                    # Price predictions
                    price_pred = self.price_predictor.predict_price(symbol, prediction_horizon=5)
                    if 'error' not in price_pred:
                        predictions['price_predictions'][symbol] = price_pred
                    
                    # Pattern predictions
                    pattern_pred = self.pattern_recognizer.predict_from_patterns(symbol)
                    if 'error' not in pattern_pred:
                        predictions['pattern_predictions'][symbol] = pattern_pred
                    
                    # Ensemble prediction
                    ensemble_pred = self._create_ensemble_prediction(symbol, price_pred, pattern_pred)
                    predictions['ensemble_predictions'][symbol] = ensemble_pred
                    
                except Exception as e:
                    print(f"    ❌ Error predicting {symbol}: {e}")
                    predictions['ensemble_predictions'][symbol] = {"error": str(e)}
            
            return predictions
            
        except Exception as e:
            print(f"Error generating ML predictions: {e}")
            return {"error": str(e)}
    
    def _create_ensemble_prediction(self, symbol: str, price_pred: Dict, pattern_pred: Dict) -> Dict:
        """
        Create ensemble prediction combining price and pattern predictions
        """
        try:
            ensemble = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'price_prediction': price_pred,
                    'pattern_prediction': pattern_pred
                }
            }
            
            # Extract key metrics
            price_confidence = price_pred.get('confidence', 0) if 'error' not in price_pred else 0
            pattern_confidence = pattern_pred.get('combined_confidence', 0) if 'error' not in pattern_pred else 0
            
            price_direction = price_pred.get('direction', 'NEUTRAL') if 'error' not in price_pred else 'NEUTRAL'
            pattern_direction = pattern_pred.get('predicted_direction', 'neutral') if 'error' not in pattern_pred else 'neutral'
            
            # Combine predictions
            if price_confidence > 0 and pattern_confidence > 0:
                # Both models available
                combined_confidence = (price_confidence + pattern_confidence) / 2
                
                # Direction consensus
                if price_direction == 'UP' and pattern_direction == 'bullish':
                    final_direction = 'BULLISH'
                    combined_confidence *= 1.2  # Boost for consensus
                elif price_direction == 'DOWN' and pattern_direction == 'bearish':
                    final_direction = 'BEARISH'
                    combined_confidence *= 1.2
                else:
                    final_direction = 'NEUTRAL'
                    combined_confidence *= 0.8  # Reduce for disagreement
                
            elif price_confidence > 0:
                # Only price prediction
                combined_confidence = price_confidence * 0.8
                final_direction = 'BULLISH' if price_direction == 'UP' else 'BEARISH' if price_direction == 'DOWN' else 'NEUTRAL'
                
            elif pattern_confidence > 0:
                # Only pattern prediction
                combined_confidence = pattern_confidence * 0.7
                final_direction = 'BULLISH' if pattern_direction == 'bullish' else 'BEARISH' if pattern_direction == 'bearish' else 'NEUTRAL'
                
            else:
                # No reliable predictions
                combined_confidence = 0.3
                final_direction = 'NEUTRAL'
            
            # Cap confidence
            combined_confidence = min(combined_confidence, 0.95)
            
            # Additional metrics
            ensemble.update({
                'ml_confidence': combined_confidence,
                'ml_direction': final_direction,
                'prediction_strength': self._calculate_prediction_strength(combined_confidence, final_direction),
                'recommendation': self._generate_ml_recommendation(symbol, combined_confidence, final_direction, price_pred),
                'risk_assessment': self._assess_ml_risk(combined_confidence, final_direction)
            })
            
            return ensemble
            
        except Exception as e:
            return {"error": f"Ensemble prediction failed: {e}"}
    
    def _calculate_prediction_strength(self, confidence: float, direction: str) -> str:
        """
        Calculate prediction strength category
        """
        if direction == 'NEUTRAL':
            return 'NEUTRAL'
        elif confidence >= 0.8:
            return 'STRONG'
        elif confidence >= 0.6:
            return 'MODERATE'
        elif confidence >= 0.4:
            return 'WEAK'
        else:
            return 'VERY_WEAK'
    
    def save_all_models(self):
        """
        Save all trained models to disk
        """
        try:
            # Save price prediction models
            self.price_predictor.save_models(f"{self.model_dir}/price_models.joblib")
            
            # Save pattern recognition models
            pattern_models_path = f"{self.model_dir}/pattern_models.joblib"
            joblib.dump(self.pattern_recognizer.pattern_models, pattern_models_path)
            
            # Save model registry
            registry_path = f"{self.model_dir}/model_registry.joblib"
            joblib.dump({
                'model_registry': self.model_registry,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'last_saved': datetime.now().isoformat()
            }, registry_path)
            
            print(f"✅ All models saved to {self.model_dir}")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_all_models(self):
        """
        Load all models from disk
        """
        try:
            # Load price prediction models
            price_loaded = self.price_predictor.load_models(f"{self.model_dir}/price_models.joblib")
            
            # Load pattern recognition models
            pattern_models_path = f"{self.model_dir}/pattern_models.joblib"
            if os.path.exists(pattern_models_path):
                self.pattern_recognizer.pattern_models = joblib.load(pattern_models_path)
                pattern_loaded = True
            else:
                pattern_loaded = False
            
            # Load model registry
            registry_path = f"{self.model_dir}/model_registry.joblib"
            if os.path.exists(registry_path):
                registry_data = joblib.load(registry_path)
                self.model_registry = registry_data.get('model_registry', {})
                self.training_history = registry_data.get('training_history', {})
                self.performance_metrics = registry_data.get('performance_metrics', {})
                registry_loaded = True
            else:
                registry_loaded = False
            
            success = price_loaded or pattern_loaded or registry_loaded
            
            if success:
                print(f"✅ Models loaded from {self.model_dir}")
                print(f"   Price models: {'✅' if price_loaded else '❌'}")
                print(f"   Pattern models: {'✅' if pattern_loaded else '❌'}")
                print(f"   Registry: {'✅' if registry_loaded else '❌'}")
            else:
                print(f"⚠️ No models found in {self.model_dir}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
