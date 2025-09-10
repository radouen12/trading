import requests
import pandas as pd
from datetime import datetime, timedelta
from config import Config

# Simple TextBlob fallback
class SimpleSentiment:
    def __init__(self, text):
        self.text = text.lower()
    
    @property
    def sentiment(self):
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'buy', 'bullish', 'up', 'gain', 'profit', 'strong']
        negative_words = ['bad', 'terrible', 'sell', 'bearish', 'down', 'loss', 'weak', 'decline', 'drop']
        
        positive_count = sum(1 for word in positive_words if word in self.text)
        negative_count = sum(1 for word in negative_words if word in self.text)
        
        if positive_count > negative_count:
            polarity = 0.5 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            polarity = -0.5 - (negative_count - positive_count) * 0.1
        else:
            polarity = 0.0
        
        class Sentiment:
            def __init__(self, pol):
                self.polarity = max(-1, min(1, pol))
                self.subjectivity = 0.5
        
        return Sentiment(polarity)

# Try to import TextBlob, fallback to simple version
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = SimpleSentiment
    TEXTBLOB_AVAILABLE = False

class SentimentAnalyzer:
    def __init__(self):
        self.config = Config()
        
        # Sample news headlines for demo (when API not available)
        self.demo_headlines = {
            'AAPL': [
                "Apple reports strong quarterly earnings",
                "iPhone sales exceed expectations",
                "Apple stock upgraded by analysts"
            ],
            'MSFT': [
                "Microsoft cloud revenue grows significantly",
                "Azure continues market expansion",
                "Microsoft AI investments paying off"
            ],
            'GOOGL': [
                "Google advertising revenue stable",
                "Alphabet focuses on AI development",
                "Search business remains strong"
            ],
            'TSLA': [
                "Tesla delivery numbers mixed",
                "Electric vehicle competition increases",
                "Tesla stock shows volatility"
            ],
            'NVDA': [
                "NVIDIA AI chip demand surges",
                "Data center revenue accelerates",
                "NVIDIA stock reaches new highs"
            ]
        }
    
    def get_news_sentiment(self, symbol):
        """Get news sentiment for a symbol"""
        try:
            # Try to get real news if API available
            if self.config.NEWS_API_KEY and self.config.NEWS_API_KEY != "demo":
                return self._get_real_news_sentiment(symbol)
            else:
                return self._get_demo_news_sentiment(symbol)
                
        except Exception as e:
            return self._get_demo_news_sentiment(symbol)
    
    def _get_real_news_sentiment(self, symbol):
        """Get real news sentiment (requires API key)"""
        try:
            # This would use actual News API
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock",
                'apiKey': self.config.NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                headlines = [article['title'] for article in data.get('articles', [])]
                return self._analyze_headlines(headlines)
            else:
                return self._get_demo_news_sentiment(symbol)
                
        except Exception as e:
            return self._get_demo_news_sentiment(symbol)
    
    def _get_demo_news_sentiment(self, symbol):
        """Get demo news sentiment"""
        headlines = self.demo_headlines.get(symbol, [
            f"{symbol} stock shows mixed signals",
            f"Market analysis for {symbol}",
            f"{symbol} trading update"
        ])
        
        return self._analyze_headlines(headlines)
    
    def _analyze_headlines(self, headlines):
        """Analyze sentiment of headlines"""
        try:
            if not headlines:
                return {
                    'sentiment_score': 50,
                    'sentiment_label': 'NEUTRAL',
                    'confidence': 50,
                    'article_count': 0
                }
            
            sentiments = []
            
            for headline in headlines:
                try:
                    blob = TextBlob(headline)
                    polarity = blob.sentiment.polarity
                    sentiments.append(polarity)
                except:
                    sentiments.append(0)  # Neutral if analysis fails
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Convert to 0-100 scale
            sentiment_score = 50 + (avg_sentiment * 50)
            
            # Determine label
            if sentiment_score > 60:
                sentiment_label = 'POSITIVE'
                confidence = min(90, 50 + (sentiment_score - 50))
            elif sentiment_score < 40:
                sentiment_label = 'NEGATIVE'
                confidence = min(90, 50 + (50 - sentiment_score))
            else:
                sentiment_label = 'NEUTRAL'
                confidence = 50
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'article_count': len(headlines),
                'raw_sentiments': sentiments
            }
            
        except Exception as e:
            return {
                'sentiment_score': 50,
                'sentiment_label': 'NEUTRAL',
                'confidence': 50,
                'article_count': 0
            }
    
    def analyze_symbol(self, symbol):
        """Analyze sentiment for a symbol"""
        try:
            news_sentiment = self.get_news_sentiment(symbol)
            
            # Generate signals based on sentiment
            signals = []
            sentiment_score = news_sentiment['sentiment_score']
            confidence = news_sentiment['confidence']
            
            if sentiment_score > 70:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Positive news sentiment',
                    'confidence': confidence,
                    'source': 'news'
                })
            elif sentiment_score < 30:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Negative news sentiment',
                    'confidence': confidence,
                    'source': 'news'
                })
            else:
                signals.append({
                    'type': 'NEUTRAL',
                    'reason': 'Neutral news sentiment',
                    'confidence': 50,
                    'source': 'news'
                })
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment_label': news_sentiment['sentiment_label'],
                'confidence': confidence,
                'signals': signals,
                'article_count': news_sentiment['article_count'],
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return self._get_default_sentiment_analysis(symbol)
    
    def _get_default_sentiment_analysis(self, symbol):
        """Default sentiment analysis"""
        return {
            'symbol': symbol,
            'sentiment_score': 50,
            'sentiment_label': 'NEUTRAL',
            'confidence': 50,
            'signals': [{
                'type': 'NEUTRAL',
                'reason': 'No sentiment data available',
                'confidence': 50,
                'source': 'default'
            }],
            'article_count': 0,
            'analysis_time': datetime.now()
        }
    
    def analyze_all_symbols(self):
        """Analyze sentiment for all symbols"""
        results = {}
        
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        for symbol in all_symbols[:10]:  # Limit for performance
            try:
                results[symbol] = self.analyze_symbol(symbol)
            except:
                results[symbol] = self._get_default_sentiment_analysis(symbol)
        
        return results
    
    def get_market_sentiment(self):
        """Get overall market sentiment"""
        try:
            # Analyze major indices
            major_symbols = ['SPY', 'QQQ', 'IWM']
            sentiments = []
            
            for symbol in major_symbols:
                try:
                    analysis = self.analyze_symbol(symbol)
                    sentiments.append(analysis['sentiment_score'])
                except:
                    sentiments.append(50)  # Neutral default
            
            if sentiments:
                overall_sentiment = sum(sentiments) / len(sentiments)
                
                if overall_sentiment > 60:
                    sentiment_label = 'BULLISH'
                elif overall_sentiment < 40:
                    sentiment_label = 'BEARISH'
                else:
                    sentiment_label = 'NEUTRAL'
                
                return {
                    'overall_sentiment': sentiment_label,
                    'sentiment_score': overall_sentiment,
                    'confidence': abs(overall_sentiment - 50) + 50,
                    'analysis_time': datetime.now()
                }
            else:
                return {
                    'overall_sentiment': 'NEUTRAL',
                    'sentiment_score': 50,
                    'confidence': 50,
                    'analysis_time': datetime.now()
                }
                
        except Exception as e:
            return {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 50,
                'confidence': 50,
                'analysis_time': datetime.now()
            }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_symbol('AAPL')
    print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.1f})")
    print(f"Signals: {result['signals']}")
    
    market_sentiment = analyzer.get_market_sentiment()
    print(f"Market sentiment: {market_sentiment['overall_sentiment']}")
