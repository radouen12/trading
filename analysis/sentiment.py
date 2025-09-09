import requests
import pandas as pd
from datetime import datetime, timedelta
import re
from config import Config
import time
from functools import wraps

# Safe import of TextBlob with fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("⚠️ TextBlob not available. Install with: pip install textblob")
    TEXTBLOB_AVAILABLE = False
    
    # Fallback TextBlob class
    class TextBlob:
        def __init__(self, text):
            self.text = text
        
        @property
        def sentiment(self):
            # Simple fallback sentiment (neutral)
            class Sentiment:
                polarity = 0.0
                subjectivity = 0.5
            return Sentiment()

# Retry decorator for network requests
def retry_request(max_retries=3, delay=1, backoff=2):
    """Decorator to retry network requests with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, requests.Timeout, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"❌ Failed after {max_retries} retries: {e}")
                        raise
                    
                    wait_time = delay * (backoff ** (retries - 1))
                    print(f"⚠️ Request failed, retrying in {wait_time}s... (attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class SentimentAnalyzer:
    def __init__(self):
        self.config = Config()
        self.news_cache = {}
        
    def get_news_sentiment(self, symbol, days_back=3):
        """Get news sentiment for a symbol"""
        try:
            # Use NewsAPI if properly configured
            if self.config.is_news_api_available():
                return self.get_newsapi_sentiment(symbol, days_back)
            else:
                # Fallback to Yahoo Finance news scraping
                print(f"⚠️ News API not configured, using Yahoo Finance fallback for {symbol}")
                return self.get_yahoo_news_sentiment(symbol, days_back)
                
        except Exception as e:
            print(f"❌ Error getting news sentiment for {symbol}: {e}")
            return self.get_default_sentiment()
    
    @retry_request(max_retries=3, delay=2)
    def get_newsapi_sentiment(self, symbol, days_back=3):
        """Get sentiment from NewsAPI with retry logic"""
        try:
            # Convert symbol to company name for better search
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft', 
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'NVDA': 'Nvidia',
                'META': 'Meta Facebook',
                'BTC-USD': 'Bitcoin',
                'ETH-USD': 'Ethereum'
            }
            
            search_term = company_names.get(symbol, symbol.replace('-USD', '').replace('=X', ''))
            
            # NewsAPI request with extended timeout
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_term,
                'apiKey': self.config.NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'pageSize': 20
            }
            
            # Add timeout and better error handling
            response = requests.get(url, params=params, timeout=30, 
                                  headers={'User-Agent': 'Trading-System/1.0'})
            
            if response.status_code == 200:
                news_data = response.json()
                
                # Validate response structure
                if 'articles' not in news_data:
                    print(f"⚠️ Unexpected NewsAPI response format for {symbol}")
                    return self.get_yahoo_news_sentiment(symbol, days_back)
                
                articles = news_data['articles']
                
                if articles:
                    return self.analyze_articles_sentiment(articles, symbol)
            elif response.status_code == 429:
                print(f"⚠️ NewsAPI rate limit exceeded for {symbol}")
                time.sleep(60)  # Wait before fallback
                return self.get_yahoo_news_sentiment(symbol, days_back)
            elif response.status_code == 401:
                print(f"⚠️ NewsAPI authentication failed - check API key")
                return self.get_yahoo_news_sentiment(symbol, days_back)
            elif response.status_code == 400:
                print(f"⚠️ NewsAPI bad request for {symbol} - check parameters")
                return self.get_yahoo_news_sentiment(symbol, days_back)
            else:
                print(f"⚠️ NewsAPI error {response.status_code} for {symbol}")
            
            return self.get_default_sentiment()
            
        except Exception as e:
            print(f"❌ NewsAPI error for {symbol}: {e}")
            return self.get_default_sentiment()
    
    @retry_request(max_retries=2, delay=1)
    def get_yahoo_news_sentiment(self, symbol, days_back=3):
        """Fallback: Get sentiment from Yahoo Finance news with retry logic"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            # Try to get news with timeout handling
            try:
                # Set a shorter timeout for yfinance
                news = ticker.news
                if news and len(news) > 0:
                    # Convert Yahoo news format to our format
                    articles = []
                    for item in news[:10]:  # Limit to 10 articles
                        # Validate item structure
                        if not isinstance(item, dict):
                            continue
                            
                        title = item.get('title', '')
                        summary = item.get('summary', '')
                        timestamp = item.get('providerPublishTime', 0)
                        
                        if title:  # Only add if we have a title
                            try:
                                pub_date = datetime.fromtimestamp(timestamp).isoformat()
                            except (ValueError, OSError):
                                pub_date = datetime.now().isoformat()
                                
                            articles.append({
                                'title': title,
                                'description': summary,
                                'publishedAt': pub_date
                            })
                    
                    if articles:
                        return self.analyze_articles_sentiment(articles, symbol)
                else:
                    print(f"📰 No news found for {symbol} via Yahoo Finance")
            except Exception as yf_error:
                print(f"⚠️ Yahoo Finance error for {symbol}: {yf_error}")
            
            # If no news available, return neutral sentiment
            return self.get_neutral_sentiment()
            
        except Exception as e:
            print(f"❌ Yahoo news error for {symbol}: {e}")
            return self.get_default_sentiment()
    
    def analyze_articles_sentiment(self, articles, symbol):
        """Analyze sentiment of news articles"""
        try:
            sentiments = []
            article_count = 0
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Combine title and description
                text = f"{title} {description}".strip()
                
                if text and len(text) > 10:
                    # Use TextBlob for sentiment analysis
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity  # -1 to 1
                    subjectivity = blob.sentiment.subjectivity  # 0 to 1
                    
                    # Classify sentiment
                    if polarity > 0.1:
                        sentiment_label = 'positive'
                        positive_count += 1
                    elif polarity < -0.1:
                        sentiment_label = 'negative'
                        negative_count += 1
                    else:
                        sentiment_label = 'neutral'
                        neutral_count += 1
                    
                    sentiments.append({
                        'text': text[:200],  # First 200 chars
                        'polarity': polarity,
                        'subjectivity': subjectivity,
                        'sentiment': sentiment_label,
                        'published': article.get('publishedAt', '')
                    })
                    
                    article_count += 1
            
            if article_count == 0:
                return self.get_neutral_sentiment()
            
            # Calculate overall sentiment
            avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
            avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
            
            # Determine overall sentiment
            if avg_polarity > 0.2:
                overall_sentiment = 'positive'
                sentiment_score = min(75 + (avg_polarity * 25), 95)
            elif avg_polarity < -0.2:
                overall_sentiment = 'negative'  
                sentiment_score = max(25 - (abs(avg_polarity) * 25), 5)
            else:
                overall_sentiment = 'neutral'
                sentiment_score = 50
            
            return {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment,
                'sentiment_score': sentiment_score,
                'polarity': avg_polarity,
                'subjectivity': avg_subjectivity,
                'article_count': article_count,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'articles': sentiments[:5],  # Top 5 articles
                'confidence': self.calculate_sentiment_confidence(sentiments),
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
            return self.get_default_sentiment()
    
    def calculate_sentiment_confidence(self, sentiments):
        """Calculate confidence in sentiment analysis"""
        if not sentiments:
            return 50
        
        # More articles = higher confidence
        article_confidence = min(len(sentiments) * 10, 40)
        
        # Agreement between articles = higher confidence
        polarities = [s['polarity'] for s in sentiments]
        if len(polarities) > 1:
            polarity_std = pd.Series(polarities).std()
            agreement_confidence = max(30 - (polarity_std * 50), 10)
        else:
            agreement_confidence = 20
        
        # Subjectivity affects confidence (lower subjectivity = more factual)
        avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
        objectivity_confidence = max(30 - (avg_subjectivity * 20), 10)
        
        total_confidence = min(article_confidence + agreement_confidence + objectivity_confidence, 90)
        return total_confidence
    
    def get_market_sentiment(self):
        """Get overall market sentiment from major indices"""
        try:
            market_symbols = ['SPY', 'QQQ', 'IWM', 'VIX']
            market_sentiment = {}
            
            for symbol in market_symbols:
                sentiment = self.get_news_sentiment(symbol, days_back=2)
                market_sentiment[symbol] = sentiment
            
            # Calculate overall market mood
            sentiments = [s['sentiment_score'] for s in market_sentiment.values() if s['sentiment_score'] != 50]
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                
                if avg_sentiment > 60:
                    market_mood = 'bullish'
                elif avg_sentiment < 40:
                    market_mood = 'bearish'
                else:
                    market_mood = 'neutral'
            else:
                market_mood = 'neutral'
                avg_sentiment = 50
            
            return {
                'market_mood': market_mood,
                'market_sentiment_score': avg_sentiment,
                'individual_sentiments': market_sentiment,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error getting market sentiment: {e}")
            return {
                'market_mood': 'neutral',
                'market_sentiment_score': 50,
                'individual_sentiments': {},
                'analysis_time': datetime.now()
            }
    
    def get_crypto_sentiment(self):
        """Get sentiment for major cryptocurrencies"""
        try:
            crypto_symbols = ['BTC-USD', 'ETH-USD']
            crypto_sentiment = {}
            
            for symbol in crypto_symbols:
                sentiment = self.get_news_sentiment(symbol, days_back=1)  # Crypto moves fast
                crypto_sentiment[symbol] = sentiment
            
            return crypto_sentiment
            
        except Exception as e:
            print(f"❌ Error getting crypto sentiment: {e}")
            return {}
    
    def generate_sentiment_signals(self, sentiment_data):
        """Generate trading signals based on sentiment"""
        signals = []
        
        sentiment_score = sentiment_data.get('sentiment_score', 50)
        confidence = sentiment_data.get('confidence', 50)
        overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
        
        # Strong positive sentiment
        if sentiment_score > 70 and confidence > 60:
            signals.append({
                'type': 'BUY',
                'reason': f'Strong positive news sentiment (score: {sentiment_score:.0f})',
                'confidence': min(confidence + 10, 80),
                'timeframe': 'daily',
                'sentiment_driven': True
            })
        
        # Strong negative sentiment
        elif sentiment_score < 30 and confidence > 60:
            signals.append({
                'type': 'SELL',
                'reason': f'Strong negative news sentiment (score: {sentiment_score:.0f})',
                'confidence': min(confidence + 5, 75),
                'timeframe': 'daily',
                'sentiment_driven': True
            })
        
        # Contrarian signals (when sentiment is extreme)
        elif sentiment_score > 85:
            signals.append({
                'type': 'SELL',
                'reason': 'Contrarian: Extremely positive sentiment may indicate top',
                'confidence': 60,
                'timeframe': 'weekly',
                'sentiment_driven': True,
                'contrarian': True
            })
        
        elif sentiment_score < 15:
            signals.append({
                'type': 'BUY',
                'reason': 'Contrarian: Extremely negative sentiment may indicate bottom',
                'confidence': 65,
                'timeframe': 'weekly', 
                'sentiment_driven': True,
                'contrarian': True
            })
        
        return signals
    
    def analyze_all_symbols(self):
        """Analyze sentiment for all symbols"""
        print("📰 Starting comprehensive sentiment analysis...")
        
        # Prioritize major symbols for sentiment analysis
        priority_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                          'SPY', 'QQQ', 'BTC-USD', 'ETH-USD']
        
        results = {}
        
        for i, symbol in enumerate(priority_symbols):
            print(f"📰 Analyzing sentiment for {symbol} ({i+1}/{len(priority_symbols)})")
            sentiment = self.get_news_sentiment(symbol)
            sentiment['signals'] = self.generate_sentiment_signals(sentiment)
            results[symbol] = sentiment
        
        # Get overall market sentiment
        print("📊 Analyzing overall market sentiment...")
        results['market_sentiment'] = self.get_market_sentiment()
        
        print("✅ Sentiment analysis complete!")
        return results
    
    # Default sentiment methods
    def get_default_sentiment(self):
        return {
            'symbol': 'UNKNOWN',
            'overall_sentiment': 'neutral',
            'sentiment_score': 50,
            'polarity': 0,
            'subjectivity': 0.5,
            'article_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'articles': [],
            'confidence': 30,
            'analysis_time': datetime.now()
        }
    
    def get_neutral_sentiment(self):
        return {
            'symbol': 'UNKNOWN',
            'overall_sentiment': 'neutral',
            'sentiment_score': 50,
            'polarity': 0,
            'subjectivity': 0.5,
            'article_count': 1,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 1,
            'articles': [{'text': 'No significant news found', 'sentiment': 'neutral'}],
            'confidence': 40,
            'analysis_time': datetime.now()
        }

if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test single symbol
    print("Testing AAPL sentiment analysis...")
    aapl_sentiment = analyzer.get_news_sentiment('AAPL')
    
    print(f"Overall sentiment: {aapl_sentiment['overall_sentiment']}")
    print(f"Sentiment score: {aapl_sentiment['sentiment_score']}")
    print(f"Article count: {aapl_sentiment['article_count']}")
    
    # Generate signals
    signals = analyzer.generate_sentiment_signals(aapl_sentiment)
    print(f"Sentiment signals: {signals}")
