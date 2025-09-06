import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sqlite3
from config import Config
import logging

class DataFetcher:
    def __init__(self):
        self.config = Config()
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing price data"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            # Create prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    asset_type TEXT,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Create analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi REAL,
                    macd REAL,
                    signal_line REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    volume_sma REAL,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database setup error: {e}")
    
    def fetch_real_time_stocks(self):
        """Fetch real-time stock data"""
        try:
            all_symbols = self.config.STOCK_SYMBOLS
            tickers = yf.Tickers(' '.join(all_symbols))
            
            stock_data = {}
            for symbol in all_symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.tail(1).iloc[0]
                        stock_data[symbol] = {
                            'symbol': symbol,
                            'price': latest['Close'],
                            'change': latest['Close'] - hist['Close'].iloc[-2] if len(hist) > 1 else 0,
                            'change_pct': ((latest['Close'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                            'volume': latest['Volume'],
                            'high': latest['High'],
                            'low': latest['Low'],
                            'timestamp': datetime.now(),
                            'asset_type': 'stock'
                        }
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"⚠️  Error fetching {symbol}: {e}")
                    continue
            
            return stock_data
            
        except Exception as e:
            print(f"❌ Error fetching stock data: {e}")
            return {}
    
    def fetch_real_time_crypto(self):
        """Fetch real-time crypto data"""
        try:
            crypto_data = {}
            for symbol in self.config.CRYPTO_SYMBOLS:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.tail(1).iloc[0]
                        crypto_data[symbol] = {
                            'symbol': symbol,
                            'price': latest['Close'],
                            'change': latest['Close'] - hist['Close'].iloc[-2] if len(hist) > 1 else 0,
                            'change_pct': ((latest['Close'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                            'volume': latest['Volume'],
                            'high': latest['High'],
                            'low': latest['Low'],
                            'timestamp': datetime.now(),
                            'asset_type': 'crypto'
                        }
                    time.sleep(0.1)
                except Exception as e:
                    print(f"⚠️  Error fetching crypto {symbol}: {e}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            print(f"❌ Error fetching crypto data: {e}")
            return {}
    
    def fetch_real_time_forex(self):
        """Fetch real-time forex data"""
        try:
            forex_data = {}
            for symbol in self.config.FOREX_SYMBOLS:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.tail(1).iloc[0]
                        forex_data[symbol] = {
                            'symbol': symbol,
                            'price': latest['Close'],
                            'change': latest['Close'] - hist['Close'].iloc[-2] if len(hist) > 1 else 0,
                            'change_pct': ((latest['Close'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                            'volume': latest.get('Volume', 0),
                            'high': latest['High'],
                            'low': latest['Low'],
                            'timestamp': datetime.now(),
                            'asset_type': 'forex'
                        }
                    time.sleep(0.1)
                except Exception as e:
                    print(f"⚠️  Error fetching forex {symbol}: {e}")
                    continue
            
            return forex_data
            
        except Exception as e:
            print(f"❌ Error fetching forex data: {e}")
            return {}
    
    def fetch_all_assets(self):
        """Fetch real-time data for all asset types"""
        print("🔄 Fetching real-time data...")
        
        all_data = {}
        
        # Fetch stocks
        stock_data = self.fetch_real_time_stocks()
        all_data.update(stock_data)
        print(f"✅ Fetched {len(stock_data)} stocks")
        
        # Fetch crypto
        crypto_data = self.fetch_real_time_crypto()
        all_data.update(crypto_data)
        print(f"✅ Fetched {len(crypto_data)} crypto pairs")
        
        # Fetch forex
        forex_data = self.fetch_real_time_forex()
        all_data.update(forex_data)
        print(f"✅ Fetched {len(forex_data)} forex pairs")
        
        # Store in database
        self.store_data(all_data)
        
        return all_data
    
    def store_data(self, data):
        """Store fetched data in database"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            for symbol, info in data.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO prices 
                    (symbol, timestamp, open, high, low, close, volume, asset_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    info['timestamp'],
                    info.get('open', info['price']),
                    info['high'],
                    info['low'], 
                    info['price'],
                    info['volume'],
                    info['asset_type']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error storing data: {e}")
    
    def get_historical_data(self, symbol, period="30d", interval="1h"):
        """Get historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Simple check for weekdays during market hours
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            return self.config.MARKET_OPEN <= current_time <= self.config.MARKET_CLOSE
        return False

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()
    data = fetcher.fetch_all_assets()
    print(f"\n📊 Successfully fetched data for {len(data)} assets")
    
    # Show sample data
    for i, (symbol, info) in enumerate(data.items()):
        if i < 5:  # Show first 5
            print(f"{symbol}: ${info['price']:.2f} ({info['change_pct']:+.2f}%)")
