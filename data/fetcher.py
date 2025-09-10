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
        """Setup SQLite database"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            # Create market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    change_pct REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database setup complete")
            
        except Exception as e:
            print(f"⚠️ Database setup error: {e}")
    
    def fetch_real_time_data(self):
        """Fetch real-time data for all symbols"""
        all_data = {}
        
        # Combine all symbols
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        for symbol in all_symbols:
            try:
                data = self._fetch_symbol_data(symbol)
                if data:
                    all_data[symbol] = data
            except Exception as e:
                print(f"⚠️ Error fetching {symbol}: {e}")
                continue
        
        return all_data
    
    def _fetch_symbol_data(self, symbol):
        """Fetch data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0,
                'open': hist['Open'].iloc[-1],
                'high': hist['High'].iloc[-1],  
                'low': hist['Low'].iloc[-1],
                'change_pct': change_pct,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return None
    
    def fetch_real_time_stocks(self):
        """Fetch stock data"""
        return self._fetch_symbols_batch(self.config.STOCK_SYMBOLS)
    
    def fetch_real_time_crypto(self):
        """Fetch crypto data"""
        return self._fetch_symbols_batch(self.config.CRYPTO_SYMBOLS)
    
    def fetch_real_time_forex(self):
        """Fetch forex data"""
        return self._fetch_symbols_batch(self.config.FOREX_SYMBOLS)
    
    def _fetch_symbols_batch(self, symbols):
        """Fetch data for a batch of symbols"""
        data = {}
        for symbol in symbols:
            try:
                symbol_data = self._fetch_symbol_data(symbol)
                if symbol_data:
                    data[symbol] = symbol_data
            except:
                continue
        return data
    
    def store_market_data(self, market_data):
        """Store market data in database"""
        try:
            conn = sqlite3.connect(self.config.DB_PATH)
            cursor = conn.cursor()
            
            for symbol, data in market_data.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, change_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    data['timestamp'],
                    data['open'],
                    data['high'],
                    data['low'],
                    data['price'],
                    data['volume'],
                    data['change_pct']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Database storage error: {e}")
    
    def get_historical_data(self, symbol, period="1mo"):
        """Get historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period)
        except:
            return pd.DataFrame()

if __name__ == "__main__":
    fetcher = DataFetcher()
    data = fetcher.fetch_real_time_data()
    print(f"Fetched data for {len(data)} symbols")
