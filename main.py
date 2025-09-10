#!/usr/bin/env python3
"""
Real-Time Trading System - Main Entry Point
Phase 4: Complete AI Trading System

Run with: streamlit run main.py
"""

import sys
import os
import subprocess
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'yfinance', 
        'pandas',
        'numpy',
        'plotly',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'logs',
        'analysis',
        'engine', 
        'ui',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure ready")

def run_dashboard():
    """Launch the Phase 4 Complete AI Trading System"""
    dashboard_path = current_dir / "ui" / "phase4_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Phase 4 dashboard file not found!")
        return False
    
    print("🚀 Launching Phase 4 COMPLETE AI Trading System...")
    print("🧠 Machine Learning: Price prediction & pattern recognition")
    print("🤖 Automation Engine: Automated signal processing & execution (PAPER MODE)")
    print("📊 Regime Detection: Advanced market regime classification")
    print("🔄 Arbitrage Detection: Cross-asset opportunity scanner")
    print("🛡️ Risk Monitor: Real-time portfolio risk management")
    print("📡 Signal Processing: Advanced signal validation & filtering")
    print("📈 Features: Real-time data + Portfolio + Risk management")
    print("📡 Dashboard will open in your browser")
    print("🔄 Data updates every 60 seconds")
    print("⚡ Analysis runs every 10 minutes")
    print("\n" + "="*60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main application launcher"""
    print("🚀 AI Trading System - Phase 4 COMPLETE")
    print("="*60)
    
    # Check system requirements
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    # Setup project structure
    setup_directories()
    
    # Initialize configuration
    try:
        from config import Config
        config = Config()
        
        print(f"✅ Configuration loaded:")
        print(f"   Database: {config.DB_PATH}")
        print(f"   Log file: {config.LOG_FILE}")
        
    except Exception as e:
        print(f"⚠️ Configuration warning: {e}")
        print("Continuing with default configuration...")
    
    # Display startup information
    print("\n📊 System Configuration:")
    try:
        from config import Config
        config = Config()
        
        # Check API configuration
        api_issues = config.validate_api_keys()
        if api_issues:
            print("\n⚠️  Configuration Notes:")
            for issue in api_issues:
                print(f"   - {issue}")
            print("\n💡 To enable full functionality:")
            print("   1. Copy .env.template to .env")
            print("   2. Add your API keys to the .env file")
            print("   3. Restart the application")
            print("\n🚀 System will continue with available features...")
        else:
            print("✅ All API keys configured properly")
        
        print(f"💰 Total Capital: ${config.TOTAL_CAPITAL:,}")
        print(f"📈 Max Position: {config.MAX_POSITION_SIZE*100:.1f}%")
        print(f"⚠️  Daily Loss Limit: {config.MAX_DAILY_LOSS*100:.1f}%")
        print(f"📊 Tracking Assets:")
        print(f"   - {len(config.STOCK_SYMBOLS)} Stocks")
        print(f"   - {len(config.CRYPTO_SYMBOLS)} Crypto pairs")
        print(f"   - {len(config.FOREX_SYMBOLS)} Forex pairs")
        
    except Exception as e:
        print(f"⚠️ Could not load configuration: {e}")
    
    print("\n⚡ Phase 4 Features (COMPLETED):")
    print("✅ Machine Learning Integration - Price prediction & pattern recognition")
    print("✅ Automated Execution Engine - Signal processing & paper trading")
    print("✅ Market Regime Detection - Advanced regime classification")
    print("✅ Arbitrage Detection - Cross-asset opportunity scanner")
    print("✅ Risk Monitoring System - Real-time portfolio risk management")
    print("✅ Signal Processing - Advanced signal validation & filtering")
    print("✅ Technical Analysis - RSI, MACD, Bollinger Bands")
    print("✅ Seasonal Analysis - Monthly performance patterns")
    print("✅ Sentiment Analysis - News sentiment tracking")
    print("✅ Correlation Analysis - Portfolio risk assessment")
    print("✅ Real-time Data Fetching - Stocks, Crypto, Forex")
    print("✅ Portfolio Management - Position tracking & risk management")
    print("✅ Enhanced Dashboard - Multi-tab professional interface")
    
    print("\n🔄 Auto-Features:")
    print("✅ Real-time price data updates")
    print("✅ Capital-based position sizing")
    print("✅ Multi-timeframe suggestions")
    print("✅ Advanced risk management")
    print("✅ Portfolio correlation tracking")
    print("✅ Auto-refreshing enhanced dashboard")
    
    print("\n🔮 Future Enhancements (Beyond Phase 4):")
    print("🔜 Reinforcement Learning Trading Agents")
    print("🔜 Options Trading Support")
    print("🔜 Real-Time Live Trading with Broker Integration")
    print("🔜 Multi-Account Management")
    print("🔜 Mobile Application")
    print("🔜 Cloud Deployment & Scaling")
    
    # Quick system test
    print("\n🧪 Quick System Test:")
    try:
        from data.fetcher import DataFetcher
        from engine.portfolio import PortfolioManager
        from engine.suggester import EnhancedSuggestionEngine
        
        # Test data fetcher
        fetcher = DataFetcher()
        print("✅ Data fetcher initialized")
        
        # Test portfolio manager
        portfolio = PortfolioManager()
        stats = portfolio.get_portfolio_stats()
        print(f"✅ Portfolio manager working (${stats.get('total_value', 0):,.2f})")
        
        # Test suggestion engine
        suggester = EnhancedSuggestionEngine()
        print("✅ AI suggestion engine initialized")
        
        print("✅ All core systems operational")
        
    except Exception as e:
        print(f"⚠️ System test warning: {e}")
        print("Some features may not work properly")
    
    # Launch Phase 4 dashboard
    input("\n📡 Press Enter to launch PHASE 4 COMPLETE AI Trading System...")
    run_dashboard()

if __name__ == "__main__":
    main()
