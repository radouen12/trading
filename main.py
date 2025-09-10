#!/usr/bin/env python3
"""
Real-Time Trading System - Main Entry Point
Phase 1: Foundation with Streamlit Dashboard

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
    print("📈 Phase 3 Features: Backtesting + Performance + Optimization + Alerts")
    print("🧠 Phase 2 Features: Technical + Seasonal + Sentiment + Correlation")
    print("📊 Phase 1 Features: Real-time data + Portfolio + Risk management")
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
    """Main application launcher with health check"""
    print("🚀 AI Trading System - Phase 4 COMPLETE")
    print("="*60)
    
    # Run health check first
    print("🔍 Running system health check...")
    try:
        from utils.health_check import SystemHealthCheck
        health_check = SystemHealthCheck()
        system_ready = health_check.run_full_check()
        
        if not system_ready:
            print("\n❌ System is not ready for operation.")
            health_check.get_fix_suggestions()
            input("\nPress Enter to continue anyway (not recommended) or Ctrl+C to exit...")
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
        print("Continuing without health check...")
    
    # Check system requirements
    if not check_dependencies():
        return
    
    # Setup project structure with improved error handling
    setup_directories()
    
    # Initialize configuration directories
    try:
        from config import Config
        config = Config()
        dir_info = config.initialize_directories()
        
        print(f"✅ Directory setup complete:")
        print(f"   Data: {dir_info['data_directory']}")
        print(f"   Logs: {dir_info['logs_directory']}")
        print(f"   Database: {dir_info['db_path']}")
        
    except Exception as e:
        print(f"❌ Error initializing directories: {e}")
        print("Continuing with default configuration...")
    
    # Display startup information with comprehensive configuration check
    print("\n📊 System Configuration:")
    from config import Config
    config = Config()
    
    # Check API configuration
    api_issues = config.validate_api_keys()
    if api_issues:
        print("\n⚠️  Configuration Issues:")
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
    
    print("\n⚡ Phase 4 Features (COMPLETED):")
    print("✅ Machine Learning Integration - Price prediction & pattern recognition")
    print("✅ Automated Execution Engine - Signal processing & paper trading")
    print("✅ Market Regime Detection - Advanced regime classification")
    print("✅ Arbitrage Detection - Cross-asset opportunity scanner")
    print("✅ Risk Monitoring System - Real-time portfolio risk management")
    print("✅ Signal Processing - Advanced signal validation & filtering")
    print("\n⚡ Phase 3 Features (INTEGRATED):")
    print("✅ Advanced Backtesting Engine - Strategy testing & validation")
    print("✅ Performance Analytics Suite - 15+ professional metrics")
    print("✅ Email Alert System - Real-time notifications")
    print("✅ Portfolio Optimization - Modern portfolio theory")
    print("✅ Risk Management Tools - VaR, Sharpe, drawdown analysis")
    print("✅ Enhanced Dashboard - Multi-tab professional interface")
    
    print("\n🔄 Auto-Features:")
    print("✅ Real-time price data")
    print("✅ Capital-based position sizing")
    print("✅ Multi-timeframe suggestions")
    print("✅ Advanced risk management")
    print("✅ Portfolio correlation tracking")
    print("🔄 Auto-refreshing enhanced dashboard")
    
    print("\n🔮 Future Enhancements (Beyond Phase 4):")
    print("🔜 Reinforcement Learning Trading Agents")
    print("🔜 Options Trading Support")
    print("🔜 Real-Time Paper Trading with Live Execution")
    print("🔜 Multi-Account Management")
    print("🔜 Mobile Application")
    print("🔜 Cloud Deployment & Scaling")
    
    # Launch Phase 4 dashboard
    input("\n📡 Press Enter to launch PHASE 4 COMPLETE AI Trading System...")
    run_dashboard()

if __name__ == "__main__":
    main()
