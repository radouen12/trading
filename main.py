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
    """Launch the Streamlit dashboard"""
    dashboard_path = current_dir / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return False
    
    print("🚀 Launching Real-Time Trading Dashboard...")
    print("📡 Dashboard will open in your browser")
    print("🔄 Data updates every 60 seconds")
    print("⚡ Analysis runs every 10 minutes")
    print("\n" + "="*50)
    
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
    print("🚀 Real-Time Trading System - Phase 1")
    print("="*50)
    
    # Check system requirements
    if not check_dependencies():
        return
    
    # Setup project structure
    setup_directories()
    
    # Display startup information
    print("\n📊 System Configuration:")
    from config import Config
    config = Config()
    
    print(f"💰 Total Capital: ${config.TOTAL_CAPITAL:,}")
    print(f"📈 Max Position: {config.MAX_POSITION_SIZE*100:.1f}%")
    print(f"⚠️  Daily Loss Limit: {config.MAX_DAILY_LOSS*100:.1f}%")
    print(f"📊 Tracking Assets:")
    print(f"   - {len(config.STOCK_SYMBOLS)} Stocks")
    print(f"   - {len(config.CRYPTO_SYMBOLS)} Crypto pairs")
    print(f"   - {len(config.FOREX_SYMBOLS)} Forex pairs")
    
    print("\n⚡ Features Available:")
    print("✅ Real-time price data")
    print("✅ Capital-based position sizing")
    print("✅ Multi-timeframe suggestions")
    print("✅ Risk management")
    print("✅ Portfolio tracking")
    print("🔄 Auto-refreshing dashboard")
    
    print("\n🔮 Coming in Phase 2:")
    print("🔜 Advanced technical analysis")
    print("🔜 Seasonal pattern detection")
    print("🔜 News sentiment analysis")
    print("🔜 Enhanced risk management")
    
    # Launch dashboard
    input("\n📡 Press Enter to launch dashboard...")
    run_dashboard()

if __name__ == "__main__":
    main()
