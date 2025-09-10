#!/usr/bin/env python3
"""
Quick system test to check if the trading system works
"""
import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_imports():
    """Test if all critical imports work"""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError as e:
        print(f"❌ Streamlit missing: {e}")
        return False
    
    try:
        from config import Config
        config = Config()
        print("✅ Config loads")
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False
    
    try:
        from data.fetcher import DataFetcher
        print("✅ DataFetcher loads")
    except Exception as e:
        print(f"❌ DataFetcher failed: {e}")
        return False
    
    try:
        from engine.portfolio import PortfolioManager
        portfolio = PortfolioManager()
        stats = portfolio.get_portfolio_stats()
        print(f"✅ Portfolio Manager works: {stats.get('total_value', 0)}")
    except Exception as e:
        print(f"❌ Portfolio Manager failed: {e}")
        return False
    
    try:
        from analysis import TechnicalAnalyzer, SeasonalAnalyzer
        if TechnicalAnalyzer is None or SeasonalAnalyzer is None:
            print("⚠️ Some analysis modules are None")
        else:
            print("✅ Analysis modules available")
    except Exception as e:
        print(f"❌ Analysis modules failed: {e}")
        return False
    
    return True

def test_dashboard():
    """Test if dashboard can be imported"""
    try:
        from ui.phase4_dashboard import main
        print("✅ Phase 4 dashboard imports successfully")
        return True
    except Exception as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING TRADING SYSTEM")
    print("=" * 40)
    
    imports_ok = test_imports()
    dashboard_ok = test_dashboard()
    
    print("\n" + "=" * 40)
    if imports_ok and dashboard_ok:
        print("✅ SYSTEM APPEARS TO WORK")
        print("Try running: streamlit run ui/phase4_dashboard.py")
    else:
        print("❌ SYSTEM HAS ISSUES - See errors above")
        print("Run the fixes below before using the system")
