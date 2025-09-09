"""
System Health Check - Diagnostic Tool
Validates system configuration and dependencies
"""

import sys
import os
import importlib
from pathlib import Path
from datetime import datetime

class SystemHealthCheck:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.passed.append(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        else:
            self.issues.append(f"❌ Python version {version.major}.{version.minor}.{version.micro} not supported. Need Python 3.8+")
    
    def check_required_packages(self):
        """Check if required packages are installed"""
        required_packages = [
            'streamlit',
            'pandas', 
            'numpy',
            'yfinance',
            'plotly',
            'requests',
            'sqlite3'  # Built-in
        ]
        
        optional_packages = [
            'dotenv',
            'nltk',
            'textblob',
            'scikit-learn'
        ]
        
        # Check required packages
        for package in required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                else:
                    importlib.import_module(package)
                self.passed.append(f"✅ {package}")
            except ImportError:
                self.issues.append(f"❌ Missing required package: {package}")
        
        # Check optional packages
        for package in optional_packages:
            try:
                if package == 'dotenv':
                    from dotenv import load_dotenv
                else:
                    importlib.import_module(package)
                self.passed.append(f"✅ {package} (optional)")
            except ImportError:
                self.warnings.append(f"⚠️ Optional package not installed: {package}")
    
    def check_configuration(self):
        """Check system configuration"""
        try:
            from config import Config
            config = Config()
            
            # Check API keys
            api_issues = config.validate_api_keys()
            if api_issues:
                for issue in api_issues:
                    self.warnings.append(f"⚠️ {issue}")
            else:
                self.passed.append("✅ API configuration")
            
            # Check directories
            try:
                dir_info = config.initialize_directories()
                if dir_info['data_success'] and dir_info['logs_success']:
                    self.passed.append(f"✅ Directories: {dir_info['data_directory']}")
                else:
                    self.warnings.append(f"⚠️ Directory setup had issues")
            except Exception as e:
                self.issues.append(f"❌ Directory initialization failed: {e}")
            
            # Check capital configuration
            if config.TOTAL_CAPITAL > 0:
                self.passed.append(f"✅ Capital configured: ${config.TOTAL_CAPITAL:,}")
            else:
                self.warnings.append("⚠️ Capital not configured")
                
        except Exception as e:
            self.issues.append(f"❌ Configuration check failed: {e}")
    
    def check_data_fetcher(self):
        """Test data fetcher functionality"""
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            
            # Try to fetch a small sample
            import yfinance as yf
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d", interval="1h")
            
            if not test_data.empty:
                self.passed.append("✅ Data fetching works")
            else:
                self.warnings.append("⚠️ Data fetching returned empty results")
                
        except Exception as e:
            self.issues.append(f"❌ Data fetcher test failed: {e}")
    
    def check_database_connectivity(self):
        """Test database operations"""
        try:
            from config import Config
            import sqlite3
            
            config = Config()
            
            # Test database connection
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self.passed.append(f"✅ Database connectivity: {config.DB_PATH}")
            else:
                self.issues.append("❌ Database test query failed")
                
        except Exception as e:
            self.issues.append(f"❌ Database test failed: {e}")
    
    def check_portfolio_manager(self):
        """Test portfolio manager functionality"""
        try:
            from engine.portfolio import PortfolioManager
            
            portfolio = PortfolioManager()
            stats = portfolio.get_portfolio_stats()
            
            if stats and isinstance(stats, dict) and 'total_value' in stats:
                self.passed.append("✅ Portfolio manager working")
            else:
                self.issues.append("❌ Portfolio manager returned invalid data")
                
        except Exception as e:
            self.issues.append(f"❌ Portfolio manager test failed: {e}")
    
    def check_analysis_modules(self):
        """Check analysis module imports"""
        try:
            from analysis import TechnicalAnalyzer, SeasonalAnalyzer, SentimentAnalyzer, CorrelationAnalyzer
            
            available_count = sum([
                TechnicalAnalyzer is not None,
                SeasonalAnalyzer is not None,
                SentimentAnalyzer is not None,
                CorrelationAnalyzer is not None
            ])
            
            if available_count == 4:
                self.passed.append("✅ All analysis modules available")
            elif available_count > 0:
                self.warnings.append(f"⚠️ Only {available_count}/4 analysis modules available")
            else:
                self.issues.append("❌ No analysis modules available")
                
        except Exception as e:
            self.issues.append(f"❌ Analysis modules test failed: {e}")
    
    def run_full_check(self):
        """Run all health checks"""
        print("🔍 Running System Health Check...")
        print("=" * 50)
        
        self.check_python_version()
        self.check_required_packages()
        self.check_configuration()
        self.check_database_connectivity()
        self.check_data_fetcher()
        self.check_portfolio_manager()
        self.check_analysis_modules()
        
        # Display results
        print(f"\n✅ PASSED ({len(self.passed)}):")
        for item in self.passed:
            print(f"  {item}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for item in self.issues:
                print(f"  {item}")
        
        # Overall status
        print("\n" + "=" * 50)
        if self.issues:
            print("❌ SYSTEM NOT READY - Fix critical issues before proceeding")
            return False
        elif self.warnings:
            print("⚠️ SYSTEM PARTIALLY READY - Some features may not work")
            return True
        else:
            print("✅ SYSTEM FULLY READY - All checks passed!")
            return True
    
    def get_fix_suggestions(self):
        """Provide suggestions for fixing issues"""
        if not self.issues and not self.warnings:
            return
            
        print("\n🔧 FIX SUGGESTIONS:")
        print("=" * 50)
        
        # Package installation suggestions
        missing_packages = [issue.split(': ')[1] for issue in self.issues if 'Missing required package' in issue]
        if missing_packages:
            print(f"📦 Install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
        
        # API key suggestions
        api_warnings = [w for w in self.warnings if 'API' in w or 'demo mode' in w]
        if api_warnings:
            print(f"\n🔑 Configure API keys:")
            print(f"   1. Copy .env.template to .env")
            print(f"   2. Add your API keys to .env file")
            print(f"   3. Get free keys from:")
            print(f"      - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
            print(f"      - News API: https://newsapi.org/register")
        
        # Directory suggestions
        dir_issues = [issue for issue in self.issues if 'Directory' in issue]
        if dir_issues:
            print(f"\n📁 Fix directory issues:")
            print(f"   1. Check folder permissions")
            print(f"   2. Run as administrator if needed")
            print(f"   3. Free up disk space")

if __name__ == "__main__":
    health_check = SystemHealthCheck()
    system_ready = health_check.run_full_check()
    health_check.get_fix_suggestions()
    
    sys.exit(0 if system_ready else 1)
