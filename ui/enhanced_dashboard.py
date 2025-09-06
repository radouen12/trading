#!/usr/bin/env python3
"""
Enhanced Real-Time Trading Dashboard - Phase 2 Complete
Features: Technical Analysis + Seasonal Intelligence + News Sentiment + Correlation Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import yfinance as yf

# Import system components
from config import Config
from data.fetcher import DataFetcher
from engine.portfolio import PortfolioManager
from engine.suggester import EnhancedSuggestionEngine
from analysis.technical import TechnicalAnalyzer
from analysis.seasonal import SeasonalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.correlation import CorrelationAnalyzer

class EnhancedTradingDashboard:
    def __init__(self):
        self.config = Config()
        self.data_fetcher = DataFetcher()
        self.portfolio_manager = PortfolioManager()
        self.enhanced_engine = EnhancedSuggestionEngine()
        
        # Analysis engines
        self.technical_analyzer = TechnicalAnalyzer()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="🚀 Enhanced Trading Dashboard - Phase 2",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now() - timedelta(hours=1)
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {
                'technical_analysis': {},
                'seasonal_analysis': {},
                'sentiment_analysis': {},
                'correlation_analysis': {},
                'market_summary': {},
                'last_analysis': datetime.now() - timedelta(hours=1)
            }
        
        if 'enhanced_suggestions' not in st.session_state:
            st.session_state.enhanced_suggestions = []
            
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL', 'MSFT', 'BTC-USD']
    
    def run(self):
        """Main dashboard application"""
        self.show_header()
        self.show_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Portfolio & Market", 
            "📈 Technical Analysis", 
            "🗓️ Seasonal Intelligence",
            "🌐 Market Intelligence", 
            "📉 Technical Charts"
        ])
        
        with tab1:
            self.show_portfolio_and_market_tab()
        
        with tab2:
            self.show_technical_analysis_tab()
        
        with tab3:
            self.show_seasonal_analysis_tab()
        
        with tab4:
            self.show_market_intelligence_tab()
        
        with tab5:
            self.show_technical_charts_tab()
    
    def show_header(self):
        """Display dashboard header"""
        st.title("🚀 Enhanced Real-Time Trading Dashboard")
        st.subheader("🧠 Phase 2 Complete: Multi-Factor AI Analysis")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            last_update = st.session_state.last_update
            if datetime.now() - last_update < timedelta(minutes=2):
                st.success(f"🟢 Data: {last_update.strftime('%H:%M:%S')}")
            else:
                st.warning(f"🟡 Data: {last_update.strftime('%H:%M:%S')}")
        
        with col2:
            analysis_time = st.session_state.analysis_results.get('last_analysis', datetime.now() - timedelta(hours=1))
            if datetime.now() - analysis_time < timedelta(minutes=15):
                st.success(f"🧠 Analysis: {analysis_time.strftime('%H:%M:%S')}")
            else:
                st.info(f"⚪ Analysis: {analysis_time.strftime('%H:%M:%S')}")
        
        with col3:
            portfolio_stats = self.portfolio_manager.get_portfolio_stats()
            if portfolio_stats['position_count'] > 0:
                st.info(f"💼 Positions: {portfolio_stats['position_count']}")
            else:
                st.info("💼 No Positions")
        
        with col4:
            suggestions_count = len(st.session_state.enhanced_suggestions)
            if suggestions_count > 0:
                st.success(f"💡 Suggestions: {suggestions_count}")
            else:
                st.info("💡 No Suggestions")
    
    def show_sidebar(self):
        """Display sidebar controls"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Capital configuration
            st.subheader("💰 Capital Settings")
            new_capital = st.number_input(
                "Total Capital ($)", 
                value=self.config.TOTAL_CAPITAL,
                min_value=1000,
                max_value=10000000,
                step=1000
            )
            
            if new_capital != self.config.TOTAL_CAPITAL:
                self.config.TOTAL_CAPITAL = new_capital
                self.portfolio_manager.cash = new_capital
                st.success(f"Capital updated to ${new_capital:,}")
            
            st.divider()
            
            # Control buttons
            st.subheader("🎮 Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    self.fetch_latest_data()
            
            with col2:
                if st.button("🧠 Run Analysis", use_container_width=True):
                    self.run_comprehensive_analysis()
            
            if st.button("💡 Generate Enhanced Suggestions", use_container_width=True):
                self.generate_enhanced_suggestions()
            
            st.divider()
            
            # Auto-refresh
            self.setup_auto_refresh()
    
    def show_portfolio_and_market_tab(self):
        """Display portfolio and market overview"""
        st.header("📊 Portfolio & Market Overview")
        
        # Portfolio statistics
        portfolio_stats = self.portfolio_manager.get_portfolio_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Value",
                f"${portfolio_stats['total_value']:,.2f}",
                f"{portfolio_stats['total_pnl']:+.2f}"
            )
        
        with col2:
            st.metric(
                "Available Cash",
                f"${portfolio_stats['available_cash']:,.2f}",
                f"{portfolio_stats['utilization']*100:.1f}% used"
            )
        
        with col3:
            st.metric(
                "Positions",
                portfolio_stats['position_count'],
                f"Max: {portfolio_stats['max_position_size']:,.0f}"
            )
        
        with col4:
            st.metric(
                "Daily P&L",
                f"${portfolio_stats['daily_pnl']:+.2f}",
                f"{(portfolio_stats['daily_pnl']/self.config.TOTAL_CAPITAL)*100:+.2f}%"
            )
        
        st.divider()
        
        # Enhanced suggestions display
        st.subheader("💡 AI-Powered Enhanced Suggestions")
        
        if st.session_state.enhanced_suggestions:
            for i, suggestion in enumerate(st.session_state.enhanced_suggestions[:6]):
                with st.expander(f"🎯 {suggestion['symbol']} - {suggestion['action']} ({suggestion['confidence']:.0f}% confidence)"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Entry:** ${suggestion['entry_price']:.2f}")
                        st.write(f"**Target:** ${suggestion['target_price']:.2f}")
                        st.write(f"**Stop Loss:** ${suggestion['stop_loss']:.2f}")
                    
                    with col2:
                        st.write(f"**Timeframe:** {suggestion['timeframe'].title()}")
                        st.write(f"**Risk/Reward:** 1:{suggestion['risk_reward']:.2f}")
                        st.write(f"**Signal Sources:** {suggestion.get('signal_sources', 1)}")
                    
                    with col3:
                        # Analysis breakdown
                        if 'analysis_breakdown' in suggestion:
                            breakdown = suggestion['analysis_breakdown']
                            st.write("**Analysis Scores:**")
                            st.write(f"Technical: {breakdown.get('technical_score', 50):.0f}%")
                            st.write(f"Seasonal: {breakdown.get('seasonal_score', 50):.0f}%")
                            st.write(f"Sentiment: {breakdown.get('sentiment_score', 50):.0f}%")
                    
                    st.write(f"**Reasoning:** {suggestion['reasoning']}")
                    
                    # Execute button
                    if st.button(f"Execute {suggestion['action']} {suggestion['symbol']}", key=f"execute_{i}"):
                        self.execute_enhanced_trade(suggestion)
        else:
            st.info("🧠 Click 'Generate Enhanced Suggestions' to get AI-powered trading ideas")
        
        st.divider()
        
        # Market data overview
        st.subheader("📈 Live Market Data")
        
        if st.session_state.market_data:
            # Create market data tabs
            stock_tab, crypto_tab, forex_tab = st.tabs(["📈 Stocks", "₿ Crypto", "💱 Forex"])
            
            with stock_tab:
                self.show_market_data_table('stock')
            
            with crypto_tab:
                self.show_market_data_table('crypto')
            
            with forex_tab:
                self.show_market_data_table('forex')
        else:
            st.warning("📡 No market data available. Click 'Refresh Data' to fetch latest prices.")
    
    def show_market_data_table(self, asset_type):
        """Display market data table for specific asset type"""
        filtered_data = {
            symbol: data for symbol, data in st.session_state.market_data.items()
            if data.get('asset_type') == asset_type
        }
        
        if filtered_data:
            df_data = []
            for symbol, data in filtered_data.items():
                df_data.append({
                    'Symbol': symbol,
                    'Price': f"${data['price']:.2f}",
                    'Change': f"{data['change_pct']:+.2f}%",
                    'Volume': f"{data['volume']:,}",
                    'High': f"${data['high']:.2f}",
                    'Low': f"${data['low']:.2f}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"No {asset_type} data available")
    
    def show_technical_analysis_tab(self):
        """Display technical analysis results"""
        st.header("📈 Technical Analysis")
        
        if st.session_state.analysis_results.get('technical_analysis'):
            technical_results = st.session_state.analysis_results['technical_analysis']
            
            # Symbol selector
            symbols = list(technical_results.keys())
            selected_symbol = st.selectbox("Select Symbol for Analysis", symbols)
            
            if selected_symbol and selected_symbol in technical_results:
                symbol_data = technical_results[selected_symbol]
                
                # Display key indicators
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi = symbol_data.get('rsi', 50)
                    st.metric("RSI", f"{rsi:.1f}", help="Relative Strength Index")
                
                with col2:
                    macd = symbol_data.get('macd', 0)
                    st.metric("MACD", f"{macd:.4f}", help="Moving Average Convergence Divergence")
                
                with col3:
                    support = symbol_data.get('support_level', 0)
                    st.metric("Support", f"${support:.2f}", help="Support Level")
                
                with col4:
                    resistance = symbol_data.get('resistance_level', 0)
                    st.metric("Resistance", f"${resistance:.2f}", help="Resistance Level")
                
                # Technical signals
                st.subheader("🎯 Technical Signals")
                signals = symbol_data.get('signals', [])
                
                if signals:
                    for signal in signals:
                        signal_type, reason, confidence = signal
                        color = "green" if signal_type == "BUY" else "red"
                        st.write(f":{color}[{signal_type}] {reason} (Confidence: {confidence}%)")
                else:
                    st.info("No technical signals generated")
        else:
            st.info("🔄 Run technical analysis to see detailed indicators and signals")
            if st.button("📊 Run Technical Analysis Now"):
                self.run_technical_analysis()
    
    def show_seasonal_analysis_tab(self):
        """Display seasonal analysis results"""
        st.header("🗓️ Seasonal Intelligence")
        
        if st.session_state.analysis_results.get('seasonal_analysis'):
            seasonal_results = st.session_state.analysis_results['seasonal_analysis']
            
            # Sector rotation signals
            if 'sector_rotation' in seasonal_results:
                st.subheader("🔄 Sector Rotation Signals")
                
                sector_signals = seasonal_results['sector_rotation']
                for sector, signal in sector_signals.items():
                    recommendation = signal['recommendation']
                    favorable = signal['seasonal_favorable']
                    confidence = signal['confidence']
                    
                    color = "green" if recommendation == "OVERWEIGHT" else "blue"
                    status = "Favorable" if favorable else "Neutral"
                    
                    st.write(f":{color}[{sector}] {recommendation} - {status} (Confidence: {confidence}%)")
        else:
            st.info("🗓️ Run seasonal analysis to see historical patterns and seasonal signals")
            if st.button("🗓️ Run Seasonal Analysis Now"):
                self.run_seasonal_analysis()
    
    def show_market_intelligence_tab(self):
        """Display market intelligence and correlation analysis"""
        st.header("🌐 Market Intelligence Hub")
        
        # Market sentiment overview
        if st.session_state.analysis_results.get('sentiment_analysis'):
            sentiment_results = st.session_state.analysis_results['sentiment_analysis']
            
            st.subheader("📰 Market Sentiment")
            
            if 'market_sentiment' in sentiment_results:
                market_sentiment = sentiment_results['market_sentiment']
                mood = market_sentiment.get('market_mood', 'neutral')
                score = market_sentiment.get('market_sentiment_score', 50)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Market Mood", mood.title())
                
                with col2:
                    st.metric("Sentiment Score", f"{score:.0f}/100")
        else:
            st.info("🔗 Run sentiment analysis to see market intelligence")
            if st.button("📰 Run Sentiment Analysis Now"):
                self.run_sentiment_analysis()
    
    def show_technical_charts_tab(self):
        """Display technical charts with indicators"""
        st.header("📉 Technical Charts")
        
        # Symbol selector
        all_symbols = (self.config.STOCK_SYMBOLS + 
                      self.config.CRYPTO_SYMBOLS + 
                      self.config.FOREX_SYMBOLS)
        
        selected_symbol = st.selectbox("Select Symbol for Chart", all_symbols)
        
        # Chart period selector
        period = st.selectbox("Chart Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
        
        if selected_symbol:
            try:
                # Fetch data for chart
                ticker = yf.Ticker(selected_symbol)
                data = ticker.history(period=period, interval="1h" if period in ["1d", "5d"] else "1d")
                
                if not data.empty:
                    # Create candlestick chart
                    fig = go.Figure(data=go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=selected_symbol
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_symbol} Price Chart",
                        height=600,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning(f"No chart data available for {selected_symbol}")
                    
            except Exception as e:
                st.error(f"Error creating chart: {e}")
    
    # Enhanced data and analysis methods
    def fetch_latest_data(self):
        """Fetch latest market data"""
        with st.spinner("Fetching real-time data..."):
            try:
                market_data = self.data_fetcher.fetch_all_assets()
                st.session_state.market_data = market_data
                st.session_state.last_update = datetime.now()
                
                # Update position prices
                self.portfolio_manager.update_position_prices(market_data)
                
                st.success("✅ Data updated successfully!")
                return True
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return False
    
    def run_comprehensive_analysis(self):
        """Run all analysis engines"""
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Technical analysis
                technical_results = self.technical_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['technical_analysis'] = technical_results
                
                # Seasonal analysis
                seasonal_results = self.seasonal_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['seasonal_analysis'] = seasonal_results
                
                # Sentiment analysis
                sentiment_results = self.sentiment_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['sentiment_analysis'] = sentiment_results
                
                # Correlation analysis
                correlation_results = self.correlation_analyzer.analyze_all_correlations()
                st.session_state.analysis_results['correlation_analysis'] = correlation_results
                
                st.session_state.analysis_results['last_analysis'] = datetime.now()
                
                st.success("✅ Comprehensive analysis complete!")
                return True
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                return False
    
    def generate_enhanced_suggestions(self):
        """Generate enhanced trading suggestions"""
        with st.spinner("🧠 Generating enhanced suggestions..."):
            try:
                if not st.session_state.market_data:
                    self.fetch_latest_data()
                
                # Generate enhanced suggestions
                suggestions = self.enhanced_engine.generate_comprehensive_suggestions(
                    st.session_state.market_data,
                    self.portfolio_manager.positions
                )
                
                st.session_state.enhanced_suggestions = suggestions
                st.success(f"✅ Generated {len(suggestions)} enhanced suggestions!")
                
            except Exception as e:
                st.error(f"Error generating suggestions: {e}")
    
    def run_technical_analysis(self):
        """Run technical analysis only"""
        with st.spinner("Running technical analysis..."):
            try:
                technical_results = self.technical_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['technical_analysis'] = technical_results
                st.success("✅ Technical analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    def run_seasonal_analysis(self):
        """Run seasonal analysis only"""
        with st.spinner("Running seasonal analysis..."):
            try:
                seasonal_results = self.seasonal_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['seasonal_analysis'] = seasonal_results
                st.success("✅ Seasonal analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    def run_sentiment_analysis(self):
        """Run sentiment analysis only"""
        with st.spinner("Running sentiment analysis..."):
            try:
                sentiment_results = self.sentiment_analyzer.analyze_all_symbols()
                st.session_state.analysis_results['sentiment_analysis'] = sentiment_results
                st.success("✅ Sentiment analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    def run_correlation_analysis(self):
        """Run correlation analysis only"""
        with st.spinner("Running correlation analysis..."):
            try:
                correlation_results = self.correlation_analyzer.analyze_all_correlations()
                st.session_state.analysis_results['correlation_analysis'] = correlation_results
                st.success("✅ Correlation analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    def execute_enhanced_trade(self, suggestion):
        """Execute an enhanced trading suggestion"""
        try:
            # Calculate position size first
            position_info = self.portfolio_manager.calculate_position_size(
                suggestion['symbol'],
                suggestion['entry_price'],
                suggestion['stop_loss'],
                suggestion['confidence'],
                suggestion['timeframe']
            )
            
            success, message = self.portfolio_manager.add_position(
                symbol=suggestion['symbol'],
                shares=position_info['shares'],
                entry_price=suggestion['entry_price'],
                stop_loss=suggestion['stop_loss'],
                target_price=suggestion['target_price'],
                confidence=suggestion['confidence'],
                timeframe=suggestion['timeframe'],
                reasoning=suggestion['reasoning']
            )
            
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ Cannot execute trade: {message}")
                
        except Exception as e:
            st.error(f"❌ Trade execution error: {e}")
    
    def should_refresh_data(self):
        """Check if data should be refreshed"""
        time_since_update = datetime.now() - st.session_state.last_update
        return time_since_update.total_seconds() > self.config.REAL_TIME_INTERVAL
    
    def setup_auto_refresh(self):
        """Setup auto-refresh functionality"""
        # Enhanced auto-refresh with analysis updates
        if st.checkbox("🔄 Auto-refresh (60s)", value=False):
            if self.should_refresh_data():
                self.fetch_latest_data()
                # Run quick analysis every 10 minutes
                time_since_analysis = datetime.now() - st.session_state.analysis_results.get('last_analysis', datetime.now() - timedelta(minutes=11))
                if time_since_analysis.total_seconds() > 600:  # 10 minutes
                    st.info("Running periodic analysis update...")
                    self.run_comprehensive_analysis()
                st.rerun()

def main():
    """Main enhanced application entry point"""
    try:
        dashboard = EnhancedTradingDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.write("**Error Details:**")
            st.code(str(e))
            st.write("**Session State:**")
            st.write(st.session_state)

if __name__ == "__main__":
    main()
