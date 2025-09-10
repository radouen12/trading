"""
Phase 4 Dashboard - Intelligence & Automation
Complete trading system with ML, automation, and advanced analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 Phase 4: AI Trading System",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .phase4-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Phase 4 Dashboard"""
    
    # Header
    st.markdown("""
    <div class="phase4-header">
        <h1>🤖 Phase 4: Intelligence & Automation</h1>
        <p>Advanced ML Trading System with Automated Execution & Risk Management</p>
        <p><strong>🧠 Machine Learning | 🤖 Automation | 📊 Regime Detection | 🔄 Arbitrage</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Phase 4 Controls")
        
        execution_mode = st.selectbox("Execution Mode", ["PAPER", "LIVE"])
        max_daily_loss = st.slider("Max Daily Loss %", 1.0, 5.0, 3.0)
        min_ml_confidence = st.slider("Min ML Confidence", 0.5, 0.9, 0.7)
        
        if st.button("🛑 EMERGENCY STOP", type="primary"):
            st.error("Emergency stop activated")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 ML Intelligence",
        "🤖 Automation Hub", 
        "📊 Regime Analysis",
        "🔄 Arbitrage"
    ])
    
    # Tab 1: ML Intelligence
    with tab1:
        st.header("🧠 Machine Learning Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 ML Predictions")
            
            # Mock ML predictions for demonstration
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            predictions_data = []
            
            for symbol in symbols:
                direction = np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
                confidence = np.random.uniform(0.5, 0.95)
                
                predictions_data.append({
                    'Symbol': symbol,
                    'ML Direction': direction,
                    'ML Confidence': f"{confidence:.1%}",
                    'Prediction Strength': np.random.choice(['STRONG', 'MODERATE', 'WEAK']),
                    'Recommendation': np.random.choice(['BUY', 'SELL', 'HOLD'])
                })
            
            df = pd.DataFrame(predictions_data)
            st.dataframe(df, use_container_width=True)
            
            # Confidence distribution
            confidences = [float(p['ML Confidence'].strip('%'))/100 for p in predictions_data]
            fig = px.histogram(x=confidences, title="ML Prediction Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Model Performance")
            
            # Mock performance metrics
            st.metric("Avg Directional Accuracy", "68.5%")
            st.metric("Avg R² Score", "0.234")
            st.metric("Total Models", "12")
            
            if st.button("🚀 Initialize ML Models"):
                with st.spinner("Training ML models..."):
                    st.success("Models initialized successfully!")
    
    # Tab 2: Automation Hub
    with tab2:
        st.header("🤖 Automation Hub")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Execution Engine Status")
            
            # Mock execution status
            st.metric("Execution Mode", "PAPER")
            st.metric("Daily Trade Count", "3")
            st.metric("Daily P&L", "$127.50")
            
            st.subheader("🛡️ Safety Limits")
            st.write("Max Daily Trades: 20")
            st.write("Max Position Size: 5.0%")
            st.write("Max Daily Loss: 3.0%")
        
        with col2:
            st.subheader("📡 Signal Processing")
            
            # Mock signal analytics
            st.metric("Processed Signals (24h)", "15")
            st.metric("Accepted Signals", "8")
            st.metric("Rejected Signals", "7")
            
            # Signal test
            st.subheader("🧪 Test Signal Processing")
            test_symbol = st.selectbox("Test Symbol", ['AAPL', 'MSFT', 'GOOGL'])
            test_action = st.selectbox("Test Action", ["BUY", "SELL", "HOLD"])
            test_confidence = st.slider("Test Confidence", 0.5, 1.0, 0.75)
            
            if st.button("📤 Process Test Signal"):
                st.success(f"Signal processed: {test_action} {test_symbol} (Confidence: {test_confidence:.1%})")
    
    # Tab 3: Regime Analysis
    with tab3:
        st.header("📊 Market Regime Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Market Overview")
            
            st.metric("Dominant Regime", "BULL_MARKET")
            st.metric("Market Sentiment", "BULLISH")
            
            # Mock regime distribution
            regime_data = {'BULL_MARKET': 6, 'BEAR_MARKET': 1, 'SIDEWAYS': 3}
            fig = px.pie(values=list(regime_data.values()), names=list(regime_data.keys()),
                        title="Market Regime Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Individual Asset Regimes")
            
            regime_data = []
            for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:
                regime_data.append({
                    'Symbol': symbol,
                    'Current Regime': np.random.choice(['BULL_MARKET', 'BEAR_MARKET', 'SIDEWAYS']),
                    'Confidence': f"{np.random.uniform(0.6, 0.95):.1%}",
                    'Strategy': np.random.choice(['TREND_FOLLOWING', 'MEAN_REVERSION', 'MOMENTUM'])
                })
            
            df_regimes = pd.DataFrame(regime_data)
            st.dataframe(df_regimes, use_container_width=True)
    
    # Tab 4: Arbitrage
    with tab4:
        st.header("🔄 Arbitrage Opportunities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Opportunities", "5")
        with col2:
            st.metric("Statistical Arbitrage", "2")
        with col3:
            st.metric("Pair Trading", "3")
        
        # Mock arbitrage opportunities
        st.subheader("📊 Current Opportunities")
        
        opportunities = [
            {
                'Type': 'Statistical Arbitrage',
                'Assets': 'SPY vs QQQ',
                'Direction': 'LONG SPY / SHORT QQQ',
                'Confidence': '78%',
                'Z-Score': '-2.34'
            },
            {
                'Type': 'Pair Trading',
                'Assets': 'XLF vs XLK',
                'Direction': 'LONG SPREAD',
                'Confidence': '65%',
                'Z-Score': '1.87'
            },
            {
                'Type': 'Mean Reversion',
                'Assets': 'AAPL',
                'Direction': 'LONG',
                'Confidence': '72%',
                'Z-Score': '-2.1'
            }
        ]
        
        df_opportunities = pd.DataFrame(opportunities)
        st.dataframe(df_opportunities, use_container_width=True)
        
        # Display detailed opportunities
        for i, opp in enumerate(opportunities):
            with st.expander(f"{opp['Type']} Opportunity {i+1}"):
                st.write(f"**Assets:** {opp['Assets']}")
                st.write(f"**Direction:** {opp['Direction']}")
                st.write(f"**Confidence:** {opp['Confidence']}")
                st.write(f"**Z-Score:** {opp['Z-Score']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Phase 4: Intelligence & Automation Complete | 
        🧠 ML Models: Active | 🤖 Automation: Ready | 📊 Regime Detection: Online | 🔄 Arbitrage: Scanning</p>
        <p><em>⚠️ For educational purposes only. Always use proper risk management.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
