            # Simulate some performance (random walk)
            change = performance[-1] * 0.001 * (1 if i % 3 == 0 else -0.5)
            performance.append(performance[-1] + change)
        
        # Create performance DataFrame
        perf_df = pd.DataFrame({
            'Date': dates[:len(performance)],
            'Portfolio Value': performance
        })
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf_df['Date'],
            y=perf_df['Portfolio Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Add horizontal line for initial capital
        fig.add_hline(
            y=self.config.TOTAL_CAPITAL,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def fetch_latest_data(self):
        """Fetch latest market data"""
        with st.spinner("Fetching real-time data..."):
            try:
                market_data = self.data_fetcher.fetch_all_assets()
                st.session_state.market_data = market_data
                st.session_state.last_update = datetime.now()
                
                # Update position prices
                self.portfolio_manager.update_position_prices(market_data)
                
                return True
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return False
    
    def should_refresh_data(self):
        """Check if data should be refreshed"""
        time_since_update = datetime.now() - st.session_state.last_update
        return time_since_update.total_seconds() > self.config.REAL_TIME_INTERVAL
    
    def execute_trade(self, suggestion):
        """Execute a trading suggestion"""
        try:
            success, message = self.portfolio_manager.add_position(
                symbol=suggestion['symbol'],
                shares=suggestion['shares'],
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
    
    def close_position(self, symbol):
        """Close a position"""
        try:
            if symbol in self.portfolio_manager.positions:
                position = self.portfolio_manager.positions[symbol]
                
                # Calculate P&L
                pnl = position['unrealized_pnl']
                
                # Add cash back
                self.portfolio_manager.cash += position['current_value']
                
                # Update daily P&L
                self.portfolio_manager.daily_pnl += pnl
                self.portfolio_manager.total_pnl += pnl
                
                # Remove position
                del self.portfolio_manager.positions[symbol]
                
                st.success(f"✅ Closed {symbol} with P&L: ${pnl:+,.2f}")
            else:
                st.error(f"❌ Position {symbol} not found")
                
        except Exception as e:
            st.error(f"❌ Error closing position: {e}")
    
    def setup_auto_refresh(self):
        """Setup auto-refresh functionality"""
        # Auto-refresh every 60 seconds
        if st.checkbox("🔄 Auto-refresh (60s)", value=True):
            time.sleep(1)
            if self.should_refresh_data():
                self.fetch_latest_data()
                st.rerun()

def main():
    """Main application entry point"""
    try:
        dashboard = TradingDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")

if __name__ == "__main__":
    main()
