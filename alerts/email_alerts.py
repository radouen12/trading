"""
Email Alert System - Phase 3
Send email notifications for trading signals and alerts
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

class EmailAlerts:
    """
    Email notification system for trading alerts
    """
    
    def __init__(self, smtp_server: str = None, smtp_port: int = None):
        # Load configuration from environment/config
        from config import Config
        config = Config()
        
        self.smtp_server = smtp_server or config.SMTP_SERVER
        self.smtp_port = smtp_port or config.SMTP_PORT
        self.sender_email = config.EMAIL_SENDER
        self.sender_password = config.EMAIL_PASSWORD
        self.recipient_email = config.EMAIL_RECIPIENT
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration on initialization
        if self._is_configured():
            self.logger.info(f"Email alerts configured for {self.recipient_email}")
        else:
            self.logger.warning("Email alerts not configured - check environment variables")
    
    def configure_smtp(self, sender_email: str, sender_password: str, recipient_email: str):
        """Configure SMTP settings"""
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
        self.logger.info(f"Email alerts configured for {recipient_email}")
    
    def _is_configured(self) -> bool:
        """Check if email is properly configured"""
        return all([
            self.sender_email,
            self.sender_password, 
            self.recipient_email,
            self.smtp_server,
            self.smtp_port
        ])
    
    def send_trading_signal_alert(self, signals: List[Dict]) -> bool:
        """Send alert for new trading signals"""
        
        if not self._is_configured():
            self.logger.warning("Email not configured, skipping alert")
            return False
        
        try:
            # Create email content
            subject = f"🚨 New Trading Signals - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            body = self._create_signal_email_body(signals)
            
            # Send email
            return self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send trading signal alert: {e}")
            return False
    
    def send_position_alert(self, alert_type: str, symbol: str, details: Dict) -> bool:
        """Send alert for position events (stop loss, target, etc.)"""
        
        if not self._is_configured():
            return False
        
        try:
            subject = f"📊 Position Alert: {alert_type} - {symbol}"
            body = self._create_position_alert_body(alert_type, symbol, details)
            
            return self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send position alert: {e}")
            return False
    
    def send_daily_report(self, portfolio_summary: Dict, top_signals: List[Dict]) -> bool:
        """Send daily portfolio and signal summary"""
        
        if not self._is_configured():
            return False
        
        try:
            subject = f"📈 Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
            body = self._create_daily_report_body(portfolio_summary, top_signals)
            
            return self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send daily report: {e}")
            return False
    
    def send_risk_alert(self, risk_type: str, details: Dict) -> bool:
        """Send risk management alerts"""
        
        if not self._is_configured():
            return False
        
        try:
            subject = f"⚠️ Risk Alert: {risk_type}"
            body = self._create_risk_alert_body(risk_type, details)
            
            return self._send_email(subject, body, urgent=True)
            
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {e}")
            return False
    
    def _create_signal_email_body(self, signals: List[Dict]) -> str:
        """Create email body for trading signals"""
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #1f2937; color: white; padding: 15px; text-align: center; }}
                .signal {{ background-color: #f3f4f6; margin: 10px 0; padding: 15px; border-radius: 8px; }}
                .buy {{ border-left: 5px solid #10b981; }}
                .sell {{ border-left: 5px solid #ef4444; }}
                .confidence {{ font-weight: bold; }}
                .high-confidence {{ color: #10b981; }}
                .medium-confidence {{ color: #f59e0b; }}
                .low-confidence {{ color: #ef4444; }}
                .footer {{ margin-top: 20px; padding: 15px; background-color: #f9fafb; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 New Trading Signals</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h3>📊 Signal Summary</h3>
            <p><strong>Total Signals:</strong> {len(signals)}</p>
            <p><strong>High Confidence (80%+):</strong> {len([s for s in signals if s.get('confidence', 0) >= 80])}</p>
            <p><strong>Medium Confidence (60-79%):</strong> {len([s for s in signals if 60 <= s.get('confidence', 0) < 80])}</p>
        """
        
        # Add individual signals
        for signal in signals[:10]:  # Limit to top 10 signals
            confidence = signal.get('confidence', 0)
            confidence_class = 'high-confidence' if confidence >= 80 else 'medium-confidence' if confidence >= 60 else 'low-confidence'
            action_class = 'buy' if signal.get('action') == 'BUY' else 'sell'
            
            html_body += f"""
            <div class="signal {action_class}">
                <h4>{signal.get('symbol', 'N/A')} - {signal.get('action', 'N/A')}</h4>
                <p><strong>Price:</strong> ${signal.get('price', 0):.2f}</p>
                <p><strong>Target:</strong> ${signal.get('target', 0):.2f}</p>
                <p><strong>Stop Loss:</strong> ${signal.get('stop_loss', 0):.2f}</p>
                <p><strong>Position Size:</strong> ${signal.get('position_size', 0):,.2f}</p>
                <p class="confidence {confidence_class}"><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p><strong>Strategy:</strong> {signal.get('strategy', 'AI Enhanced')}</p>
                <p><strong>Reasoning:</strong> {signal.get('reasoning', 'Multi-factor analysis')}</p>
            </div>
            """
        
        html_body += """
            <div class="footer">
                <p>⚠️ This is an automated alert from your AI Trading System</p>
                <p>Always perform your own analysis before making trading decisions</p>
                <p>Never risk more than you can afford to lose</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _create_position_alert_body(self, alert_type: str, symbol: str, details: Dict) -> str:
        """Create position alert email body"""
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="background-color: #dc2626; color: white; padding: 15px; text-align: center;">
                <h2>📊 Position Alert: {alert_type}</h2>
                <h3>{symbol}</h3>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Alert Details:</h3>
                <p><strong>Symbol:</strong> {symbol}</p>
                <p><strong>Alert Type:</strong> {alert_type}</p>
                <p><strong>Current Price:</strong> ${details.get('current_price', 0):.2f}</p>
                <p><strong>Entry Price:</strong> ${details.get('entry_price', 0):.2f}</p>
                <p><strong>Triggered Level:</strong> ${details.get('trigger_price', 0):.2f}</p>
                <p><strong>P&L:</strong> ${details.get('pnl', 0):,.2f}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px;">
                <h4>📋 Recommended Action:</h4>
                <p>{details.get('recommendation', 'Review position and take appropriate action.')}</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _create_daily_report_body(self, portfolio_summary: Dict, top_signals: List[Dict]) -> str:
        """Create daily report email body"""
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="background-color: #1f2937; color: white; padding: 15px; text-align: center;">
                <h2>📈 Daily Trading Report</h2>
                <p>{datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <h3>💰 Portfolio Summary</h3>
            <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px;">
                <p><strong>Total Value:</strong> ${portfolio_summary.get('total_value', 0):,.2f}</p>
                <p><strong>Daily P&L:</strong> ${portfolio_summary.get('daily_pnl', 0):,.2f} ({portfolio_summary.get('daily_pnl_pct', 0):.2f}%)</p>
                <p><strong>Cash Available:</strong> ${portfolio_summary.get('cash_available', 0):,.2f}</p>
                <p><strong>Open Positions:</strong> {portfolio_summary.get('open_positions', 0)}</p>
            </div>
            
            <h3>🎯 Top Signals Today</h3>
        """
        
        for signal in top_signals[:5]:
            html_body += f"""
            <div style="background-color: #f9fafb; margin: 10px 0; padding: 10px; border-left: 3px solid #10b981;">
                <p><strong>{signal.get('symbol')} - {signal.get('action')}</strong></p>
                <p>Confidence: {signal.get('confidence', 0):.1f}%</p>
                <p>Price: ${signal.get('price', 0):.2f}</p>
            </div>
            """
        
        html_body += """
        </body>
        </html>
        """
        
        return html_body
    
    def _create_risk_alert_body(self, risk_type: str, details: Dict) -> str:
        """Create risk alert email body"""
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="background-color: #dc2626; color: white; padding: 15px; text-align: center;">
                <h2>⚠️ RISK ALERT</h2>
                <h3>{risk_type}</h3>
            </div>
            
            <div style="background-color: #fef2f2; padding: 15px; border: 2px solid #dc2626; border-radius: 8px; margin: 20px 0;">
                <h4>🚨 Immediate Attention Required</h4>
                <p><strong>Risk Type:</strong> {risk_type}</p>
                <p><strong>Severity:</strong> {details.get('severity', 'High')}</p>
                <p><strong>Description:</strong> {details.get('description', 'Risk threshold breached')}</p>
                <p><strong>Current Value:</strong> {details.get('current_value', 'N/A')}</p>
                <p><strong>Threshold:</strong> {details.get('threshold', 'N/A')}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px;">
                <h4>📋 Recommended Actions:</h4>
                <ul>
                    {' '.join([f'<li>{action}</li>' for action in details.get('recommended_actions', [])])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _send_email(self, subject: str, body: str, urgent: bool = False) -> bool:
        """Send email using SMTP"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            if urgent:
                msg['X-Priority'] = '1'  # High priority
            
            # Attach HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                text = msg.as_string()
                server.sendmail(self.sender_email, self.recipient_email, text)
            
            self.logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _is_configured(self) -> bool:
        """Check if email is properly configured"""
        return all([self.sender_email, self.sender_password, self.recipient_email])
    
    def test_connection(self) -> bool:
        """Test email configuration"""
        
        if not self._is_configured():
            self.logger.error("Email not configured")
            return False
        
        try:
            test_subject = "✅ Trading System Email Test"
            test_body = """
            <html>
            <body>
                <h2>Email Alert System Test</h2>
                <p>If you receive this email, your alert system is working correctly!</p>
                <p>Timestamp: {}</p>
            </body>
            </html>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            return self._send_email(test_subject, test_body)
            
        except Exception as e:
            self.logger.error(f"Email test failed: {e}")
            return False
