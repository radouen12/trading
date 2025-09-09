"""
Alert System Module - Phase 3
Email and SMS notifications for trading signals
"""

from .email_alerts import EmailAlerts
from .sms_alerts import SMSAlerts
from .alert_manager import AlertManager

__all__ = ['EmailAlerts', 'SMSAlerts', 'AlertManager']
