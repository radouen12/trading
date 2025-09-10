"""
Automation Module - Phase 4
Automated execution capabilities with safeguards
"""

from .execution_engine import ExecutionEngine
from .risk_monitor import RiskMonitor
from .signal_processor import SignalProcessor

__all__ = [
    'ExecutionEngine',
    'RiskMonitor', 
    'SignalProcessor'
]
