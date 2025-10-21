# risk/advanced_risk_engine.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdvancedRiskEngine:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = []
        self.trade_history = []
        
        # Risk parameters
        self.max_position_size = 0.1  # Max 10% of account per trade
        self.daily_loss_limit = 0.03  # Max 3% daily loss
        self.max_consecutive_losses = 3
        self.consecutive_losses = 0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        
        logger.info("🛡️ ADVANCED RISK ENGINE INITIALIZED")
    
    def calculate_position_size(self, signal, confidence, current_price, volatility):
        """Advanced position sizing using Kelly Criterion"""
        # Base risk (1% of account)
        base_risk = self.current_balance * 0.01
        
        # Confidence adjustment
        confidence_multiplier = confidence ** 2  # Square confidence for conservative sizing
        
        # Volatility adjustment (higher volatility = smaller position)
        volatility_multiplier = 1.0 / max(volatility, 0.01)
        
        # Signal strength adjustment
        if 'STRONG' in signal:
            signal_multiplier = 1.5
        else:
            signal_multiplier = 1.0
            
        # Calculate position size
        position_size = (base_risk * confidence_multiplier * 
                        volatility_multiplier * signal_multiplier)
        
        # Apply maximum position size limit
        max_size = self.current_balance * self.max_position_size
        position_size = min(position_size, max_size)
        
        # Convert to BTC amount
        btc_amount = position_size / current_price
        
        logger.info(f"📊 Position Size: ${position_size:.2f} ({btc_amount:.6f} BTC)")
        return btc_amount, position_size
    
    def can_enter_trade(self, signal_type):
        """Check if we can enter a new trade"""
        # Daily loss limit check
        if self.daily_pnl < -self.current_balance * self.daily_loss_limit:
            logger.warning("🚨 Daily loss limit reached")
            return False
            
        # Consecutive losses check
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("🚨 Max consecutive losses reached")
            return False
            
        # Existing position check
        if any(pos['status'] == 'OPEN' for pos in self.positions):
            logger.info("⏳ Existing position open")
            return False
            
        return True
    
    def record_trade(self, signal, entry_price, position_size, confidence):
        """Record a new trade"""
        trade = {
            'entry_time': datetime.now(),
            'signal': signal,
            'entry_price': entry_price,
            'position_size': position_size,
            'confidence': confidence,
            'status': 'OPEN'
        }
        
        self.positions.append(trade)
        logger.info(f"📝 Trade recorded: {signal} at ${entry_price:.2f}")
    
    def update_trade(self, exit_price, pnl):
        """Update trade on exit"""
        if self.positions:
            trade = self.positions[-1]
            trade.update({
                'exit_time': datetime.now(),
                'exit_price': exit_price,
                'pnl': pnl,
                'status': 'CLOSED'
            })
            
            self.trade_history.append(trade)
            self.current_balance += pnl
            self.daily_pnl += pnl
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            logger.info(f"💰 Trade closed: PnL ${pnl:.2f}")
    
    def calculate_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        if not self.trade_history:
            return {}
            
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade,
            'current_balance': self.current_balance,
            'consecutive_losses': self.consecutive_losses
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (should be called daily)"""
        if datetime.now().date() > self.last_reset.date():
            self.daily_pnl = 0.0
            self.last_reset = datetime.now()
            logger.info("🔄 Daily metrics reset")

# Test
if __name__ == "__main__":
    risk_engine = AdvancedRiskEngine()
    print("✅ Advanced Risk Engine Ready!")