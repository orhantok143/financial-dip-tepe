# strategies/professional_strategy.py
class ProfessionalTradingStrategy:
    def __init__(self):
        self.risk_per_trade = 0.02  # %2 risk
        self.max_daily_loss = 0.05  # %5 max daily loss
        self.consecutive_losses = 0
        
    def calculate_position_size(self, account_balance, confidence, volatility):
        """Kelly Criterion + Volatility adjusted position sizing"""
        # Kelly fraction
        kelly_f = confidence - (1 - confidence)
        
        # Volatility adjustment
        vol_adjustment = 1.0 / (volatility * np.sqrt(252))
        
        # Position size
        position_size = account_balance * self.risk_per_trade * kelly_f * vol_adjustment
        
        # Maximum %10 of account
        return min(position_size, account_balance * 0.10)
    
    def should_enter_trade(self, signal, confidence, market_regime):
        """Trade entry koşulları"""
        if confidence < 0.65:
            return False
            
        if self.consecutive_losses >= 3:
            return False  # Stop after 3 consecutive losses
            
        if market_regime == 'HIGH_VOLATILITY' and confidence < 0.75:
            return False
            
        return True