# production/professional_trader.py
import time
import schedule
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import os
import requests

# Professional logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/professional_trader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProfessionalLiveTrader:
    def __init__(self, initial_balance=10000.0):
        self.setup_directories()
        
        # Import professional modules
        from production.professional_predictor import ProfessionalPredictor
        from risk.advanced_risk_engine import AdvancedRiskEngine
        from data_engine.advanced_data_loader import ProfessionalDataLoader
        from data_engine.advanced_feature_engineer import AdvancedFeatureEngineer
        
        # Initialize components
        self.predictor = ProfessionalPredictor()
        self.risk_engine = AdvancedRiskEngine(initial_balance)
        self.data_loader = ProfessionalDataLoader()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Trading state
        self.current_position = None
        self.signal_history = []
        self.is_running = False
        self.position_history = []  # YENİ: Trade geçmişi
        
        logger.info("🚀 PROFESSIONAL LIVE TRADER INITIALIZED")
        logger.info(f"💰 Initial Balance: ${initial_balance:,.2f}")
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs('logs', exist_ok=True)
        os.makedirs('signals', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('risk', exist_ok=True)
    
    def get_live_market_data(self):
        """Get live market data from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '15m',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"📊 Live data fetched: {len(df)} candles")
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"❌ Live data error: {e}")
            return None

    def get_current_price(self):
        """Şu anki fiyatı al - YENİ FONKSİYON"""
        try:
            df = self.get_live_market_data()
            if df is not None:
                return df['close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"❌ Current price error: {e}")
            return None
    
    def calculate_market_volatility(self, df):
        """Calculate current market volatility"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        return volatility
    
    def generate_professional_signal(self):
        """Generate professional trading signal"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"🔍 Generating professional signal: {current_time}")
        
        # Get live data
        df = self.get_live_market_data()
        if df is None:
            logger.error("❌ Failed to get live data")
            return
        
        # Calculate features
        df_with_features = self.feature_engineer.calculate_all_indicators(df)
        
        # Get AI prediction
        signal, confidence = self.predictor.predict(df_with_features)
        
        # Calculate market volatility
        volatility = self.calculate_market_volatility(df)
        current_price = df['close'].iloc[-1]
        
        # Risk management check
        if not self.risk_engine.can_enter_trade(signal):
            logger.info("⏸️ Risk management blocked trade entry")
            signal = 'HOLD'
        
        # Create professional signal
        professional_signal = {
            'timestamp': current_time,
            'signal': signal,
            'confidence': confidence,
            'price': current_price,
            'volatility': volatility,
            'position_size': 0.0,
            'risk_adjusted': True
        }
        
        # Calculate position size if not HOLD
        if signal != 'HOLD':
            position_size, usd_size = self.risk_engine.calculate_position_size(
                signal, confidence, current_price, volatility
            )
            professional_signal['position_size'] = position_size
            professional_signal['usd_size'] = usd_size
            
            # YENİ: Pozisyon oluştur
            if not self.current_position:  # Sadece pozisyon yoksa
                self.current_position = {
                    'type': signal,
                    'entry_price': current_price,
                    'size': position_size,
                    'time': datetime.now(),
                    'usd_size': usd_size
                }
                logger.info(f"🆕 Yeni Pozisyon: {signal} - ${current_price:,.2f}")
            
            # Record trade
            self.risk_engine.record_trade(signal, current_price, position_size, confidence)
        
        # Save signal
        self.save_professional_signal(professional_signal)
        self.signal_history.append(professional_signal)
        
        logger.info(f"🎯 PROFESSIONAL SIGNAL: {signal} "
                   f"| Confidence: {confidence:.1%} "
                   f"| Price: ${current_price:,.2f} "
                   f"| Size: {professional_signal.get('position_size', 0):.6f} BTC")
    
    def save_professional_signal(self, signal):
        """Save professional signal to CSV"""
        try:
            filename = f"signals/professional_signals_{datetime.now().strftime('%Y%m%d')}.csv"
            
            header = "timestamp,signal,confidence,price,volatility,position_size,risk_adjusted\n"
            row = (f"{signal['timestamp']},{signal['signal']},{signal['confidence']:.3f},"
                  f"{signal['price']:.2f},{signal['volatility']:.4f},"
                  f"{signal.get('position_size', 0):.6f},{signal['risk_adjusted']}\n")
            
            if not os.path.exists(filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(header)
            
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(row)
                
        except Exception as e:
            logger.error(f"❌ Signal save error: {e}")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        metrics = self.risk_engine.calculate_portfolio_metrics()
        
        report = f"""
📊 DAILY PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d')}
============================================
💰 Current Balance: ${metrics.get('current_balance', 0):,.2f}
📈 Total Trades: {metrics.get('total_trades', 0)}
🎯 Win Rate: {metrics.get('win_rate', 0):.1%}
📊 Total PnL: ${metrics.get('total_pnl', 0):.2f}
📉 Consecutive Losses: {metrics.get('consecutive_losses', 0)}
🕒 Report Time: {datetime.now().strftime('%H:%M:%S')}
        """
        
        logger.info(report)
        
        # Save to file
        report_filename = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def start_professional_trading(self):
        """Start professional trading system"""
        logger.info("🎯 STARTING PROFESSIONAL TRADING SYSTEM")
        
        # Schedule tasks - YENİ: Trade performans kontrolü eklendi
        schedule.every(30).minutes.do(self.generate_professional_signal)  # Every 30 minutes
        schedule.every(1).hours.do(self.risk_engine.reset_daily_metrics)  # Hourly risk reset
        schedule.every(24).hours.do(self.generate_daily_report)  # Daily report
        schedule.every(5).minutes.do(self.check_trade_performance)  # YENİ: Her 5 dakikada trade kontrol
        
        self.is_running = True
        
        # Initial signal
        self.generate_professional_signal()
        
        logger.info("✅ Professional trading system started")
        logger.info("⏰ Signal generation: Every 30 minutes")
        logger.info("📊 Risk reset: Every hour") 
        logger.info("📈 Daily report: Every 24 hours")
        logger.info("🔍 Trade performance: Every 5 minutes")  # YENİ
        logger.info("⏹️ Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Professional trading stopped by user")
            self.generate_daily_report()  # Final report

    def check_trade_performance(self):
        """Açık trade'lerin performansını kontrol et"""
        if self.current_position:
            current_price = self.get_current_price()
            if current_price is None:
                return
                
            entry_price = self.current_position['entry_price']
            
            pnl_percent = (current_price - entry_price) / entry_price * 100
            if self.current_position['type'] == 'SELL':
                pnl_percent = -pnl_percent  # Short position için tersi
            
            logger.info(f"📊 Açık Trade: {pnl_percent:+.2f}%")
            
            # Take-profit / Stop-loss kontrolü
            if pnl_percent >= 2.0:  # %2 kar
                self.close_trade("TAKE_PROFIT")
            elif pnl_percent <= -1.5:  # %1.5 zarar
                self.close_trade("STOP_LOSS")

    def close_trade(self, reason):
        """Trade'i kapat ve performansı kaydet"""
        if self.current_position:
            current_price = self.get_current_price()
            if current_price is None:
                return
                
            entry_price = self.current_position['entry_price']
            
            # PnL hesapla
            if self.current_position['type'] == 'BUY':
                pnl = (current_price - entry_price) * self.current_position['size']
            else:  # SELL
                pnl = (entry_price - current_price) * self.current_position['size']
            
            # Performans kaydı
            trade_result = {
                'entry_time': self.current_position['time'],
                'exit_time': datetime.now(),
                'type': self.current_position['type'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'reason': reason,
                'success': pnl > 0
            }
            
            # Risk engine'e gerçek PnL'i bildir
            self.risk_engine.update_trade(current_price, pnl)
            
            logger.info(f"🔚 Trade Kapandı: {reason} | PnL: ${pnl:+.2f}")
            self.position_history.append(trade_result)
            self.current_position = None

# Test and run
if __name__ == "__main__":
    trader = ProfessionalLiveTrader(initial_balance=10000.0)
    trader.start_professional_trading()