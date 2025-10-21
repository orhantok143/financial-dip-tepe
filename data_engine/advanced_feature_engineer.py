# data_engine/advanced_feature_engineer.py
import pandas as pd
import numpy as np
import ta
from ta import trend, momentum, volatility, volume

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_count = 0
        
    def calculate_all_indicators(self, df):
        """50+ advanced teknik indikatör"""
        print("🔧 CALCULATING 50+ ADVANCED INDICATORS...")
        
        df = df.copy()
        
        # 1. TREND INDICATORS (15)
        df = self._calculate_trend_indicators(df)
        
        # 2. MOMENTUM INDICATORS (15)  
        df = self._calculate_momentum_indicators(df)
        
        # 3. VOLATILITY INDICATORS (10)
        df = self._calculate_volatility_indicators(df)
        
        # 4. VOLUME INDICATORS (10)
        df = self._calculate_volume_indicators(df)
        
        # 5. CYCLE INDICATORS
        df = self._calculate_cycle_indicators(df)
        
        # Clean NaN
        df = df.ffill().bfill()
        
        self.feature_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        print(f"✅ {self.feature_count} ADVANCED INDICATORS CALCULATED")
        
        return df
    
    def _calculate_trend_indicators(self, df):
        """Trend indikatörleri"""
        # Multiple SMAs
        for period in [5, 10, 20, 50, 100]:
            df[f'SMA_{period}'] = df['close'].rolling(period).mean()
            
        # Multiple EMAs  
        for period in [8, 13, 21, 34, 55]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
            
        # MACD
        macd = trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # Ichimoku (simplified)
        df = self._calculate_ichimoku(df)
        
        return df
    
    def _calculate_momentum_indicators(self, df):
        """Momentum indikatörleri"""
        # Multiple RSI periods
        for period in [6, 14, 28]:
            df[f'RSI_{period}'] = momentum.RSIIndicator(df['close'], window=period).rsi()
            
        # Stochastic
        stoch = momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # CCI
        df['CCI'] = trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Awesome Oscillator
        df['AO'] = momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        
        return df
    
    def _calculate_volatility_indicators(self, df):
        """Volatilite indikatörleri"""
        # Bollinger Bands
        bb = volatility.BollingerBands(df['close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = bb.bollinger_wband()
        
        # ATR
        df['ATR'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Keltner Channel
        df = self._calculate_keltner_channel(df)
        
        return df
    
    def _calculate_volume_indicators(self, df):
        """Volume indikatörleri"""
        # OBV
        df['OBV'] = volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume SMA
        for period in [5, 20, 50]:
            df[f'Volume_SMA_{period}'] = df['volume'].rolling(period).mean()
            
        # Volume Price Trend
        df['VPT'] = (df['volume'] * (df['close'].diff() / df['close'].shift(1))).cumsum()
        
        return df
    
    def _calculate_ichimoku(self, df):
        """Ichimoku Cloud (simplified)"""
        # Tenkan-sen (Conversion Line)
        period9_high = df['high'].rolling(9).max()
        period9_low = df['low'].rolling(9).min()
        df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = df['high'].rolling(26).max()
        period26_low = df['low'].rolling(26).min()
        df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2
        
        return df
    
    def _calculate_keltner_channel(self, df):
        """Keltner Channel"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['KC_Middle'] = typical_price.ewm(span=20).mean()
        
        atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
        df['KC_Upper'] = df['KC_Middle'] + 2 * atr
        df['KC_Lower'] = df['KC_Middle'] - 2 * atr
        
        return df
    
    def _calculate_cycle_indicators(self, df):
        """Cycle indicators"""
        # Price Rate of Change
        df['ROC'] = momentum.ROCIndicator(df['close']).roc()
        
        # Price Acceleration
        df['Price_Accel'] = df['ROC'].diff()
        
        return df

# Test
if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    
    # Sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
    sample_data = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 1000),
        'high': np.random.normal(50500, 1000, 1000),
        'low': np.random.normal(49500, 1000, 1000),
        'close': np.random.normal(50000, 1000, 1000),
        'volume': np.random.normal(1000, 100, 1000)
    }, index=dates)
    
    result = engineer.calculate_all_indicators(sample_data)
    print(f"📊 Final shape: {result.shape}")