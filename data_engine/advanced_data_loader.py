# data_engine/advanced_data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalDataLoader:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.timeframes = ['15m', '1h', '4h', '1d']
        
    def load_all_timeframes(self):
        """Tüm timeframe'leri yükle"""
        print("📈 LOADING MULTI-TIMEFRAME DATA...")
        
        dataframes = {}
        for tf in self.timeframes:
            try:
                file_path = f"{self.data_path}/{tf}.csv"
                df = pd.read_csv(file_path)
                
                # Standardize columns
                df = self.standardize_columns(df)
                
                # Basic cleaning
                df = self.clean_data(df)
                
                dataframes[tf] = df
                print(f"   ✅ {tf.upper()}: {len(df):,} records")
                
            except Exception as e:
                print(f"   ❌ {tf} error: {e}")
                
        return dataframes
    
    def standardize_columns(self, df):
        """Kolon isimlerini standardize et"""
        column_mapping = {
            'Open time': 'timestamp', 'Open': 'open', 
            'High': 'high', 'Low': 'low', 'Close': 'close', 
            'Volume': 'volume', 'Close time': 'close_time'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Timestamp conversion
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def clean_data(self, df):
        """Veri temizleme"""
        # Numeric conversion
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Fill NaN
        df = df.ffill().bfill()
        
        return df
    
    def remove_outliers(self, df, n_std=3):
        """Aykırı değerleri temizle"""
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] > mean - n_std * std) & (df[col] < mean + n_std * std)]
        return df

# Test
if __name__ == "__main__":
    loader = ProfessionalDataLoader()
    data = loader.load_all_timeframes()
    print(f"✅ Loaded {len(data)} timeframes")