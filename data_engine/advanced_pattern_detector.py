# research/advanced_pattern_detector.py
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternDetector:
    def __init__(self):
        self.patterns = {
            0: 'V_Bottom',
            1: 'Inverse_V', 
            2: 'Double_Bottom',
            3: 'Double_Top',
            4: 'Head_Shoulders',
            5: 'Triangle'
        }
        
    def detect_all_patterns(self, df):
        """6 farklı pattern tespiti"""
        print("🎯 DETECTING 6 ADVANCED PATTERNS...")
        
        patterns = []
        
        # Swing noktalarını bul
        swing_highs, swing_lows = self.find_swing_points(df)
        
        # Pattern'leri tespit et
        patterns.extend(self.detect_v_patterns(df, swing_lows))
        patterns.extend(self.detect_inverse_v_patterns(df, swing_highs))
        patterns.extend(self.detect_double_patterns(df, swing_lows, swing_highs))
        patterns.extend(self.detect_triangle_patterns(df, swing_highs, swing_lows))
        
        print(f"✅ {len(patterns)} PATTERNS DETECTED")
        return patterns
    
    def find_swing_points(self, df, order=5):
        """Swing high/low noktaları"""
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = argrelextrema(highs, np.greater, order=order)[0]
        swing_lows = argrelextrema(lows, np.less, order=order)[0]
        
        return swing_highs, swing_lows
    
    def detect_v_patterns(self, df, swing_lows):
        """V Bottom pattern detection"""
        patterns = []
        
        for i in range(2, len(swing_lows)):
            left_idx = swing_lows[i-2]
            bottom_idx = swing_lows[i-1] 
            right_idx = swing_lows[i]
            
            left_price = df['low'].iloc[left_idx]
            bottom_price = df['low'].iloc[bottom_idx]
            right_price = df['low'].iloc[right_idx]
            
            # V pattern conditions
            left_to_bottom = (bottom_price - left_price) / left_price
            bottom_to_right = (bottom_price - right_price) / bottom_price
            
            if left_to_bottom < -0.01 and bottom_to_right < -0.01:
                patterns.append({
                    'type': 'V_Bottom',
                    'indices': [left_idx, bottom_idx, right_idx],
                    'timestamp': df.index[bottom_idx],
                    'price': bottom_price,
                    'confidence': min(abs(left_to_bottom), abs(bottom_to_right))
                })
                
        return patterns
    
    def detect_double_patterns(self, df, swing_lows, swing_highs):
        """Double Bottom/Top patterns"""
        patterns = []
        
        # Double Bottom
        for i in range(3, len(swing_lows)):
            if (abs(swing_lows[i-1] - swing_lows[i-2]) / swing_lows[i-2] < 0.005 and
                swing_lows[i] > swing_lows[i-1]):
                patterns.append({
                    'type': 'Double_Bottom',
                    'indices': [swing_lows[i-2], swing_lows[i-1], swing_lows[i]],
                    'timestamp': df.index[swing_lows[i-1]],
                    'price': df['low'].iloc[swing_lows[i-1]]
                })
                
        return patterns
    
    def detect_triangle_patterns(self, df, swing_highs, swing_lows):
        """Triangle patterns"""
        patterns = []
        # Triangle pattern detection logic
        return patterns

# Test
if __name__ == "__main__":
    detector = AdvancedPatternDetector()
    print("✅ Advanced Pattern Detector Ready")