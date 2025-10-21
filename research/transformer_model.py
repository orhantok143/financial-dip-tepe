# research/transformer_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, Flatten, Input, 
                                   MultiHeadAttention, LayerNormalization,
                                   GlobalAveragePooling1D)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, TensorBoard)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional Encoding layer for Transformer"""
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices, cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Interleave sines and cosines
        pos_encoding = tf.reshape(
            tf.stack([sines, cosines], axis=2), 
            [position, d_model]
        )
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer Block with Multi-Head Attention"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        # Multi-Head Attention
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class FinancialTransformer:
    def __init__(self, sequence_length=20, d_model=64, num_heads=8, dff=256, num_layers=3):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.n_patterns = 6  # 6 farklı pattern
        
        print("🧠 FINANCIAL TRANSFORMER INITIALIZED")
        print(f"📊 Sequence Length: {sequence_length}")
        print(f"🎯 Number of Patterns: {self.n_patterns}")
    
    def build_model(self, n_features):
        """Build Transformer model for financial data"""
        print("🏗️ BUILDING TRANSFORMER MODEL...")
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, n_features))
        
        # Feature projection to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Positional Encoding
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Multiple Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                rate=0.1
            )(x)
        
        # Global Average Pooling
        x = GlobalAveragePooling1D()(x)
        
        # Classification Head
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer - 6 patterns
        outputs = Dense(self.n_patterns, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Optimizer with weight decay
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        
        print("✅ TRANSFORMER MODEL BUILT SUCCESSFULLY")
        return model

class AdvancedDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def load_and_preprocess_data(self):
        """Load BTC data and create advanced features"""
        print("📈 LOADING AND PREPROCESSING DATA...")
        
        try:
            # Load BTC data
            df = pd.read_csv('data/15m.csv')
            print(f"📊 Original data shape: {df.shape}")
            
            # Basic preprocessing
            df = self._basic_preprocessing(df)
            
            # Calculate advanced features
            df = self._calculate_advanced_features(df)
            
            # Create sequences and labels
            X, y = self._create_sequences(df)
            
            print(f"✅ Preprocessing complete:")
            print(f"   📊 Sequences: {X.shape}")
            print(f"   🏷️ Labels: {y.shape}")
            print(f"   🔢 Features: {X.shape[2]}")
            print(f"   🎯 Patterns: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            return self._create_sample_data()
    
    def _basic_preprocessing(self, df):
        """Basic data preprocessing"""
        # Rename columns
        df = df.rename(columns={
            'Open time': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Keep only necessary columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _calculate_advanced_features(self, df):
        """Calculate 30+ advanced technical indicators"""
        print("📊 CALCULATING ADVANCED FEATURES...")
        
        # 1. PRICE FEATURES
        # Multiple moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ROC_{period}'] = df['close'].pct_change(period)  # Rate of Change
        
        # 2. VOLATILITY FEATURES
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 3. MOMENTUM INDICATORS
        # Multiple RSI periods
        for period in [6, 14, 28]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['close'], period)
        
        # Stochastic
        df['Stoch_K'] = self._calculate_stochastic(df, 14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df, 14)
        
        # 4. VOLUME INDICATORS
        df['Volume_SMA'] = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # 5. TREND INDICATORS
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 6. CUSTOM FEATURES
        # Price action features
        df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
        df['Open_Close_Range'] = (df['close'] - df['open']) / df['open']
        df['Body_Size'] = abs(df['close'] - df['open']) / df['open']
        
        # Volatility
        df['Volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        self.feature_columns = [col for col in df.columns if col not in ['timestamp']]
        print(f"✅ {len(self.feature_columns)} ADVANCED FEATURES CREATED")
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df, period=14):
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        return 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        return 100 * ((high_max - df['close']) / (high_max - low_min))
    
    def _create_sequences(self, df, sequence_length=20):
        """Create sequences for training"""
        print("🔄 CREATING SEQUENCES...")
        
        X, y = [], []
        
        # Use all features
        feature_data = df[self.feature_columns].values
        
        # Create sequences (for now, use simple labeling)
        # In real scenario, you'd detect patterns here
        for i in range(sequence_length, len(feature_data)):
            sequence = feature_data[i-sequence_length:i]
            X.append(sequence)
            
            # Simple trend-based labeling (replace with actual pattern detection)
            current_price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-5]
            
            price_change = (current_price - prev_price) / prev_price
            
            if price_change > 0.02:
                label = 0  # Uptrend
            elif price_change < -0.02:
                label = 1  # Downtrend
            else:
                label = 2  # Sideways
        
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        print("📝 CREATING SAMPLE DATA...")
        
        # Generate sample data
        n_samples = 10000
        sequence_length = 20
        n_features = 30
        
        X = np.random.randn(n_samples, sequence_length, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 classes
        
        return X, y

def train_transformer_model():
    """Train the transformer model"""
    print("🚀 TRAINING TRANSFORMER MODEL")
    print("=" * 60)
    
    # Initialize components
    preprocessor = AdvancedDataPreprocessor()
    transformer = FinancialTransformer()
    
    # Load and preprocess data
    X, y = preprocessor.load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📚 DATA SPLIT:")
    print(f"   🏋️ Training: {X_train.shape}")
    print(f"   📋 Validation: {X_val.shape}")
    print(f"   🧪 Test: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Build model
    model = transformer.build_model(n_features=X_train.shape[2])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
        ModelCheckpoint('models/transformer_model.h5', monitor='val_accuracy', 
                       save_best_only=True, save_weights_only=False),
        TensorBoard(log_dir='logs/transformer')
    ]
    
    # Train model
    print("\n🎯 STARTING TRAINING...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 MODEL EVALUATION:")
    train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
    val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)[1]
    test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    
    print(f"   ✅ Training Accuracy: {train_accuracy:.4f}")
    print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
    print(f"   ✅ Test Accuracy: {test_accuracy:.4f}")
    
    # Save artifacts
    joblib.dump(scaler, 'models/transformer_scaler.pkl')
    joblib.dump(preprocessor.feature_columns, 'models/feature_columns.pkl')
    
    print("\n💾 MODEL AND ARTIFACTS SAVED:")
    print("   models/transformer_model.h5")
    print("   models/transformer_scaler.pkl")
    print("   models/feature_columns.pkl")
    
    return model, history

if __name__ == "__main__":
    model, history = train_transformer_model()
    
    print("\n🎉 TRANSFORMER MODEL TRAINING COMPLETED!")
    print("🚀 Next step: Implement advanced pattern detection for 6 patterns")