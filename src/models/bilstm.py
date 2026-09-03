"""
Bidirectional LSTM Architecture for Multi-Class Drowsiness Classification.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_bilstm_model(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Construct Bidirectional LSTM model with Batch Normalization and L2 regularization.

    Args:
        input_shape: (timesteps, features), e.g. (30, 5) or (30, 2) or (30, 1).
        num_classes: Number of output classes (default: 3).
        l2_reg: L2 regularization factor.
        dropout_rate: Dropout rate for regularization.

    Returns:
        Compiled or uncompiled tf.keras.Model instance.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Layer 1: First Bidirectional LSTM Layer
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg))),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Layer 2: Second Bidirectional LSTM Layer
        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Layer 3: Dense Fusion Layer 1
        Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(max(0.0, dropout_rate - 0.1)),

        # Layer 4: Dense Fusion Layer 2
        Dense(32, activation="relu"),

        # Output Softmax Layer
        Dense(num_classes, activation="softmax")
    ], name="BiLSTM_Drowsiness_Detector")

    return model
