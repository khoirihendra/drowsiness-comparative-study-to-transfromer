"""
Standard Unidirectional LSTM Architecture for Drowsiness Classification.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Construct Standard LSTM model with Batch Normalization and L2 regularization.

    Args:
        input_shape: (timesteps, features), e.g. (30, 5) or (30, 1).
        num_classes: Number of output classes (default: 3).
        l2_reg: L2 regularization factor.
        dropout_rate: Dropout rate.

    Returns:
        tf.keras.Model instance.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Layer 1: First Standard LSTM Layer
        LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Layer 2: Second Standard LSTM Layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Layer 3: Dense Layer 1
        Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(max(0.0, dropout_rate - 0.1)),

        # Layer 4: Dense Layer 2
        Dense(32, activation="relu"),

        # Output Softmax Layer
        Dense(num_classes, activation="softmax")
    ], name="LSTM_Drowsiness_Detector")

    return model
