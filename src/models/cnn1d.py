"""
1D Convolutional Neural Network (1D-CNN) for Temporal Drowsiness Classification.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_1dcnn_model(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.4,
    **kwargs
) -> tf.keras.Model:
    """
    Construct 1D-CNN model for temporal sequence processing.

    Args:
        input_shape: (timesteps, features).
        num_classes: Number of output classes (default: 3).
        l2_reg: L2 regularization factor.
        dropout_rate: Dropout rate.

    Returns:
        tf.keras.Model instance.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Block 1: Conv1D + BatchNorm + MaxPool + Dropout
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        # Block 2: Conv1D + BatchNorm + MaxPool + Dropout
        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        # Flatten 2D temporal representation into 1D vector
        Flatten(),

        # Dense classification head
        Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(max(0.0, dropout_rate - 0.1)),

        Dense(32, activation="relu"),

        # Output Softmax Layer
        Dense(num_classes, activation="softmax")
    ], name="CNN1D_Drowsiness_Detector")

    return model
