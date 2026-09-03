"""
Time-Series Transformer Architecture (Multi-Head Self-Attention) for Drowsiness Classification.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


def transformer_encoder_block(
    inputs: tf.Tensor,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.3
) -> tf.Tensor:
    """
    Core Transformer Encoder block with Multi-Head Self Attention,
    Layer Normalization, Feed-Forward Network, and Residual Connections.
    """
    # 1. Multi-Head Attention Sub-layer
    norm1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(norm1, norm1)
    res1 = layers.Add()([attn, inputs])

    # 2. Feed-Forward Sub-layer
    norm2 = layers.LayerNormalization(epsilon=1e-6)(res1)
    ff = layers.Dense(ff_dim, activation="relu")(norm2)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)
    res2 = layers.Add()([ff, res1])

    return res2


def build_transformer_model(
    input_shape: Tuple[int, int],
    num_classes: int = 3,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    dropout: float = 0.3,
    l2_reg: float = 0.001
) -> tf.keras.Model:
    """
    Construct Time-Series Transformer architecture.

    Args:
        input_shape: (timesteps, features).
        num_classes: Number of output classes (default: 3).
        head_size: Attention key/query dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        num_transformer_blocks: Number of stacked encoder layers.
        dropout: Attention & dense dropout rate.
        l2_reg: L2 regularization factor.

    Returns:
        tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Linear projection to expand feature dimension to 64
    x = layers.Dense(64)(inputs)

    # Stack Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

    # Temporal aggregation via Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification Head
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(32, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer_Drowsiness_Detector")
    return model
