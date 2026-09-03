"""
Model definitions for Drowsiness Detection.
"""

MODEL_BUILDERS = {}

try:
    from .bilstm import build_bilstm_model
    from .lstm import build_lstm_model
    from .bigru import build_bigru_model
    from .cnn1d import build_1dcnn_model
    from .transformer import build_transformer_model

    MODEL_BUILDERS.update({
        "bilstm": build_bilstm_model,
        "lstm": build_lstm_model,
        "bigru": build_bigru_model,
        "cnn1d": build_1dcnn_model,
        "transformer": build_transformer_model,
    })
except ImportError:
    pass

try:
    from .xgboost_model import train_xgboost_model, evaluate_xgboost_model
except ImportError:
    pass

