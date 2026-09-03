from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

from config import XGBOOST_PARAMS, SEED



def flatten_temporal_features(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D sequence array (samples, timesteps, features) into
    2D tabular array (samples, timesteps * features) for tabular tree models.
    """
    num_samples = X.shape[0]
    num_flat_features = X.shape[1] * X.shape[2]
    return X.reshape(num_samples, num_flat_features)


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    custom_params: Optional[Dict] = None,
    verbose: Union[bool, int] = False
) -> "xgb.XGBClassifier":
    """
    Train an XGBoost Classifier on flattened temporal sequence features.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please run 'pip install xgboost'.")

    # Flatten if 3D
    if X_train.ndim == 3:
        X_train_flat = flatten_temporal_features(X_train)
    else:
        X_train_flat = X_train


    params = XGBOOST_PARAMS.copy()
    if custom_params:
        params.update(custom_params)

    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        if X_val.ndim == 3:
            X_val_flat = flatten_temporal_features(X_val)
        else:
            X_val_flat = X_val

        eval_set = [(X_train_flat, y_train), (X_val_flat, y_val)]
        model.fit(X_train_flat, y_train, eval_set=eval_set, verbose=verbose)
    else:
        model.fit(X_train_flat, y_train, verbose=verbose)

    return model


def evaluate_xgboost_model(
    model: "xgb.XGBClassifier",
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities and class indices using trained XGBoost model.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please run 'pip install xgboost'.")

    if X_test.ndim == 3:
        X_test_flat = flatten_temporal_features(X_test)
    else:
        X_test_flat = X_test

    pred_probs = model.predict_proba(X_test_flat)
    pred_classes = np.argmax(pred_probs, axis=1)
    return pred_probs, pred_classes
