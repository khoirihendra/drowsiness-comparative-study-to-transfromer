"""Lazy training dependencies and portable preprocessing + predictor bundles."""
import math
from pathlib import Path
import numpy as np
from src.provenance import write_json
from src.research_data import FeatureTransform, summary_features


def fit_classifier(name, x, y, xv, yv, out, args):
    if name in ('logistic', 'xgboost'):
        z = summary_features(x)
        if name == 'logistic':
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(z)
            model = LogisticRegression(max_iter=1000, C=args.logistic_c, random_state=args.seed)
            model.fit(scaler.transform(z), y)
            if not np.array_equal(model.classes_, [0, 1, 2]):
                raise ValueError('Training classifier must contain all three classes')
            np.savez_compressed(out / 'logistic.npz', mean=scaler.mean_, scale=scaler.scale_,
                                coef=model.coef_, intercept=model.intercept_)
            info = dict(iterations=model.n_iter_.tolist(), optimizer_updates=None)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05,
                                  subsample=.8, colsample_bytree=.8, reg_lambda=5,
                                  objective='multi:softprob', num_class=3,
                                  eval_metric='mlogloss', random_state=args.seed, n_jobs=4)
            model.fit(z, y, eval_set=[(summary_features(xv), yv)], verbose=False)
            model.save_model(out / 'xgboost.json')
            info = dict(optimizer_updates=None, history=model.evals_result())
        write_json(out / 'training.json', info)
        return

    import tensorflow as tf
    from src.models import MODEL_BUILDERS
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model = MODEL_BUILDERS[name](input_shape=x.shape[1:], num_classes=3,
                                 positional_encoding=not args.no_positional_encoding)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1.),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    steps = args.steps_per_epoch or math.ceil(len(x) / args.batch_size)
    if args.max_updates:
        steps = min(steps, args.max_updates)
    epochs = math.ceil(args.max_updates / steps) if args.max_updates else args.epochs

    class UpdateBudget(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if args.max_updates and int(self.model.optimizer.iterations.numpy()) >= args.max_updates:
                self.model.stop_training = True

    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(out / 'model.keras'),
                 monitor='val_loss', save_best_only=True), UpdateBudget(),
                 tf.keras.callbacks.CSVLogger(str(out / 'history.csv'))]
    # No early stopping in budget mode: thinning arms receive equal updates.
    if not args.max_updates:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                         patience=args.patience, restore_best_weights=True))
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(
        min(len(x), 20000), seed=args.seed, reshuffle_each_iteration=True)
    ds = ds.repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    validation = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(args.batch_size)
    history = model.fit(ds, steps_per_epoch=steps, epochs=epochs,
                        validation_data=validation, callbacks=callbacks, verbose=2).history
    write_json(out / 'training.json', dict(history=history, steps_per_epoch=steps,
               optimizer_updates=int(model.optimizer.iterations.numpy()),
               best_epoch=int(np.argmin(history['val_loss']) + 1),
               checkpoint_selection='minimum validation loss'))


def load_predictor(out):
    out = Path(out)
    if (out / 'logistic.npz').exists():
        with np.load(out / 'logistic.npz', allow_pickle=False) as a:
            mean, scale, coef, intercept = (a[k].copy() for k in ('mean', 'scale', 'coef', 'intercept'))

        def predict(z):
            logits = ((summary_features(z) - mean) / scale) @ coef.T + intercept
            exp = np.exp(logits - logits.max(1, keepdims=True))
            return exp / exp.sum(1, keepdims=True)
        return predict
    if (out / 'xgboost.json').exists():
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(out / 'xgboost.json')
        return lambda z: model.predict_proba(summary_features(z))
    import tensorflow as tf
    from src.models.transformer import SinusoidalPosition  # Register serialization.
    model = tf.keras.models.load_model(out / 'model.keras', compile=False)
    return lambda z: np.asarray(model(z, training=False))


def predict_bundle(out, x, usable, observed, batch_size=256):
    """Accept raw EAR/MAR/degree angles AND masks, not normalized inputs."""
    transform = FeatureTransform.load(Path(out) / 'preprocessing.json')
    predictor = load_predictor(out)
    result = []
    for start in range(0, len(x), batch_size):
        sl = slice(start, start + batch_size)
        result.append(predictor(transform.transform(x[sl], usable[sl], observed[sl])))
    if not result:
        raise ValueError('No windows to predict')
    return np.concatenate(result)
