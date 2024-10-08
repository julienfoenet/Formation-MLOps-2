import os
import time
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model_with_io(features_path: str, model_registry_folder: str, model_path: str) -> None:
    features = pd.read_parquet(features_path)

    train_model(features, model_registry_folder, model_path)


def train_model(features: pd.DataFrame, model_registry_folder: str, model_path: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    model = RandomForestRegressor(n_estimators=1, max_depth=10, n_jobs=1)
    model.fit(X, y)

    with open(os.path.join(model_registry_folder, 'version'), 'w') as f:
        json.dump({'latest': model_path}, f)

    joblib.dump(model, os.path.join(model_registry_folder, model_path))


def predict_with_io(features_path: str, model_registry_folder: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)

    with open(os.path.join(model_registry_folder, 'version'), 'r') as f:
        model_version = json.load(f)
    model_path = model_version['latest']

    features = predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features
