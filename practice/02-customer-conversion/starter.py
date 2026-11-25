from pathlib import Path
import numpy as np
import pandas as pd
DATA = Path(__file__).resolve().parent / 'data'

def build_features(data_dir=DATA):
    raise NotImplementedError

def train_model(features):
    raise NotImplementedError

def evaluate(model, X_test, y_test):
    raise NotImplementedError

if __name__ == '__main__':
    features = build_features()
    assert features.shape == (300, 14)
    assert features.customer_id.is_unique
    model, name, scores, X_test, y_test = train_model(features)
    assert len(X_test) == 45
    print(name, scores, evaluate(model, X_test, y_test))
    # TODO: construct prediction table
