from functools import partial
from prepare import prepare_data
from sklearn.metrics import f1_score
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True, preprocess=False)
    X_train = TabularDataset(pd.concat([X_train, y_train], axis=1))
    if pd.concat([y_train, y_test]).nunique() > 2:
        func = f1_weighted
        eval_metric = "f1_weighted"
    else:
        func = f1_score
        eval_metric = "f1"
    predictor = TabularPredictor(label="target", path=f"./tmp/autogluon_{data_idx}_{random_seed}", eval_metric=eval_metric)
    predictor = predictor.fit(X_train, time_limit=60, presets='best_quality')
    y_hat = predictor.predict(TabularDataset(X_test))
    print(f"!RESULT {data_idx}_{random_seed} F1 score", func(y_test, y_hat))

if __name__ == "__main__":
    for i in range(0, 39):
        run(i, 1)