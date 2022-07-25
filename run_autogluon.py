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
    if X_train["target"].nunique() > 2:
        func = f1_weighted
    else:
        func = f1_score
    scorer = make_scorer(name="f1", score_func=func, optimum=1, greater_is_better=True)
    predictor = TabularPredictor(label="target", path=f"./tmp/autogluon_{data_idx}_{random_seed}", eval_metric=scorer)
    predictor = predictor.fit(X_train, time_limit=120, presets='best_quality')
    y_hat = predictor.predict(TabularDataset(X_test))
    print("F1 score", func(y_test, y_hat))

if __name__ == "__main__":
    run(0, 1)