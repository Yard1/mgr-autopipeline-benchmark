from functools import partial
from prepare import prepare_data
from sklearn.metrics import f1_score
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import argparse
from os import makedirs
import contextlib

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    print(f"!FITTING {data_idx}_{random_seed}")
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True, preprocess=False)
    X_train = TabularDataset(pd.concat([X_train, y_train], axis=1))
    if pd.concat([y_train, y_test]).nunique() > 2:
        func = f1_weighted
        eval_metric = "f1_weighted"
    else:
        func = f1_score
        eval_metric = "f1"
    predictor = TabularPredictor(label="target", path=f"./tmp/autogluon_{data_idx}_{random_seed}/out", eval_metric=eval_metric, ag_fit_args={'num_cpus': NUM_CORES_YOU_WANT})
    predictor = predictor.fit(X_train, time_limit=3600, presets='best_quality')
    y_hat = predictor.predict(TabularDataset(X_test))
    print(predictor.leaderboard())
    print(f"!NUM_EVALUATED {data_idx}_{random_seed} {len(predictor.leaderboard())}")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", func(y_test, y_hat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        help="seed to use")
    args, _ = parser.parse_known_args()
    random_seed = args.seed
    for data_idx in range(39):
        makedirs(f"./tmp/autogluon_{data_idx}_{random_seed}/", exist_ok=True)
        with open(f"./tmp/autogluon_{data_idx}_{random_seed}/log.txt", 'w') as f:
            with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                run(data_idx, random_seed)