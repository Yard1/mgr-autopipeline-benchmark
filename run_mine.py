from functools import partial
from os import makedirs
import shutil
from prepare import prepare_data
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from automl.search.automl import AutoML
from ray.tune.progress_reporter import JupyterNotebookReporter
import contextlib
import ray
import argparse

import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_RESULT_DELIM"] = "/"
os.environ["TUNE_FORCE_TRIAL_CLEANUP_S"] = "10"

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    print(f"!FITTING {data_idx}_{random_seed}")
    ray.init(num_cpus=16)
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True, preprocess=False)
    y_train = y_train.astype("category")
    y_test = y_test.astype("category")
    if y_train.nunique() > 2:
        func = f1_weighted
        eval_metric = "f1_weighted"
        scorer = f1_weighted
    else:
        func = f1_score
        eval_metric = "f1"
        scorer = partial(f1_score, zero_division=0)
    am = AutoML(
        random_state=random_seed,
        level="common",
        test_size=0.1,
        target_metric=eval_metric,
        trainer_config={
            "secondary_level": "uncommon",
            "cache": True,
            # "early_stopping": False,
            "return_test_scores_during_tuning": False,
            "tuning_time": 3600,
            "stacking_level": 0,
            "tune_kwargs": {"max_concurrent": 3, "trainable_n_jobs": 5, "verbose": 0, "fail_fast": False, "raise_on_failed_trial": False}
        },
        stacking_cv=5,
        cv=5,
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        am.fit(X_train, y_train)
    pred = am.best_pipeline_
    print(am.results_)
    print(f"!NUM_EVALUATED {data_idx}_{random_seed} {len(am.results_)}")
    y_hat = pred.predict(X_test)
    print(f"!RESULT {data_idx}_{random_seed} F1 score", scorer(y_test, y_hat))
    ray.shutdown()
    shutil.rmtree("/tmp/joblib", ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        help="seed to use")
    args, _ = parser.parse_known_args()
    random_seed = args.seed
    for i in range(0, 39):
        makedirs(f"./tmp/mine_{i}_{random_seed}/", exist_ok=True)
        with open(f"./tmp/mine_{i}_{random_seed}/log.txt", 'w') as f:
            with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                run(i, random_seed)