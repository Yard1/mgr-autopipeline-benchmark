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

import os
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_RESULT_DELIM"] = "/"
os.environ["TUNE_FORCE_TRIAL_CLEANUP_S"] = "10"

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    print(f"!FITTING {data_idx}_{random_seed}")
    ray.init()
    X, y = prepare_data(data_idx, random_seed, True, preprocess=False, split=False)
    y = y.astype("category")
    if y.nunique() > 2:
        func = f1_weighted
        eval_metric = "f1_weighted"
    else:
        func = f1_score
        eval_metric = "f1"
    am = AutoML(
        random_state=random_seed,
        level="common",
        test_size=0.25,
        target_metric=eval_metric,
        trainer_config={
            "secondary_level": "uncommon",
            "cache": True,
            "early_stopping": True,
            "return_test_scores_during_tuning": False,
            "tuning_time": 60,
            "stacking_level": 0,
            "tune_kwargs": {"max_concurrent": 1, "trainable_n_jobs": 5, "verbose": 0, "fail_fast": False, "raise_on_failed_trial": False}
        },
        stacking_cv=5,
        cv=5,
    )
    makedirs(f"./tmp/mine_{data_idx}_{random_seed}/", exist_ok=True)
    with open(f"./tmp/mine_{data_idx}_{random_seed}/log.txt", 'w') as f:
        with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                am.fit(X, y)
    best_test = am.results_.iloc[0][f"Test {eval_metric}"]
    best_single_test = am.score_pipeline(am.best_id_non_ensemble_)[eval_metric]
    print(f"best_test {best_test} best_single_test {best_single_test}")
    final_test = best_single_test if best_single_test > best_test else best_test
    am.results_.to_csv(f"./tmp/mine_{data_idx}_{random_seed}/results.csv")
    print(am.results_)
    print(f"!NUM_EVALUATED {data_idx}_{random_seed} {len(am.results_)}")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", final_test)
    ray.shutdown()
    shutil.rmtree("/tmp/joblib", ignore_errors=True)

if __name__ == "__main__":
    for i in range(0, 39):
        run(i, 1)