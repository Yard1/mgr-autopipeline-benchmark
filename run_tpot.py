from functools import partial
from os import makedirs
from prepare import prepare_data
from tpot import TPOTClassifier
from sklearn.metrics import f1_score, make_scorer
import argparse
from os import makedirs
import contextlib

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    print(f"!FITTING {data_idx}_{random_seed}")
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True, preprocess=True)
    if y_train.nunique() > 2:
        scorer = f1_weighted
    else:
        scorer = partial(f1_score, zero_division=0)
        scorer.__name__ = "f1"
    makedirs(f"./tmp/tpot_{data_idx}_{random_seed}/out", exist_ok=True)
    pipeline_optimizer = TPOTClassifier(max_time_mins=1, cv=5, memory="auto", log_file=f"./tmp/tpot_{data_idx}_{random_seed}/out/log.txt",
                                    random_state=random_seed, verbosity=3, n_jobs=16, scoring=make_scorer(scorer))
    pipeline_optimizer.fit(X_train, y_train)
    y_hat = pipeline_optimizer.predict(X_test)
    print(pipeline_optimizer.evaluated_individuals_)
    print(f"!NUM_EVALUATED {data_idx}_{random_seed} {len(pipeline_optimizer.evaluated_individuals_)}")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", f1_weighted(y_test, y_hat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        help="seed to use")
    args, _ = parser.parse_known_args()
    random_seed = args.seed
    for data_idx in range(39):
        makedirs(f"./tmp/tpot_{data_idx}_{random_seed}/", exist_ok=True)
        with open(f"./tmp/tpot_{data_idx}_{random_seed}/log.txt", 'w') as f:
            with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                run(i, random_seed)