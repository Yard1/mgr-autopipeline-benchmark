from functools import partial
from os import makedirs
from prepare import prepare_data
from tpot import TPOTClassifier
from sklearn.metrics import f1_score, make_scorer

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
    makedirs(f"./tmp/tpot_{data_idx}_{random_seed}/", exist_ok=True)
    with open(f"./tmp/tpot_{data_idx}_{random_seed}/log.txt", "w") as f:
        pass
    pipeline_optimizer = TPOTClassifier(max_time_mins=1, cv=5, memory="auto", log_file=f"./tmp/tpot_{data_idx}_{random_seed}/log.txt",
                                    random_state=random_seed, verbosity=3, n_jobs=4, scoring=make_scorer(scorer))
    pipeline_optimizer.fit(X_train, y_train)
    y_hat = pipeline_optimizer.predict(X_test)
    print(pipeline_optimizer.evaluated_individuals_)
    print(f"!NUM_EVALUATED {data_idx}_{random_seed} {len(pipeline_optimizer.evaluated_individuals_)}")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", f1_weighted(y_test, y_hat))

if __name__ == "__main__":
    for i in range(0, 1):
        run(i, 1)