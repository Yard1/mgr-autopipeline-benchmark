from functools import partial
from prepare import prepare_data
from tpot import TPOTClassifier
from sklearn.metrics import f1_score, make_scorer
from joblib import dump

f1_weighted = partial(f1_score, zero_division=0, average="weighted")
f1_weighted.__name__ = "f1_weighted"

def run(data_idx, random_seed):
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True, preprocess=True)
    if y_train.nunique() > 2:
        scorer = f1_weighted
    else:
        scorer = partial(f1_score, zero_division=0)
        scorer.__name__ = "f1"
    pipeline_optimizer = TPOTClassifier(max_time_mins=1, cv=5, memory="auto",
                                    random_state=random_seed, verbosity=2, n_jobs=4, scoring=make_scorer(scorer))
    pipeline_optimizer.fit(X_train, y_train)
    y_hat = pipeline_optimizer.predict(X_test)
    dump(pipeline_optimizer, f"./tmp/tpot_{data_idx}_{random_seed}/opt.pkl")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", f1_weighted(y_test, y_hat))

if __name__ == "__main__":
    for i in range(0, 39):
        run(i, 1)