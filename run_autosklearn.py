from functools import partial
from prepare import prepare_data
import autosklearn.classification
import autosklearn.metrics
from sklearn.metrics import f1_score
from joblib import dump

f1_weighted = partial(f1_score, zero_division=0, average="weighted")

def run(data_idx, random_seed):
    X_train, X_test, y_train, y_test = prepare_data(data_idx, random_seed, True)
    if y_train.nunique() > 2:
        scorer = f1_weighted
    else:
        scorer = partial(f1_score, zero_division=0)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        tmp_folder=f'./tmp/autosklearn_tmp_{data_idx}_{random_seed}',
        n_jobs=4,
        # Each one of the 4 jobs is allocated 2GB
        memory_limit=2048,
        seed=random_seed,
        metric=autosklearn.metrics.make_scorer(
            'f1_weighted', scorer
        ),
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    dump(pipeline_optimizer, f"./tmp/autosklearn_tmp_{data_idx}_{random_seed}/opt.pkl")
    print(f"!RESULT {data_idx}_{random_seed} F1 score", scorer(y_test, y_hat))

if __name__ == "__main__":
    run(0,1)