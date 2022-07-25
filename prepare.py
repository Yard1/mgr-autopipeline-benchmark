import pandas as pd
from datasets import get_classical_ml_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from abc import abstractproperty
from typing import Callable, Dict, Optional
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer
from category_encoders import TargetEncoder

def drop_all_na_pandas(X: pd.DataFrame, y=None):
    return X.dropna(axis=1, how="all")

def get_preprocessor(data: pd.DataFrame) -> Pipeline:
    cat_columns = list(data.select_dtypes("category").columns)
    num_categories = sum(len(data[col].unique()) for col in cat_columns)
    if num_categories < 20000:
        return Pipeline([
            ("dropna", FunctionTransformer(drop_all_na_pandas)),
            ("column",
                ColumnTransformer([
                    ("numerical",
                    Pipeline([("impute", SimpleImputer()),
                            ("scale", StandardScaler())]),
                    make_column_selector(dtype_exclude="category")),
                    ("categorical",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(drop="if_binary",
                                        dtype=np.bool,
                                        sparse=False),
                        )
                    ]), make_column_selector(dtype_include="category"))
                ]))
        ])
    else:
        return Pipeline([
            ("dropna", FunctionTransformer(drop_all_na_pandas)),
            ("column",
                ColumnTransformer([
                    ("numerical", SimpleImputer(),
                    make_column_selector(dtype_exclude="category")),
                    ("categorical",
                    TargetEncoder(drop_invariant=True, smoothing=300),
                    make_column_selector(dtype_include="category"))
                ])), (
                    "scale",
                    StandardScaler(),
                )
        ])

def prepare_data(data_idx, random_seed, stratify,*, preprocess=False, split=True):
    data = pd.read_parquet(get_classical_ml_datasets()[data_idx])
    y = pd.Series(LabelEncoder().fit_transform(data["target"]), name="target")
    X = data.drop("target", axis=1)
    if preprocess:
        X = get_preprocessor(data).fit_transform(X)
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed, test_size=0.25, stratify=y if stratify else None)
        return X_train, X_test, y_train, y_test
    return X, y

