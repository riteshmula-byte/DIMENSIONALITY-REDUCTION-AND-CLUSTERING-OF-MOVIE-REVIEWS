"""Single shared data pipeline: load reviews -> TF-IDF -> TruncatedSVD.

Every method (SNE, t-SNE, K-medoids) consumes the same `Features` object, so the
load/vectorise/reduce code exists exactly once.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_NAME = "labeledTrainData.tsv"


def find_data(root: Path) -> Path:
    for p in (root / DATA_NAME, root / "data" / DATA_NAME):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Place {DATA_NAME} in {root} or {root / 'data'}")


@dataclass
class Features:
    n: int
    ids: np.ndarray
    labels: np.ndarray          # sentiment, 0/1
    Z: np.ndarray               # (n, n_components) SVD space
    tfidf: object               # sparse (n, n_features)
    terms: np.ndarray           # vocabulary
    cum_variance: np.ndarray    # cumulative explained variance of the SVD


def load_reviews(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=["id", "sentiment", "review"])


def build_features(reviews: pd.DataFrame, n: int, max_features=5000,
                   n_components=50, seed=42) -> Features:
    """Use the first `n` reviews (nested subsets, as in the original notebooks)."""
    sub = reviews.iloc[:n]
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform(sub["review"])
    k = min(n_components, len(sub) - 1, X.shape[1])
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = svd.fit_transform(X)
    return Features(n=len(sub), ids=sub["id"].to_numpy(),
                    labels=sub["sentiment"].to_numpy(), Z=Z, tfidf=X,
                    terms=vec.get_feature_names_out(),
                    cum_variance=svd.explained_variance_ratio_.cumsum())
