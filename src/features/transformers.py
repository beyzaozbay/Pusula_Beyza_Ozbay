from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -------- Utilities --------
def _tokenize_cell(s: object) -> list[str]:
    """Tokenize a multi-label cell into normalized tokens."""
    if s is None:
        return []
    if isinstance(s, float) and np.isnan(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    # Split by common separators
    tokens = re.split(r"[;,/|]+", s)
    out = []
    # Basic TR -> ASCII-ish normalization for robustness
    tr_map = str.maketrans({
        "ı":"i","İ":"i","ç":"c","ğ":"g","ö":"o","ş":"s","ü":"u",
        "Ç":"c","Ğ":"g","Ö":"o","Ş":"s","Ü":"u"
    })
    for t in tokens:
        t = t.strip().lower().translate(tr_map)
        # Keep alnum, space, plus, minus; drop others
        t = re.sub(r"[^a-z0-9 \+\-]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            out.append(t)
    return out

def _safe_name(tok: str) -> str:
    """Make a token safe for use in column names."""
    s = tok.lower()
    s = re.sub(r"[^a-z0-9\+]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# -------- Transformer --------
class MultiLabelBinarizerDF(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible multi-label binarizer for multiple text columns.
    - Parameters are stored without mutation in __init__ (important for cloning).
    - Vocab is learned per column on fit.
    - Transform outputs a dense np.ndarray (int8) with 0/1 indicators.
    """
    def __init__(self, columns=None):
        self.columns = columns  # do not mutate here

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            raise ValueError("`columns` must be provided.")
        # Keep an internal copy; never mutate public params
        self._columns_ = list(self.columns)
        self.vocab_ = {}
        for col in self._columns_:
            ser = X[col] if col in X.columns else pd.Series([], dtype=object)
            col_tokens = set()
            # Build per-column vocabulary
            for s in ser.fillna(""):
                for tok in _tokenize_cell(s):
                    col_tokens.add(tok)
            self.vocab_[col] = sorted(col_tokens)
        # Build feature names (stable order: by column, then token)
        self.feature_names_ = []
        for col in self._columns_:
            for tok in self.vocab_[col]:
                self.feature_names_.append(f"{col}__{_safe_name(tok)}")
        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if self.n_features_out_ == 0:
            return np.zeros((n, 0), dtype=np.int8)
        # Pre-tokenize inputs column-wise for speed
        tokenized = {}
        for col in self._columns_:
            ser = X[col] if col in X.columns else pd.Series([""]*n, dtype=object)
            tokenized[col] = ser.fillna("").map(_tokenize_cell).tolist()
        # Fill matrix
        out = np.zeros((n, self.n_features_out_), dtype=np.int8)
        j = 0
        for col in self._columns_:
            vocab = self.vocab_[col]
            for tok in vocab:
                for i, toks in enumerate(tokenized[col]):
                    if tok in toks:
                        out[i, j] = 1
                j += 1
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)
