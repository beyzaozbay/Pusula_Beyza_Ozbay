from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re, unicodedata
from scipy.sparse import csr_matrix

def _norm_token(s):
    if s is None: return None
    t = str(s).strip().lower()
    # Türkçe karakterleri sadeleştir + punctuation temizle
    t = (t.replace("ı","i").replace("İ","i")
           .replace("ş","s").replace("ğ","g")
           .replace("ç","c").replace("ö","o").replace("ü","u"))
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or None

def _split_cell(x):
    if pd.isna(x): return []
    parts = re.split(r"[;,/|]+", str(x))
    return [p.strip() for p in parts if p.strip()]

class MultiLabelBinarizerDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        # clone() uyumu: __init__ içinde parametreyi **hiç değiştirme**
        self.columns = columns

    def fit(self, X, y=None):
        data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        self.vocab_ = {}
        self._order_ = []
        self.feature_names_ = []
        seen = set()  # tüm feature adlarında global tekillik

        for col in self.columns:
            toks_series = data[col].apply(_split_cell)
            norm_series = toks_series.apply(lambda lst: [_norm_token(t) for t in lst])
            vocab = sorted({t for lst in norm_series for t in lst if t})
            self.vocab_[col] = vocab
            for tok in vocab:
                name = f"mlb__{col}__{tok}"
                if name in seen:
                    continue
                seen.add(name)
                self._order_.append((col, tok))
                self.feature_names_.append(name)
        return self

    def transform(self, X):
        data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        n_rows = len(data)
        n_cols = len(self._order_)
        out = np.zeros((n_rows, n_cols), dtype=np.int8)

        tokenized = {}
        for col in self.columns:
            toks_series = data[col].apply(_split_cell)
            tokenized[col] = toks_series.apply(
                lambda lst: { _norm_token(t) for t in lst if _norm_token(t) }
            )

        for j, (col, tok) in enumerate(self._order_):
            col_sets = tokenized[col]
            for i, toks in enumerate(col_sets):
                if tok in toks:
                    out[i, j] = 1
        return csr_matrix(out)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)
