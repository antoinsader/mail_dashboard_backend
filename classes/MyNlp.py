


from collections import Counter
import hashlib
import itertools
import os
import pickle
from typing import Iterable, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import spacy


# ==========================
#       DATA CLASSES
# ==========================

@dataclass(frozen=True)
class EmailDoc:
    id:str
    text:str
    html: Optional[str]
    sender: Optional[str]
    subject: Optional[str]
    language: Optional[str]


@dataclass
class Dataset:
    docs : List[EmailDoc]

    def by_lang(self, lang:str) -> List[EmailDoc]:
        return [d for d in self.docs if d.language == lang]

    def ids(self) -> List[str]:
        return [d.id for  d in self.docs]


@dataclass(frozen=True)
class TokenKind:
    remove_punc: bool = True
    remove_stop: bool = True
    min_alpha_len: int = 3
    use_lemma: bool = True


TOKEN_KINDS : Dict[str, TokenKind]  = {
    "nlp": TokenKind(),
    "tfidf": TokenKind(min_alpha_len=1, use_lemma=False),
    "summarization": TokenKind(remove_stop=False, min_alpha_len=1, use_lemma=False)
}

# ==========================
#       STORAGE
# ==========================

class CacheMall:
    def __init__(self, base_ds_dir):
        self.root_dir = base_ds_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _store_key(self, store_type, store_name, version=1):
        raw = f"{store_type}::{store_name}::{version}".encode()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return os.path.join(self.root_dir, f"{store_type}__{store_name}__v{version}__hsh{digest}.pkl")

    def store_exists(self, store_type, store_name, version ):
        return os.path.exists(self._store_key(store_type, store_name, version))

    def load_store(self, store_type, store_name, version):
        with open(self._store_key(store_type, store_name, version ), "rb") as f:
            return pickle.load(f)

    def save_in_store(self, obj, store_type, store_name, version):
        pth = self._store_key(store_type, store_name, version)
        with open(pth, "wb") as f:
            pickle.dump(obj, f)
        return pth

# ==========================
#       UTILS
# ==========================

def most_common(tokens_iter: Iterable[Iterable[str]], n: int = 50):
    all_toks = list(itertools.chain.from_iterable(tokens_iter))
    return Counter(all_toks).most_common(n)





# ==========================
#       NLP LANGUAGES
# ==========================

SPACY_NLPS_BY_LANG = {
    "en": spacy.load("en_core_web_sm", disable=['ner', 'parser']),
    "it": spacy.load("it_core_news_sm", disable=['ner', 'parser']),
}




# ==========================
#       TOKENIZER
# ==========================


