from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
import pickle
from pydoc import text
from typing import Iterable, List, Optional, Dict, Any, Tuple

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
import torch
from tqdm import tqdm

import spacy.lang.en.stop_words as en_sw
from scipy.sparse import vstack, csr_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
#       DATA CLASSES
# ==========================

@dataclass(frozen=True)
class Paths:
    base_dir : Path

    @property
    def details_folder(self):
        p = self.base_dir / "details"
        p.mkdir(parents=True, exist_ok=True)
        return p 

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

class ArtifactStore:
    
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, namespace:str, name:str, version:str) -> Path:
        raw = f"{namespace}::{name}::{version}".encode()
        digest= hashlib.sha256(raw).hexdigest()[:16]
        return self.root / f"{namespace}__{name}__v{version}__{digest}.pkl"

    def has(self, namespace: str, name: str, version: str) -> bool:
        return self._key(namespace, name, version).exists()


    def load(self, namespace: str, name: str, version: str) -> Any:
        with open(self._key(namespace, name, version), "rb") as f:
          return pickle.load(f)

    def save(self, obj: Any, namespace: str, name: str, version: str) -> Path:
        path = self._key(namespace, name, version)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path


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

@dataclass
class Tokenized:
    """
        List of document ids and for each document id there are a list of tokens tuples (lemma, text)
    """
    ids: List[str]
    tokens: List[List[Tuple[str, str]]] # (lemma_lower, text_lower)

class TokenizerService:
    NAMESPACE = "tokens"


    def __init__(self, store: ArtifactStore, version: str = "1"):
        self.store = store
        self.version = version


    def _artifact_name(self, dataset_name: str) -> str:
        return f"{dataset_name}__raw"


    def run(self, dataset_name: str, ds: Dataset, force: bool = False) -> Tokenized:
        name = self._artifact_name(dataset_name)
        if self.store.has(self.NAMESPACE, name, self.version) and not force:
            return self.store.load(self.NAMESPACE, name, self.version)

        tokens_per_doc = []
        ids: List[str] = []
        by_lang: Dict[str, List[int]] = {}


        for i, d in enumerate(ds.docs):
            by_lang.setdefault(d.language, []).append(i)

        for lang, idxs in by_lang.items():
            nlp = SPACY_NLPS_BY_LANG.get(lang)
            if not nlp:
                continue
            texts = [ds.docs[i].text for i in idxs]

            for doc, i in tqdm(zip(nlp.pipe(texts, batch_size=128), idxs), total=len(idxs), desc=f"tok[{lang}]"):
                pairs = [(t.lemma_.lower(), t.text.lower()) for t in doc]
                tokens_per_doc.append(pairs)
                ids.append(ds.docs[i].id)


        result = Tokenized(ids=ids, tokens=tokens_per_doc)
        self.store.save(result, self.NAMESPACE, name, self.version)
        return result


    def view(self, result: Tokenized, kind: str = "nlp") -> Dict[str, List[str]]:
        cfg: TokenKind = TOKEN_KINDS[kind]
        out: Dict[str, List[str]] = {}
        STOP = en_sw.STOP_WORDS
        for i, doc_id in enumerate(result.ids):
            toks = [
                (lemma if cfg.use_lemma else text)
                for (lemma, text) in result.tokens[i]
                if (not cfg.remove_stop or lemma not in STOP)
                and (not cfg.remove_punc or text.isalpha())
                and len(text) >= cfg.min_alpha_len
            ]
            out[doc_id] = toks
        return out



# ==========================
#       tf idf
# ==========================
@dataclass
class TfidfArtifacts:
    ids: List[str]
    tfidf: csr_matrix
    vectorizer: TfidfVectorizer

class TfIdfService:
    NAMESPACE = "tfidf"

    def __init__(self, store: ArtifactStore, max_features: int = 5000, version: str = "1"):
        self.store = store
        self.max_features = max_features
        self.version = version

    def _name(self, dataset_name: str) -> str:
        return f"{dataset_name}__m{self.max_features}"

    def run(self, dataset_name: str, texts_by_id: Dict[str, List[str]], force: bool = False) -> TfidfArtifacts:
        name = self._name(dataset_name)
        if self.store.has(self.NAMESPACE, name, self.version) and not force:
            return self.store.load(self.NAMESPACE, name, self.version)


        ids = list(texts_by_id.keys())
        tokens = [" ".join(texts_by_id[_id]) for _id in ids]
        vec = TfidfVectorizer(max_features=self.max_features)
        vec.fit(tokens)


        batch = 10000
        mats = []
        for i in range(0, len(tokens), batch):
            mats.append(vec.transform(tokens[i:i+batch]))
        tfidf = vstack(mats)
        art = TfidfArtifacts(ids=ids, tfidf=tfidf, vectorizer=vec)
        self.store.save(art, self.NAMESPACE, name, self.version)
        return art


    def top_terms(self, art: TfidfArtifacts, doc_index: int, k: int = 5, thresh: float = 0.001) -> List[Tuple[str, float]]:
        row = art.tfidf[doc_index].toarray().ravel()
        feats = art.vectorizer.get_feature_names_out()
        idxs = np.argsort(row)[-k:][::-1]
        return [(feats[i], float(row[i])) for i in idxs if row[i] > thresh]


@dataclass
class Embeddings:
    ids: List[str]
    x: torch.Tensor # shape [N, D]


class EmbeddingService:
    NAMESPACE = "embeddings"


    def __init__(self, store: ArtifactStore, model_name: str = "all-MiniLM-L6-v2", version: str = "1"):
        self.store = store
        self.model_name = model_name
        self.version = version
        self._model = None


    def _name(self, dataset_name: str) -> str:
        return f"{dataset_name}__{self.model_name}"


    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model


    def run(self, dataset_name: str, ids: List[str], senders: List[str], subjects: List[str], texts: List[str], force: bool = False) -> Embeddings:
        name = self._name(dataset_name)
        if self.store.has(self.NAMESPACE, name, self.version) and not force:
            return self.store.load(self.NAMESPACE, name, self.version)


        out = []
        bs = 512
        for i in range(0, len(ids), bs):
            a = self.model.encode(senders[i:i+bs], convert_to_tensor=True)
            b = self.model.encode(subjects[i:i+bs], convert_to_tensor=True)
            c = self.model.encode(texts[i:i+bs], convert_to_tensor=True)
            out.append(0.3 * a + 0.3 * b + 0.4 * c)
        x = torch.cat(out, dim=0)
        emb = Embeddings(ids=ids, x=x)
        self.store.save(emb, self.NAMESPACE, name, self.version)
        return emb

@dataclass
class Topics:
    topic_to_ids: Dict[int, List[str]]
    info: Dict[int, Dict[str, int]] # {topic: {name, count}}


class TopicService:
    NAMESPACE = "topics"


    def __init__(self, store: ArtifactStore, version: str = "1"):
        self.store = store
        self.version = version
        self._model = BERTopic(verbose=False)


    def _name(self, dataset_name: str) -> str:
        return f"{dataset_name}"


    def run(self, dataset_name: str, ids: List[str], texts: List[str], embeddings: np.ndarray, force: bool = False) -> Topics:
        name = self._name(dataset_name)
        if self.store.has(self.NAMESPACE, name, self.version) and not force:
            return self.store.load(self.NAMESPACE, name, self.version)


        topics, _ = self._model.fit_transform(texts, embeddings=embeddings)
        topic_to_ids: Dict[int, List[str]] = {}
        for i, tid in enumerate(topics):
            topic_to_ids.setdefault(int(tid), []).append(ids[i])


        info_df = self._model.get_topic_info()
        info = {int(r["Topic"]): {"name": r["Name"], "count": int(r["Count"])} for _, r in info_df.iterrows()}
        obj = Topics(topic_to_ids=topic_to_ids, info=info)
        self.store.save(obj, self.NAMESPACE, name, self.version)
        return obj


@dataclass
class Summaries:
    ids: List[str]
    lines: List[str]


class SummarizeService:
    """Placeholder for your chosen summarization approach.
    Swap in TextRank / KeyBERT / LLM as needed, but keep the interface stable.
    """
    NAMESPACE = "summaries"


    def __init__(self, store, version: str = "1"):
        self.store = store
        self.version = version


    def _name(self, dataset_name: str) -> str:
        return f"{dataset_name}"


    def run(self, dataset_name: str, ids: List[str], texts: List[str], force: bool = False) -> Summaries:
        # TODO: implement; simple baseline below
        if not texts:
            return Summaries(ids=[], lines=[])
        lines = [t.strip().split("\n")[0][:280] for t in texts]
        return Summaries(ids=ids, lines=lines)

@dataclass
class NlpWorkspace:
    dataset_name: str
    dataset: Dataset
    paths: Paths = DEFAULT_PATHS


    def __post_init__(self):
        self.store = ArtifactStore(self.paths.artifacts)
        self.tokenizer = TokenizerService(self.store)
        self.tfidf = TfIdfService(self.store)
        self.embedder = EmbeddingService(self.store)
        self.topics = TopicService(self.store)
        self.summarizer = SummarizeService(self.store)
        self._token_view: Optional[Dict[str, List[str]]] = None
        self._tfidf_art = None
        self._emb = None
        self._topics = None


    # ---------- Tokens
    def get_tokens(self, kind: str = "nlp", force: bool = False) -> Dict[str, List[str]]:
        t = self.tokenizer.run(self.dataset_name, self.dataset, force=force)
        self._token_view = self.tokenizer.view(t, kind=kind)
        return self._token_view


    # ---------- TF-IDF
    def get_tfidf(self, force: bool = False):
        if self._token_view is None:
            self.get_tokens(kind="tfidf", force=force)
        art = self.tfidf.run(self.dataset_name, self._token_view, force=force)
        self._tfidf_art = art
        return art


    # ---------- Embeddings
    def get_embeddings(self, force: bool = False):
        ids = self.dataset.ids()
        senders = [d.sender or "" for d in self.dataset.docs]
        subjects = [d.subject or "" for d in self.dataset.docs]
        texts = [d.text for d in self.dataset.docs]
        emb = self.embedder.run(self.dataset_name, ids, senders, subjects, texts, force=force)
        self._emb = emb
        return emb


    # ---------- Topics
    def get_topics(self, force: bool = False):
        if self._emb is None:
            self.get_embeddings(force=force)
        ids = self._emb.ids
        texts = [d.text for d in self.dataset.docs]
        x = self._emb.x.detach().cpu().numpy()
        tp = self.topics.run(self.dataset_name, ids, texts, x, force=force)
        self._topics = tp
        return tp


    # ---------- Summaries
    def get_summaries(self, force: bool = False):
        ids = self.dataset.ids()
        texts = [d.text for d in self.dataset.docs]
        return self.summarizer.run(self.dataset_name, ids, texts, force=force)


    # ---------- Convenience helpers
    def tfidf_top_terms(self, doc_index: int, k: int = 5, thresh: float = 0.001):
        art = self.get_tfidf()
        return self.tfidf.top_terms(art, doc_index, k=k, thresh=thresh)
