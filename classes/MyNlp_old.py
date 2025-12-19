
import itertools
import os
from collections import Counter, defaultdict
from re import S
from tabnanny import verbose


from tqdm import tqdm
import numpy as np
import torch
import faiss
import langid
import spacy
from scipy.sparse import vstack
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic

from utils import get_pkl, save_pkl


from dataclasses import dataclass

@dataclass
class TokenKindConfig:
    remove_punc: bool = True
    remove_stop: bool = True
    min_alpha_len: int = 3
    use_lemma: bool = True

TOKEN_KINDS = {
    "nlp": TokenKindConfig(),
    "tfidf": TokenKindConfig(remove_punc=False, remove_stop=False, min_alpha_len=1, use_lemma=False),
    "summarization": TokenKindConfig(remove_stop=False),
}



class NlpDoc():
    def __init__(self, id, text, html, sender, subject):
        self.id = id
        self.text = text
        self.html = html
        self.sender = sender 
        self.subject = subject
        self.language, _  = langid.classify(text)

        self._tokens_raw = None
        self._tokens_cache = {}

    def set_tokens_raw(self,tokens_raw):
        self._tokens_raw = tokens_raw


    def get_tokens(self, kind='nlp'):
        if kind in self._tokens_cache:
            return self._tokens_cache[kind]
        if self._tokens_raw is None:
            return []
        if kind not in TOKEN_KINDS:
            raise ValueError("token kind does not exists")


        cfg = TOKEN_KINDS.get(kind)
        tokens = [
            t[0] if cfg.use_lemma else t[1]
            for t in self._tokens_raw
            if (not cfg.remove_stop or t[0] not in spacy.lang.en.stop_words.STOP_WORDS)
            and (not cfg.remove_punc or t[1].isalpha())
            and len(t[1]) >= cfg.min_alpha_len
        ]

        self._tokens_cache[kind] = tokens
        return tokens

class NlpDocs():
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def get_all(self):
        return self.docs

    def get_docs_language(self, language):
        return [doc  for  doc in self.docs if doc.language==language]




class TokenizerManager:
    def __init__(self, language_nlps, tokens_path, tokens_ids_path):
        self.tokens_path = tokens_path
        self.tokens_ids_path = tokens_ids_path
        self.tokens= None
        self.tokens_ids = None
        self.languages_nlp = language_nlps

    def tokenize_docs(self, my_docs, remove_punc=True, remove_stop=True, force=False):
        if os.path.exists(self.tokens_path) and not force:
            self.tokens = np.load(self.tokens_path, allow_pickle=True)
            self.tokens_ids = get_pkl(self.tokens_ids_path)

            c = 0
            id_to_tokens = dict(zip(self.tokens_ids, self.tokens))
            all = my_docs.get_all()
            for doc in all:
                if doc.id in id_to_tokens:
                    doc.set_tokens(id_to_tokens[doc.id])
                    c+=1
            print(f"Tokens loaded from {self.tokens_path}, {c}/{len(my_docs)} docs has been tokenized")
            return self.tokens
        for lang, nlp in self.languages_nlp.items():
            lang_docs = my_docs.get_docs_language(lang)
            texts = [doc.text for doc in lang_docs]
            print(f"Tokenizing {len(texts)} for lang {lang}")
            docs_nlp = nlp.pipe(texts, batch_size=100)
            print(f"We have {len(texts)} docs_nlp")
            for doc_obj, spacy_doc in tqdm(zip(lang_docs, docs_nlp), total=len(lang_docs)):
                tokens = [
                    (token.lemma_.lower(), token.text.lower())
                    for token in spacy_doc
        
                ]
                doc_obj.set_tokens(tokens)
        updated_docs = my_docs.get_all()

        self.tokens = np.array([doc.tokens for doc in updated_docs], dtype=object)
        self.tokens_ids = [doc.id for doc in updated_docs]
        print(f"Tokens are here with len : {len(self.tokens)}")
        np.save(self.tokens_path, self.tokens)
        save_pkl(self.tokens_ids, self.tokens_ids_path)
        print(f"Tokens saved at {self.tokens_path}")
        return self.tokens
class TfIdfManager:
    def __init__(self, tfidf_path, vectorizer_path, ids_path):
        self.tfidf_path = tfidf_path
        self.vectorizer_path = vectorizer_path
        self.ids_path = ids_path
        
        self.vectorizer = None
        self.tfidf = None
        self.tfidf_ids = None

    def build_tfidf(self, docs, force=False):
        if os.path.exists(self.tfidf_path) and not force:
            self.tfidf = get_pkl(self.tfidf_path)
            self.vectorizer = get_pkl(self.vectorizer_path)
            self.tfidf_ids = get_pkl(self.ids_path)
            return self.tfidf, self.vectorizer


        print(f"docs: {docs}")
        english_docs = docs.get_docs_language("en")
        print(f"docs len: {len(english_docs)}")
        tokens = [" ".join(doc.tokens) for doc in english_docs]
        self.tfidf_ids = [doc.id for doc in english_docs]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.vectorizer.fit(tokens)


        tfidf_batches = []
        batch_size = 10000
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens) )
            batch = tokens[start : end]
            tfidf_batches.append(self.vectorizer.transform(batch))
        self.tfidf = vstack(tfidf_batches)
        save_pkl(self.tfidf, self.tfidf_path)
        save_pkl(self.vectorizer, self.vectorizer_path)
        save_pkl(self.tfidf_ids, self.ids_path)
        return self.tfidf, self.vectorizer


class EmbedingManager:
    def __init__(self, embs_path, ids_path):
        self.embs_path = embs_path
        self.ids_path = ids_path
        self.model = None
        self.embs = None
        self.texts = None
        self.embs_ids = None


    def load_dense_model(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # all-mpnet-base-v2 is more accurate but slower
        # "paraphrase-multilingual-MiniLM-L12-v2" → supports 50+ languages, still compact & fast.

    def embed_docs(self, docs, force=False):
        if self.model is None:
            self.load_dense_model()

        if os.path.exists(self.embs_path) and not force:
            self.embs = get_pkl(self.embs_path)
            self.embs_ids = get_pkl(self.ids_path)
            english_docs = docs.get_docs_language("en")
            doc_map = {doc.id: doc.text for doc in english_docs}
            self.texts = [doc_map[_id] for _id in self.embs_ids if _id in doc_map]
            return self.embs


        english_docs = docs.get_docs_language("en")
        self.embs_ids = [doc.id for doc in english_docs]
        senders = [doc.sender for doc in english_docs]
        subjects = [doc.subject for doc in english_docs]
        self.texts = [doc.text for doc in english_docs]


        all_embs = []
        batch_size = 512
        for start in range(0, len(english_docs) , batch_size):
            end = min(start + batch_size, len(english_docs))
            senders_emb = self.model.encode(senders[start:end], convert_to_tensor=True)
            subjects_emb = self.model.encode(subjects[start:end], convert_to_tensor=True)
            text_emb = self.model.encode(self.texts[start:end], convert_to_tensor=True)
            combined = 0.3 * senders_emb + 0.3 * subjects_emb + 0.4 * text_emb
            all_embs.append(combined)
        self.embs = torch.cat(all_embs, dim=0)
        save_pkl(self.embs, self.embs_path)
        save_pkl(self.embs_ids, self.ids_path)
        return self.embs

class TopicModelManager():
    def __init__(self, topics_path):
        self.topics_path = topics_path
        self.topic_model = BERTopic(verbose=True)
        self.topics = None

    def generate_topics(self, emb_factory, force=False):
        if os.path.exists(self.topics_path) and not force:
            self.topics = get_pkl(self.topics_path)
            return self.topics

        texts = emb_factory.texts
        ids = emb_factory.embs_ids

        topics, _ = self.topic_model.fit_transform(texts, embeddings=emb_factory.embs.cpu().numpy())
        topics_mails = defaultdict(list)
        for i, email_id in enumerate(ids):
            topics_mails[int(topics[i])].append(email_id)


        topics_info = self.topic_model.get_topic_info()
        topic_map = {
            row["Topic"]: {"name": row["Name"], "count": int(row["Count"])}
            for _, row in topics_info.iterrows()
        }

        topics_res = {"topics": topic_map, "topic_mails": topics_mails}
        save_pkl(topics_res, self.topics_path)
        self.topics = topics_res
        return topics_res


class NlpProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.base_dir = dataset.base_dir
        os.makedirs(self.base_dir, exist_ok=True)


        # === PATHS ===

        self.tokens_path  =  os.path.join(self.base_dir, "_tokens.npy")
        self.tokens_ids_path = os.path.join(self.base_dir, "_tokens_ids.pkl")
        self.tfidf_path = os.path.join(self.base_dir, "_tfidf.pkl") 
        self.tfidf_vectorizer_path = os.path.join(self.base_dir, "_tfidf_vectorizer.pkl") 
        self.tfidf_ids_path = os.path.join(self.base_dir, "_tfidf_ids.pkl") 
        self.embs_path = os.path.join(self.base_dir, "_embs.pkl") 
        self.embs_ids_path = os.path.join(self.base_dir, "_embs_ids.pkl") 
        self.topics_path = os.path.join(self.base_dir, "_topics.pkl") 


        self.languages_nlps = {
            "en": spacy.load("en_core_web_sm", disable=["ner", "parser"]),
            "it": spacy.load("it_core_news_sm", disable=["ner", "parser"])
        }

        # === DOCS ====
        ds = get_pkl(dataset.pkl_path)
        self.docs = NlpDocs([
            NlpDoc(i + 1, mail.content_clean, mail.content_html, mail.sender_name, mail.subject)
            for i, mail in enumerate(ds)
            if mail.content_clean and len(mail.content_clean) > 3
        ])
        self.mail_dicts = [
            mail.to_dict(i + 1)
            for i, mail in enumerate(ds)
            if mail.content_clean and len(mail.content_clean) > 3
        ]



        # === FACTORIES ===
        self.tokenizer_factory = TokenizerManager(self.languages_nlps, self.tokens_path, self.tokens_ids_path )
        self.tfidf_factory = TfIdfManager(self.tfidf_path, self.tfidf_vectorizer_path, self.tfidf_ids_path)
        self.emb_factory = EmbedingManager(self.embs_path, self.embs_ids_path)
        self.topics_factory = TopicModelManager(self.topics_path)



    def get_tokens(self,kind="nlp", force=False):
        if self.tokenizer_factory.tokens is not None and not force:
            return self.tokenizer_factory.tokens
        return self.tokenizer_factory.tokenize_docs(self.docs, force=force)

    def get_tfidf(self, force=False):
        if self.tfidf_factory.tfidf is not None and not force:
            return self.tfidf_factory.tfidf, self.tfidf_factory.vectorizer
        self.get_tokens()
        return self.tfidf_factory.build_tfidf(self.docs,force=force)

    def get_embedings(self, force=False):
        if self.emb_factory.embs is not None and not force:
            return self.emb_factory.embs
        self.get_tokens()
        return self.emb_factory.embed_docs(self.docs, force=force)

    def get_topics(self, force=False):
        if self.topics_factory.topics is not None and not force:
            return self.topics_factory.topics
        self.get_embedings()
        return self.topics_factory.generate_topics(self.emb_factory, force=force)



    def get_most_common(self, count=50):
        tokens = self.get_tokens()
        print(f"Tokens count: {len(tokens)}")
        all_toks = list(itertools.chain.from_iterable(tokens))
        word_counts =Counter(all_toks)
        most_common = word_counts.most_common(count)
        return most_common

    def get_languages(self):
        langs = {}
        
        for doc in self.docs.get_all():
            langs[doc.language] = langs.get(doc.language, 0 ) + 1
        return langs

    def get_doc_top_tfidf_terms(self, doc_idx, top_num=5, score_threshold=.001):
        self.get_tfidf()
        vectorizer, tfidf ,tfidf_ids = self.tfidf_factory.vectorizer, self.tfidf_factory.tfidf, self.tfidf_factory.tfidf_ids

        if  vectorizer is None or tfidf is None or tfidf_ids is None:
            raise RuntimeError("Wrong")

        if doc_idx > len(tfidf_ids) or doc_idx < 0:
            raise RuntimeError("Doc idx is invalid")

        doc_id = tfidf_ids[doc_idx]
        doc_obj = next(doc for doc in self.docs.get_docs_language("en") if doc.id == doc_id)
        if doc_obj is None:
            raise RuntimeError("Doc is invalid")


        feature_names = vectorizer.get_feature_names_out()
        row = tfidf[doc_idx].toarray().flatten()

        top_indices = np.argsort(row)[-top_num:][::-1]
        top_terms = [{"term": feature_names[i], "tfidf_score": float(row[i])} for i in top_indices if row[i] > score_threshold]


        return {
            "doc_id": doc_id,
            "html": doc_obj.html,
            "doc_text": doc_obj.text,
            "language": doc_obj.language,
            "tokens": doc_obj.tokens,
            "top_terms": top_terms
        }
