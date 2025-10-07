from classes.MyNlp import NlpProcessor
from db import TABLES






class DatasetModel:
    def __init__(self, row=None ):
        if row is not None:
            self.id = row["id"]
            self.ds_name= row["ds_name"]
            self.pkl_path = row["pkl_path"]
            self.user_id = row["user_id"]
            self.base_dir = row["base_dir"]

    def create_manually(self, ds_name, pkl_path, user_id, base_dir):
        self.ds_name= ds_name
        self.pkl_path = pkl_path
        self.user_id = user_id
        self.base_dir = base_dir

    def get_dict(self):
        return {
            "id": self.id,
            "ds_name": self.ds_name
        }

class DatasetController:
    def __init__(self):
        self.table = TABLES["datasets"]

    def create(self, ds_name, user_id, pkl_path, base_dir):
        return self.table.insert(ds_name=ds_name, user_id=user_id, pkl_path=pkl_path, base_dir=base_dir)


    def get_list_by_user(self, user_id):
        rows = self.table.select_where(user_id=user_id)
        datasets = [
            DatasetModel(row) for row in rows
        ]
        return datasets


    def validate_ds_user(self, user_id, ds_id):
        global nlp_processors

        if ds_id not in nlp_processors.keys():
            raise RuntimeError(f"Ds id: {ds_id} is not valid ")

        result = self.table.select_where(user_id = user_id, id=ds_id)[0]

        if result is None:
            raise RuntimeError(f"Ds id: {ds_id} does not exist ")
        return result["id"], nlp_processors[ds_id]

    def get_ds_mails(self, user_id, ds_id):
        """
            Return list of mails 
            Each item is mail.to_dict with the index of the document inside the nlp_proc
            The mail is an instance of MyEmail which has to_dict
            to_dict is showing 
              mail_id (here the index in the docs + 1), message_id, subject, sender_name, sender_mail, content_clean, content_html
        """
        ds_id, nlp_proc = self.validate_ds_user(user_id, ds_id)
        return nlp_proc.mail_dicts

    def get_ds_stats(self, user_id, ds_id, most_common_count=50):
        """
            Returning most common list each item has token (the token) and count (how many times appeared in the ds)
            languages a dictionary having as keys "en", "it",... and as value showing how many documents
        """
        ds_id, nlp_proc = self.validate_ds_user(user_id, ds_id)
        most_commons = nlp_proc.get_most_common(most_common_count)
        langs = nlp_proc.get_languages()
        res = {
            "most_common":[{"token": mc[0], "count": mc[1]} for mc in most_commons],
            "languages": langs
        }
        return res


    def get_doc_tfidf_terms(self, user_id, ds_id, doc_idx):
        """
        tfidf_terms: returning dict for the specific doc_idx:
                [doc_id, html, doc_text, language, tokens, top_terms]
        top_terms having list and each item having: {term, tfidf_score}
        """
        ds_id, nlp_proc = self.validate_ds_user(user_id, ds_id)
        tfidf_terms = nlp_proc.get_doc_top_tfidf_terms(doc_idx)
        return tfidf_terms

    def get_ds_topics(self, user_id, ds_id):
        """
            topics_res having: {"topics" , "topic_mails"}
            "topics_res.topics": dictionary of the topics, each item has {name, count}
            "topics_res.topic_mails": dictionary having topic_id and the list of mail ids of this topic
        """
        ds_id, nlp_proc = self.validate_ds_user(user_id, ds_id)
        topics_res = nlp_proc.get_topics()
        return topics_res

nlp_processors = {}
def init_user_nlp_processors():
    global nlp_processors
    rows = TABLES["datasets"].select_all()

    datasets = [
        DatasetModel(row) for row in rows
    ]
    for ds in datasets:
        if ds.id not in nlp_processors:
            nlp_processors[ds.id] = NlpProcessor(ds)

