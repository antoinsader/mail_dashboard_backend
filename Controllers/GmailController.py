
import time
import requests
from api_config import DS_ALL_DIR, GOOGLE_USERINFO_BASE_URL, LITE_DB_PATH
from classes.IMapMail import ImapMail
import os

class GmailController:
    def __init__(self):

        self.imap = None
        self.user = None
        self.email = None

    def connect(self,access_token):
        self.access_token = access_token
        self.headers={"Authorization": f"Bearer {self.access_token}"}
        try:
            r = requests.get(GOOGLE_USERINFO_BASE_URL, headers=self.headers)
            r.raise_for_status()
            self.email = r.json()["email"]
            self.user = r.json()
            self.imap= ImapMail(self.email, self.access_token)
        except requests.HTTPError as e:
            raise RuntimeError(f"Failed to fetch profile: {e}") from e
        return True

    @staticmethod
    def get_user_info(access_token):
        if access_token is None:
            return {}

        headers={"Authorization": f"Bearer {access_token}"}
        try:
            r= requests.get(GOOGLE_USERINFO_BASE_URL, headers=headers)
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"Failed to fetch profile: {e}") from e
        return r.json()

    def get_inbox_meta(self, access_token):
        self.connect(access_token)
        if self.imap is None:
            raise RuntimeError("Could not connect to imap")

        meta = self.imap.get_inbox_meta()
        return meta
    def get_mails_based_on_criteria(self,  access_token,  criteria):
        """
            Construct criteria body from the parameters 
            and fetch emails based on this criteria
            return list of MyEmail dict instances 
            MyEmail dict has: message_id, subject, sender_name, sender_mail, content_clean, content_html

            if save_name is set, the dataset will be saved
        """
        self.connect(access_token)
        if self.imap is None:
            raise RuntimeError("Could not connect to imap")


        sender, subject, date_from, date_to, only_unseen = criteria.get("sender"), criteria.get("subject"), criteria.get("date_from"), criteria.get("date_to"), criteria.get("only_unseen")

        mails = self.imap.get_mails_criteria(sender, subject, date_from, date_to, only_unseen)


        return mails
