import email
from email.utils import parseaddr, parsedate_to_datetime
from email.header import decode_header, make_header

from bs4 import BeautifulSoup
import re
import regex

from classes.TextProcess import TextProcess
import langid



def _decode_header( value):
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value

def _decode_sender(sender):
    if not sender:
        return "", ""
    name, addr = parseaddr(sender)
    clean_name = _decode_header(name).strip()
    return clean_name, addr.strip()

def parse_header_meta(raw_mail):
    try:
        msg = email.message_from_bytes(raw_mail)
        name, email_addr = _decode_sender(msg.get("From"))
        subject = _decode_header(msg.get("Subject"))

        date_header = msg.get("Date")
        dt = None
        if date_header:
            try:
                dt = parsedate_to_datetime(date_header).date()
            except Exception:
                pass
        return (name, email_addr, subject, dt)
    except Exception:
        return None


class MyEmail():
    def __init__(self,raw_mail, mail_id):
        assert raw_mail is not None
        r = email.message_from_bytes(raw_mail)
        self.raw_message= r
        self.mail_id = mail_id

        self.message_id = r.get("Message-ID")
        self.subject = r.get("Subject")
        self.sender_name, self.sender_mail = _decode_sender(r.get("From"))
        self.to = r.get("To")
        # self.content = self._extract_content(r)
        self.contains_attachement = False
        self.content_text, self.content_clean, self.content_html = self._parse_mail(r)
        self.language, _  = langid.classify(self.content_clean)

    def _parse_mail(self, msg):
        # content_clean, content_html, content_text_all = None, None

        plain_text_parts = []
        clean_text_parts = []
        html_parts = []
        self.contains_attachement = False
        for part in msg.walk():
            content_type = part.get_content_type()
            filename = part.get_filename()

            if filename:
                self.contains_attachement = True
                continue
            payload = part.get_payload(decode=True)

            if not payload:
                continue
    
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = payload.decode(charset, errors="ignore")
            except Exception:
                payload = payload.decode("utf-8", errors="ignore")


            if content_type == "text/plain":
                text_body = payload.strip()
                plain_text_parts.append(text_body)

                txt_pr = TextProcess(text_body)
                txt_pr = txt_pr.email_text_body_clean()

                text_body = txt_pr.text
                clean_text_parts.append(text_body)


            elif content_type == "text/html":
                soup = BeautifulSoup(payload, "html.parser")
                if soup.body:
                    body_html = "".join(str(c) for c in soup.body.contents)  # keep inner HTML
                else:
                    body_html = payload  # fallback if no <body>
                html_parts.append(body_html)


        content_text = " ".join(plain_text_parts) if plain_text_parts else ""
        content_clean = " ".join(clean_text_parts) if clean_text_parts else ""
        content_html = " ".join(html_parts) if html_parts else ""
        return content_text, content_clean, content_html


    def preprocess_content_clean(self, lowering=True):
        """
            Make sure no html tags 
            Lowering
            remove url
            Remove urls, remove html tags, keep alphanum only, remove extra spaces
        """
        text = self.content_clean
        if not text:
            return ""
        
        if lowering:
            text = text.lower()

        txt_pr = TextProcess(text)
        txt_pr = txt_pr.email_text_body_clean()

        text = txt_pr.text
        self.content_clean = text
        return text


    def to_dict(self, mail_id = None):
        return {
            "mail_id": mail_id,
            "message_id": self.message_id,
            "subject": self.subject,
            "sender_name": self.sender_name,
            "sender_mail": self.sender_mail,
            "content_clean": self.content_clean,
            "content_html": self.content_html,
        }