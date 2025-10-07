import re
from bs4 import BeautifulSoup
import regex
class TextProcess():
    def __init__(self, text):
        self.text= text


    def remove_url(self):
        self.text = re.sub(r"http\S+|www\.\S+", "", self.text)
        return self

    def remove_html_tags(self, separator=" "):
        self.text = BeautifulSoup(self.text, "html.parser").get_text(separator=" ")
        return self


    def keep_only_alphanumeric_and_whitespace(self):
        self.text = re.sub(r"[^a-z0-9\s]", " ", self.text)
        return self

    def collapse_multiple_white_spaces(self):
        self.text = re.sub(r"\s+", " ", self.text).strip()
        return self

    def clean_slash(self, replace=" "):
        """
            Replace every \n, \t any slash with one char
        """
        self.text = re.sub(r"\s+", replace, self.text )
        return self

    def clean_non_ascii(self, replace=""):
        self.text = re.sub(r"[^\x20-\x7E]", replace, self.text)
        return self

    def clean_emojis(self, replace=""):
        self.text = regex.sub(r"\p{So}", replace, self.text)
        return self

    def clean_repeated_strange_chars(self, replace=""):
        self.text = re.sub(r"(.)\1{4,}", replace, self.text)
        return self

    def strip(self):
        self.text = self.text.strip()
        return self


    def email_text_body_clean(self):
        self.clean_slash()
        self.remove_html_tags()
        self.clean_non_ascii()
        self.clean_emojis()
        self.collapse_multiple_white_spaces()
        self.remove_url()
        self.strip()
        return self
        