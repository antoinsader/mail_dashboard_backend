import pickle
import urllib.parse
import requests
import os

from api_config import CLIENT_ID, CLIENT_SECRET, GOOGLE_LOGIN_BASE_URL, REDIRECT_URI, SCOPES, TOKEN_URL

def save_pkl(ar, fp):
    with open(fp, 'wb') as f:
        pickle.dump(ar, f)
def get_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

def remove_file(fp):
    if os.path.exists(fp):
        os.remove(fp)
        return True

def get_google_login_url():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent"
    }
    return GOOGLE_LOGIN_BASE_URL +"?" + urllib.parse.urlencode(params)


def exchange_code_for_token(code):
    data = {
        "code": code, 
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    r = requests.post(TOKEN_URL, data=data)
    r.raise_for_status()
    return r.json()
