
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
from api_config import IS_PROD, MAX_WORKERS
from classes.MyEmail import MyEmail, parse_header_meta
 
from imapclient import IMAPClient
from datetime import datetime



class ImapMail:
    def __init__(self, email, access_token):
        self.email = email
        self.access_token = access_token
        self.imap_client= None
        self.max_workers = MAX_WORKERS
        self.connect()


    def connect(self):
        try:
            assert IS_PROD == False, f"You should upload your certificate : IS_PROD: {IS_PROD}"
            context = ssl.create_default_context()
            context.check_hostname  = False
            context.verify_mode = ssl.CERT_NONE

            print(f"imap connecting...")
            self.imap_client = IMAPClient("imap.gmail.com", ssl=True, ssl_context=context)
            self.imap_client.oauth2_login(self.email, self.access_token)
            print(f"Authenticated")
            self.imap_client.select_folder("INBOX")
            print(f"Inbox selected")
        except Exception as e:
            print(f"imap CONNECTION error: {e}")
            raise RuntimeError(f"Imap connect error: {e}")

    def _fetch_emails(self, uids, batch_size=50):
        """
            Returns list of instances from the class MyEmail
        """
        all_emails = []
        for start in range(0, len(uids), batch_size):
            batch_uids = uids[start:start+batch_size]
            data = self.imap_client.fetch(batch_uids, ['RFC822'])
            # Parallel decoding
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(MyEmail, d[b'RFC822'], uid) for uid, d in data.items()]
                for future in as_completed(futures):
                    all_emails.append(future.result())
        return all_emails

    def get_inbox_meta(self):
        """
        Fetch all senders, subjects, and dates from the inbox for autocomplete and datepickers.
        Returns:
        {
            senders_emails: [str,..],
            senders_names: [str,..],
            subjects: [str, ...],
            min_date: str,
            max_date: str
        }
        """
        try:
            uids = self.imap_client.search(['ALL'])
            if not uids:
                return {"senders": [], "subjects": [], "min_date": None, "max_date": None}

            senders_names_set = set()
            senders_emails_set = set()
            subjects_set = set()
            dates_list = []

            batch_size = 200
            for start in range(0, len(uids), batch_size):
                batch_uids = uids[start:start+batch_size]
                data = self.imap_client.fetch(batch_uids, ['RFC822'])

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(parse_header_meta,  d[b'RFC822'])
                        for _, d in data.items()
                    ]
                    for future in as_completed(futures):
                        meta = future.result()
                        if meta is None:
                            continue
                        sender_name, sender_email, subject, date = meta

                        if sender_name:
                            senders_names_set.add(sender_name)
                        if sender_email:
                            senders_emails_set.add(sender_email)
                        if subject:
                            subjects_set.add(subject)
                        if date:
                            dates_list.append(date)

            senders_names = sorted(list(senders_names_set))
            senders_emails = sorted(list(senders_emails_set))
            subjects = sorted(list(subjects_set))
            min_date = str(min(dates_list)) if dates_list else None
            max_date = str(max(dates_list)) if dates_list else None

            return {
                "senders_names":senders_names,
                "senders_emails": senders_emails,
                "subjects": subjects,
                "min_date": min_date,
                "max_date": max_date
            }

        except Exception as e:
            raise RuntimeError(f"Error fetching inbox metadata: {e}")
    

    def get_mails_criteria(self, sender=None, subject=None, date_from=None, date_to=None, only_unseen=None ):
        search_params = []
        if sender is not None:
            search_params.extend(['FROM', sender])
        if subject is not None:
            search_params.extend(['SUBJECT', subject])
        if date_from is not None:
            dt = datetime.strptime(date_from, "%Y-%m-%d")
            search_params.extend(['SINCE', dt])
        if date_to is not None:
            dt = datetime.strptime(date_to, "%Y-%m-%d")
            search_params.extend(['BEFORE', dt])
        if only_unseen:
            search_params.append('UNSEEN')
        if not search_params:
            search_params.append('ALL')

        print(f"Seach params:{search_params}")

        uids = self.imap_client.search(search_params)

        return self._fetch_emails(uids, batch_size=100)