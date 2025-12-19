from db import TABLES

class UserController:
    def __init__(self):
        self.table = TABLES["users"]

    def create(self, code, gmail):
        return self.table.insert(code=code, gmail=gmail)


    def get_by_code(self, code):
        results = self.table.select_where(code=code)
        return results[0] if results else None

    def get_by_id(self, user_id):
        results = self.table.select_where(id=user_id)
        return results[0] if results else None

    def get_by_gmail(self, gmail):
        users = self.table.select_where(gmail=gmail)
        return users[0] if users and len(users) > 0 else None 

