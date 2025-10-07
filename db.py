
import sqlite3
import re


from api_config import LITE_DB_PATH

class TableBase:
    def __init__(self, table_name, columns, foreign_keys=[]):
        self.table_name = table_name
        self.columns  = columns
        self.foreign_keys= foreign_keys

    def create_table_str(self):
        cols_query = [col.create_column_str() for col in self.columns  ]
        fk_query = [fk.create_fk_str() for fk in self.foreign_keys]
        defs = ", ".join(cols_query + fk_query)
        q  = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({defs}); "
        return q

    def get_columns_dict(self):
        return [col.as_dict() for col in self.columns]

    def get_foreign_keys_dict(self):
        return [fk.as_dict() for fk in self.foreign_keys]

    def get_columns_names(self):
        return [col.column_name for col in self.columns]

    def insert(self, **kwargs):
        """
        Example: table.insert(code="abc", other_col=123)
        """
        col_names = []
        values = []
        for col in self.columns:
            if col.column_name in kwargs:
                col_names.append(col.column_name)
                values.append(kwargs[col.column_name])




        placeholders = ", ".join("?" for _ in col_names)
        columns_str = ", ".join(col_names)

        query = f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"

        conn = sqlite3.connect(LITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return last_id

    def select_all(self):
        conn = sqlite3.connect(LITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name}")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in rows]

    def select_where(self, **criteria):
        """
        Example: table.select_where(code="abc", id=5)
        """
        if not criteria:
            return self.select_all()

        cols = []
        values = []
        for k, v in criteria.items():
            assert k in self.get_columns_names()
            cols.append(f"{k}=?")
            values.append(v)

        where_clause = " AND ".join(cols)
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"

        conn = sqlite3.connect(LITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, values)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in rows]

    def update(self, values: dict, criteria: dict):
        """
        Example: table.update({"code": "newval"}, {"id": 5})
        """

        set_clause = ", ".join(f"{k}=?" for k in values)
        set_values = list(values.values())

        where_clause = " AND ".join(f"{k}=?" for k in criteria)
        where_values = list(criteria.values())

        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {where_clause}"

        conn = sqlite3.connect(LITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, set_values + where_values)
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        return affected

    def delete(self, **criteria):
        """
        Example: table.delete(id=5)
        """
        if not criteria:
            raise ValueError("Delete requires at least one criterion!")

        where_clause = " AND ".join(f"{k}=?" for k in criteria)
        values = list(criteria.values())

        query = f"DELETE FROM {self.table_name} WHERE {where_clause}"

        conn = sqlite3.connect(LITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        return affected


class ColumnBase:
    def __init__(self, column_name, column_type, primary=False, autoincrement=False, unique=False, allow_null=False ):
        self.column_name = column_name
        self.column_type = column_type
        self.primary = primary
        self.autoincrement = autoincrement
        self.unique = unique
        self.allow_null = allow_null

    def create_column_str(self):
        parts = [
            self.column_name,
            self.column_type,
            "PRIMARY KEY" if self.primary else '',
            "AUTOINCREMENT" if self.autoincrement else '',
            "UNIQUE" if self.unique else "",
            "NOT NULL" if not self.allow_null else ''
        ]
        return " ".join(filter(None, parts))

    def as_dict(self):
        return {
            "name": self.column_name,
            "type": self.column_type,
            "primary": self.primary,
            "autoincrement": self.autoincrement,
            "unique": self.unique,
            "not_null": not self.allow_null
        }

class ForeignKeyBase:
    def __init__(self, from_column, to_table, to_column):
        self.from_column=from_column
        self.to_table = to_table
        self.to_column = to_column

    def create_fk_str(self):
        return f" FOREIGN KEY ({self.from_column}) REFERENCES {self.to_table}({self.to_column}) "

    def as_dict(self):
        return {
            "from": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column
        }


TABLES = {
    "users":    
 TableBase(
        "users",
        columns=[
            ColumnBase("id", "integer", primary=True, autoincrement=True),
            ColumnBase("code", "TEXT", unique=True),
            ColumnBase("gmail", "TEXT", unique=True),
        ]
    ),
 "datasets":
    TableBase(
        "datasets",
        columns=[
            ColumnBase("id", "integer", primary=True, autoincrement=True),
            ColumnBase("ds_name", "TEXT", unique=True),
            ColumnBase("pkl_path", "TEXT", unique=True),
            ColumnBase("base_dir", "TEXT", unique=True),
            ColumnBase("user_id", "INTEGER"),
        ],
        foreign_keys=[
            ForeignKeyBase("user_id", "users", "id")
        ]
    )
}





def init_db():
    conn = sqlite3.connect(LITE_DB_PATH)
    cursor = conn.cursor()

    for table in TABLES.values():
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name=?;", (table.table_name, ))
        exists = cursor.fetchone()
        if not exists:
            print(f"Creating table {table.table_name}")
            cursor.execute(table.create_table_str())
            continue

        # cursor.execute(f"PRAGMA table_info({table.table_name})")
        # db_columns = cursor.fetchall()
        # db_cols_dict = [
        #     {
        #         "name": col[1],
        #         "type": col[2].upper(),
        #         "not_null": bool(col[3]),
        #         "primary": bool(col[5])
        #     } for col in db_columns
        # ]
        # expected_cols = table.get_columns_dict()

        # if len(db_cols_dict) != len(expected_cols):
        #     raise RuntimeError(f"Table {table.table_name} column count mismatch")

        # for exp_col in expected_cols:
        #     db_col = next ((c for c in  db_cols_dict if c['name'].lower() == exp_col['name'].lower()), None)
        #     if not db_col:
        #         raise RuntimeError(f"Table: {table.table_name}, column: {exp_col['name']} was not found, make sure to run migrate.py")

        #     for k in ["name", "type", "primary"]:
        #         if str(db_col[k]).lower() != str(exp_col[k]).lower():
        #             raise RuntimeError(f"Table: {table.table_name}, column: {exp_col['name']} mismatch with {k}; expected: {exp_col[k]}, got: {db_col[k]}  ! make sure to run migrate.py")



        # cursor.execute(f"PRAGMA foreign_key_list({table.table_name})")
        # db_fks = cursor.fetchall()
        # db_fks_dict = [
        #     {
        #         "from": fk[3],
        #         "to_table": fk[2],
        #         "to_column": fk[4],
        #     } for fk in db_fks
        # ]
        # expected_fks = table.get_foreign_keys_dict()

        # if len(db_fks_dict) != len(expected_fks):
        #     raise RuntimeError(f"Table {table.table_name} fks count mismatch")


        # for exp_fk in expected_fks:
            
        #     db_fk = next ((c for c in  db_fks_dict if c['from'].lower() == exp_fk['from'].lower() and   db_fks_dict if c['to_table'].lower() == exp_fk['to_table'].lower()  and db_fks_dict if c['to_column'].lower() == exp_fk['to_column'].lower() ), None)
        #     if not db_fk:
        #         raise RuntimeError(f"Table: {exp_fk} fk was not found in db")





    conn.commit()
    conn.close()
