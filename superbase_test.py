import os
from supabase import create_client, Client
# Supabaseの設定
url: str = "https://ycnpvgqdogzhjhiilvbs.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InljbnB2Z3Fkb2d6aGpoaWlsdmJzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTU3MDkxMDksImV4cCI6MjAzMTI4NTEwOX0.w32xNB9Wv81yD5X3mvvSUuKb2ydvMrqvDXFg7DWa0L0"
supabase: Client = create_client(url, key)

# テーブル名とカラム情報
table_name = "count"
column_name = "person"
column_type = "int8"

# データを書き込む関数
def insert_data(value: int):
    data = {column_name: value}
    res = supabase.table(table_name).insert(data).execute()
    print(f"Inserted data: {res}")

# 使用例
insert_data(10)
insert_data(20)
insert_data(30)