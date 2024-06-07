import torch
import cv2
import os
from supabase import create_client, Client
import time
# Supabaseの設定
url: str = "https://ycnpvgqdogzhjhiilvbs.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InljbnB2Z3Fkb2d6aGpoaWlsdmJzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTU3MDkxMDksImV4cCI6MjAzMTI4NTEwOX0.w32xNB9Wv81yD5X3mvvSUuKb2ydvMrqvDXFg7DWa0L0"
supabase: Client = create_client(url, key)

# テーブル名とカラム情報
table_name = "count"
column_name = "person"
column_type = "int8"

# 保存先のディレクトリ
save_directory = "captured_images"

# 保存先のディレクトリが存在しない場合は作成する
os.makedirs(save_directory, exist_ok=True)

# YOLOv5モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイス
time.sleep(2)    
if not cap.isOpened():
    print("Webカメラを開くことができませんでした")
    exit()

# 写真を撮影する
ret, frame = cap.read()
if not ret:
    print("フレームを取得できませんでした")
    exit()

# 撮影した写真を保存する
image_path = os.path.join(save_directory, "captured_image.jpg")
cv2.imwrite(image_path, frame)
print(f"写真を {image_path} に保存しました")

# 保存した写真を読み込む
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画像をモデルに渡して推論
results = model(img_rgb)

# 結果の取得
results_df = results.pandas().xyxy[0]

# 人間のクラスIDは0（COCOデータセットのクラスID）
human_results = results_df[results_df['name'] == 'person']

# 人数をカウント
person_count = len(human_results)

# 既存のレコードを確認する
query = supabase.table(table_name).select('*').eq(column_name, person_count).execute()
existing_data = query.data

if existing_data:
    # 既存のレコードがある場合は更新する
    res = supabase.table(table_name).update({"person": person_count}).eq(column_name, person_count).execute()
    print(f"Updated data: {res}")
else:
    # 既存のレコードがない場合は挿入する
    data = {column_name: person_count}
    res = supabase.table(table_name).insert(data).execute()
    print(f"Inserted data: {res}")

# リソースを解放
cap.release()