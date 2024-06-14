import torch
import cv2
import os
from supabase import create_client, Client
import time
from config import supabase_url, supabase_key
from datetime import datetime

# Supabaseの設定(APIキーを取得する)
url: str = supabase_url
key: str = supabase_key
supabase: Client = create_client(url, key)

# superbase上のテーブル名とカラム情報を指定する
table_name = "count"
column_name = "person"
column_type = "int8"

# 画像の保存先のディレクトリを指定
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

# 画像をモデルに渡して推論
results = model(frame)

# 結果の取得
results_df = results.pandas().xyxy[0]

# 人間のクラスIDは0（COCOデータセットのクラスID）
human_results = results_df[results_df['name'] == 'person']

# 人数をカウント
person_count = len(human_results)

# 現在のタイムスタンプを取得
current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 既存のレコードを確認する
query = supabase.table(table_name).select('*').limit(1).execute()
existing_data = query.data

if existing_data:
    # 既存のレコードがある場合は更新する
    res = supabase.table(table_name).update({"person": person_count, "time": current_timestamp}).eq('person', existing_data[0]['person']).execute()
    print(f"Updated data: {res}")
else:
    # 既存のレコードがない場合は挿入する
    data = {column_name: person_count, "time": current_timestamp}
    res = supabase.table(table_name).insert(data).execute()
    print(f"Inserted data: {res}")

# 枠線の描画
for index, row in human_results.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 撮影した写真を保存する
image_path = os.path.join(save_directory, "captured_image.jpg")
cv2.imwrite(image_path, frame)
print(f"写真を {image_path} に保存しました")
print(f"人数: {person_count}")
print(f"タイムスタンプ: {current_timestamp}")

# リソースを解放
cap.release()