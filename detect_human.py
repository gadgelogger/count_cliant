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

# 枠線の描画
for index, row in human_results.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 現在のタイムスタンプを取得
current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 撮影した写真を保存する（枠線描画後のフレームを使用）
image_filename = "captured_image.jpg"
image_path = os.path.join(save_directory, image_filename)
cv2.imwrite(image_path, frame)
print(f"写真を {image_path} に保存しました")

# 撮影した画像をSupabaseのStorageにアップロード
with open(image_path, 'rb') as f:
    image_data = f.read()
    res = supabase.storage.from_('count').update(path=image_filename, file=image_data, file_options={"upsert": "true"})
    print(f"画像をSupabaseのStorageにアップロードしました: {res}")

# アップロードした画像のURLを取得
image_url = supabase.storage.from_('count').get_public_url(image_filename)

# 24時間分のデータを保存するために、最近の24レコードを取得
recent_records = supabase.table(table_name).select('*').order('time', desc=True).limit(24).execute()

if len(recent_records.data) == 24:
    # 最も古いレコードを削除
    oldest_record = recent_records.data[-1]
    supabase.table(table_name).delete().eq('time', oldest_record['time']).execute()

# 新しいレコードを挿入（画像のURLを含む）
data = {column_name: person_count, "time": current_timestamp, "image_url": image_url}
res = supabase.table(table_name).insert(data).execute()
print(f"Inserted data: {res}")

print(f"人数: {person_count}")
print(f"タイムスタンプ: {current_timestamp}")

# リソースを解放
cap.release()