import torch
import cv2
import numpy as np
import os
from supabase import create_client, Client
import time
from config import supabase_url, supabase_key
from datetime import datetime
from picamera2 import Picamera2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Supabaseの設定
url: str = supabase_url
key: str = supabase_key
supabase: Client = create_client(url, key)

# Supabase上のテーブル名とカラム情報を指定する
table_name = "count"
column_name = "person"
column_type = "int8"

# 画像の保存先のディレクトリを指定
save_directory = "captured_images"

# 保存先のディレクトリが存在しない場合は作成する
os.makedirs(save_directory, exist_ok=True)

# YOLO11モデルのロードをSAHIを使ったスライス推論に変更
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="yolo11x.pt",
    confidence_threshold=0.3,
    device="cpu"  # または 'cuda:0'
)

# Picamera2の設定
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

# カメラの安定化を待つ
time.sleep(2)

# 写真を撮影する
frame = picam2.capture_array()
# 画像を上下反転
frame = cv2.flip(frame, 0)

# SAHIを使ったスライス推論
result = get_sliced_prediction(
    frame,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 結果の取得
human_count = len(result.object_prediction_list)

# 枠線の描画
for object_prediction in result.object_prediction_list:
    box = object_prediction.bbox
    x1, y1, x2, y2 = map(int, [box.minx, box.miny, box.maxx, box.maxy])
    confidence = object_prediction.score
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 現在のタイムスタンプを取得
current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 撮影した写真を保存する（枠線描画後のフレームを使用）
image_filename = "captured_image.jpg"
image_path = os.path.join(save_directory, image_filename)
cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print(f"写真を {image_path} に保存しました")

# 撮影した画像をSupabaseのStorageにアップロード
with open(image_path, 'rb') as f:
    image_data = f.read()
    res = supabase.storage.from_('count').upload(path=image_filename, file=image_data, file_options={"upsert": "true"})
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
data = {column_name: human_count, "time": current_timestamp, "image_url": image_url}
res = supabase.table(table_name).insert(data).execute()
print(f"Inserted data: {res}")

print(f"人数: {human_count}")
print(f"タイムスタンプ: {current_timestamp}")

# カメラを停止
picam2.stop()

# デバッグ用：保存した画像を表示
debug_image = cv2.imread(image_path)
cv2.imshow("Captured Image", debug_image)
cv2.waitKey(0)
cv2.destroyAllWindows()