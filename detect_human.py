import cv2
import numpy as np
import os
from supabase import create_client, Client
import time
from config import supabase_url, supabase_key
from datetime import datetime
from picamera2 import Picamera2
import json
from ultralytics import YOLO

# Supabaseの設定
url: str = supabase_url
key: str = supabase_key
supabase: Client = create_client(url, key)

# Supabaseテーブルの設定
table_name = "count"
column_name = "person"
column_type = "int8"

# 画像保存用ディレクトリの作成
save_directory = "captured_images"
os.makedirs(save_directory, exist_ok=True)

# キャリブレーションデータの読み込み
with open('/home/gadgelogger/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

# 元の解像度
original_dim = tuple(calibration_data["DIM"])  # 640x480
K = np.array(calibration_data["K"])
D = np.array(calibration_data["D"])

# 新しい解像度
new_dim = (2592, 1944)

# カメラ行列をスケーリング
scale_x = new_dim[0] / original_dim[0]
scale_y = new_dim[1] / original_dim[1]
K_scaled = K.copy()
K_scaled[0, 0] *= scale_x  # fx
K_scaled[1, 1] *= scale_y  # fy
K_scaled[0, 2] *= scale_x  # cx
K_scaled[1, 2] *= scale_y  # cy

# 魚眼レンズ補正のマップを作成
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_scaled, D, np.eye(3), K_scaled, new_dim, cv2.CV_16SC2
)

# YOLOモデルのロード
model = YOLO("yolo11x.pt")

# Picamera2の初期化
picam2 = Picamera2()

# 解像度を2592x1944に設定し、フォーマットをRGB888に指定
config = picam2.create_still_configuration(main={"size": new_dim, "format": 'RGB888'})
picam2.configure(config)

# カメラスタート
picam2.start()

# カメラの準備完了を待機
time.sleep(2)

try:
    # フレームを取得
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # 魚眼補正を適用
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # YOLOを用いて人検出を実行
    results = model(undistorted_frame, classes=[0], conf=0.3)
    human_count = len(results[0].boxes)  # 検出された人数をカウント

    # 検出結果を描画
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(undistorted_frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 現在のタイムスタンプを取得
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 画像を上書き保存 (ファイル名を固定)
    image_filename = "captured_image.jpg"
    image_path = os.path.join(save_directory, image_filename)
    cv2.imwrite(image_path, undistorted_frame)  # 色空間変換なしで保存
    print(f"画像が上書き保存されました: {image_path}")

    # Supabase Storageに画像をアップロード
    with open(image_path, 'rb') as f:
        image_data = f.read()
        res = supabase.storage.from_('count').upload(path=image_filename, file=image_data, file_options={"upsert": "true"})
        print(f"Supabase Storageに画像がアップロードされました: {res}")

    # 画像URLを取得
    image_url = supabase.storage.from_('count').get_public_url(image_filename)

    # 24時間以上の古いデータを削除
    recent_records = supabase.table(table_name).select('*').order('time', desc=True).limit(24).execute()

    if len(recent_records.data) == 24:
        # 最も古いレコードを削除
        oldest_record = recent_records.data[-1]
        supabase.table(table_name).delete().eq('time', oldest_record['time']).execute()

    # Supabaseテーブルにデータを挿入
    data = {column_name: human_count, "time": current_timestamp, "image_url": image_url}
    res = supabase.table(table_name).insert(data).execute()
    print(f"挿入されたデータ: {res}")

    print(f"検出された人数: {human_count}")
    print(f"タイムスタンプ: {current_timestamp}")

except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()