import torch
import cv2

# YOLOv5モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイス

if not cap.isOpened():
    print("Webカメラを開くことができませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # フレームをRGBに変換
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 画像をモデルに渡して推論
    results = model(img_rgb)

    # 結果の取得
    results_df = results.pandas().xyxy[0]

    # 人間のクラスIDは0（COCOデータセットのクラスID）
    human_results = results_df[results_df['name'] == 'person']

    # 検出結果の表示
    for index, row in human_results.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = f"Person {confidence:.2f}"

        # バウンディングボックスとラベルの描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # フレームの表示
    cv2.imshow('WebCam', frame)

    # 'q'キーを押したらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
