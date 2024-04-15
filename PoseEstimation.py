import cv2
from ultralytics import YOLO

# モデルの読み込み。姿勢推論用のモデルデータを読み込む
model = YOLO("yolov8n-pose.pt")

# 本体のウェブカメラからキャプチャする設定の場合
video_path = 0  # 本体に付属のカメラを指定
capture = cv2.VideoCapture(video_path)
# 一人だけの動画の場合
# capture = cv2.VideoCapture("001.mp4")
# 二人の動画の場合
# capture = cv2.VideoCapture("002.mp4")

# keypointの位置毎の名称定義
KEYPOINTS_NAMES = [
    "nose",  # 0
    "eye(L)",  # 1
    "eye(R)",  # 2
    "ear(L)",  # 3
    "ear(R)",  # 4
    "shoulder(L)",  # 5
    "shoulder(R)",  # 6
    "elbow(L)",  # 7
    "elbow(R)",  # 8
    "wrist(L)",  # 9
    "wrist(R)",  # 10
    "hip(L)",  # 11
    "hip(R)",  # 12
    "knee(L)",  # 13
    "knee(R)",  # 14
    "ankle(L)",  # 15
    "ankle(R)",  # 16
]

while capture.isOpened():
    success, frame = capture.read()
    if success:
        # 推論を実行
        results = model(frame)

        annotatedFrame = results[0].plot()

        # 検出オブジェクトの名前、バウンディングボックス座標を取得
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes

        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            print(f"name={name}")
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        if len(results[0].keypoints) == 0:
            continue


        # 姿勢分析結果のキーポイントを取得する
        keypoints = results[0].keypoints
        confs = keypoints.conf[0].tolist()  # 推論結果:1に近いほど信頼度が高い
        xys = keypoints.xy[0].tolist()  # 座標
        # print(list(zip(xys, confs)))

        for index, keypoint in enumerate(zip(xys, confs)):  # ZIP＆LIST化して（X,Y座標＋信頼度）で置いてる
            score = keypoint[1]
            print(keypoint)

            # スコアが0.5以下なら描画しない
            if score < 0.5:
                continue

            x = int(keypoint[0][0])
            y = int(keypoint[0][1])
            # print(f"Keypoint Name={KEYPOINTS_NAMES[index]}, X={x}, Y={y}, Score={score:.4}")
            # 紫の四角を描画
            annotatedFrame = cv2.rectangle(
                annotatedFrame,
                (x, y),
                (x + 3, y + 3),
                (255, 0, 255),
                cv2.FILLED,
                cv2.LINE_AA,
            )
            # キーポイントの部位名称を描画
            #annotatedFrame = cv2.putText(
            #    annotatedFrame,
             #   KEYPOINTS_NAMES[index],
              #  (x + 5, y),
               # fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #fontScale=1.3,
                #color=(255, 0, 255),
                #thickness=2,
                #lineType=cv2.LINE_AA,
            #)

        print("------------------------------------------------------")

        cv2.imshow("YOLOv8 human pose estimation", annotatedFrame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows
