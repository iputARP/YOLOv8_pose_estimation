import cv2
from ultralytics import YOLO
import math

# モデルの読み込み。姿勢推論用のモデルデータを読み込む
model = YOLO("yolov8n-pose.pt")

# 本体のウェブカメラからキャプチャする設定
# 0  本体に付属のカメラを指定
# "001.mp4"  一人Verの動画
# "002.mp4"  二人Verの動画
# "003.mp4"  三人Verの動画
capture = cv2.VideoCapture("003.mp4")

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

STATE_LIST = [  # 状態リスト（INDEXで取り出す）
    "Unknown",  # 0
    "Standing",  # 1
    "Sitting on th Chair",  # 2
    "Waving both hands over head",  # 3
    "Raising one hand",  # 4
]

def angle(top, mid, bot): # その関節の角度の求め方（真っ直ぐ伸ばした状態からの絶対値）topが一番上でbotが一番下
    return abs(abs(math.degrees(math.atan2(bot[1] - mid[1], bot[0] - mid[0]) - math.atan2(top[1] - mid[1], top[0] - mid[0]))) - 180)

def result(xys): # キーポイントの座標リストを与えて姿勢推定結果を出す
    # xys[KEYPOINTS_NAMES]＝[X座標,Y座標]
    if ((xys[5][1]+xys[6][1])/2 < (xys[13][1]+xys[14][1])/2) and ((xys[11][1]+xys[12][1])/2 < (xys[13][1]+xys[14][1])/2):
        # 「肩の高さ>腰の高さ>ひざの高さ」だったら最低限立ってます。
        if angle(xys[11],xys[13],xys[15]) + angle(xys[12],xys[14],xys[16]) + angle(xys[5],xys[11],xys[13]) + angle(xys[6],xys[12],xys[14]) > 40:
            # もし座っていたら
            return 2
        else:  # もし立っているとして
            # ここにサムズアップポーズのIf文を入れたいが、キーポイントが足りない？
            # 手のどちらか片方は少なくとも上げている
            if (xys[9][1] < xys[7][1] and xys[7][1] < xys[5][1]) or (xys[10][1] < xys[8][1] and xys[8][1] < xys[6][1]):
                if (xys[9][1] < xys[7][1] and xys[7][1] < xys[5][1]) and (xys[10][1] < xys[8][1] and xys[8][1] < xys[6][1]):
                    # 両手のどちらも上げている
                    return 3
                else:  # 片手だけ上げている
                    return 4
            else:  # 立っているだけでした
                return 1

    else:  # 何のポーズか分かりません
        return 0


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


        def pose_estimate(XYs, Confs, annotatedFrame):  # 姿勢推定の関数
            for index, keypoint in enumerate(zip(XYs, Confs)):  # ZIP＆LIST化して（X,Y座標＋信頼度）で置いてる
                score = keypoint[1]
                # print(keypoint)

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
                # annotatedFrame = cv2.putText(
                #    annotatedFrame,
                #   KEYPOINTS_NAMES[index],
                #  (x + 5, y),
                # fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                # fontScale=1.3,
                # color=(255, 0, 255),
                # thickness=2,
                # lineType=cv2.LINE_AA,
                # )


        # 姿勢分析結果のキーポイントを取得する
        keypoints = results[0].keypoints
        for i in range(len(keypoints)):
            confs = keypoints.conf[i].tolist()  # 推論結果:1に近いほど信頼度が高い
            xys = keypoints.xy[i].tolist()  # 座標
            # print(list(zip(xys, confs)))
            print(result(xys))
            print(STATE_LIST[result(xys)])
            pose_estimate(xys, confs, annotatedFrame)


        print("------------------------------------------------------")

        cv2.imshow("YOLOv8 human pose estimation", annotatedFrame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows
