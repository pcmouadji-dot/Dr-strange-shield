import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import numpy as np
import shield

start=time.time()
def is_open_hand(pts):
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))   # (tip, middle joint)
    up = 0
    for tip, pip in fingers:
        d_tip = math.hypot(pts[tip][0] - pts[0][0], pts[tip][1] - pts[0][1])
        d_pip = math.hypot(pts[pip][0] - pts[0][0], pts[pip][1] - pts[0][1])
        if d_tip > d_pip:        # tip is farther from the wrist than the joint
            up += 1
    return up >= 3

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)
#alpha=0

video = cv2.VideoCapture(0)
start = time.time()

while True:
    target=0
    ret, img = video.read()
    if not ret:
        break
    img=cv2.flip(img, 1)
    t = time.time() - start
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = img.shape
    layer=np.zeros_like(img)
    for hand in result.hand_landmarks:
            pts = [(lm.x * w, lm.y * h) for lm in hand]

            ids = (0, 5, 9, 13,17)
            cx = int(sum(pts[i][0] for i in ids) / len(ids))
            cy = int(sum(pts[i][1] for i in ids) / len(ids))
            dx = pts[9][0] - pts[0][0]
            dy = pts[9][1] - pts[0][1]
            angle = math.degrees(math.atan2(dy, dx)) + 90
            print(f"Calculated Angle: {angle:.2f}")

            radius = int(math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1])*1.1 )
            if is_open_hand(pts):
                shield.draw_shield(layer, (cx, cy), radius, t,angle_deg=angle)
                #target=1

   # alpha += (target - alpha) * 0.1 this was in order to make the shield fade and appear fading but the problem is it will work with only one hande (making it two hands will requare  a lotttttt of work and i don't have it)
    #if alpha > 0.3:
   #     shield.draw_shield(layer, (cx, cy), radius, t,alpha=alpha)

    img = shield.composite(img, layer)
    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
        break

landmarker.close()
video.release()
cv2.destroyAllWindows()