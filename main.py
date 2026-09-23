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
trail_points = []
# BGR Color Schemes & Geometry for 1 to 5 fingers
THEMES = {
    1: {  # Frost / Ice (3-Sided Triangle)
        "ORANGE": (255, 230, 100),
        "GOLD": (255, 200, 0),
        "HOT": (255, 255, 255),
        "sides": 3,
    },
    2: {  # Earth / Nature (4-Sided Square)
        "ORANGE": (0, 200, 50),
        "GOLD": (100, 255, 100),
        "HOT": (200, 255, 200),
        "sides": 4,
    },
    3: {  # Arcane / Void (5-Sided Pentagon)
        "ORANGE": (200, 0, 200),
        "GOLD": (255, 100, 255),
        "HOT": (255, 220, 255),
        "sides": 5,
    },
    4: {  # Fire / Magma (6-Sided Hexagon)
        "ORANGE": (0, 0, 255),
        "GOLD": (0, 140, 255),
        "HOT": (100, 255, 255),
        "sides": 6,
    },
    5: {  # Divine / Light (8-Sided Octagon)
        "ORANGE": (0, 180, 255),
        "GOLD": (100, 220, 255),
        "HOT": (255, 255, 255),
        "sides": 8,
    },
}

def count_raised_fingers(pts):
    # 1. Four fingers: Index (8), Middle (12), Ring (16), Pinky (20)
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))
    count = 0

    for tip, pip in fingers:
        d_tip = math.hypot(pts[tip][0] - pts[0][0], pts[tip][1] - pts[0][1])
        d_pip = math.hypot(pts[pip][0] - pts[0][0], pts[pip][1] - pts[0][1])
        if d_tip > d_pip:
            count += 1

    # 2. Thumb check: Compare thumb tip (4) distance from pinky base (17) vs thumb IP joint (3)
    d_thumb_tip = math.hypot(pts[4][0] - pts[17][0], pts[4][1] - pts[17][1])
    d_thumb_ip = math.hypot(pts[3][0] - pts[17][0], pts[3][1] - pts[17][1])

    if d_thumb_tip > d_thumb_ip:
        count += 1

    return count
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


            # 1. Update Motion Trail
            wrist = (int(pts[0][0]), int(pts[0][1]))
            trail_points.append(wrist)
            if len(trail_points) > 15:
                trail_points.pop(0)


            # 2. Gesture Mode Detection
            fingers_up = count_raised_fingers(pts)

            if fingers_up in THEMES:
                ids = (0, 5, 9, 13, 17)
                cx = int(sum(pts[i][0] for i in ids) / len(ids))
                cy = int(sum(pts[i][1] for i in ids) / len(ids))


                radius = int(
                    math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]) * 1.1
                )

                theme = THEMES[fingers_up]
                shield.draw_shield(
                    layer,
                    (cx, cy),
                    radius,
                    t,
                    colors=theme,
                    sides=theme["sides"],
                )
            radius = int(math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1])*1.1 )
            #if is_open_hand(pts):
             #   shield.draw_shield(layer, (cx, cy), radius, t,angle_deg=angle)
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