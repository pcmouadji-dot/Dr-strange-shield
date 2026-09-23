import math

import cv2
import numpy as np



ORANGE = (0, 0, 255)
GOLD = (35, 35, 255)
HOT = (80, 80, 255)

GLOW = 0.8
DARKEN = 0.6


def polar(center, r, deg):
    a = math.radians(deg)
    return int(center[0] + r * math.cos(a)), int(center[1] + r * math.sin(a))


def dashed_ring(layer, center, r, start_deg, n, fill, color, thickness):
    step = 360 / n
    for i in range(n):
        a0 = start_deg + i * step
        cv2.ellipse(
            layer,
            center,
            (int(r), int(r)),
            0,
            a0,
            a0 + step * fill,
            color,
            thickness,
            cv2.LINE_AA,
        )


def polygon(layer, center, r, start_deg, sides, color, thickness):
    pts = [polar(center, r, start_deg + i * 360 / sides) for i in range(sides)]
    cv2.polylines(
        layer,
        [np.array(pts, np.int32).reshape(-1, 1, 2)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_shield(layer, center, R, t, alpha=1.0, angle_deg=0):
    def tint(c):
        return tuple(int(v * alpha) for v in c)

    th = max(1, int(R * 0.018))
    AA = cv2.LINE_AA

    # Outer double ring
    cv2.circle(layer, center, int(R), tint(ORANGE), th * 2, AA)
    cv2.circle(layer, center, int(R * 0.94), tint(GOLD), th, AA)

    # 1. Tick marks (adding hand rotation)
    for i in range(60):
        deg = i * 6 + t * 10 + angle_deg
        inner = 0.84 if i % 5 == 0 else 0.88
        cv2.line(
            layer,
            polar(center, R * inner, deg),
            polar(center, R * 0.92, deg),
            tint(GOLD),
            1,
            AA,
        )

    # 2. Outer dashed ring (adding hand rotation)
    dashed_ring(
        layer,
        center,
        R * 0.78,
        t * 35 + angle_deg,
        16,
        0.65,
        tint(ORANGE),
        th * 2,
    )
    cv2.circle(layer, center, int(R * 0.70), tint(GOLD), th, AA)

    # 3. Rotating 8-point star squares (adding hand rotation)
    for off in (0, 45):
        polygon(
            layer, center, R * 0.70, -t * 25 + off + angle_deg, 4, tint(ORANGE), th
        )

    # 4. Inner dashed ring (adding hand rotation)
    dashed_ring(
        layer, center, R * 0.50, -t * 50 + angle_deg, 10, 0.5, tint(GOLD), th * 2
    )
    cv2.circle(layer, center, int(R * 0.38), tint(GOLD), th, AA)

    # 5. Runes (adding hand rotation)
    for i in range(8):
        deg = i * 45 + t * 20 + angle_deg
        cv2.circle(
            layer,
            polar(center, R * 0.60, deg),
            max(2, int(R * 0.04)),
            tint(GOLD),
            th,
            AA,
        )
        cv2.line(
            layer,
            polar(center, R * 0.38, deg),
            polar(center, R * 0.55, deg),
            tint(ORANGE),
            1,
            AA,
        )

    # 6. Glowing core
    cv2.circle(layer, center, max(2, int(R * 0.06)), tint(HOT), -1, AA)

    # 7. Sparks (adding hand rotation)
    for i in range(14):
        deg = i * 360 / 14 + t * 60 + angle_deg
        d = R * (1.04 + 0.06 * math.sin(t * 4 + i * 1.7))
        cv2.circle(
            layer, polar(center, d, deg), max(1, int(R * 0.015)), tint(HOT), -1, AA
        )


def composite(img, layer):
    layer = layer.astype(np.float32)
    glow = cv2.GaussianBlur(layer, (0, 0), 10)
    total = np.clip(layer + glow * GLOW, 0, 255)
    m = total[:, :, 2:3] / 255.0
    out = img.astype(np.float32) * (1.0 - m) + total
    return np.clip(out, 0, 255).astype(np.uint8)