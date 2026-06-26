import cv2
import numpy as np
from pathlib import Path

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
]

SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def draw_boxes(img, boxes, labels=None, color=(0, 255, 0), thickness=2):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        c = COLORS[i % len(COLORS)] if len(boxes) > 1 else color
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)
        if labels and i < len(labels):
            label = str(labels[i])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), c, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def draw_keypoints(img, keypoints, conf_thresh=0.3):
    h, w = img.shape[:2]
    for person_kps in keypoints:
        kps = person_kps[:, :2] if person_kps.shape[1] >= 2 else person_kps
        confs = person_kps[:, 2] if person_kps.shape[1] >= 3 else np.ones(len(person_kps))
        pts = []
        for i, (kp, conf) in enumerate(zip(kps, confs)):
            if conf < conf_thresh:
                pts.append(None)
                continue
            x = int(kp[0] * w) if kp[0] < 1 else int(kp[0])
            y = int(kp[1] * w) if kp[1] < 1 else int(kp[1])
            pts.append((x, y))
            cv2.circle(img, (x, y), 4, COLORS[i % len(COLORS)], -1)
            cv2.putText(img, KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else str(i),
                        (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for (i, j) in SKELETON:
            i, j = i - 1, j - 1
            if i < len(pts) and j < len(pts) and pts[i] is not None and pts[j] is not None:
                cv2.line(img, pts[i], pts[j], (0, 255, 255), 2)
    return img


def make_comparison_grid(images, labels, max_w=1200):
    if not images:
        return None
    n = len(images)
    h, w = images[0].shape[:2]
    aspect = w / h
    disp_w = min(max_w // n, int(max_w * 0.48))
    disp_h = int(disp_w / aspect)
    rows = []
    row = []
    for i, (img, label) in enumerate(zip(images, labels)):
        resized = cv2.resize(img, (disp_w, disp_h))
        cv2.putText(resized, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        row.append(resized)
        if len(row) == 2 or i == n - 1:
            rows.append(np.hstack(row))
            row = []
    return np.vstack(rows) if len(rows) > 1 else rows[0]


def save_output(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path
