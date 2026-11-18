# Mall & Retail Foot Traffic Analysis
# Combined code: foot_traffic.py + plot_traffic.py

# ============================================================
# Listing 1: foot_traffic.py
# ============================================================

# foot_traffic.py

import csv
import cv2
from ultralytics import YOLO

VIDEO_PATH = "mall_entrance.mp4"
OUTPUT_VIDEO = "output_annotated.mp4"
CSV_PATH = "foot_traffic_log.csv"
CONF = 0.40
LINE_RATIO = 0.5

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = int(height * LINE_RATIO)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

prev_cy = {}
count_in = 0
count_out = 0
unique_ids = set()
rows = []
frame_no = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    result = model.track(
        frame, persist=True, classes=[0], conf=CONF,
        tracker="bytetrack.yaml", verbose=False
    )[0]

    people_now = 0

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            unique_ids.add(tid)
            people_now += 1

            if tid in prev_cy:
                if prev_cy[tid] < line_y <= cy:
                    count_in += 1
                elif prev_cy[tid] >= line_y > cy:
                    count_out += 1

            prev_cy[tid] = cy

            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                frame, "ID " + str(tid),
                (p1[0], p1[1] - 6), FONT, 0.5,
                (0, 255, 0), 1
            )
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

    text = ("IN: " + str(count_in) + " OUT: "
            + str(count_out) + " NOW: " + str(people_now))
    cv2.putText(
        frame, text, (10, 30), FONT, 0.8,
        (0, 0, 255), 2
    )

    writer.write(frame)
    rows.append([
        frame_no, round(frame_no / fps, 2),
        people_now, count_in, count_out
    ])

cap.release()
writer.release()

with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "time_s", "people_in_frame",
                "total_in", "total_out"])
    w.writerows(rows)

print("Entries:", count_in, "Exits:", count_out)
print("Unique people seen:", len(unique_ids))


# ============================================================
# Listing 2: plot_traffic.py
# ============================================================

# plot_traffic.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("foot_traffic_log.csv")
df["minute"] = (df["time_s"] // 60).astype(int)
per_min = df.groupby("minute")["people_in_frame"].mean()

per_min.plot(kind="bar", figsize=(8, 4))
plt.xlabel("Minute of video")
plt.ylabel("Average people in frame")
plt.title("Foot Traffic per Minute")
plt.tight_layout()
plt.savefig("traffic_per_minute.png", dpi=200)
plt.show()
