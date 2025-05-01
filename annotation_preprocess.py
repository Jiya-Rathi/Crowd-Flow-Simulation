import os
import numpy as np
from collections import defaultdict

# --- SETTINGS --- #
ANNOT_DIR = "./annotations"
OUT_DIR = "./annotations_with_ids"
os.makedirs(OUT_DIR, exist_ok=True)
DIST_THRESHOLD = 30  # Adjust based on frame rate/motion

def assign_ids_for_file(file_path, output_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # Parse lines into frame-wise format
    frame_annotations = defaultdict(list)
    for line in lines:
        frame, x, y = map(int, line.strip().split(","))
        frame_annotations[frame].append((x, y))

    next_id = 0
    active_tracks = {}
    output_lines = []

    for frame_num in range(1, max(frame_annotations.keys()) + 1):
        current_points = frame_annotations[frame_num]
        current_assigned = [False] * len(current_points)
        new_active_tracks = {}

        for track_id, (px, py) in active_tracks.items():
            min_dist = float("inf")
            match_idx = -1
            for i, (cx, cy) in enumerate(current_points):
                if current_assigned[i]:
                    continue
                dist = np.linalg.norm([px - cx, py - cy])
                if dist < min_dist and dist < DIST_THRESHOLD:
                    min_dist = dist
                    match_idx = i
            if match_idx != -1:
                cx, cy = current_points[match_idx]
                new_active_tracks[track_id] = (cx, cy)
                current_assigned[match_idx] = True
                output_lines.append(f"{frame_num},{track_id},{cx},{cy}")

        for i, (cx, cy) in enumerate(current_points):
            if not current_assigned[i]:
                track_id = next_id
                next_id += 1
                new_active_tracks[track_id] = (cx, cy)
                output_lines.append(f"{frame_num},{track_id},{cx},{cy}")

        active_tracks = new_active_tracks

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Processed: {os.path.basename(file_path)} → {os.path.basename(output_path)}")

# --- Process All Annotation Files --- #
for fname in os.listdir(ANNOT_DIR):
    if not fname.endswith(".txt"):
        continue
    input_path = os.path.join(ANNOT_DIR, fname)
    output_path = os.path.join(OUT_DIR, fname.replace(".txt", "_with_ids.txt"))
    assign_ids_for_file(input_path, output_path)
