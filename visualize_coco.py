import os
import cv2
import argparse
from collections import defaultdict

def parse_annotations(txt_path):
    frame_dict = defaultdict(list)
    with open(txt_path, 'r') as f:
        for line in f:
            frame, obj_id, x, y = map(int, line.strip().split(','))
            frame_dict[frame].append((x, y, obj_id))
    return frame_dict

def draw_boxes(image, annotations, box_size=30):
    H, W = image.shape[:2]  # Image dimensions for safety

    for x_center, y_center, obj_id in annotations:
        cx, cy = int(round(x_center)), int(round(y_center))

        half_size = box_size // 2
        x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
        x2, y2 = min(W - 1, cx + half_size), min(H - 1, cy + half_size)

        # Optional: Skip drawing if box would be out of bounds (extra safety)
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ Skipping invalid box for ID {obj_id} at ({cx},{cy})")
            continue
        print(obj_id, " ", (x1, y1), " " , (x2, y2))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f'ID {obj_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return image


def visualize_sequence(seq_id, annotation_file, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_data = parse_annotations(annotation_file)
    output_seq_dir = os.path.join(output_dir, seq_id)
    os.makedirs(output_seq_dir, exist_ok=True)
    for frame_id, annotations in frame_data.items():
        filename = f"{frame_id:05d}.jpg"
        image_path = os.path.join(image_dir, seq_id, filename)
        if not os.path.exists(image_path):
            print(f"Skipping missing frame: {filename}")
            continue

        image = cv2.imread(image_path)
        annotated = draw_boxes(image, annotations)
        cv2.imwrite(os.path.join(output_seq_dir, filename), annotated)

    print(f"All frames annotated and saved in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    visualize_sequence(args.seq_id, args.annotation_file, args.image_dir, args.output_dir)
