import os
import cv2
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
import numpy as np

def draw_bboxes_on_frame(image_tensor, centers, box_size=20):
    image = to_pil_image(image_tensor)
    draw = ImageDraw.Draw(image)
    for x, y in centers:
        x1, y1 = x - box_size // 2, y - box_size // 2
        x2, y2 = x + box_size // 2, y + box_size // 2
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image

def extract_background(image_tensor, centers, box_size=20):
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    for x, y in centers:
        x, y = int(x), int(y)
        x1, y1 = max(0, x - box_size // 2), max(0, y - box_size // 2)
        x2, y2 = min(img_np.shape[1], x + box_size // 2), min(img_np.shape[0], y + box_size // 2)
        mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(img_uint8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

def process_sequence(seq_id, image_dir, ann_path, bbox_dir, bg_dir, sequence_length=30):
    sequence_folder = os.path.join(image_dir, seq_id)
    os.makedirs(os.path.join(bbox_dir, seq_id), exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        ann_lines = [list(map(int, line.strip().split(','))) for line in f if len(line.strip().split(',')) == 3]


    for frame_num in range(1, sequence_length + 1):
        frame_id = f"{frame_num:05d}"
        img_path = os.path.join(sequence_folder, frame_id + ".jpg")
        if not os.path.exists(img_path):
            continue

        image = read_image(img_path).float() / 255.0
        frame_annotations = [(x, y) for f, x, y in ann_lines if f == frame_num]

        out_img_path = os.path.join(bbox_dir, seq_id, frame_id + ".jpg")
        if not os.path.exists(out_img_path):
            image_with_boxes = draw_bboxes_on_frame(image, frame_annotations)
            image_with_boxes.save(out_img_path)

    # Extract background from frame 1
    bg_path = os.path.join(sequence_folder, "00001.jpg")
    if os.path.exists(bg_path):
        image = read_image(bg_path).float() / 255.0
        centers = [(x, y) for f, x, y in ann_lines if f == 1]
        bg_image = extract_background(image, centers)
        cv2.imwrite(os.path.join(bg_dir, f"{seq_id}.jpg"), bg_image[:, :, ::-1])  # RGB → BGR

    print(f"Finished: {seq_id}")

def process_all_sequences(annotation_dir, image_dir, bbox_dir="bbox_sequences", bg_dir="backgrounds"):
    ann_files = [f for f in os.listdir(annotation_dir) if f.endswith(".txt")]

    for fname in sorted(ann_files):
        seq_id = os.path.splitext(fname)[0]
        ann_path = os.path.join(annotation_dir, fname)
        process_sequence(seq_id, image_dir, ann_path, bbox_dir, bg_dir)

if __name__ == "__main__":
    annotation_dir = "./annotations"
    image_dir = "./sequences"
    process_all_sequences(annotation_dir, image_dir)
