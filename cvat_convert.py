import os
import json
import itertools

def center_to_bbox(x_center, y_center, box_size=30):
    x = x_center - box_size / 2
    y = y_center - box_size / 2
    return x, y, box_size, box_size

def prepare_coco_format_per_sequence(input_folder, output_folder='output_coco', box_size=30):
    os.makedirs(output_folder, exist_ok=True)

    annotation_id = itertools.count(1)  # Unique ID counter for annotations

    for input_file in os.listdir(input_folder):
        if not input_file.endswith('.txt'):
            continue

        sequence_name = os.path.splitext(input_file)[0]
        input_path = os.path.join(input_folder, input_file)

        images = []
        annotations = []
        categories = [
            {"id": 1, "name": "object", "supercategory": "none"}
        ]

        frames_seen = {}

        with open(input_path, 'r') as fin:
            for line in fin:
                parts = list(map(float, line.strip().split(',')))
                frame_id, object_id, x_center, y_center = parts[:4]

                x, y, w, h = center_to_bbox(x_center, y_center, box_size)

                # Add image entry only once per frame
                if int(frame_id) not in frames_seen:
                    file_name = f"{int(frame_id):05d}.jpg"  # Frame 0 -> 000001.jpg, frame 1 -> 000002.jpg
                    image_info = {
                        "id": int(frame_id),  # +1 to start IDs at 1
                        "width": 1920,
                        "height": 1080,
                        "file_name": file_name
                    }
                    frames_seen[int(frame_id)] = image_info
                    images.append(image_info)

                annotations.append({
                    "id": next(annotation_id),
                    "image_id": int(frame_id),
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "track_id": int(object_id)
                })

        output_json = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        output_json_path = os.path.join(output_folder, f"{sequence_name}.json")
        with open(output_json_path, 'w') as fout:
            json.dump(output_json, fout, indent=4)

        print(f"Created corrected COCO JSON: {output_json_path}")

if __name__ == '__main__':
    input_folder = "./annotations_with_ids/"
    prepare_coco_format_per_sequence(input_folder)
