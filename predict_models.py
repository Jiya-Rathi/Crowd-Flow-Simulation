import os
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw
from model_trajectory_transformer import TrajectoryTransformer

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------- Utility Functions --------- #
def draw_prediction_on_background(background_path, predicted_coords, output_dir, box_size=20):
    os.makedirs(output_dir, exist_ok=True)
    bg_img = Image.open(background_path).convert("RGB")

    for i, (x, y) in enumerate(predicted_coords):
        img = bg_img.copy()
        draw = ImageDraw.Draw(img)
        x1, y1 = x - box_size // 2, y - box_size // 2
        x2, y2 = x + box_size // 2, y + box_size // 2
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Future {i+1}", fill="yellow")
        img.save(os.path.join(output_dir, f"future_{i+1}.jpg"))

# --------- Prediction Logic --------- #
def predict_future_trajectory(seq_id, person_id, weights_path):
    ann_file    = f"./annotations_with_ids/{seq_id}_with_ids.txt"
    bg_file     = f"./backgrounds/{seq_id}.jpg"
    output_dir  = f"./future_frames/{seq_id}/person_{person_id}"

    # --- checks ---
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return
    if not os.path.exists(bg_file):
        print(f"Background image not found: {bg_file}")
        return

    # --- load model ---
    model = TrajectoryTransformer().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- read all detections for this person ---
    data = []
    with open(ann_file, 'r') as f:
        for line in f:
            fr, pid, x, y = map(float, line.strip().split(',')[:4])
            if int(pid) == person_id:
                data.append((int(fr), x/1920.0, y/1080.0))

    if not data:
        print(f"No detections for person {person_id}")
        return

    data.sort(key=lambda x: x[0])
    full_map = {fr:(x,y) for fr,x,y in data}
    max_frame = data[-1][0]

    # --- build exactly 25 consecutive frames up to max_frame ---
    frames = list(range(max_frame-24, max_frame+1))
    seq_vals = []
    last_xy = None
    for fr in frames:
        if fr in full_map:
            last_xy = full_map[fr]
        # repeat last seen if missing
        seq_vals.append(last_xy)

    # --- build dx,dy + abs inputs ---
    combined = []
    for i in range(1, len(seq_vals)):
        x0,y0 = seq_vals[i-1]
        x1,y1 = seq_vals[i]
        combined.append([x1-x0, y1-y0, x1, y1])
    # combined has length 24

    # --- tensor prep ---
    seq = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)   # [1,24,4]
    seq = torch.cat([seq, seq[:, -1:, :]], dim=1)                    # [1,25,4]
    seq = seq.unsqueeze(1).to(device)                                # [1,1,25,4]
    
    # Compute avg disp over last 10
    disps = seq[0, 0, -10:, :2]        # [10, 2]
    avg_disp = disps.mean(dim=0)      # [2]
    
    # Adjusted start
    '''
    last_pos = torch.tensor(seq_vals[-1], dtype=torch.float32).to(device)
    if avg_disp.norm() > 1e-4:  # small threshold to detect static
        last_pos = last_pos + avg_disp
    last_pos = last_pos.unsqueeze(0)   # [1, 2]
    '''
    # Confirm last input used in sequence (i.e., last [x,y] in input)
    true_last = combined[-1][2:]  # x1, y1 from last row of combined
    last_pos = torch.tensor(true_last, dtype=torch.float32).unsqueeze(0).to(device)
    print("True frame 30 position:", seq_vals[-1])
    print("Last input position used in input sequence:", true_last)


    # --- rollout 5 future steps ---
    preds = []
    with torch.no_grad():
        for _ in range(5):
            delta = model(seq).squeeze(1)  # [1,2]
            last_pos = last_pos + delta    # [1,2]
            preds.append(last_pos.squeeze(0).cpu().numpy().copy())
    
            new_in = torch.cat([delta.squeeze(0), last_pos.squeeze(0)], dim=0)  # [4]
            new_in = new_in.view(1, 1, 1, 4).to(device)  # [1,1,1,4]
    
            seq = torch.cat([seq[:, :, 1:, :], new_in], dim=2)

    # --- unnormalize & draw ---
    preds = np.array(preds)
    preds[:,0] *= 1920.0
    preds[:,1] *= 1080.0

    draw_prediction_on_background(bg_file, preds, output_dir)
    print(f"Predictions saved in: {output_dir}")

# --------- CLI --------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=str, default="00001", help="Sequence ID (default: 00001)")
    parser.add_argument('--person_id', type=int, default=40, help="Person ID (default: 40)")
    parser.add_argument('--weights', type=str, default='trajectory_transformer.pth', help="Path to model weights")
    args = parser.parse_args()

    predict_future_trajectory(args.seq_id, args.person_id, args.weights)
