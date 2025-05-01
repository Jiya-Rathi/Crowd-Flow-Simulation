import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2
from PIL import Image, ImageDraw

# --------- Model Definition --------- #
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # [B, T, D] or [B, N, T, D]
        if x.dim() == 4:
            B, N, T, D = x.shape
            x = x.view(B * N, T, D)
            out, _ = self.lstm(x)
            out = self.norm(out[:, -1, :])
            pred = self.fc(out)
            return pred.view(B, N, 2)
        elif x.dim() == 3:
            out, _ = self.lstm(x)             # [B, T, H]
            out = self.norm(out[:, -1, :])    # [B, H]
            pred = self.fc(out)               # [B, 2]
            return pred
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

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
def predict_future_trajectory(seq_id, person_id):
    annotation_file = f"./annotations_with_ids/{seq_id}_with_ids.txt"
    background_path = f"./backgrounds/{seq_id}.jpg"
    output_dir = f"./future_frames/{seq_id}/person_{person_id}"

    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return
    if not os.path.exists(background_path):
        print(f"❌ Background image not found: {background_path}")
        return

    # Parse last 25 frames of the given person
    person_data = []
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            frame, pid, x, y = parts
            if int(pid) == person_id:
                person_data.append((int(frame), x / 1920.0, y / 1080.0))  # normalized

    person_data.sort()
    if len(person_data) < 25:
        print(f"❌ Not enough data (found {len(person_data)} frames) for person {person_id}")
        return

    last_25 = person_data[-25:]
    combined_seq = []

    for i in range(1, len(last_25)):
        dx = last_25[i][1] - last_25[i-1][1]
        dy = last_25[i][2] - last_25[i-1][2]
        x = last_25[i][1]
        y = last_25[i][2]
        combined_seq.append([dx, dy, x, y])

    # Shape: [1, 24, 4]
    seq = torch.tensor(combined_seq, dtype=torch.float32).unsqueeze(0).to(device)
    # Pad to make 25 steps (like training)
    seq = torch.cat([seq, seq[:, -1:, :]], dim=1)  # [1, 25, 4]

    # Last known absolute position
    last_pos = torch.tensor([last_25[-1][1], last_25[-1][2]], dtype=torch.float32).unsqueeze(0).to(device)

    predicted_positions = []
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            delta = model(seq)  # [1, 2]
            last_pos = last_pos + delta
            predicted_positions.append(last_pos.squeeze().cpu().numpy())
            new_input = torch.cat([delta, last_pos], dim=-1).unsqueeze(1)  # [1, 1, 4]
            seq = torch.cat([seq[:, 1:], new_input], dim=1)

    # Unnormalize to pixel coords
    predicted_positions = np.array(predicted_positions)
    predicted_positions[:, 0] *= 1920.0
    predicted_positions[:, 1] *= 1080.0

    draw_prediction_on_background(background_path, predicted_positions, output_dir)
    print(f"✅ Predictions saved in: {output_dir}")


# --------- Main CLI --------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=str, required=True)
    parser.add_argument('--person_id', type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load("trajectory_lstm.pth", map_location=device))
    predict_future_trajectory(args.seq_id, args.person_id)
