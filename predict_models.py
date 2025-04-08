import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision.transforms.functional import to_pil_image
from project_3_models import CrowdDataset, ConvLSTM, SocialLSTM, TCNN
from torch.utils.data import DataLoader

# ---------- CLI ARG ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['conv_lstm', 'social_lstm', 'tcnn'],
                    help="Model to use for prediction")
args = parser.parse_args()

# ---------- SETUP ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- DATA ---------- #
image_dir = "./sequences/"
annotation_dir = "./annotations/"
test_dataset = CrowdDataset("testlist_copy.txt", image_dir, annotation_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------- BBOX PREDICTOR ---------- #
class BBoxLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, bbox_seq):
        pred_coords = []
        for i in range(bbox_seq.shape[2]):
            track = bbox_seq[0, :, i, :]
            if torch.isnan(track).any():
                pred_coords.append(torch.tensor([float('nan'), float('nan')]))
                continue
            out, _ = self.lstm(track.unsqueeze(0))
            next_coord = self.fc(out[:, -1])
            pred_coords.append(next_coord.squeeze(0))
        return torch.stack(pred_coords, dim=0)

# ---------- PREDICTION UTILS ---------- #
def predict_future(model, initial_frames, steps=10):
    model.eval()
    preds = []
    inputs = initial_frames[:, -20:].clone()  # [1, 20, 3, H, W]

    with torch.no_grad():
        for _ in range(steps):
            out = model(inputs)  # [1, 3, H, W]
            preds.append(out.cpu())
            inputs = torch.cat((inputs[:, 1:], out.unsqueeze(1)), dim=1)

    return torch.stack(preds, dim=1)  # [1, 10, 3, H, W]

def predict_future_bboxes(bbox_model, initial_bboxes, steps=5):
    bbox_model.eval()
    preds = []
    bbox_seq = initial_bboxes[:, -20:].clone()  # [1, 20, N, 2]

    with torch.no_grad():
        for _ in range(steps):
            pred = bbox_model(bbox_seq)             # [N, 2]
            preds.append(pred)

            pred = pred.unsqueeze(0).unsqueeze(0)   # [1, 1, N, 2]
            bbox_seq = torch.cat((bbox_seq[:, 1:], pred), dim=1)  # [1, 20, N, 2]

    return torch.stack(preds, dim=1)  # [1, 10, N, 2]

def save_predicted_frames(future_frames, output_dir="results/frames"):
    os.makedirs(output_dir, exist_ok=True)
    for t in range(future_frames.shape[1]):
        img = future_frames[0, t]
        img_pil = to_pil_image(img.clamp(0, 1))
        img_pil.save(os.path.join(output_dir, f"frame_{t+1:03d}.png"))

def save_predicted_bboxes(future_bboxes, output_path="results/bboxes/predicted_bboxes.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for t in range(future_bboxes.shape[1]):
            f.write(f"Frame {t+1}\n")
            coords = future_bboxes[0, t].cpu().numpy()

            coords = np.atleast_2d(coords)  # handles single bbox [2] → [1, 2]

            for i, coord in enumerate(coords):
                if coord.shape[0] == 2:
                    x, y = coord
                    if not (np.isnan(x) or np.isnan(y)):
                        f.write(f"{i},{x:.2f},{y:.2f}\n")
            f.write("\n")


# ---------- MAIN ---------- #
if __name__ == "__main__":
    # Select model
    model_name = args.model
    model_map = {
        "conv_lstm": ConvLSTM,
        "social_lstm": SocialLSTM,
        "tcnn": TCNN
    }
    model_class = model_map[model_name]
    model = model_class().to(device)

    # Load trained weights (adjust filename as needed)
    weight_path = f"{model_name}_checkpoint.pth"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"✅ Loaded weights from {weight_path}")

    bbox_model = BBoxLSTM().to(device)

    # Take one batch
    for frames, _, ann_seq, _ in test_loader:
        frames = frames.to(device)
        ann_seq = ann_seq[0]
        bboxes = ann_seq[:, :, 1:3].unsqueeze(0).to(device)
        break

    # Predict & save
    future_frames = predict_future(model, frames, steps=5)
    future_bboxes = predict_future_bboxes(bbox_model, bboxes, steps=5)

    save_predicted_frames(future_frames)
    save_predicted_bboxes(future_bboxes)

    print("Prediction complete. Results saved in results folder.")
