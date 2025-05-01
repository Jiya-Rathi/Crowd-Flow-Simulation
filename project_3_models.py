import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
from skimage.metrics import structural_similarity as ssim
import cv2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL import ImageDraw
# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Dataset Preparation --- #
'''
class CrowdDataset(Dataset):
    def __init__(self, data_list, image_dir, annotation_dir, sequence_length=30, max_objects=150):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sequence_length = sequence_length
        self.max_objects = max_objects

        with open(data_list, 'r') as f:
            self.sequence_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, idx):
        sequence_id = self.sequence_files[idx]
        ann_path = os.path.join(self.annotation_dir, f"{sequence_id}_with_ids.txt")

        # Parse annotations
        ann_per_frame = {}
        with open(ann_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split(',')))
                frame_id, track_id, x, y = parts[:4]
                frame_id = int(frame_id)
                if frame_id not in ann_per_frame:
                    ann_per_frame[frame_id] = []
                ann_per_frame[frame_id].append((int(track_id), x, y))

        # Build [T, 150, 3] tensor
        sequence_tensor = torch.full((self.sequence_length, self.max_objects, 3), -1.0)

        for frame_num in range(1, self.sequence_length + 1):
            objs = ann_per_frame.get(frame_num, [])
            for obj in objs:
                track_id, x, y = obj
                if track_id < self.max_objects:
                    sequence_tensor[frame_num - 1, track_id, 0] = track_id
                    sequence_tensor[frame_num - 1, track_id, 1] = x
                    sequence_tensor[frame_num - 1, track_id, 2] = y
        
        sequence_tensor[..., 1] /= 1920.0  # Normalize x
        sequence_tensor[..., 2] /= 1080.0  # Normalize y

        # Split into input 25 + target 5
        input_seq = sequence_tensor[:25]   # [25, 150, 3]
        target_seq = sequence_tensor[25:30, :, 1:]  # [5, 150, 2] (only x,y)

        return input_seq[:, :, 1:], target_seq
'''

class CrowdDataset(Dataset):
    def __init__(self, data_list, image_dir, annotation_dir, sequence_length=30, max_objects=150):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sequence_length = sequence_length
        self.max_objects = max_objects

        with open(data_list, 'r') as f:
            self.sequence_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, idx):
        sequence_id = self.sequence_files[idx]
        ann_path = os.path.join(self.annotation_dir, f"{sequence_id}_with_ids.txt")

        # Parse annotations
        ann_per_frame = {}
        with open(ann_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split(',')))
                frame_id, track_id, x, y = parts[:4]
                frame_id = int(frame_id)
                if frame_id not in ann_per_frame:
                    ann_per_frame[frame_id] = []
                ann_per_frame[frame_id].append((int(track_id), x, y))

        # Build [T, 150, 2] tensor for positions
        pos_tensor = torch.full((self.sequence_length, self.max_objects, 2), -1.0)
        for frame_num in range(1, self.sequence_length + 1):
            objs = ann_per_frame.get(frame_num, [])
            for track_id, x, y in objs:
                if track_id < self.max_objects:
                    pos_tensor[frame_num - 1, track_id, 0] = x / 1920.0  # normalize x
                    pos_tensor[frame_num - 1, track_id, 1] = y / 1080.0  # normalize y

        # Compute displacements for input (t = 0 to 24 → dxdy of t=1 to 25)
        input_pos = pos_tensor[:26]  # [26, 150, 2]
        displacement_input = input_pos[1:] - input_pos[:-1]  # [25, 150, 2]

        # Ground truth future positions (not displacements)
        future_pos = pos_tensor[26:31]  # [5, 150, 2]
        combined_input = torch.cat([displacement_input, input_pos[1:]], dim=-1)  # [25, 150, 4]
        return combined_input, input_pos[25], future_pos  # [25, 150, 4], [150,2], [5,150,2]

        #return displacement_input, input_pos[25], future_pos  # [25,150,2], [150,2], [5,150,2]



    
# Dataset Paths
image_dir = "./sequences/" 
annotation_dir = "./annotations_with_ids/"

train_dataset = CrowdDataset("trainlist_copy.txt", image_dir, annotation_dir)
test_dataset = CrowdDataset("testlist_copy.txt", image_dir, annotation_dir)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# --- 2. Model Architectures --- #

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [B, 150, 25, 4]
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        pred = self.fc(out)
        return pred.view(B, N, 2)

def masked_mse(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask.unsqueeze(-1)  # [B, 150, 5, 2]
    return loss.sum() / mask.sum().clamp(min=1)

def masked_l2(pred, target, mask, clip_thresh = 1000):
    dist = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # [B, 150, 5]
    dist = dist * mask
    if clip_thresh is not None:
        dist = torch.clamp(dist, max=clip_thresh)
    return dist.sum() / mask.sum().clamp(min=1)

def train(model, dataloader, epochs=50, clip_thresh=1000):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_mse, total_l2_clipped = 0, 0

        for full_seq, last_pos, future in dataloader:
            # full_seq: [B, 25, 150, 4] → [B, 150, 25, 4]
            seq = full_seq.permute(0, 2, 1, 3).to(device)  
            last_pos = last_pos.to(device)                # [B, 150, 2]
            future = future.permute(0, 2, 1, 3).to(device) # [B, 150, 5, 2]

            preds = []
            curr_pos = last_pos.clone()

            for _ in range(5):
                delta = model(seq)            # [B, 150, 2]
                curr_pos = curr_pos + delta   # [B, 150, 2]
                preds.append(curr_pos)

                dxdy = delta.unsqueeze(2)     # [B, 150, 1, 2]
                absxy = curr_pos.unsqueeze(2) # [B, 150, 1, 2]
                new_input = torch.cat([dxdy, absxy], dim=-1)  # [B, 150, 1, 4]

                seq = torch.cat((seq[:, :, 1:], new_input), dim=2)

            pred_tensor = torch.stack(preds, dim=1).permute(1, 2, 0, 3)  # [B, 150, 5, 2]

            # Unnormalize
            pred_tensor[..., 0] *= 1920.0
            pred_tensor[..., 1] *= 1080.0
            future[..., 0] *= 1920.0
            future[..., 1] *= 1080.0

            mask = (future[..., 0] != -1920).float()

            loss_mse = masked_mse(pred_tensor, future, mask)
            loss_l2 = masked_l2(pred_tensor, future, mask, clip_thresh)

            optimizer.zero_grad()
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_mse += loss_mse.item()
            total_l2_clipped += loss_l2.item()

        print(f"Epoch {epoch+1}, MSE: {total_mse / len(dataloader):.2f}, "
              f"L2 (clipped): {total_l2_clipped / len(dataloader):.2f} px")



def evaluate(model, dataloader, clip_thresh=1000):
    model.eval()
    total_mse, total_l2_clipped = 0, 0

    with torch.no_grad():
        for full_seq, last_pos, future in dataloader:
            seq = full_seq.permute(0, 2, 1, 3).to(device)
            last_pos = last_pos.to(device)
            future = future.permute(0, 2, 1, 3).to(device)

            preds = []
            curr_pos = last_pos.clone()

            for _ in range(5):
                delta = model(seq)
                curr_pos = curr_pos + delta
                preds.append(curr_pos)

                dxdy = delta.unsqueeze(2)
                absxy = curr_pos.unsqueeze(2)
                new_input = torch.cat([dxdy, absxy], dim=-1)
                seq = torch.cat((seq[:, :, 1:], new_input), dim=2)

            pred_tensor = torch.stack(preds, dim=1).permute(1, 2, 0, 3)

            pred_tensor[..., 0] *= 1920.0
            pred_tensor[..., 1] *= 1080.0
            future[..., 0] *= 1920.0
            future[..., 1] *= 1080.0

            mask = (future[..., 0] != -1920).float()
            loss_mse = masked_mse(pred_tensor, future, mask)
            loss_l2 = masked_l2(pred_tensor, future, mask, clip_thresh)

            total_mse += loss_mse.item()
            total_l2_clipped += loss_l2.item()

    print(f"Eval MSE: {total_mse / len(dataloader):.2f}")
    print(f"Eval L2 (clipped): {total_l2_clipped / len(dataloader):.2f} pixels")

