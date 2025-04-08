import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from project_3_models import CrowdDataset
from predict_future import BBoxLSTM
import os

# -------- Setup -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- Dataset -------- #
image_dir = "./sequences/"
annotation_dir = "./annotations/"
train_dataset = CrowdDataset("trainlist_copy.txt", image_dir, annotation_dir)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# -------- Model -------- #
model = BBoxLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -------- Training Config -------- #
EPOCHS = 50
input_len = 20
pred_len = 10

print("Starting BBoxLSTM training (20 → 10)...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for _, _, ann_seq, _ in train_loader:
        ann_seq = ann_seq[0]                          # [30, N, 3]
        bboxes = ann_seq[:, :, 1:3]                   # [30, N, 2]
        bboxes = bboxes.permute(1, 0, 2)              # [N, 30, 2]

        for track in bboxes:
            if torch.isnan(track).any():
                continue

            # Use first 20 → predict next 10
            inputs = track[:input_len].unsqueeze(0).to(device)   # [1, 20, 2]
            targets = track[input_len:input_len+pred_len].to(device)  # [10, 2]

            if targets.shape[0] < pred_len:
                continue  # skip incomplete sequences

            # Autoregressive prediction
            preds = []
            seq = inputs.clone()
            for _ in range(pred_len):
                out = model(seq)                   # [1, 2]
                preds.append(out.squeeze(0))
                out = out.unsqueeze(1)             # [1, 1, 2]
                seq = torch.cat((seq[:, 1:], out), dim=1)  # roll forward

            pred_tensor = torch.stack(preds, dim=0)  # [10, 2]
            loss = criterion(pred_tensor, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

# -------- Save Model -------- #
save_path = "bbox_lstm_checkpoint.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved trained BBoxLSTM to {save_path}")
