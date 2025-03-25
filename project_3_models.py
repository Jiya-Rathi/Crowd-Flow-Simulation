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

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Dataset Preparation --- #
class CrowdDataset(Dataset):
    def __init__(self, data_list, image_dir, annotation_dir, sequence_length=30):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sequence_length = sequence_length
        
        with open(data_list, 'r') as f:
            self.sequence_files = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.sequence_files)
    '''
    def __getitem__(self, idx):
        images = []
        annotations = []
        
        for i in range(self.sequence_length):
            frame_id = self.sequence_files[idx + i]
            img_path = os.path.join(self.image_dir , frame_id, f"{frame_id}.jpg")
            ann_path = os.path.join(self.annotation_dir, f"{frame_id}.txt")
            
            image = read_image(img_path).float() / 255.0  # Normalize to [0,1]
            images.append(image)
            
            with open(ann_path, 'r') as f:
                ann = [list(map(int, line.strip().split(','))) for line in f.readlines()]
                ann = torch.tensor(ann, dtype=torch.float32)
                annotations.append(ann)
        
        images = torch.stack(images)
        return images[:-1], images[-1], annotations[:-1], annotations[-1]
    '''
    def __getitem__(self, idx):
        images = []
        annotations = []
        
        sequence_id = self.sequence_files[idx]  # Get the sequence folder name (e.g., "00001")
        sequence_folder = os.path.join(self.image_dir, sequence_id)  # Path to sequence folder
        
        for frame_num in range(1, self.sequence_length + 1):  # Frames from 1 to 30
            frame_id = f"{frame_num:05d}"  # Ensure zero-padding (e.g., 00001.jpg, 00002.jpg)
            img_path = os.path.join(sequence_folder, f"{frame_id}.jpg")
            ann_path = os.path.join(self.annotation_dir, f"{sequence_id}.txt")
    
            image = read_image(img_path).float() / 255.0  # Normalize image
            images.append(image)
    
            with open(ann_path, 'r') as f:
                ann = [list(map(int, line.strip().split(','))) for line in f.readlines()]
                ann = torch.tensor(ann, dtype=torch.float32)
                annotations.append(ann)
    
        images = torch.stack(images)
        return images[:-1], images[-1], annotations[:-1], annotations[-1]

    
# Dataset Paths
image_dir = "./sequences/" 
annotation_dir = "./annotations/"

train_dataset = CrowdDataset("trainlist_copy.txt", image_dir, annotation_dir)
test_dataset = CrowdDataset("testlist_copy.txt", image_dir, annotation_dir)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# --- 2. Model Architectures --- #

# Convolutional LSTM (ConvLSTM)
class ConvLSTM(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=256):
        super(ConvLSTM, self).__init__()
        
        # ----- ENCODER -----
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Reduce H, W by 2
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce further
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Reduce H, W by 2
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce further
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Reduce H, W by 2
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch, 128, 1, 1)
        )

        # ----- LSTM -----
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=2, batch_first=True)

        # ----- DECODER -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (2,2)
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (4,4)
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (8,8)
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (16,16)
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (32,32)
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (64,64)
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (128,128)
            nn.ReLU(),
            
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (256,256)
            nn.ReLU(),
            
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (512,512)
            nn.ReLU(),
            
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample → (1024,1024)
            nn.ReLU(),
            
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1),  # Final adjustment → (1080, 1920)
            nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True),
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        #print(f"Input shape: {x.shape}")  # Debugging

        x = x.view(batch_size * seq_len, c, h, w)  
        x = self.encoder(x)  
        #print(f"Shape after CNN Encoder: {x.shape}")  # Expected: (batch*seq, 128, 1, 1)

        x = x.view(batch_size, seq_len, -1)  # Flatten for LSTM
        #print(f"Shape before LSTM: {x.shape}")  # Expected: (batch, seq, 128)
        
        x, _ = self.lstm(x)  
        #print(f"Shape after LSTM: {x.shape}")  # Expected: (batch, seq, hidden_dim)
        
        x = x[:, -1, :].view(batch_size, -1, 1, 1)  # Take last sequence output
        x = self.decoder(x)

        return x

# Social LSTM (S-LSTM)
class SocialLSTM(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128, hidden_dim=256):
        super(SocialLSTM, self).__init__()

        # CNN Encoder: turns each frame into a feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, feature_dim, 1, 1]
        )

        # LSTM operates on encoded features
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)

        # Final decoder to image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape

        # Reshape: [B*T, C, H, W] → CNN Encoder
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.encoder(x)                          # [B*T, F, 1, 1]
        x = x.view(batch_size, seq_len, -1)          # [B, T, F]

        x, _ = self.lstm(x)                           # [B, T, hidden_dim]
        x = self.fc(x[:, -1, :])                      # [B, feature_dim]
        x = x.view(batch_size, -1, 1, 1)              # [B, feature_dim, 1, 1]
        x = self.decoder(x)                           # [B, 3, 1080, 1920]
        return x


# Temporal Convolutional Neural Network (TCNN)
class TCNN(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128, hidden_dim=256):
        super(TCNN, self).__init__()
        
        # --- CNN Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, H/4, W/4]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, H/8, W/8]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, H/16, W/16]
            
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),  # [B, feature_dim, H/32, W/32]
            nn.AdaptiveAvgPool2d((1, 1))  # [B, feature_dim, 1, 1]
        )

        # --- Temporal Conv (1D) ---
        self.tcn = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, feature_dim)

        # --- Simple Decoder back to image (optional or minimal) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # [B, 3, 8, 8] (or upsample more)
            nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, C, H, W]
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size * seq_len, c, h, w)     # [B*T, C, H, W]
        x = self.encoder(x)                           # [B*T, F, 1, 1]
        x = x.view(batch_size, seq_len, -1)           # [B, T, F]
        x = x.permute(0, 2, 1)                         # [B, F, T]
        x = self.tcn(x)                                # [B, hidden_dim, T]
        x = self.fc(x[:, :, -1])                       # [B, feature_dim]

        x = x.view(batch_size, -1, 1, 1)               # [B, F, 1, 1]
        x = self.decoder(x)                            # [B, 3, 1080, 1920]
        return x



# --- 3. Training Pipeline --- #
def train(model, train_loader, epochs=50):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

# --- 4. Testing Pipeline --- #
def evaluate(model, test_loader):
    model.eval()
    total_loss, total_ssim, total_psnr = 0, 0, 0
    with torch.no_grad():
        for inputs, targets, _, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse = nn.MSELoss()(outputs, targets).item()
            
            targets_np = targets.squeeze(0).permute(1, 2, 0).cpu().numpy()
            outputs_np = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
            H, W = targets.shape[-2:]  
            win_size = min(7, H, W) if min(H, W) >= 7 else max(3, min(H, W))  # Ensure valid win_size
            if win_size % 2 == 0:
                win_size -= 1  # Ensure win_size is odd
    
            ssim_val = ssim(targets_np, outputs_np, 
                            data_range=outputs_np.max() - outputs_np.min(), 
                            win_size=win_size, channel_axis=-1)
    
            psnr_val = cv2.PSNR(targets_np, outputs_np)
    
            total_loss += mse
            total_ssim += ssim_val
            total_psnr += psnr_val
    print(f"Test Loss: {total_loss / len(test_loader)}")
    print(f"Test SSIM: {total_ssim / len(test_loader)}")
    print(f"Test PSNR: {total_psnr / len(test_loader)}")
