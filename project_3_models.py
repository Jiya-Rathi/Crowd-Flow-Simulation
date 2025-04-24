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

class CrowdDataset(Dataset):
    def __init__(self, data_list, image_dir, annotation_dir, sequence_length=30, save_bbox_dir="./bbox_sequences"):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.sequence_length = sequence_length
        self.save_bbox_dir = save_bbox_dir
        
        with open(data_list, 'r') as f:
            self.sequence_files = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.sequence_files)

    def draw_bboxes_on_frame(self, image_tensor, centers, box_size=20):
        image = to_pil_image(image_tensor)
        draw = ImageDraw.Draw(image)
        for x, y in centers:
            x1, y1 = x - box_size // 2, y - box_size // 2
            x2, y2 = x + box_size // 2, y + box_size // 2
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        return image

    def extract_background(self, image_tensor, centers, box_size=20):
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

    def __getitem__(self, idx):
        images = []
        annotations = []
        
        sequence_id = self.sequence_files[idx]
        sequence_folder = os.path.join(self.image_dir, sequence_id)
        ann_path = os.path.join(self.annotation_dir, f"{sequence_id}.txt")

        # Load and parse annotation file once
        with open(ann_path, 'r') as f:
            ann_lines = [list(map(int, line.strip().split(','))) for line in f.readlines()]
        ann_tensor = torch.tensor(ann_lines, dtype=torch.float32)

        bg_path = f"backgrounds/{sequence_id}.jpg"
        if not os.path.exists(bg_path):
            frame_id = f"{1:05d}"
            img_path = os.path.join(sequence_folder, f"{frame_id}.jpg")
            image = read_image(img_path).float() / 255.0
            frame1_annotations = [(x, y) for f, x, y in ann_lines if f == 1]
            inpainted_bg = self.extract_background(image, frame1_annotations)
            os.makedirs("backgrounds", exist_ok=True)
            cv2.imwrite(bg_path, inpainted_bg[:, :, ::-1])  # RGB to BGR for OpenCV

        for frame_num in range(1, self.sequence_length + 1):
            frame_id = f"{frame_num:05d}"
            img_path = os.path.join(sequence_folder, f"{frame_id}.jpg")
            image = read_image(img_path).float() / 255.0
            images.append(image)

            # Get (x, y) for this frame
            frame_annotations = [(x, y) for f, x, y in ann_lines if f == frame_num]
            annotations.append(torch.tensor(frame_annotations, dtype=torch.float32))

            # Save image with bbox if not already done
            save_path = os.path.join(self.save_bbox_dir, sequence_id)
            os.makedirs(save_path, exist_ok=True)
            bbox_path = os.path.join(save_path, f"{frame_id}.jpg")

            if not os.path.exists(bbox_path):  # avoid re-saving
                img_with_bbox = self.draw_bboxes_on_frame(image, frame_annotations)
                img_with_bbox.save(bbox_path)

        images = torch.stack(images)
        return images[:25], images[25:], annotations[:25], annotations[25:]


    
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



def train(model, train_loader, epochs=100):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        epoch_loss = 0

        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)            # [1, 20, 3, H, W]
            targets = targets.to(device)          # [1, 10, 3, H, W]

            optimizer.zero_grad()

            # Autoregressive 10-step prediction
            preds = []
            seq = inputs.clone()
            for _ in range(targets.shape[1]):     # 10 steps
                out = model(seq)                  # [1, 3, H, W]
                preds.append(out)
                seq = torch.cat((seq[:, 1:], out.unsqueeze(1)), dim=1)

            pred_tensor = torch.stack(preds, dim=1)  # [1, 10, 3, H, W]
            loss = criterion(pred_tensor, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")


# --- 4. Testing Pipeline --- #
def evaluate(model, test_loader):
    model.eval()
    total_loss, total_ssim, total_psnr = 0, 0, 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets, _, _ in test_loader:
            inputs = inputs.to(device)   # [1, 20, 3, H, W]
            targets = targets.to(device) # [1, 10, 3, H, W]

            preds = []
            seq = inputs.clone()
            for _ in range(targets.shape[1]):
                out = model(seq)                  # [1, 3, H, W]
                preds.append(out)
                seq = torch.cat((seq[:, 1:], out.unsqueeze(1)), dim=1)

            pred_tensor = torch.stack(preds, dim=1)   # [1, 10, 3, H, W]

            # Compute loss
            loss = criterion(pred_tensor, targets)
            total_loss += loss.item()

            # Compute SSIM/PSNR for each frame pair
            for i in range(targets.shape[1]):
                t_np = targets[0, i].permute(1, 2, 0).cpu().numpy()
                p_np = pred_tensor[0, i].permute(1, 2, 0).cpu().numpy()

                # SSIM
                from skimage.metrics import structural_similarity as ssim
                import cv2

                win_size = min(7, t_np.shape[0], t_np.shape[1])
                if win_size % 2 == 0:
                    win_size -= 1
                win_size = max(3, win_size)

                ssim_val = ssim(t_np, p_np, data_range=1.0, win_size=win_size, channel_axis=-1)
                psnr_val = cv2.PSNR(t_np, p_np)

                total_ssim += ssim_val
                total_psnr += psnr_val

    num_samples = len(test_loader.dataset) * targets.shape[1]

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Test SSIM: {total_ssim / num_samples:.4f}")
    print(f"Test PSNR: {total_psnr / num_samples:.2f} dB")

