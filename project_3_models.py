import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------- Device ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------- Dataset ------------------- #
class CrowdDataset(Dataset):
    def __init__(self, list_file, annotation_dir,
                 sequence_length=30, input_len=25, pred_len=5, max_objects=150):
        self.annotation_dir = annotation_dir
        self.sequence_length = sequence_length
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.max_objects = max_objects

        with open(list_file, 'r') as f:
            seq_ids = [line.strip() for line in f]

        self.samples = []
        for sid in seq_ids:
            for start in range(0, self.sequence_length - self.total_len + 1):
                self.samples.append((sid, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, start = self.samples[idx]
        ann_path = os.path.join(self.annotation_dir, f"{seq_id}_with_ids.txt")

        # parse annotations
        ann = {}
        with open(ann_path, 'r') as f:
            for line in f:
                fr, tid, x, y = map(float, line.strip().split(',')[:4])
                fr, tid = int(fr), int(tid)
                ann.setdefault(fr, []).append((tid, x, y))

        # build pos tensor [total_len, max_objects, 2]
        pos = torch.full((self.total_len, self.max_objects, 2), -1.0, dtype=torch.float32)
        for i in range(self.total_len):
            fr_i = start + 1 + i
            for tid, x, y in ann.get(fr_i, []):
                if 0 <= tid < self.max_objects:
                    pos[i, tid, 0] = x / 1920.0
                    pos[i, tid, 1] = y / 1080.0

        # input: displacements + absolute
        inp = pos[: self.input_len + 1]                      # [input_len+1, N, 2]
        disp = inp[1:] - inp[:-1]                            # [input_len, N, 2]
        inp_feats = torch.cat([disp, inp[1:]], dim=-1)       # [input_len, N, 4]

        last_pos = inp[-1]                                   # [N, 2]
        future = pos[self.input_len:]                        # [pred_len, N, 2]
        return inp_feats, last_pos, future

ann_dir   = "./annotations_with_ids/"
train_ds  = CrowdDataset('trainlist_copy.txt', ann_dir)
test_ds   = CrowdDataset('testlist_copy.txt',  ann_dir)
train_loader  = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=2)
# ---------- Positional Encoding ---------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B*N, T, D]
        return x + self.pe[:x.size(1)]

# --------- Transformer Model ------------ #
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=heads,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, 2)

    def forward(self, x):  # x: [B, N, T, D]
        B, N, T, D = x.shape
        x = x.reshape(B*N, T, D)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        h = self.norm(x[:, -1, :])
        out = self.fc(h)
        return out.reshape(B, N, 2)

def stillness_penalty(preds, prev_pos, mask, threshold=5.0):
    """
    Adds penalty when previous positions were static but model predicts movement.
    """
    d_pred = torch.norm(preds - prev_pos.unsqueeze(2), dim=-1)  # [B, N, T]
    penalty = torch.clamp(d_pred - threshold, min=0.0)
    static_mask = (mask.sum(-1) > 0).float().unsqueeze(-1).expand_as(penalty)
    return (penalty * static_mask).sum() / static_mask.sum().clamp(min=1)

# --------- Losses ------------------------ #
def masked_mse(pred, tgt, mask):
    loss = (pred - tgt)**2
    loss = loss * mask.unsqueeze(-1)
    return loss.sum() / mask.sum().clamp(min=1)

def masked_l2(pred, tgt, mask, clip=1000):
    d = torch.norm(pred - tgt, dim=-1)
    d = torch.clamp(d, max=clip)
    return (d * mask).sum() / mask.sum().clamp(min=1)

def velocity_consistency_loss(preds, input_disps, mask):
    pred_disps = preds[:, :, 1:] - preds[:, :, :-1]  # [B, N, T-1, 2]
    ref_disp = input_disps[:, :, -1:]                # [B, N, 1, 2]
    ref_disp = ref_disp.expand_as(pred_disps)
    loss = (pred_disps - ref_disp)**2
    return (loss * mask[:, :, 1:].unsqueeze(-1)).sum() / mask[:, :, 1:].sum().clamp(min=1)

def direction_consistency_loss(preds, input_disps, mask):
    pred_vecs = preds[:, :, 1:] - preds[:, :, :-1]
    ref_vecs = input_disps[:, :, -1:].expand_as(pred_vecs)
    pred_norm = pred_vecs / (pred_vecs.norm(dim=-1, keepdim=True) + 1e-6)
    ref_norm  = ref_vecs / (ref_vecs.norm(dim=-1, keepdim=True) + 1e-6)
    cos_sim = (pred_norm * ref_norm).sum(dim=-1)  # [B, N, T-1]
    loss = 1 - cos_sim
    return (loss * mask[:, :, 1:]).sum() / mask[:, :, 1:].sum().clamp(min=1)

def first_step_alignment_loss(preds, input_disps, mask):
    pred_disp = preds[:, :, 0] - preds[:, :, 0].detach()  # placeholder — we’ll use delta directly
    ref_disp = input_disps[:, :, -1]  # last known input displacement
    loss = (pred_disp - ref_disp)**2
    return (loss * mask[:, :, 0].unsqueeze(-1)).sum() / mask[:, :, 0].sum().clamp(min=1)

def angular_loss(preds, last_pos, input_disps, mask):
    pred_vec = preds[:, :, 0] - last_pos  # [B, N, 2]
    ref_vec  = input_disps[:, :, -1]      # [B, N, 2]

    pred_norm = pred_vec / (pred_vec.norm(dim=-1, keepdim=True) + 1e-6)
    ref_norm  = ref_vec / (ref_vec.norm(dim=-1, keepdim=True) + 1e-6)
    cos_sim = (pred_norm * ref_norm).sum(dim=-1)  # [B, N]

    ang_diff = 1 - cos_sim
    return (ang_diff * mask[:, :, 0]).sum() / mask[:, :, 0].sum().clamp(min=1)

def delta_magnitude_regularizer(preds, mask, max_delta=0.01):
    first_disp = preds[:, :, 0] - preds[:, :, 0].detach()
    mag = torch.norm(first_disp, dim=-1)
    excess = torch.clamp(mag - max_delta, min=0.0)
    return (excess * mask[:, :, 0]).sum() / mask[:, :, 0].sum().clamp(min=1)

# --------- Train/Evaluate --------------- #
def train(model, loader, epochs=75, lr=0.001, stillness_weight=0.1):
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        sm, sl = 0.0, 0.0
        for inp, lp, fut in loader:
            seq = inp.permute(0,2,1,3).to(device)   # [B, N, T, 4]
            cur = lp.to(device)                     # [B, N, 2]
            fut = fut.permute(0,2,1,3).to(device)   # [B, N, T, 2]
            preds = []

            for _ in range(fut.size(2)):
                d = model(seq)                      # [B, N, 2]
                cur = cur + d
                preds.append(cur)
                new = torch.cat([d.unsqueeze(2), cur.unsqueeze(2)], dim=-1)
                seq = torch.cat([seq[:,:,1:,:], new], dim=2)

            pt = torch.stack(preds, dim=2)          # [B, N, T, 2]
            pt[...,0]*=1920; pt[...,1]*=1080
            ft = fut.clone(); ft[...,0]*=1920; ft[...,1]*=1080
            mask = (ft[...,0]!=-1920).float()

            mse = masked_mse(pt, ft, mask)
            l2 = masked_l2(pt, ft, mask)
            penalty = stillness_penalty(pt, lp.to(device), mask)
            vel_loss = velocity_consistency_loss(pt, inp[..., :2].permute(0, 2, 1, 3).to(device), mask)
            dir_loss = direction_consistency_loss(pt, inp[..., :2].permute(0, 2, 1, 3).to(device), mask)
            first_step_loss = first_step_alignment_loss(pt, inp[..., :2].permute(0, 2, 1, 3).to(device), mask)
            ang_loss = angular_loss(pt, lp.to(device) * torch.tensor([1920,1080], device=device), inp[..., :2].permute(0, 2, 1, 3).to(device), mask)
            delta_loss = delta_magnitude_regularizer(pt, mask)
            total_loss = mse + stillness_weight * penalty + 0.05 * vel_loss + 0.05 * dir_loss + 0.05*first_step_loss + 0.1*ang_loss + 0.05*delta_loss

            #total_loss = mse + stillness_weight * penalty

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sm += mse.item()
            sl += l2.item()
        print(f"Epoch {ep}/{epochs} — MSE: {sm/len(loader):.3f}, L2: {sl/len(loader):.3f}")


def evaluate(model, loader, stillness_weight=0.1):
    model.eval()
    tm, tl = 0.0, 0.0
    with torch.no_grad():
        for inp, lp, fut in loader:
            seq = inp.permute(0,2,1,3).to(device)
            cur = lp.to(device)
            fut = fut.permute(0,2,1,3).to(device)
            preds = []

            for _ in range(fut.size(2)):
                d = model(seq)
                cur = cur + d
                preds.append(cur)
                new = torch.cat([d.unsqueeze(2), cur.unsqueeze(2)], dim=-1)
                seq = torch.cat([seq[:,:,1:,:], new], dim=2)

            pt = torch.stack(preds, dim=2)
            pt[...,0]*=1920; pt[...,1]*=1080
            ft = fut.clone(); ft[...,0]*=1920; ft[...,1]*=1080
            mask = (ft[...,0]!=-1920).float()

            mse = masked_mse(pt, ft, mask)
            l2 = masked_l2(pt, ft, mask)
            penalty = stillness_penalty(pt, lp.to(device), mask)
            total_loss = mse + stillness_weight * penalty

            tm += mse.item()
            tl += l2.item()
    print(f"Eval — MSE: {tm/len(loader):.3f}, L2: {tl/len(loader):.3f}")
