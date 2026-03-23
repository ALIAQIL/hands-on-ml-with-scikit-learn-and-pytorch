"""
Fully Convolutional Network (FCN) - PyTorch Implementation
===========================================================
This implements a simple FCN for semantic segmentation.
We train it on a toy dataset (colored shapes) so you can run it
without downloading any large datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# 1.  TOY DATASET  (colored shapes on black bg)
# ─────────────────────────────────────────────
# Classes:  0 = background,  1 = rectangle,  2 = circle

def make_sample(img_size=128):
    """Generate one (image, mask) pair with random shapes."""
    img  = np.zeros((3, img_size, img_size), dtype=np.float32)
    mask = np.zeros((img_size, img_size),    dtype=np.int64)

    # ── random rectangle (class 1, red-ish) ──────────────────────
    x1, y1 = np.random.randint(5, 50, size=2)
    x2, y2 = x1 + np.random.randint(20, 60), y1 + np.random.randint(20, 60)
    x2, y2 = min(x2, img_size-1), min(y2, img_size-1)
    img[0, y1:y2, x1:x2] = 0.9   # red channel
    img[1, y1:y2, x1:x2] = 0.2
    mask[y1:y2, x1:x2]   = 1

    # ── random circle (class 2, blue-ish) ────────────────────────
    cx, cy = np.random.randint(60, 110, size=2)
    r      = np.random.randint(15, 35)
    yy, xx = np.ogrid[:img_size, :img_size]
    circle = (xx - cx)**2 + (yy - cy)**2 <= r**2
    img[2, circle] = 0.9          # blue channel
    img[0, circle] = 0.1
    mask[circle]   = 2

    return torch.tensor(img), torch.tensor(mask)


def make_batch(batch_size=8, img_size=128):
    imgs, masks = zip(*[make_sample(img_size) for _ in range(batch_size)])
    return torch.stack(imgs), torch.stack(masks)


# ─────────────────────────────────────────────
# 2.  FCN MODEL
# ─────────────────────────────────────────────

class FCN(nn.Module):
    """
    Simple FCN:
      Encoder  →  3 conv blocks that shrink the spatial size
      Decoder  →  transposed convolutions that bring it back
      Skip connections combine encoder + decoder features
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # ── Encoder (Downsampling) ────────────────────────────────
        self.enc1 = self._block(3,   32)   # 128 → 64
        self.enc2 = self._block(32,  64)   # 64  → 32
        self.enc3 = self._block(64, 128)   # 32  → 16

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = self._block(128, 256)  # stays 16×16

        # ── Decoder (Upsampling) ─────────────────────────────────
        # Each upconv doubles spatial size; channel count = enc + upconv
        self.up3    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3   = self._block(256, 128)   # 16 → 32  (128 skip + 128 up)

        self.up2    = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec2   = self._block(128, 64)    # 32 → 64  (64 skip + 64 up)

        self.up1    = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        self.dec1   = self._block(64,  32)    # 64 → 128 (32 skip + 32 up)

        # ── Final 1×1 conv → per-pixel class scores ──────────────
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────────
        e1 = self.enc1(x)           # [B,  32, 128, 128]
        e2 = self.enc2(self.pool(e1))  # [B,  64,  64,  64]
        e3 = self.enc3(self.pool(e2))  # [B, 128,  32,  32]

        # ── Bottleneck ───────────────────────────────────────────
        b  = self.bottleneck(self.pool(e3))  # [B, 256,  16,  16]

        # ── Decoder + Skip Connections ───────────────────────────
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))  # [B,128,32,32]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 64,64,64]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32,128,128]

        return self.head(d1)   # [B, num_classes, 128, 128]


# ─────────────────────────────────────────────
# 3.  TRAINING
# ─────────────────────────────────────────────

def train(num_epochs=20, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model     = FCN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        imgs, masks = make_batch(batch_size)
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)          # [B, 3, 128, 128]
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{num_epochs}  |  Loss: {loss.item():.4f}")

    print("\nTraining complete!")
    return model, losses


# ─────────────────────────────────────────────
# 4.  VISUALISE RESULTS
# ─────────────────────────────────────────────

def visualise(model, losses, n_samples=3):
    device = next(model.parameters()).device
    model.eval()

    fig = plt.figure(figsize=(14, 4 * (n_samples + 1)))
    fig.patch.set_facecolor("#0f0f1a")

    color_map = np.array([
        [15,  15,  26],   # 0 background – dark navy
        [220, 60,  60],   # 1 rectangle  – red
        [60,  120, 220],  # 2 circle     – blue
    ], dtype=np.uint8)

    # ── sample rows ──────────────────────────────────────────────
    with torch.no_grad():
        for row in range(n_samples):
            img, mask = make_sample()
            pred = model(img.unsqueeze(0).to(device))
            pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy()

            img_np  = img.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()

            for col, (data, title) in enumerate([
                (img_np,              "Input Image"),
                (color_map[mask_np],  "Ground Truth"),
                (color_map[pred_mask],"Prediction"),
            ]):
                ax = fig.add_subplot(n_samples + 1, 3, row * 3 + col + 1)
                ax.imshow(data if col == 0 else data)
                ax.set_title(title, color="white", fontsize=10, pad=4)
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")

    # ── loss curve ───────────────────────────────────────────────
    ax_loss = fig.add_subplot(n_samples + 1, 1, n_samples + 1)
    ax_loss.set_facecolor("#16162a")
    ax_loss.plot(losses, color="#5599ff", linewidth=2)
    ax_loss.set_title("Training Loss", color="white", fontsize=11)
    ax_loss.set_xlabel("Epoch", color="#aaaacc")
    ax_loss.set_ylabel("Cross-Entropy Loss", color="#aaaacc")
    ax_loss.tick_params(colors="#aaaacc")
    for spine in ax_loss.spines.values():
        spine.set_edgecolor("#333355")

    # ── legend ───────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=np.array([15, 15, 26])/255,  label="Background"),
        mpatches.Patch(color=np.array([220, 60, 60])/255, label="Rectangle (class 1)"),
        mpatches.Patch(color=np.array([60, 120, 220])/255,label="Circle (class 2)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               facecolor="#16162a", labelcolor="white",
               edgecolor="#333355", fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("FCN — Semantic Segmentation on Toy Dataset",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/fcn_results.png",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved visualisation → fcn_results.png")


# ─────────────────────────────────────────────
# 5.  RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model, losses = train(num_epochs=30, batch_size=8, lr=1e-3)
    visualise(model, losses, n_samples=3)