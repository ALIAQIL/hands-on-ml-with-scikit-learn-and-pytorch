import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt

# ================= DDP init (MUON requires this) =================
if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=0, world_size=1)

from muon import MuonWithAuxAdam

# ================= Data (toy NLP) =================
torch.manual_seed(0)
vocab_size = 5000
seq_len = 32
n_classes = 2
batch_size = 256

X = torch.randint(0, vocab_size, (batch_size, seq_len))
y = torch.randint(0, n_classes, (batch_size,))

# ================= Model =================
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.body = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.body(x)
        x = x.mean(dim=1)
        return self.head(x)

# ================= Metrics =================
def grad_norm(model):
    return torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))

def param_norm(model):
    return torch.sqrt(sum(p.data.norm()**2 for p in model.parameters()))

# ================= Training =================
def train(make_optimizer, steps=150):
    model = TinyTransformer()
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer(model)

    losses, gnorms, pnorms, times = [], [], [], []

    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()

        t0 = time.time()
        opt.step()
        times.append(time.time() - t0)

        losses.append(loss.item())
        gnorms.append(grad_norm(model).item())
        pnorms.append(param_norm(model).item())

    return losses, gnorms, pnorms, times

# ================= Optimizers =================
def adam(m):   return optim.Adam(m.parameters(), lr=3e-4)
def adamw(m):  return optim.AdamW(m.parameters(), lr=3e-4)
def adamax(m): return optim.Adamax(m.parameters(), lr=3e-4)
def nadam(m):  return optim.NAdam(m.parameters(), lr=3e-4)

def muon(m):
    hidden = [p for p in m.body.parameters() if p.ndim >= 2]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for p in m.parameters() if id(p) not in hidden_ids]

    return MuonWithAuxAdam([
        dict(params=hidden, use_muon=True, lr=0.03),
        dict(params=rest,   use_muon=False, lr=3e-4),
    ])

# ================= Run =================
results = {
    "Adam":   train(adam),
    "AdamW":  train(adamw),
    "Adamax": train(adamax),
    "NAdam":  train(nadam),
    "Muon":   train(muon),
}

# ================= Plots =================
def plot(idx, ylabel, log=False):
    plt.figure()
    for name, vals in results.items():
        plt.plot(vals[idx], label=name)
    if log:
        plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot(0, "Loss", log=True)
plot(1, "Gradient norm", log=True)
plot(2, "Parameter norm")
plot(3, "Step time (s)")
