import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt

# ================= DDP init (required by MUON) =================
if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=0, world_size=1)

from muon import MuonWithAuxAdam

# ================= Dataset =================
torch.manual_seed(0)
X = torch.randn(512, 20)
y = torch.randn(512, 1)

# ================= Model =================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.body(x))

# ================= Metrics =================
def grad_norm(model):
    return torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))

def param_norm(model):
    return torch.sqrt(sum(p.data.norm()**2 for p in model.parameters()))

# ================= Training =================
def train(make_optimizer, steps=200):
    model = MLP()
    loss_fn = nn.MSELoss()
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
def adam(m):    return optim.Adam(m.parameters(), lr=1e-3)
def adamw(m):   return optim.AdamW(m.parameters(), lr=1e-3)
def adamax(m):  return optim.Adamax(m.parameters(), lr=1e-3)
def nadam(m):   return optim.NAdam(m.parameters(), lr=1e-3)

def muon(m):
    hidden = [p for p in m.body.parameters() if p.ndim >= 2]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for p in m.parameters() if id(p) not in hidden_ids]

    return MuonWithAuxAdam([
        dict(params=hidden, use_muon=True, lr=0.05),
        dict(params=rest,   use_muon=False, lr=1e-3),
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
    for k, v in results.items():
        plt.plot(v[idx], label=k)
    if log: plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot(0, "Loss", log=True)
plot(1, "Gradient norm", log=True)
plot(2, "Parameter norm")
plot(3, "Step time (s)")
