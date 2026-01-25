import os, time, math, pickle, torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader, IterableDataset

# --- 1. SETUP ---
if not dist.is_initialized():
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29500"
    dist.init_process_group("gloo", rank=0, world_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# --- 2. WRAPPER FOR STREAMING DATA ---
class StreamingWikiDataset(IterableDataset):
    def __init__(self, split="train", seq_len=256):
        self.split = split
        self.seq_len = seq_len
        # Load the streaming dataset
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)

    def __iter__(self):
        for ex in self.dataset:
            # Skip empty lines or very short headers
            if len(ex["text"]) < 100:
                continue
            
            tokens = tokenizer(
                ex["text"], 
                truncation=True, 
                max_length=self.seq_len, 
                padding="max_length", 
                return_tensors="pt"
            )
            yield tokens["input_ids"].squeeze(0)

def get_loader(split="train", batch_size=4, seq_len=256):
    dataset = StreamingWikiDataset(split=split, seq_len=seq_len)
    # Important: Do not use shuffle=True for IterableDataset in the DataLoader;
    # shuffle logic should be handled within the Dataset itself if needed.
    return DataLoader(dataset, batch_size=batch_size)

# --- 3. OPTIMIZER BUILDERS ---
def get_muon_opt(model):
    hidden = [p for p in model.transformer.h.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if id(p) not in {id(lp) for lp in hidden}]
    return MuonWithAuxAdam([
        {"params": hidden, "use_muon": True, "lr": 0.02}, 
        {"params": rest, "use_muon": False, "lr": 6e-4}, 
    ])

def get_adamw_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))

# --- 4. FULL BENCHMARK SUITE ---
def run_full_benchmark(steps=300):
    results = {}
    
    for name, opt_factory in [("AdamW", get_adamw_opt), ("Muon-Hybrid", get_muon_opt)]:
        print(f"\n[PHASE 1] Training {name} on WikiText-103...")
        config = GPT2Config(vocab_size=len(tokenizer), n_layer=4, n_head=4, n_embd=256)
        model = GPT2LMHeadModel(config).to(device)
        optimizer = opt_factory(model)
        
        # Reset loader for each optimizer comparison
        loader = iter(get_loader(split="train"))
        
        history = []
        for i in range(steps):
            try:
                batch = next(loader).to(device)
            except StopIteration:
                # If we run out of data before steps are done (unlikely with Wiki-103)
                break
                
            loss = model(batch, labels=batch).loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history.append(loss.item())
            if i % 50 == 0:
                print(f"Step {i}/{steps} | Loss: {loss.item():.4f}")
        
        # Save results and weight state
        results[name] = history
        save_path = f"model_{name.lower().replace('-', '_')}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model.state_dict(), f)
        print(f"✓ Weight state saved to {save_path}")

    # --- 5. PLOTTING ---
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.title("Real Data Training Comparison (WikiText-103)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("optimizer_loss_plot.png")
    print("\n✓ Plot saved to 'optimizer_loss_plot.png'")
    
    return results

# --- 6. VALIDATION (PERPLEXITY) ---
def validate_saved_models():
    print("\n[PHASE 2] Validating Models for Perplexity (PPL)...")
    
    for opt_name in ["adamw", "muon_hybrid"]:
        filename = f"model_{opt_name}.pkl"
        if not os.path.exists(filename):
            continue

        config = GPT2Config(vocab_size=len(tokenizer), n_layer=4, n_head=4, n_embd=256)
        model = GPT2LMHeadModel(config).to(device)
        
        with open(filename, "rb") as f:
            model.load_state_dict(pickle.load(f))
        
        model.eval()
        val_loader = iter(get_loader(split="validation", batch_size=8))
        total_loss = 0
        val_steps = 50
        
        with torch.no_grad():
            for _ in range(val_steps):
                try:
                    batch = next(val_loader).to(device)
                    total_loss += model(batch, labels=batch).loss.item()
                except StopIteration:
                    break
        
        avg_loss = total_loss / val_steps
        ppl = math.exp(avg_loss)
        print(f"-> {opt_name.upper()}: Val Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")

# --- EXECUTION ---
if __name__ == "__main__":
    results = run_full_benchmark(steps=300)
    validate_saved_models()