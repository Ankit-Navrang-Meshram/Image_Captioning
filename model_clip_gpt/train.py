import torch
import torch.nn as nn
from torch.optim import AdamW
from dataset import get_loader
from model import ClipGPT2Model
import os
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config
    epochs = 10
    batch_size = 32
    prefix_length = 10
    lr = 2e-5 # GPT-2 requires smaller learning rates
    
    print("Loading Data...")
    dataloader = get_loader(batch_size)
    
    print("Initializing Model...")
    model = ClipGPT2Model(prefix_length=prefix_length).to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=50256) # 50256 is GPT2 <EOS> used as PAD
    
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for features, tokens, mask in progress:
            features = features.to(device).float()
            tokens = tokens.to(device)
            
            # Forward
            logits = model(features, tokens)
            
            # --- Alignment Logic ---
            # Inputs:  [Prefix (10), Word1, Word2, Word3, <PAD>]
            # Logits:  [Pred1,       Pred2, Pred3, Pred4,  ...]
            # Targets: [Word1,       Word2, Word3, <EOS>,  ...]
            
            # We want to predict 'tokens' using the previous steps.
            # The logits corresponding to the Prefix predict the first word of the caption.
            
            # Get logits for the caption part (exclude the last prediction)
            # We take from index (prefix_length - 1) up to (total_len - 1)
            # This aligns the prefix prediction with the first word
            caption_logits = logits[:, prefix_length-1 : -1, :]
            
            # Flatten for loss
            loss = criterion(
                caption_logits.reshape(-1, caption_logits.shape[-1]), 
                tokens.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress.set_postfix({"loss": loss.item()})
            
        # Save
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(model.state_dict(), f"checkpoints/clip_gpt_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()