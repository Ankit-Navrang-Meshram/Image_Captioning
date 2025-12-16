import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import get_loader
from model import CNNtoRNN
import os

def train():
    # ----------------------
    # 1. Hyperparameters
    # ----------------------
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 32
    num_workers = 2
    
    # Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ----------------------
    # 2. Data Setup
    # ----------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Loading Dataset...")
    train_loader, dataset = get_loader(
        root_folder="data/Images",
        annotation_file="data/captions.txt",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    vocab_size = len(dataset.vocab)
    print(f"Vocab Size: {vocab_size}")

    # ----------------------
    # 3. Model & Optimization
    # ----------------------
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Ignore the <PAD> token when calculating loss
    pad_idx = dataset.vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Only optimize decoder parameters and encoder's linear layer (ResNet is frozen)
    # If you enabled train_CNN=True in model.py, pass all params here.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------
    # 4. Training Loop
    # ----------------------
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
        
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward pass
            # The model internally handles shifting input captions
            outputs = model(imgs, captions)

            # Targets:
            # We want the model to predict the *next* token.
            # Due to the architecture (Img -> SOS -> Word1), the outputs align 
            # such that outputs[t] should predict captions[t].
            # Flatten outputs to (batch*seq_len, vocab_size)
            # Flatten targets to (batch*seq_len)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), 
                captions.reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if idx % 100 == 0:
                print(f"Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # ----------------------
        # 5. Save Checkpoint
        # ----------------------
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": dataset.vocab # Optional: save vocab if you want strict reproducibility
        }
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(checkpoint, f"checkpoints/my_checkpoint_epoch_{epoch+1}.pth.tar")
        print("Checkpoint saved!")

if __name__ == "__main__":
    train()