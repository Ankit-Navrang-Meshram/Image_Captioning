import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

# Import your model and vocabulary definition
from model import CNNtoRNN
from dataset import Vocabulary  # Required to recreate the vocab class

def load_checkpoint(checkpoint_path, model, device):
    print(f"=> Loading checkpoint from {checkpoint_path}")
    # Fix: Add weights_only=False to allow loading custom classes like Vocabulary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def main():
    # ------------------------------------------------------
    # 1. Configuration (Must match your training script!)
    # ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    
    # Paths
    # Replace this with your specific image path
    TEST_IMAGE_PATH = "data/Images/3637013_c675de7705.jpg"
    CHECKPOINT_PATH = "checkpoints/my_checkpoint_epoch_10.pth.tar"
    CAPTIONS_FILE = "data/captions.txt"

    # ------------------------------------------------------
    # 2. Rebuild Vocabulary
    # ------------------------------------------------------
    # We need the exact same mapping (Word -> Int) as training.
    # Since we didn't save a vocab.pkl, we rebuild it from the CSV.
    print("Rebuilding vocabulary from source...")
    df = pd.read_csv(CAPTIONS_FILE)
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(df["caption"].tolist())
    vocab_size = len(vocab)
    print(f"Vocabulary loaded. Size: {vocab_size}")

    # ------------------------------------------------------
    # 3. Initialize and Load Model
    # ------------------------------------------------------
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model = load_checkpoint(CHECKPOINT_PATH, model, device)
    model.eval()

    # ------------------------------------------------------
    # 4. Process Image
    # ------------------------------------------------------
    # Transforms must be identical to training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    try:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        image_tensor = transform(image).unsqueeze(0) # Add batch dimension: (1, 3, 224, 224)
        image_tensor = image_tensor.to(device)
        
        # ------------------------------------------------------
        # 5. Generate Caption
        # ------------------------------------------------------
        print("\nGenerating caption...")
        with torch.no_grad():
            caption = model.caption_image(image_tensor.squeeze(0), vocab)
            
        # Join words and clean up
        sentence = " ".join(caption)
        sentence = sentence.replace("<SOS>", "").replace("<EOS>", "").strip()
        
        print(f"\nOUTPUT: {sentence}")

        # ------------------------------------------------------
        # 6. Visualize
        # ------------------------------------------------------
        plt.imshow(image)
        plt.title(sentence)
        plt.axis("off")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find image at {TEST_IMAGE_PATH}")

if __name__ == "__main__":
    main()