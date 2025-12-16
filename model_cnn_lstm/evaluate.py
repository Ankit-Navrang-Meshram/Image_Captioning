import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import CNNtoRNN
from dataset import Vocabulary  # Needed to load the vocab class structure
import nltk
from nltk.translate.bleu_score import corpus_bleu

# Download nltk tokenizer data
nltk.download('punkt', quiet=True)

def evaluate():
    # ----------------------
    # 1. Setup
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (Adjust if needed)
    IMAGES_DIR = "data/Images"
    CAPTIONS_FILE = "data/captions.txt"
    CHECKPOINT_PATH = "checkpoints/my_checkpoint_epoch_10.pth.tar" # Use your latest checkpoint
    
    # Transforms (Must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ----------------------
    # 2. Load Model & Vocab
    # ----------------------
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # Re-instantiate Vocabulary from training data
    # (In a real app, you would save/load the vocab object specifically using pickle)
    df = pd.read_csv(CAPTIONS_FILE)
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(df["caption"].tolist())
    
    # Initialize Model
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    vocab_size = len(vocab)
    
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # ----------------------
    # 3. Evaluation Loop
    # ----------------------
    # We will evaluate on the first 100 images for speed.
    # In Flickr8k, images are repeated in the CSV (5 captions per image).
    # We want unique images to generate 1 caption and compare against all 5 references.
    
    unique_images = df["image"].unique()[:100] # Test on 100 images
    
    refs = []       # List of lists of valid captions (ground truth)
    preds = []      # List of generated captions
    
    print("Generating captions...")
    for img_name in unique_images:
        # A. Get all ground truth captions for this image
        valid_captions = df[df["image"] == img_name]["caption"].tolist()
        
        # Tokenize references for BLEU (list of words)
        img_refs = [vocab.tokenizer_eng(c) for c in valid_captions]
        refs.append(img_refs)
        
        # B. Generate Caption
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).to(device)
        
        # Generate
        generated_words = model.caption_image(image_tensor, vocab)
        
        # Remove <SOS> and <EOS> if present (caption_image usually handles this, 
        # but let's be safe and strip special tokens just in case)
        filtered_generated = [w for w in generated_words if w not in ["<SOS>", "<EOS>", "<PAD>"]]
        preds.append(filtered_generated)

    # ----------------------
    # 4. Calculate Scores
    # ----------------------
    # BLEU-4 looks at 4-gram overlap. It is strict!
    bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))
    
    # BLEU-1 looks at 1-gram overlap.
    bleu1 = corpus_bleu(refs, preds, weights=(1.0, 0, 0, 0))
    
    print(f"\n--- Results ---")
    print(f"BLEU-1: {bleu1*100:.2f}")
    print(f"BLEU-4: {bleu4*100:.2f}")
    
    # Show an example
    print("\n--- Example ---")
    print(f"Prediction: {' '.join(preds[0])}")
    print(f"Reference 1: {' '.join(refs[0][0])}")

if __name__ == "__main__":
    evaluate()