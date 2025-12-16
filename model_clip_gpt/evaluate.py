import torch
import pandas as pd
import pickle
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer
from model import ClipGPT2Model
from nltk.translate.bleu_score import corpus_bleu
import nltk

# Ensure nltk data is downloaded
nltk.download('punkt', quiet=True)

def evaluate():
    # ----------------------
    # 1. Configuration
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAPTIONS_FILE = "../data/captions.txt"
    FEATURES_FILE = "features.pkl"
    CHECKPOINT_PATH = "checkpoints/clip_gpt_epoch_10.pth" # Update this to your best epoch
    PREFIX_LENGTH = 10
    
    print(f"Loading features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, 'rb') as f:
        all_features = pickle.load(f)

    # ----------------------
    # 2. Load Data & Model
    # ----------------------
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model = ClipGPT2Model(prefix_length=PREFIX_LENGTH).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
    model.eval()

    # Group captions by image
    df = pd.read_csv(CAPTIONS_FILE)
    unique_images = df["image"].unique()
    
    # We'll evaluate on the first 100 images for speed. 
    # Remove [:100] to evaluate on the whole dataset (takes longer).
    test_images = unique_images[:100] 

    references = []
    hypotheses = []

    print(f"Evaluating on {len(test_images)} images...")

    # ----------------------
    # 3. Generation Loop
    # ----------------------
    for img_name in tqdm(test_images):
        # A. Get Ground Truths
        true_captions = df[df["image"] == img_name]["caption"].tolist()
        # Tokenize references (list of list of words)
        refs = [nltk.word_tokenize(str(c).lower()) for c in true_captions]
        references.append(refs)

        # B. Get Feature
        if img_name not in all_features:
            continue
            
        feature = all_features[img_name].float().unsqueeze(0).to(device) # (1, 512)

        # C. Generate Caption (Greedy Search)
        with torch.no_grad():
            # 1. Project CLIP feature to GPT embedding space
            prefix_embed = model.clip_project(feature).view(1, PREFIX_LENGTH, -1)
            
            # 2. Generation Loop
            generated_ids = []
            curr_embeds = prefix_embed
            
            for _ in range(20): # Max length
                outputs = model.gpt(inputs_embeds=curr_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                generated_ids.append(next_token.item())
                
                # Update input for next step
                next_embed = model.gpt.transformer.wte(next_token).unsqueeze(1)
                curr_embeds = torch.cat((curr_embeds, next_embed), dim=1)

        # Decode and tokenize prediction
        caption_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        hypotheses.append(nltk.word_tokenize(caption_text.lower()))

    # ----------------------
    # 4. Calculate Metrics
    # ----------------------
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n-------------------------")
    print(f"Results for CLIP + GPT-2")
    print("-------------------------")
    print(f"BLEU-1: {bleu1 * 100:.2f}")
    print(f"BLEU-4: {bleu4 * 100:.2f}")
    
    # Show Example
    print("\n--- Example ---")
    print(f"Predicted: {' '.join(hypotheses[0])}")
    print(f"Reference: {' '.join(references[0][0])}")

if __name__ == "__main__":
    evaluate()