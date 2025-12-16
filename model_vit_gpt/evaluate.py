import torch
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu
import nltk

nltk.download('punkt', quiet=True)

def evaluate():
    # ----------------------
    # 1. Configuration
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAPTIONS_FILE = "../data/captions.txt"
    IMAGES_DIR = "../data/Images"
    # Point this to your saved checkpoint folder
    CHECKPOINT_DIR = "checkpoints/vit_gpt_epoch_5" 

    print("Loading Model & Processors...")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR).to(device)
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except OSError:
        print(f"Error: Could not find checkpoint at {CHECKPOINT_DIR}")
        print("Please check the folder name in your 'checkpoints' directory.")
        return

    # ----------------------
    # 2. Setup Evaluation
    # ----------------------
    df = pd.read_csv(CAPTIONS_FILE)
    unique_images = df["image"].unique()
    test_images = unique_images[:100] # Test on first 100 images

    references = []
    hypotheses = []

    print(f"Evaluating on {len(test_images)} images...")

    # ----------------------
    # 3. Loop
    # ----------------------
    for img_name in tqdm(test_images):
        # A. References
        true_captions = df[df["image"] == img_name]["caption"].tolist()
        refs = [nltk.word_tokenize(str(c).lower()) for c in true_captions]
        references.append(refs)

        # B. Load Image
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

        # C. Generate (Beam Search)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=50,
                num_beams=4, # Beam search size 4
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # D. Decode
        caption_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        hypotheses.append(nltk.word_tokenize(caption_text.lower()))

    # ----------------------
    # 4. Metrics
    # ----------------------
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print("\n-------------------------")
    print(f"Results for ViT + GPT-2")
    print("-------------------------")
    print(f"BLEU-1: {bleu1 * 100:.2f}")
    print(f"BLEU-4: {bleu4 * 100:.2f}")

    print("\n--- Example ---")
    print(f"Predicted: {' '.join(hypotheses[0])}")
    print(f"Reference: {' '.join(references[0][0])}")

if __name__ == "__main__":
    evaluate()