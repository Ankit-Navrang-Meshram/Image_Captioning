import torch
import clip
from PIL import Image
import os
import pickle
from tqdm import tqdm

def extract_all_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load CLIP
    # ViT-B/32 is a good balance of speed and performance
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 2. Setup Paths
    IMAGE_DIR = "../data/Images"
    OUTPUT_FILE = "features.pkl"
    
    all_features = {}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    
    print(f"Found {len(image_files)} images. Extracting features...")

    # 3. Extraction Loop
    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            try:
                # Preprocess and add batch dimension
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                
                # Encode
                feature = model.encode_image(image)
                
                # Move to CPU and flatten to save RAM
                all_features[img_name] = feature.cpu().squeeze(0)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # 4. Save
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_features, f)
    
    print(f"Done! Features saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_all_features()