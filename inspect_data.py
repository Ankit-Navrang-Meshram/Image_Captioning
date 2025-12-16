import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")

def inspect_dataset():
    # 1. Load captions
    df = pd.read_csv(CAPTIONS_FILE)
    print(f"Total captions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 2. Pick a random sample
    sample = df.sample(1).iloc[0]
    image_name = sample['image']
    caption = sample['caption']
    
    # 3. Construct full image path
    image_path = os.path.join(IMAGES_DIR, image_name)
    
    # 4. Display
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Caption: {caption}")
        plt.axis("off")
        plt.show()
        print(f"SUCCESS: Loaded {image_name}")
    except Exception as e:
        print(f"ERROR: Could not load image. Check paths.\n{e}")

if __name__ == "__main__":
    inspect_dataset()