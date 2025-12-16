import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, ViTImageProcessor

class FlickrVitDataset(Dataset):
    def __init__(self, root_dir, captions_file, max_length=50): # Increased max_length slightly
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.max_length = max_length
        
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image"]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        except:
            # Fallback for corrupted images (rare but possible)
            pixel_values = torch.zeros(3, 224, 224)

        caption = str(self.df.iloc[idx]["caption"])
        
        # --- FIX IS HERE: Append EOS Token ---
        caption = caption + self.tokenizer.eos_token
        # -------------------------------------
        
        tokenized = self.tokenizer(
            caption, 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tokenized.input_ids.squeeze(0).clone()
        # mask padding so we don't calculate loss on it
        labels[labels == self.tokenizer.pad_token_id] = -100 
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def get_loader(batch_size=16):
    dataset = FlickrVitDataset(
        root_dir="../data/Images",
        captions_file="../data/captions.txt"
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Test
if __name__ == "__main__":
    loader = get_loader(batch_size=4)
    batch = next(iter(loader))
    print("Pixel shape:", batch["pixel_values"].shape) # [4, 3, 224, 224]
    print("Labels shape:", batch["labels"].shape)      # [4, 32]