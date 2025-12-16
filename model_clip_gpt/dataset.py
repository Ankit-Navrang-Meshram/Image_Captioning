import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from transformers import GPT2Tokenizer
import os

class FlickrClipDataset(Dataset):
    def __init__(self, captions_file, features_file, prefix_length=10):
        # Load the features we saved earlier
        with open(features_file, 'rb') as f:
            self.features = pickle.load(f)
            
        # Load Captions
        self.df = pd.read_csv(captions_file)
        
        # Initialize GPT-2 Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # GPT2 doesn't have a pad token, so we use the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        self.prefix_length = prefix_length
        self.captions = self.df["caption"].tolist()
        self.images = self.df["image"].tolist()
        
        # Calculate max length for padding (simple heuristic or fixed)
        self.max_len = 40 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = str(self.captions[idx])
        
        # 1. Get Image Feature
        # Shape: (512,)
        feature = self.features[img_name] 
        
        # 2. Tokenize Caption
        # We add a space prefix because GPT2 is sensitive to leading spaces
        inputs = self.tokenizer(
            " " + caption, 
            return_tensors="pt", 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        return feature, input_ids, attention_mask

def get_loader(batch_size=32):
    dataset = FlickrClipDataset(
        captions_file="../data/captions.txt",
        features_file="features.pkl"
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Test code
if __name__ == "__main__":
    loader = get_loader()
    feat, ids, mask = next(iter(loader))
    print(f"Feature Shape: {feat.shape}") # [32, 512]
    print(f"Token Shape: {ids.shape}")    # [32, 40]