import torch
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer
from torch.optim import AdamW
from dataset import get_loader
import os
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --------------------------
    # 1. Model Initialization
    # --------------------------
    # This magic function downloads ViT and GPT2 and connects them
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224", 
        "gpt2"
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Crucial Configuration for GPT-2 Decoder
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tokenizer.eos_token_id 

    model.config.max_length = 50
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3  # Prevents "in . in . in ." loops
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # OPTIONAL: Freeze Encoder (ViT) to save memory/time
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.to(device)
    model.train()

    # --------------------------
    # 2. Setup
    # --------------------------
    dataloader = get_loader(batch_size=32) # Reduce if OOM (Out of Memory)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    
    # --------------------------
    # 3. Training Loop
    # --------------------------
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # HuggingFace models calculate loss internally if 'labels' are passed
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        
        # Using HF save_pretrained is safer and handles config files automatically
        model.save_pretrained(f"checkpoints/vit_gpt_epoch_{epoch+1}")
        tokenizer.save_pretrained(f"checkpoints/vit_gpt_epoch_{epoch+1}")
        print("Checkpoint saved.")

if __name__ == "__main__":
    train()