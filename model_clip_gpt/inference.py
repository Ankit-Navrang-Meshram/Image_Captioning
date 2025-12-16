import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from model import ClipGPT2Model
import os

def generate_caption(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load CLIP (just for feature extraction on the single image)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 2. Load Model
    model = ClipGPT2Model(prefix_length=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 3. Process Image
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = clip_model.encode_image(img).float()
    
    # 4. Generate
    # We feed the image prefix, then generate word by word
    generated_sequence = []
    
    with torch.no_grad():
        # Get the prefix embedding
        prefix_embed = model.clip_project(feature).view(1, 10, -1)
        
        # Start the loop
        current_embeds = prefix_embed
        
        for i in range(20): # Max length
            # Pass embeddings to GPT-2
            outputs = model.gpt(inputs_embeds=current_embeds)
            
            # Predict next token (Greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_sequence.append(next_token.item())
            
            # Append next token embedding for next step
            next_embed = model.gpt.transformer.wte(next_token).unsqueeze(1)
            current_embeds = torch.cat((current_embeds, next_embed), dim=1)

    caption = tokenizer.decode(generated_sequence)
    print(f"Caption: {caption}")

if __name__ == "__main__":
    # Change path to your image and trained checkpoint
    generate_caption(
        "../data/Images/1000268201_693b08cb0e.jpg", 
        "checkpoints/clip_gpt_epoch_10.pth"
    )