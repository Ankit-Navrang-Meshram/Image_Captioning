import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
from PIL import Image

def predict_caption(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model & Processors
    # Note: We load from the folder where we saved the checkpoint
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. Process Image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    # 3. Generate
    # The model has a built-in .generate() method (Beam Search is automatic!)
    generated_ids = model.generate(
        pixel_values, 
        max_length=50, 
        num_beams=4, 
        early_stopping=True
    )
    
    # 4. Decode
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nImage: {image_path}")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    # Point to the checkpoint folder created by train.py
    # e.g., checkpoints/vit_gpt_epoch_5
    predict_caption(
        "../data/Images/1000268201_693b08cb0e.jpg", 
        "checkpoints/vit_gpt_epoch_5" 
    )