from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io
import pickle
import clip
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer

# Import custom classes
from model_defs import CNNtoRNN, ClipGPT2Model

app = FastAPI()

# ---------------------------------------------------
# 1. LOAD MODELS (Global Variables)
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
models = {}

# Paths
WEIGHTS_DIR = "weights"
VOCAB_PATH = "vocab.pkl"

# -- Load Helpers --
print("Loading Vocabulary...")
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# -- Load Model A (CNN-LSTM) --
print("Loading Model A (CNN-LSTM)...")
model_a = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(vocab), num_layers=1).to(device)
checkpoint_a = torch.load(f"{WEIGHTS_DIR}/model_a.pth", map_location=device, weights_only=False)
model_a.load_state_dict(checkpoint_a["state_dict"])
model_a.eval()
models['A'] = model_a

# -- Load Model B (CLIP-GPT) --
print("Loading Model B (CLIP-GPT)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
model_b = ClipGPT2Model(prefix_length=10).to(device)
checkpoint_b = torch.load(f"{WEIGHTS_DIR}/model_b.pth", map_location=device, weights_only=False)
model_b.load_state_dict(checkpoint_b)
model_b.eval()
models['B'] = model_b
tokenizer_b = GPT2Tokenizer.from_pretrained("gpt2")

# -- Load Model C (ViT-GPT) --
print("Loading Model C (ViT-GPT)...")
model_c = VisionEncoderDecoderModel.from_pretrained(f"{WEIGHTS_DIR}/model_c_hf").to(device)
feature_extractor_c = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
tokenizer_c = GPT2Tokenizer.from_pretrained("gpt2")
models['C'] = model_c

# ---------------------------------------------------
# 2. PREDICTION ENDPOINT
# ---------------------------------------------------
@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...)):
    # Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = {}

    # --- Run Model A ---
    transform_a = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_a = transform_a(image).unsqueeze(0).to(device)
    caption_a = model_a.caption_image(img_a.squeeze(0), vocab)
    results["CNN_LSTM"] = " ".join(caption_a).replace("<SOS>", "").replace("<EOS>", "")

    # --- Run Model B ---
    img_b = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features_b = clip_model.encode_image(img_b).float()
        prefix_embed = model_b.clip_project(features_b).view(1, 10, -1)
        
        generated = []
        curr_embeds = prefix_embed
        for _ in range(20):
            out = model_b.gpt(inputs_embeds=curr_embeds)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            if next_token.item() == tokenizer_b.eos_token_id: break
            generated.append(next_token.item())
            curr_embeds = torch.cat((curr_embeds, model_b.gpt.transformer.wte(next_token).unsqueeze(1)), dim=1)
    results["CLIP_GPT"] = tokenizer_b.decode(generated)

    # --- Run Model C ---
    pixel_values = feature_extractor_c(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model_c.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    results["ViT_GPT"] = tokenizer_c.decode(output_ids[0], skip_special_tokens=True)

    return results