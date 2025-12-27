
---

```markdown
# 📸 Comparative Image Captioning System

A full-stack Deep Learning application that generates captions for images using three distinct architectures: **CNN+LSTM**, **CLIP+GPT-2**, and **ViT+GPT-2**.

This project includes a comparative study of these models and a fully containerized web application (FastAPI + Streamlit) to test them side-by-side.

## 🚀 Features
* **Model A (Baseline):** ResNet-50 Encoder + LSTM Decoder.
* **Model B (Prefix Tuning):** CLIP (ViT-B/32) Encoder + GPT-2 Decoder (Prefix method).
* **Model C (SOTA):** Vision Transformer (ViT) Encoder + GPT-2 Decoder (End-to-End Hugging Face).
* **Full Stack App:**
    * **Backend:** FastAPI for high-performance model serving.
    * **Frontend:** Streamlit for an interactive user interface.
    * **Docker:** Fully dockerized for easy deployment.

---

## 📂 Project Structure

```text
/
├── data/                    # Dataset folder (Flickr8k)
├── model_cnn_lstm/          # Model A: Training & Inference code
├── model_clip_gpt/          # Model B: Training & Inference code
├── model_vit_gpt/           # Model C: Training & Inference code
├── app_deploy/              # Full Stack Application (Dockerized)
│   ├── backend/             # FastAPI Server
│   │   └── weights/         # Trained models go here
│   ├── frontend/            # Streamlit UI
│   └── docker-compose.yml   # Orchestration
└── requirements.txt         # Dependencies

```

---

##🛠️ Setup & Installation###1. Clone Repository```bash
git clone https://github.com/Ankit-Navrang-Meshram/Image_Captioning.git
cd image-captioning-project

```

###2. Create EnvironmentWe recommend using Conda to manage dependencies.

```bash
conda create -n img_cap python=3.10 -y
conda activate img_cap
pip install torch torchvision transformers pillow pandas nltk tqdm fastapi uvicorn python-multipart streamlit
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)

```

###3. Download DatasetDownload the **Flickr8k** dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and unzip it into the `data/` folder.

```text
data/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── captions.txt

```

---

##🧠 Training the ModelsYou can train each model independently.

###Model A: CNN + LSTM```bash
cd model_cnn_lstm
python train.py
# Evaluation
python evaluate.py

```

###Model B: CLIP + GPT-2```bash
cd model_clip_gpt
# 1. Extract CLIP features first (Speeds up training)
python extract_features.py
# 2. Train
python train.py

```

###Model C: ViT + GPT-2```bash
cd model_vit_gpt
python train.py

```

---

##🐳 Deployment (The App)We use **Docker Compose** to launch the Frontend and Backend simultaneously.

###1. Prepare WeightsBefore running the app, you must move your trained checkpoints into the backend weights folder.

```bash
# Create directory
mkdir -p app_deploy/backend/weights

# Copy Model A
cp model_cnn_lstm/checkpoints/my_checkpoint_epoch_10.pth.tar app_deploy/backend/weights/model_a.pth

# Copy Model B
cp model_clip_gpt/checkpoints/clip_gpt_epoch_10.pth app_deploy/backend/weights/model_b.pth

# Copy Model C (Entire folder)
cp -r model_vit_gpt/checkpoints/vit_gpt_epoch_10 app_deploy/backend/weights/model_c_hf

# Copy Vocabulary (Required for Model A)
cp model_cnn_lstm/vocab.pkl app_deploy/backend/vocab.pkl

```

###2. Run with Docker```bash
cd app_deploy
docker-compose up --build

```

###3. Access the AppOpen your browser and go to:

* **Frontend (UI):** `http://localhost:8501`
* **Backend (API Docs):** `http://localhost:8000/docs`

---

##📊 Results (BLEU Scores)| Model | BLEU-1 | BLEU-4 | Notes |
| --- | --- | --- | --- |
| **CNN + LSTM** | ~55.0 | ~15.0 | Baseline performance. |
| **CLIP + GPT** | ~65.0 | ~22.0 | Best object detection & context. |
| **ViT + GPT** | ~68.0 | ~24.0 | State-of-the-Art performance. |

---

##📜 Credits* **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* **CLIP:** [OpenAI](https://github.com/openai/CLIP)
* **Transformers:** [Hugging Face](https://huggingface.co/)

```

```
