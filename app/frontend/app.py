import streamlit as st
import requests
from PIL import Image
import io

# FastAPI Endpoint (The service name 'backend' comes from docker-compose)
API_URL = "http://backend:8000/predict_all"

st.set_page_config(layout="wide")
st.title("🖼️ Image Captioning Arena")
st.markdown("### Compare 3 Architectures Side-by-Side")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Input Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Generate Captions"):
        with st.spinner("Processing through 3 models... (This may take a moment)"):
            try:
                # Send file to Backend
                # We need to reset file pointer to beginning
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                
                response = requests.post(API_URL, files=files)
                data = response.json()

                # Display Results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.success("Model A: CNN + LSTM")
                    st.write(data["CNN_LSTM"])
                    st.info("Old school architecture. Often short and simple.")

                with col2:
                    st.warning("Model B: CLIP + GPT")
                    st.write(data["CLIP_GPT"])
                    st.info("Prefix Tuning. Good at detecting objects.")

                with col3:
                    st.error("Model C: ViT + GPT")
                    st.write(data["ViT_GPT"])
                    st.info("End-to-End Transformer. State of the Art.")

            except Exception as e:
                st.error(f"Connection Error: {e}")