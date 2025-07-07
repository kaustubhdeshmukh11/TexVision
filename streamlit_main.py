import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# Page configuration
st.set_page_config(
    page_title="TeXVision", page_icon="📐", layout="centered"
)

st.title("TeXVision 📐")
st.write("Transform images of mathematical expressions into compile-ready LaTeX code.")

@st.cache_resource
def load_model(model_dir="models/quantized_qwen2_vl"):
    """
    Loads the quantized Qwen2-VL-7B model and processor with LoRA adapters.
    """
    # Load the processor and model
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        load_in_4bit=True,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload an image of a math expression", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating LaTeX..."):
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        # Generate tokens
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        # Decode to LaTeX string
        latex_code = processor.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Generated LaTeX Code")
    st.code(latex_code, language="latex")

    st.subheader("Preview")
    # Render LaTeX using MathJax
    st.markdown(f"$$
{latex_code}
$$")
else:
    st.info("Please upload a PNG or JPEG image to get started.")
