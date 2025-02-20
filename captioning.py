from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
# Load BLIP model
def load_model():
    """
    Load the pre-trained BLIP model from the huggingface model hub.

    This function is decorated with `@st.cache_resource`, which means that the
    model will be loaded from the model hub the first time the function is called,
    and then cached in memory for subsequent calls. This can be useful for
    avoiding unnecessary network requests and making the app feel more responsive.

    Returns:
        processor (BlipProcessor): A BlipProcessor instance
        model (BlipForConditionalGeneration): A BlipForConditionalGeneration instance
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    return processor, model

@st.cache_resource
def generate_caption(image_path: str) -> str:
    """
    Generate a caption for the given image using the BLIP model.

    This function is decorated with `@st.cache_resource`, which means that the
    model and processor will be loaded from the model hub the first time the
    function is called, and then cached in memory for subsequent calls. This
    can be useful for avoiding unnecessary network requests and making the
    app feel more responsive.

    Args:
        image_path (str): The path to the image file to be captioned

    Returns:
        str: The generated caption
    """
    processor, model = load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
