import json
from typing import Any, Dict
import chromadb
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# import generate caption function
from captioning import generate_caption

chroma_client = chromadb.PersistentClient("./chroma_db")

collection = chroma_client.get_collection(name="image_captions")


@st.cache_resource
def retrieve_captions(image_path, top_k=1):
    """
    Retrieve the most relevant captions for a given image.

    This function generates a caption for the image and then queries a database
    to find similar captions. It returns the best-matching caption along with
    the generated image description.

    Args:
        image_path (str): The path to the image file.
        top_k (int): The number of top similar captions to retrieve.

    Returns:
        dict: A dictionary containing the image description and the best caption.
    """
    # Generate a description for the image using a captioning model
    image_description = generate_caption(image_path)

    # Query the ChromaDB collection to find similar captions
    results = collection.query(
        query_texts=[image_description],
        n_results=top_k,
    )

    # Extract the best matching caption from the query results
    best_caption = results["metadatas"][0]

    # Return the image description and the best caption
    return {
        "image_description": image_description,
        "caption": best_caption,
    }


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a image caption generator tool. Generate five different captions for the image based on the provided context and description of the image.
         Do not try to describe the image, think like the caption is coming from the uploader and use positive language.
         Also, generate relevant hash tags. Format the response into a json object with two keys: captions and hashtags. Strictly follow the above naming convection.
         Only return the content.""",
        ),
        (
            "user",
            "Image Description: {image_description}, <context>Retrieved captions from database: {caption}</context>",
        ),
    ]
)

model = ChatGroq(model="llama-3.3-70b-versatile", streaming=False)

chain = prompt | model


def generate(file_path: str) -> Dict[str, Any]:
    """
    Generate five different captions for a given image.

    Args:
        file_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the captions and relevant hashtags.
    """
    # Retrieve the captions from the database
    description_and_caption_obj = retrieve_captions(file_path)

    # Use the prompt and model to generate five different captions
    response = chain.invoke(
        {
            "image_description": description_and_caption_obj["image_description"],
            "caption": description_and_caption_obj["caption"],
        }
    )

    # Parse the response as JSON and return it
    return json.loads(response.content)
