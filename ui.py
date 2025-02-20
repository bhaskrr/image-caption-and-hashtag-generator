__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from app import generate

st.title("📸 Image Caption & Hashtag Generator")

# Upload Section
file = st.file_uploader("📂 Upload an image", type=["jpg", "png"])
submit = st.button("✨ Generate")

if submit and file is not None:
    # Display Uploaded Image
    st.image(file, caption="Uploaded Image", use_column_width=True)

    # Generate Response
    response = generate(file)

    # Layout with Two Columns
    col1, col2 = st.columns(2)

    # Display Captions
    with col1:
        st.subheader("📝 Generated Captions")
        for index, item in enumerate(response["captions"]):
            st.markdown(f"**{index + 1}.** *{item}*")

    # Display Hashtags as Badges
    with col2:
        st.subheader("🏷️ Suggested Hashtags")
        hashtag_text = " ".join([f"`{item}`" for item in response["hashtags"]])
        st.markdown(hashtag_text, unsafe_allow_html=True)
