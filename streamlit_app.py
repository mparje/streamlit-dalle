import io
import requests
import os
import openai
from typing import List
from PIL import Image
import streamlit as st

size = 1024


def generate_story(prompt: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Create a short children's story based on the prompt: {prompt}",
        n=1,
        max_tokens=500,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()


def download_images(image_urls: List[str]) -> List[Image.Image]:
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        images.append(img)
    return images

 
def generate_images(text: str, num_images: int) -> List[Image.Image]:
    response = openai.Image.create(
        prompt=text,
        n=num_images,
        size=f"{size}x{size}"
    )
    image_urls = [x['url'] for x in response['data']]
    return download_images(image_urls)


def generate_variations(image: Image.Image, num_images: int) -> List[Image.Image]:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    response = openai.Image.create_variation(
        image=buffer.getvalue(),
        n=num_images,
        size=f"{size}x{size}"
    )
    image_urls = [x['url'] for x in response['data']]
    return download_images(image_urls)


def main():
    openai.api_key = os.getenv("openai_api_key")
    st.set_page_config(page_title="Story and Image Generation", page_icon="🎨", layout="wide")
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Children's Story and Image Generation")
    st.subheader("Enter a prompt for your children's story")
    text = st.text_area("Story Prompt")

    st.subheader("or upload an image to create variations")
    file_upload = st.file_uploader(
        "Image upload",
        type=["png", "jpg", "jpeg", "bmp"],
        key="file_upload"
    )
    st.subheader("Select the number of images to generate")
    num_images = st.slider("Number of Images", min_value=1, max_value=6, value=4, step=1)

    if "images" not in st.session_state:
        st.session_state.images = []

    if "generated_story" in st.session_state:
        st.markdown(f"## Generated story\n\n{st.session_state.generated_story}\n")

    if st.button("Run!"):
        if text:
            with st.spinner("Generating story using GPT-3 Davinci"):
                st.session_state.generated_story = generate_story(text)

        if file_upload is not None:
            with st.spinner("Using openAI API with image"):
                image = Image.open(file_upload)
                width, height = image.size
                square_size = min(width, height)
                left = int((width - square_size) / 2)
                top = int((height - square_size) / 2)
                right = left + square_size
                bottom = top + square_size
                image = image.crop((left, top, right, bottom))
                if square_size > 1024:
                    image = image.resize((1024, 1024))
                st.session_state.images = generate_variations(image, num_images)
                st.experimental_rerun()
        else:
            story_text = st.session_state.get("generated_story")
            if story_text:
                with st.spinner("Using openAI API with text prompt"):
                    st.session_state.images = generate_images(story_text, num_images)
                    st.experimental_rerun()

    num_columns = len(st.session_state.images)
    if num_columns > 0:
        columns = st.columns(num_columns)
        for i in range(len(st.session_state.images)):
            image = st.session_state.images[i]
            column = columns[i]
            with column:
                st.image(image, caption=f"Image {i + 1}")
                if st.button(f"Create variations for Image {i + 1}", use_container_width=True):
                    with st.spinner("Using openAI API"):
                        st.session_state.images = generate_variations(image, num_images)
                        st.experimental_rerun()
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)

                st.download_button(
                    label=f"Download Image {i + 1}",
                    data=img_io,
                    file_name="generation.png",
                    mime="image/png",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
