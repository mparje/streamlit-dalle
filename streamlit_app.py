import io
import requests
from typing import List
import streamlit as st
import openai
from PIL import Image

size = 1024


def download_images(image_urls: List[str]) -> List[Image.Image]:
    """
    Download and open a list of image files from a given list of URLs and return them as PIL Image objects.

    Args:
        image_urls (List[str]): A list of URLs from which to download images.

    Returns:
        List[PIL.Image.Image]: A list of PIL Image objects containing the downloaded images.
    """
    images = []
    for url in image_urls:
        # Get the image from the URL
        response = requests.get(url)
        # Open the image using PIL
        img = Image.open(io.BytesIO(response.content))
        images.append(img)
    return images


def generate_images(text: str, num_images: int) -> List[Image.Image]:
    """
    Generate a specified number of images from OpenAI's DALL-E 2 API using a text prompt.

    Args:
        text (str): A string describing the desired images.
        num_images (int): The number of images to generate.

    Returns:
        List[Pil.Image.Image]: A list of PIL Image objects containing the generated images.
    """
    response = openai.Image.create(
        prompt=text,
        n=num_images,
        size=f"{size}x{size}"
    )
    image_urls = [x['url'] for x in response['data']]
    return download_images(image_urls)


def generate_variations(image: Image.Image, num_images: int) -> List[Image.Image]:
    """
    Generate a specified number of variations from OpenAI's DALL-E 2 API using an input image.

    Args:
        image (PIL.Image.Image): A PIL Image object to generate variations of.
        num_images (int): The number of variations to generate.

    Returns:
        List[Pil.Image.Image]: A list of PIL Image objects containing the generated images.
    """
    # Write image to in-memory buffer as PNG
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
    openai.api_key = st.secrets["openai_api_key"]
    st.set_page_config(page_title="Image Generation", page_icon="🎨", layout="wide")
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("DALL-E 2 Image Generation")
    st.subheader("Enter a text prompt")
    text = st.text_area("Text Prompt")
    st.subheader("or upload an image to create variations")
    file_upload = st.file_uploader(
        "Image upload",
        type=["png", "jpg", "jpeg", "bmp"],
        key="file_upload"
    )
    st.subheader("Select the number of images to generate")
    num_images = st.slider("Number of Images", min_value=1, max_value=6, value=4, step=1)

    # Store the images list in the session_state of streamlit
    if "images" not in st.session_state:
        st.session_state.images = []

    if st.button("Run!"):
        if file_upload is not None:
            with st.spinner("Using openAI API with image"):
                image = Image.open(file_upload)
                width, height = image.size
                square_size = min(width, height)
                left = int((width - square_size) / 2)
                top = int((height - square_size) / 2)
                right = left + square_size
                bottom = top + square_size
                # Crop the center of the image and resize if needed
                image = image.crop((left, top, right, bottom))
                if square_size > 1024:
                    image = image.resize((1024, 1024))
                # Generate variations for the uploaded image
                st.session_state.images = generate_variations(image, num_images)
                st.experimental_rerun()
        else:
            with st.spinner("Using openAI API with text prompt"):
                # Generate the images
                st.session_state.images = generate_images(text, num_images)
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
                        # Generate variations for the current image
                        st.session_state.images = generate_variations(image, num_images)
                        st.experimental_rerun()
                # Convert the PIL Image object to PNG
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)

                # Add a download button to download the PNG file
                st.download_button(
                    label=f"Download Image {i + 1}",
                    data=img_io,
                    file_name="generation.png",
                    mime="image/png",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
