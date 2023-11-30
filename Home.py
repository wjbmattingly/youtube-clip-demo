import streamlit as st
import annoy
from sentence_transformers import SentenceTransformer
import glob
from PIL import Image

@st.cache_data
def load_images():
    files = glob.glob("images/*/*.jpg")
    files.sort()
    img_list = [Image.open(filepath).convert('RGB') for filepath in files]
    return img_list

@st.cache_resource
def load_annoy():
    annoy_index = annoy.AnnoyIndex(512, 'angular')
    annoy_index.load('index.annoy')  # Load the index
    return annoy_index

@st.cache_resource
def load_model():
    model = SentenceTransformer('clip-ViT-B-32')
    return model

st.title("CLIP Image Demo")


img_list = load_images()
annoy_index = load_annoy()
model = load_model()

query_text = st.text_input("Select an item to search for")
query_emb = model.encode([query_text], show_progress_bar=True)

closest_5_idx, closest_5_dist = annoy_index.get_nns_by_vector(query_emb[0], 5,
                                        include_distances=True)
for idx, dist in zip(closest_5_idx, closest_5_dist):
    print(idx, dist)
    st.image(img_list[idx])