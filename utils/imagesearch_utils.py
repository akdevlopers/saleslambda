



from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import open_clip
import pickle
import torch
from tqdm import tqdm
import os
import requests
import numpy as np
import mysql.connector

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()

def compute_similarity(query_img_path, dataset_embeddings, image_paths):
    try:
        query_img = Image.open(query_img_path).convert("RGB")
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = model.encode_image(query_tensor)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()

        similarities = cosine_similarity(query_embedding, dataset_embeddings).flatten()
        top_indices = similarities.argsort()[::-1][:5]
        results = [(image_paths[idx], float(similarities[idx])) for idx in top_indices]
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to process query image: {e}")



def create_embeddings(dataset_folder):
    Image_embedding_path=f"static/imageembeddings/{dataset_folder}.pkl"

    download_images(dataset_folder)
    return compute_and_cache_embeddings(str(dataset_folder),Image_embedding_path)


def load_or_create_embeddings(dataset_folder):
    Image_embedding_path=f"static/imageembeddings/{dataset_folder}.pkl"
    embeddings, paths = load_embeddings(Image_embedding_path)
    if embeddings is not None and paths is not None:
        print("Loaded cached embeddings.")
        return embeddings, paths
    else:
        print("Caching new embeddings...")
        download_images(dataset_folder)
        return compute_and_cache_embeddings(str(dataset_folder),Image_embedding_path)


def save_embeddings(embeddings, paths, vendor_id,Image_embedding_path):
    imageembedding_root_dir = f"static/imageembeddings/"
    os.makedirs(imageembedding_root_dir, exist_ok=True)
    with open(Image_embedding_path, "wb") as f:
        pickle.dump((embeddings, paths), f)

def load_embeddings(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None, None



def compute_and_cache_embeddings(dataset_folder,Image_embedding_path):
    """Compute embeddings and save them locally."""
    dataset_embeddings = []
    image_paths = []
    static_dir = os.path.join("static", dataset_folder)
    print(static_dir)
    for fname in tqdm(os.listdir(static_dir), desc="Embedding dataset images"):
        path = os.path.join(static_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(img_tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                dataset_embeddings.append(embedding.cpu().numpy())
                image_paths.append(path)
        except (UnidentifiedImageError, Exception) as e:
            print(f"Skipping: {path}")

    dataset_embeddings = np.vstack(dataset_embeddings)
    save_embeddings(dataset_embeddings, image_paths,dataset_folder,Image_embedding_path)
    return dataset_embeddings, image_paths




    # connection = mysql.connector.connect(
    #     host="localhost",
    #     user="salegrowy_crmfbonboard",
    #     password="salegrowy$123",
    #     database="salegrowy_crmfbonboard"
    # )

def download_images(vendor_id):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="aibot"
    )

    cursor = connection.cursor(dictionary=True)
    query = "SELECT _id, images FROM shopify_products WHERE vendor_id = %s"
    cursor.execute(query, (vendor_id,))
    products = cursor.fetchall()
    static_dir = f"static/{vendor_id}"
    os.makedirs(static_dir, exist_ok=True)
    # for product in products:
    #     product_id = product['_id']
    #     image_urls = product['images']
    #     print("image_urls: ",image_urls)
    #     # Some entries might have multiple image URLs separated by commas
    #     if image_urls:
    #         image_urls = [url.strip() for url in image_urls.split(",")]

    for product in products:
        product_id = product['_id']
        image_urls = product['images']
        
        if image_urls:
            image_urls = [url.strip() for url in image_urls.split(",")]

            for idx, image_url in enumerate(image_urls):
                print(f"product_id: {product_id}, image_url: {image_url}")
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()

                    # Get file extension (default to .jpg if missing)
                    ext = os.path.splitext(image_url)[-1].split('?')[0]
                    ext = ext if ext else ".jpg"

                    local_filename = f"Productimage-{product_id}{ext}"
                    full_path = os.path.join(static_dir, local_filename)

                    with open(full_path, 'wb') as f:
                        f.write(response.content)

                except Exception as e:
                    print(f"Failed to download image for product {product_id}: {e}")

    cursor.close()
    connection.close()

    return "Done"


def get_products(ProductIDs):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="aibot"
    )

    cursor = connection.cursor(dictionary=True)

    # Create placeholders (%s, %s, ...) based on list length
    placeholders = ', '.join(['%s'] * len(ProductIDs))
    query = f"SELECT * FROM shopify_products WHERE _id IN ({placeholders})"

    cursor.execute(query, tuple(ProductIDs))
    products = cursor.fetchall()
    return products

