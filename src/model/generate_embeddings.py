import os
import json
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def load_processed_data(processed_data_dir):
    data = []
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(processed_data_dir, filename), 'r', encoding='utf-8') as f:
                video_data = json.load(f)
                data.append(video_data)
    return data

def generate_embeddings(model, data):
    embeddings = []
    for video in tqdm(data, desc="Generating embeddings"):
        # Generate embedding for the combined text
        embedding = model.encode(video['combined_text'])
        embeddings.append(embedding)
    return np.array(embeddings)

def save_embeddings(embeddings, video_ids, output_file):
    np.savez(output_file, embeddings=embeddings, video_ids=video_ids)

def main():
    processed_data_dir = r'E:\Udemy\fitness-podcastai\data\processed'
    output_file = r'E:\Udemy\fitness-podcastai\data\embeddings\video_embeddings.npz'
    os.makedirs('data/embeddings', exist_ok=True)

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load processed data
    data = load_processed_data(processed_data_dir)

    # Generate embeddings
    embeddings = generate_embeddings(model, data)

    # Get video IDs
    video_ids = [video['id'] for video in data]

    # Save embeddings and video IDs
    save_embeddings(embeddings, video_ids, output_file)

    print(f"Embeddings generated and saved to {output_file}")

if __name__ == '__main__':
    main()