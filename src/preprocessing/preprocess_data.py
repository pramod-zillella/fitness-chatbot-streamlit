import os
import json
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def preprocess_data():
    raw_data_dir = r'E:\Udemy\fitness-podcastai\data\raw'
    processed_data_dir = r'E:\Udemy\fitness-podcastai\data\processed'
    os.makedirs(processed_data_dir, exist_ok=True)

    for filename in tqdm(os.listdir(raw_data_dir), desc="Preprocessing files"):
        if filename.endswith('.json'):
            with open(os.path.join(raw_data_dir, filename), 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            
            # Clean title, description, and transcript
            video_data['clean_title'] = clean_text(video_data['title'])
            video_data['clean_description'] = clean_text(video_data['description'])
            video_data['clean_transcript'] = clean_text(video_data['transcript'])
            
            # Combine cleaned text for easier processing later
            video_data['combined_text'] = (video_data['clean_title'] + ' ' + 
                                           video_data['clean_description'] + ' ' + 
                                           video_data['clean_transcript'])
            
            # Save processed data
            with open(os.path.join(processed_data_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(video_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    preprocess_data()