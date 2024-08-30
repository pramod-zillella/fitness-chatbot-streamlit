from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm

load_dotenv()

# Set up YouTube API client
youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

def get_channel_videos(channel_id):
    videos = []
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        maxResults=50,
        type='video'
    )
    while request:
        response = request.execute()
        videos.extend(response['items'])
        request = youtube.search().list_next(request, response)
    return videos

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except:
        return None

def main():
    channel_id = 'UCe0TLA0EsQbE-MjuHXevj2A'
    videos = get_channel_videos(channel_id)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    for video in tqdm(videos, desc="Processing videos"):
        video_id = video['id']['videoId']
        video_data = {
            'id': video_id,
            'title': video['snippet']['title'],
            'description': video['snippet']['description'],
            'transcript': get_video_transcript(video_id)
        }
        
        with open(f'data/raw/{video_id}.json', 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()