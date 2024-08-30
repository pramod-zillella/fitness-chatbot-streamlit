from moviepy.editor import VideoFileClip
import os

def convert_video_to_gif(video_path, gif_path, start_time=0, duration=None, resize_factor=0.5):
    """
    Convert a video file to a GIF.
    
    :param video_path: Path to the input video file
    :param gif_path: Path for the output GIF file
    :param start_time: Start time of the clip (in seconds)
    :param duration: Duration of the clip (in seconds). If None, use the whole video.
    :param resize_factor: Factor by which to resize the video (e.g., 0.5 for half size)
    """
    # Load the video clip
    clip = VideoFileClip(video_path).subclip(start_time, start_time + duration if duration else None)
    
    # Resize the clip
    clip = clip.resize(resize_factor)
    
    # Write the clip as a GIF
    clip.write_gif(gif_path, fps=10)
    
    # Close the clip to free up system resources
    clip.close()

# Example usage
video_path = r"C:\Users\pramo\Videos\Captures\Fitness AI Chatbot - Google Chrome 2024-08-30 19-21-57.mp4"
gif_path = r"C:\Users\pramo\Videos\Captures\demo.gif"

convert_video_to_gif(video_path, gif_path, start_time=0, duration=120, resize_factor=0.5)

print(f"GIF created successfully: {gif_path}")
print(f"GIF file size: {os.path.getsize(gif_path) / (1024 * 1024):.2f} MB")