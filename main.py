# used for exposing the function from inside the utils to outside the utils

from utils import read_video, save_video

def main():
    # Reading video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    # Saving video
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()