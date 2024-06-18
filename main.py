# used for exposing the function from inside the utils to outside the utils

from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    #* Reading video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Tracking the objects
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")


    #Just checking if the cropped image is working 
    '''for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)
        break'''

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control=True)
    
    #* Saving video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()