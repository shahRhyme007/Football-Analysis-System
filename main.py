# used for exposing the function from inside the utils to outside the utils

from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

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
    
    # Assigning Player Teams and colors
    team_assigner = TeamAssigner()
    # assigning team color for the first frame of the video
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    # looping through the rest of the frames and assigning team color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)

            # assigning team color for the current frame
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    # output video with annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control=True)
    
    #* Saving video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()