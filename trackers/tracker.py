from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import cv2
import numpy as np
import pandas as pd





class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()



    #* detect the objects in the frames 
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
            # break
        return detections
    
    

    #* get the object tracks from the frames given  
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # if the tracks are already saved in a pickle file return them
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks




        detections = self.detect_frames(frames)

        # initialize the track objects 
        tracks={
            "players":[], #dictionary of  each frame 
            "referees":[],
            "ball":[]
        }

        # track the objects in the frames 
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)


            # Convert detections to supervision object
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # track the objects in the frames 
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)


            # putting the output in a format so that we can utilize easily
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})


            # looping through the detections of tracks
            for frame_detection in detection_with_tracks:
                # bounding box 
                bbox = frame_detection[0].tolist()

                # class id and track id
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                # referee is a special case because it is not a player
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                # ball is a special case
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            # print(detection_supervision)
            # print(detection_with_tracks)
            
            # for saving our tracks in a pickle file
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    

# ---------------------------------------------------------------#

    # drawing the elipse function for circles 
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # drawing the ellipse for the player 
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # drawing the rectangle for the player (for numbers)
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    

    # function for having triangles on the ball
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    # function for tracking the ball along with the triangle(for each seperate team)



    # *function for having circles
    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players--> they will have a red circle below them
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                # calling
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee--> they will have a yellow circle below them
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


        

            output_video_frames.append(frame)

        return output_video_frames
     

        




