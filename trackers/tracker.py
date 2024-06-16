from ultralytics import YOLO
import supervision as sv
import pickle
import os




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


    
        




