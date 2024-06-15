from ultralytics import YOLO
import supervision as sv




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
            break
        return detections
    
    

    #* get the object tracks from the frames given  
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        # track the objects in the frames 
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)

            detection_supervision = sv.Detections.from_ultralytics(detection)
            print(detection_supervision)

            break

        




