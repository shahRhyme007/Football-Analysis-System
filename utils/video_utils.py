import cv2


#  Reading video and returns a list of frames
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    # Reading frames from the video one by one 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


#  Saving video from a list of frames 
def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    # Writing frames to the video one by one
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()