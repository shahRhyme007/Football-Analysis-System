from ultralytics import YOLO 

model = YOLO('models/best.pt')


# running the model on a video
results = model.predict('input_videos/08fd33_4.mp4',save=True)
print(results[0])
print('=====================================')

# running the model on the first frame of the video and go through the loop 
for box in results[0].boxes:
    print(box)