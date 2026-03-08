import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# 1. Initialize Models
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 for GPU
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

# 2. Paths - Using your D: drive structure
video_path = "D:/ASTAR DATASET (FINAL)/DATASET_2/Subject1/vid.avi"
target_img_path = "target.jpg"
output_path = "D:/ASTAR testing/rPPG-Toolbox/test_deid.avi"

# 3. Load Target
target_img = cv2.imread(target_img_path)
target_face = app.get(target_img)[0]

# 4. Process for Research-Grade Output
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use XVID for .avi (More compatible with the toolbox)
# Note: 'XVID' is the most common for research datasets like UBFC
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
output_path_avi = "D:/ASTAR testing/rPPG-Toolbox/test_deid.avi"
out = cv2.VideoWriter(output_path_avi, fourcc, fps, (w, h))

print(f"Starting dry run... saving to {output_path_avi}")

print("Starting 5-second dry run...")
for i in range(150): # 150 frames = 5 seconds @ 30fps
    ret, frame = cap.read()
    if not ret: break
    
    faces = app.get(frame)
    if faces:
        # Sort to find the main subject
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        frame = swapper.get(frame, faces[0], target_face, paste_back=True)
    
    out.write(frame)
    if i % 30 == 0: print(f"Processed {i//30} seconds...")

cap.release()
out.release()
print(f"Done! Check {output_path} to see the result.")