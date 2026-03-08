import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import shutil
from tqdm import tqdm

# 1. Initialize GPU-accelerated Models (3060 Ti)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640)) 
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

# 2. Configure Paths
source_root = "D:/ASTAR DATASET (FINAL)/DATASET_2"
output_root = "D:/ASTAR DATASET (FINAL)/DATASET_2_DEID"

# 3. Load Matched Target Identities (Clean, Unwatermarked)
target_m_img = cv2.imread("male_AI_generated_face.jpg")
target_f_img = cv2.imread("female_AI_generated_face.jpg")

# Extract face embeddings for both
target_m_face = app.get(target_m_img)[0]
target_f_face = app.get(target_f_img)[0]

# 4. Manual Demographic Mapping
# Categorize your subjects here based on your manual check
males = [
    "subject3", "subject4", "subject5", "subject8", "subject9", 
    "subject10", "subject12", "subject13", "subject14", "subject16", 
    "subject17", "subject18", "subject20", "subject22", "subject23", 
    "subject24", "subject25", "subject27", "subject30", "subject31", 
    "subject32", "subject36", "subject39", "subject40", "subject41", 
    "subject42", "subject44", "subject45", "subject46", "subject47", 
    "subject48", "subject49"
] # All Males

females = [
    "subject1", "subject11", "subject15", "subject26", "subject33", 
    "subject34", "subject35", "subject37", "subject38", "subject43"
] # All Females

# 5. Batch Process Subjects
subjects = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]

for sub in subjects:
    print(f"\n>>> Processing {sub}")
    
    # Logic to select the right demographic target
    if sub in males:
        target_face = target_m_face
        print(f"Using MALE target for {sub}")
    elif sub in females:
        target_face = target_f_face
        print(f"Using FEMALE target for {sub}")
    else:
        # Safety fallback if you forget to add a subject to the lists
        target_face = target_m_face 
        print(f"WARNING: {sub} not in lists, defaulting to MALE")

    sub_source_dir = os.path.join(source_root, sub)
    sub_output_dir = os.path.join(output_root, sub)
    os.makedirs(sub_output_dir, exist_ok=True)

    video_in = os.path.join(sub_source_dir, "vid.avi")
    video_out = os.path.join(sub_output_dir, "vid.avi")
    gt_in = os.path.join(sub_source_dir, "ground_truth.txt")
    
    # Copy Ground Truth for rPPG evaluation
    if os.path.exists(gt_in):
        shutil.copy(gt_in, os.path.join(sub_output_dir, "ground_truth.txt"))

    cap = cv2.VideoCapture(video_in)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"Swapping {sub}"):
        ret, frame = cap.read()
        if not ret: break
        
        faces = app.get(frame)
        if faces:
            """
            Method to just plainly swap face: Yielding unfavourable results

            --> Start of code

            # Sort to primary face, then swap with demographic target
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            frame = swapper.get(frame, faces[0], target_face, paste_back=True)

            --> End of code

            """
            
            """
            Method to do a alpha blending. So I still layer the original picture, but I layer it with the original vid to give it more depth.
            """
            
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            # 1. Create the fully swapped frame
            swapped_frame = swapper.get(frame, faces[0], target_face, paste_back=True)
            
            # 2. Blend: 70% Swapped Identity + 30% Original Signal
            # 0.7 + 0.3 = 1.0 (Full opacity)
            frame = cv2.addWeighted(swapped_frame, 0.7, frame, 0.3, 0)
        
        out.write(frame)

    cap.release(); out.release()

print("\n--- ALL SUBJECTS DE-IDENTIFIED ---")