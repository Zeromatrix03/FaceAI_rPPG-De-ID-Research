import os
import subprocess
import sys
import shutil

# --- 1. CUDA & ENVIRONMENT SETUP ---
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")

# --- 2. PATHS & LISTS ---
dataset_dir = r"D:\ASTAR DATASET (FINAL)\DATASET_2"
output_root = r"D:\ASTAR DATASET (FINAL)\DATASET_2_DEID"
female_face = r"D:\ASTAR DATASET (FINAL)\AI_gen_faces\female_AI_generated_face.jpg"
male_face = r"D:\ASTAR DATASET (FINAL)\AI_gen_faces\male_AI_generated_face.jpg"

female_subjects = ["subject1", "subject11", "subject15", "subject26", "subject33", 
                   "subject34", "subject35", "subject37", "subject38", "subject43"]

# --- 3. CHURN LOGIC ---
all_folders = [d for d in os.listdir(dataset_dir) if d.startswith("subject")]

for subject in all_folders:
    # Set identity and source face
    is_female = subject in female_subjects
    source_path = female_face if is_female else male_face
    
    # Define paths
    subject_input_dir = os.path.join(dataset_dir, subject)
    subject_output_dir = os.path.join(output_root, subject)
    
    input_vid = os.path.join(subject_input_dir, "vid.avi")
    output_vid = os.path.join(subject_output_dir, "vid.avi") # Named same as input
    gt_file = os.path.join(subject_input_dir, "ground_truth.txt")

    # Create the specific subject folder in the DEID directory
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)

    # Clone ground_truth.txt if it exists
    if os.path.exists(gt_file):
        shutil.copy2(gt_file, os.path.join(subject_output_dir, "ground_truth.txt"))
        print(f"[COPIED]: Ground truth for {subject}")

    # Run the high-resolution, stable swap
    if os.path.exists(input_vid):
        print(f"[CHURNING]: {subject} ({'Female' if is_female else 'Male'})")
        
        cmd = [
            sys.executable, "facefusion.py", "headless-run",
            "-s", source_path,
            "-t", input_vid,
            "-o", output_vid,
            "--face-selector-mode", "one",
            "--reference-face-distance", "1.0",
            "--face-mask-types", "box", "region",
            "--face-mask-blur", "0.8",
            "--face-landmarker-model", "peppa_wutz",
            "--face-swapper-model", "simswap_unofficial_512",
            "--face-enhancer-model", "codeformer",
            "--face-enhancer-blend", "50",
            "--output-video-encoder", "libx264",
            "--execution-providers", "cuda"
        ]
        subprocess.run(cmd)

print("\n--- All subjects mirrored and processed! ---")