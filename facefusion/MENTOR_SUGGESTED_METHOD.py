import cv2
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
import mediapipe as mp

# 1. Setup Models & Pipeline
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
generator = torch.Generator("cuda").manual_seed(42)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,           
    requires_safety_checker=False  
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

try: pipe.enable_xformers_memory_efficient_attention()
except: pass

# 2. Refinement Logic
def generate_feathered_mask(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if results.multi_face_landmarks:
        points = [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] 
                  for lm in results.multi_face_landmarks[0].landmark]
        cv2.fillConvexPoly(mask, cv2.convexHull(np.array(points)), 255)
        return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=12))
    return None

def refine_video(orig_path, sim_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap_orig = cv2.VideoCapture(orig_path)
    cap_sim = cv2.VideoCapture(sim_path)
    
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    w, h = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    prev_refined_face = None

    while cap_orig.isOpened():
        ret1, f_orig = cap_orig.read()
        ret2, f_sim = cap_sim.read()
        if not ret1 or not ret2: break

        mask = generate_feathered_mask(f_orig)
        if mask:
            init_img = Image.fromarray(cv2.cvtColor(f_sim, cv2.COLOR_BGR2RGB)).resize((512, 512))
            mask_img = mask.resize((512, 512))
            
            if prev_refined_face is not None:
                init_img = Image.blend(init_img, prev_refined_face, 0.3)
            
            refined_face = pipe(
                prompt="highly detailed face, professional lighting, seamless skin",
                image=init_img, mask_image=mask_img,
                strength=0.20, num_inference_steps=6, guidance_scale=1.0, generator=generator
            ).images[0]

            prev_refined_face = refined_face.copy()
            res_face = cv2.cvtColor(np.array(refined_face.resize((w, h))), cv2.COLOR_RGB2BGR)
            mask_np = np.stack([np.array(mask)/255.0]*3, axis=-1)
            final_frame = (res_face * mask_np + f_orig * (1 - mask_np)).astype(np.uint8)
            out.write(final_frame)

    cap_orig.release(); cap_sim.release(); out.release()

# 3. Batch Processing Logic (Dataset Scan)
def run_batch_refinement(orig_root, sim_root, output_root):
    # Find all subject folders in the original dataset
    subjects = [d for d in os.listdir(orig_root) if os.path.isdir(os.path.join(orig_root, d))]
    
    for sub in tqdm(subjects, desc="Processing Entire Dataset"):
        orig_vid = os.path.join(orig_root, sub, "vid.avi")
        sim_vid = os.path.join(sim_root, sub, "vid.avi")
        out_vid = os.path.join(output_root, sub, "vid.avi")

        if os.path.exists(sim_vid):
            print(f"\nStarting Refinement for {sub}...")
            refine_video(orig_vid, sim_vid, out_vid)
        else:
            print(f"Skipping {sub}: No SimSwap file found.")

# SET YOUR PATHS HERE
ORIG_DIR = r"D:\ASTAR DATASET (FINAL)\DATASET_2"
SIM_DIR = r"D:\ASTAR DATASET (FINAL)\DATASET_2_DEID"
OUT_DIR = r"D:\ASTAR DATASET (FINAL)\refined_outputs"

run_batch_refinement(ORIG_DIR, SIM_DIR, OUT_DIR)