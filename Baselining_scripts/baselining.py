import sys
from unittest.mock import MagicMock

# --- STEP 1: THE EMERGENCY CPU PATCH ---
# This must be at the very top to stop the GPU-related crashes
mock_modules = ["cusignal", "cupy", "cupy.cuda"]
for module in mock_modules:
    sys.modules[module] = MagicMock()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyVHR.analysis.pipeline import Pipeline

def run_baseline_study(root_folder='DATASET_2/'):
    """
    Loops through UBFC-RPPG subjects, extracts HR using 3D MediaPipe tracking,
    and compares it to the Absolute Truth.
    """
    # 1. Initialize the 3D-aware Pipeline
    # We use 'mediapipe' because your task requires testing 3D methods.
    print("🚀 Initializing rPPG Pipeline (CPU Mode)...")
    pipe = Pipeline(methods=['POS'], estimator='mediapipe')
    
    # 2. Setup results storage
    if not os.path.exists('baseline_plots'):
        os.makedirs('baseline_plots')
    
    summary_data = []
    
    # 3. Identify Subjects
    subjects = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    subjects.sort()

    # To ensure you have results by Monday, we will process the first 3 subjects.
    # You can change this to 'subjects' to do all 42 if you have time.
    for subject in subjects[:3]: 
        print(f"\n--- Analyzing {subject} ---")
        
        video_path = os.path.join(root_folder, subject, 'vid.avi')
        gt_path = os.path.join(root_folder, subject, 'ground_truth.txt')

        if not os.path.exists(video_path) or not os.path.exists(gt_path):
            print(f"⚠️ Missing files for {subject}. Skipping.")
            continue

        try:
            # 4. Extract Heart Rate from Video
            # winsize=5 provides a good balance between stability and detail
            result = pipe.run_on_video(video_path, roi_approach='holistic', winsize=5)
            est_hr = result.bpm_estimations
            
            # 5. Load Absolute Truth (Dataset 2 Format)
            gt_data = np.loadtxt(gt_path)
            gt_hr_full = gt_data[1, :] # Line 2 is the actual HR
            
            # 6. Synchronize timing for comparison
            # We average the ground truth to match the window-based output of the AI
            fps = result.fps
            window_samples = int(5 * fps)
            step_samples = int(1 * fps)
            
            gt_windows = [np.mean(gt_hr_full[i : i + window_samples]) 
                          for i in range(0, len(gt_hr_full) - window_samples, step_samples)]
            
            # Match lengths
            min_len = min(len(gt_windows), len(est_hr))
            final_gt = np.array(gt_windows[:min_len])
            final_est = np.array(est_hr[:min_len])
            
            # 7. Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(final_gt - final_est))
            summary_data.append({'Subject': subject, 'Baseline_MAE': mae})
            print(f"✅ Success! {subject} MAE: {mae:.2f} BPM")

            # 8. Generate and Save Graph for Presentation
            plt.figure(figsize=(12, 5))
            time_axis = np.arange(min_len)
            plt.plot(time_axis, final_gt, label='Absolute Truth (Pulse Ox)', color='#2ecc71', linewidth=2)
            plt.plot(time_axis, final_est, label='rPPG Baseline (POS)', color='#e74c3c', linestyle='--')
            plt.title(f'Baseline HR Comparison: {subject} (MAE: {mae:.2f})')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Heart Rate (BPM)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'baseline_plots/{subject}_baseline.png')
            plt.close()

        except Exception as e:
            print(f"❌ Error on {subject}: {e}")

    # 9. Export Results for your Monday Report
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv('astar_baseline_results.csv', index=False)
        print("\n✨ TASK COMPLETE.")
        print("Results saved to 'astar_baseline_results.csv'")
        print("Graphs for your slides are in the '/baseline_plots' folder.")

if __name__ == "__main__":
    # Ensure this points to your actual dataset folder
    process_all_subjects = run_baseline_study('DATASET_2/')