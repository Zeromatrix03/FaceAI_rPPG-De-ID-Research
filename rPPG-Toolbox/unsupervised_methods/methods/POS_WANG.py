"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math
import numpy as np
from scipy import signal
from unsupervised_methods import utils

def _process_video(frames):
    """
    Averages the pixels in each frame to get one RGB value per time step.
    Expects frames shape: (Time, Height, Width, Channels)
    """
    # Average across Height (axis 1) and Width (axis 2)
    # Resulting shape: (Time, 3)
    RGB = np.mean(frames, axis=(1, 2))
    return RGB

def POS_WANG(frames, fs):
    WinSec = 1.6
    RGB = _process_video(frames)
    N = RGB.shape[0]
    
    # FIX 1: Initialize H as a 1D array to match the 1D signal 'h'
    H = np.zeros(N)
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            # 1. Slice and Normalize
            window_data = RGB[m:n, :]
            mean_rgb = np.mean(window_data, axis=0)
            Cn = np.true_divide(window_data, mean_rgb)
            
            # 2. Transpose for Projection: Result shape should be (3, l)
            Cn = Cn.T 
            
            # 3. Projection: (2, 3) @ (3, l) -> (2, l)
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
            S = np.matmul(projection_matrix, Cn)
            
            # 4. Calculate Alpha and combine to get 'h'
            # Standard Deviation check to avoid division by zero
            std_s1 = np.std(S[1, :])
            alpha = np.std(S[0, :]) / std_s1 if std_s1 != 0 else 0
            h = S[0, :] + alpha * S[1, :]
            
            # 5. Remove mean from the window signal
            h = h - np.mean(h)
            
            # 6. Accumulate into the full H array
            # H is 1D, h is 1D, so H[m:n] works perfectly
            H[m:n] = H[m:n] + h

    # FIX 2: Modernize the Detrending (No np.asmatrix!)
    # Reshape to (N, 1) because detrend usually expects a column
    BVP = H.reshape(-1, 1)
    BVP = utils.detrend(BVP, 100)
    
    # Flatten back to 1D for filtering
    BVP = np.asarray(BVP).flatten()
    
    # 7. Bandpass Filter (0.75Hz to 3.0Hz)
    # High-cut and Low-cut frequencies based on physiological heart rate
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    
    return BVP