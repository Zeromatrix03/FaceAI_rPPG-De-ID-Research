"""ICA
Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762
"""
import math
import numpy as np
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils

def ICA_POH(frames, FS):
    LPF = 0.7
    HPF = 2.5
    RGB = process_video(frames)

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = utils.detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    
    # Modernized call
    _, S = ica(BGRNorm.T.conj(), 3)

    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        # Reshaping if needed for compatibility
        FF = np.atleast_2d(FF)
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))

    return BVP_F

def process_video(frames):
    # vectorized mean is much faster than loops
    return np.mean(frames, axis=(1, 2))

def ica(X, Nsources, Wprev=0):
    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat

def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    
    if m < n:
        # Replaced asmatrixmul and .H
        D, U = np.linalg.eig((X @ X.T.conj()) / T)
        k = np.argsort(D)
        pu = D[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m), ibl)
        W = np.diag(bl) @ U[0:n, k[n - m:n]].T.conj()
        IW = U[0:n, k[n - m:n]] @ np.diag(ibl)
    else:
        IW = linalg.sqrtm((X @ X.T.conj()) / T)
        W = np.linalg.inv(IW)

    Y = W @ X
    R = (Y @ Y.T.conj()) / T
    C = (Y @ Y.T) / T 
    Q = np.zeros((m * m * m * m, 1), dtype=complex)
    index = 0

    for lx in range(m):
        for kx in range(m):
            for jx in range(m):
                for ix in range(m):
                    # Replaced asmatrixmul with @
                    term1 = (Y[lx, :] * np.conj(Y[kx, :]) * np.conj(Y[jx, :])) @ (Y[ix, :].T / T)
                    Q[index] = term1 - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1

    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    K = np.argsort(abs(D))[::-1] # Sort descending
    la = D[K]
    M = np.zeros((m, nem * m), dtype=complex)
    
    for i, h in enumerate(range(nem)):
        Z = U[:, K[h]].reshape((m, m))
        M[:, h*m : (h+1)*m] = la[h] * Z

    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = B.T.conj()
    V = np.eye(m).astype(complex) if Wprev == 0 else np.linalg.inv(Wprev)

    encore = 1
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip, Iq = np.arange(p, nem * m, m), np.arange(q, nem * m, m)
                g = np.array([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                # Replaced all asmatrixmul with @
                temp = B @ (g @ g.T.conj()) @ Bt
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                angles = vcp[:, K[2]]
                if angles[0] < 0: angles = -angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.array([[c, -np.conj(s)], [s, c]])
                    V[:, pair] = V[:, pair] @ G
                    M[pair, :] = G.T.conj() @ M[pair, :]
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    M[:, Ip], M[:, Iq] = temp1, temp2

    A = IW @ V
    S = V.T.conj() @ Y
    return A, S