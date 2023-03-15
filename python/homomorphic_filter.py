#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2

#%% Parameters
gammaH = 1
gammaL = 0
c = 1
D0_2 = 5 ** 2 # variance (tightness)

# Load image
IMG_PATH_DIR = "./_img/keller.jpg"
f = plt.imread(IMG_PATH_DIR)

# Tranform image to grayscale and float
if f.ndim > 2:
    f = np.mean(f, axis=2)

f = np.array(f, dtype=float)
(M, N) = f.shape

#%% Logarithmitizing 

# (Turning assumed multiplicative illumination*reflectance structure of image
# into additive one)
f_log = np.log(f + 1) # + 1: shift log curve so no negative numbers

#%% FFT
F_log = fft2(f_log)

#%% Gaussian filter in frequency domain 

# (High pass to filter out slow changing illumnation changes)
# Modified Gaussian High pass as 1-Lowpass. 
# Modification: Defined lower and upper bounds gammaL and gammaH

# Filter Dimensions 2x image dimesions
P = 2 * M
Q = 2 * N

u = np.reshape(np.arange(P), (P, 1))
v = np.reshape(np.arange(Q), (1, Q))

# Mirror t get rid of edge effects
x = np.hstack((f_log, np.fliplr(f_log)))    # mirror below
F_log = fft2(np.vstack((x, np.flipud(x))))  # mirror to the right


D = (u-P/2)**2 + (v-Q/2)**2 # Distance Function
H = (gammaH - gammaL) * (1 - np.exp(-c*D/D0_2)) + gammaL

# center filter 
H = np.roll(H, [M, N], axis=(0, 1))

#%% Apply filter to image
F_log_filtered = F_log * H

#%% Reverse transform back to spacial domain
f_log_filtered  = np.real(ifft2(F_log_filtered)) # real() just for rounding errors
f_filtered = np.exp(f_log_filtered) # un-log and shift back
f_filtered = f_filtered[:M, :N]

#%% Display results
fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
ax[0].imshow(f, cmap="gray")
ax[0].title.set_text("Original")
ax[1].imshow(f_filtered, cmap="gray")
ax[1].title.set_text("Filtered")
for a in ax:
    a.axis("off")

fig2, ax2 = plt.subplots(figsize=(5, 5), constrained_layout=True)
p = ax2.imshow(np.roll(H, [M, N], axis=(0, 1)))
ax2.title.set_text("Shifted Filter")
fig2.colorbar(p)
plt.show(block=False)

