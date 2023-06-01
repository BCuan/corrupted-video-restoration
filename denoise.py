import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_noise(ims: np.ndarray):
    pixel_mean = ims.mean(axis=(1, 2))
    mean = pixel_mean.mean(axis=0)
    std = pixel_mean.std(axis=0)

    outliers = np.any(np.abs(pixel_mean - mean) > std, axis=1)
    print("Noise frame ids: " + str(np.arange(len(outliers))[outliers]))

    return outliers


def find_noise_by_filter(ims: np.ndarray):
    ims_gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims]
    ps_list = [np.fft.fftshift(np.fft.fft2(im)) for im in ims_gray]
    mgn = [np.abs(ps) for ps in ps_list]
    agl = [np.angle(ps) for ps in ps_list]

    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(mgn[17])), cmap='gray')
    print()

