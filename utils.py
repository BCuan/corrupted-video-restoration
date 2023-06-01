import cv2
import os
import os.path as osp
import numpy as np


def vid_to_img(vid_name, save_path=None):
    if save_path:
        result_path = osp.join(save_path, osp.splitext(osp.basename(vid_name))[0])
        if not osp.exists(osp.join(result_path)):
            os.makedirs(result_path)

    ims = []
    cap = cv2.VideoCapture(vid_name)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n = 0
    ret = True
    while ret:
        # ATTENTION: default BGR for OpenCV reading. Conversion to RGB cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # may be required (especially for pretrained deep learning models).
        ret, frame = cap.read()
        if ret:
            ims.append(frame)
            if save_path:
                im_name = osp.join(result_path, str(n) + '.png')
                cv2.imwrite(im_name, frame)
        n += 1

    cap.release()

    return np.array(ims), fourcc, fps


def img_to_vid(ims, vid_name, save_path, fourcc, fps, debug=False):
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # ATTENTION: pre-compiled opencv-python is not shipped with ffmpeg due to licence issues (GPL).
    # Some encoders (e.g. AVC1 here) are therefore not supported by default.
    # Compilation from source with ffmpeg is required.
    wrt = cv2.VideoWriter(osp.join(save_path, vid_name), fourcc, fps, ims.shape[2:0:-1])  # opencv = column-major
    for i in range(ims.shape[0]):
        im = ims[i, ...]

        if debug:
            cv2.imshow('frame', im)
            cv2.waitKey(int(1000 / fps))
        wrt.write(im)

    wrt.release()

