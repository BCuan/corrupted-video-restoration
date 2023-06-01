import os.path as osp
import numpy as np
from utils import vid_to_img, img_to_vid
from denoise import find_noise, find_noise_by_filter
from rearrange import rearrange, is_reversed
import time


if __name__ == '__main__':
    result_path = 'results'
    restoration_path = osp.join(result_path, 'restored')

    debug = False
    if debug:
        raw_path = osp.join(result_path, 'raw_ims')
    else:
        raw_path = None

    vid_name = 'corrupted_video.mp4'
    ims, fourcc, fps = vid_to_img(vid_name, raw_path)

    start = time.time()

    # testing frequency filter
    # noises_fr = find_noise_by_filter(ims)

    # noise removal
    noises = find_noise(ims)
    ims = ims[~noises, ...]

    # rearrange the frames
    seq = rearrange(ims)

    ids_corrupted = np.arange(len(noises))[~noises]
    ids = ids_corrupted[seq]  # rearranged sequence of original frame ids

    # Disordered frames = undirected graph, no motion direction info can be simply acquired
    # Determining whether the sequence is reversed is very difficult without prior information
    # Unfortunately, we need an order reference
    reference = (0, 113)  # e.g. an ordered pair indicating one frame is in front of another (no need to be adjacent)
    if list(ids).index(reference[0]) > list(ids).index(reference[1]):
        seq = seq[::-1]
        ids = ids[::-1]

    print("Restored sequence: " + str(ids))

    end = time.time()
    print("Elapsed Time: " + str(end - start))

    # save the video and the sequence
    vid_basename = osp.basename(vid_name)
    txt_basename = osp.splitext(vid_basename)[0] + '.txt'

    ims = ims[seq, ...]
    img_to_vid(ims, vid_basename, restoration_path, fourcc, fps, debug)
    np.savetxt(osp.join(restoration_path, txt_basename), ids, '%d')
