import numpy as np
import cv2


def rearrange(ims: np.ndarray):
    num_fr = ims.shape[0]

    # feature extraction
    # simplest: mean BGR (OpenCV) values under spatial sampling
    # other classic features like HOG to exploit for more complex situation
    vectors = get_features(ims)

    # distance calculation: L2 norm
    diff = vectors[:, np.newaxis, ...] - vectors[np.newaxis, ...]
    dist = np.linalg.norm(diff, ord=2, axis=-1)
    # adjacency matrix
    threshold = np.amax(dist) + 1
    dist += np.identity(num_fr) * threshold

    # shortest path: dynamic programming
    idx = np.unravel_index(dist.argmin(), dist.shape)
    dist[idx] = threshold
    dist[idx[::-1]] = threshold

    seq = list(idx)
    while len(seq) < num_fr:
        head = seq[0]
        tail = seq[-1]

        p_head = np.argmin(dist[head, :])
        p_tail = np.argmin(dist[tail, :])

        flag = dist[head, p_head] < dist[tail, p_tail]

        if flag:
            seq = [p_head, ] + seq
            dist[head, :] = threshold
            dist[:, head] = threshold

            dist[tail, p_head] = threshold
            dist[p_head, tail] = threshold

        else:
            seq += [p_tail, ]
            dist[tail, :] = threshold
            dist[:, tail] = threshold

            dist[head, p_tail] = threshold
            dist[p_tail, head] = threshold

    return seq


def get_features(ims: np.ndarray):
    # HOG-style average pooling
    # Or use filter2D / conv2D instead; speed comparison can be interesting.
    grid_shape = np.array((60, 60))  # (30, 30)
    window_shape = np.array((2, 2))  # convolution
    grid_num = np.array(ims.shape[1:3] / grid_shape, dtype=int)
    assert np.sum(grid_num * grid_shape - ims.shape[1:3]) == 0

    splits = np.array(np.split(ims, grid_num[1], axis=2))
    splits = np.array(np.split(splits, grid_num[0], axis=2)).transpose((2, 0, 1, 3, 4, 5))
    mean_splits = splits.mean(axis=(3, 4))

    windows = np.lib.stride_tricks.sliding_window_view(mean_splits, window_shape=window_shape, axis=(1, 2))
    mean_windows = windows.mean(axis=(-2, -1))

    vectors = mean_windows.reshape((mean_windows.shape[0], -1))

    return vectors


def is_reversed(ims: np.ndarray):
    # audio?

    # heterogeneous motion blur

    pass

    # optical flow
    # fr_p = None
    # for im in ims:
    #     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     if fr_p:
    #         flow = cv2.calcOpticalFlowPyrLK(fr_p, im_gray)
    #     else:
    #         fr_p = im_gray
