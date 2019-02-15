"""Utility functions."""

import numpy as np
import time
import os

import torch

def img_normalize(x):
    return ((x - torch.min(x)) / (torch.max(x) - torch.min(x))).view(
                1, x.size()[0], x.size()[1]
            )

def visualize_seg(label_map, mc, one_hot=False):
    if one_hot:
        label_map = np.argmax(label_map, axis=-1)

    out = torch.zeros([label_map.size()[0], label_map.size()[1], label_map.size()[2], 3], dtype=torch.float64)

    for l in range(1, mc.NUM_CLASS):
        out[label_map==l, :] = torch.from_numpy(mc.CLS_COLOR_MAP[l])

    return out.transpose(2,3).transpose(1,2)

def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:,:,::-1])

    return out

class Timer(object):
    def __init__(self):
        self.total_time   = 0.0
        self.calls        = 0
        self.start_time   = 0.0
        self.duration     = 0.0
        self.average_time = 0.0


    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.duration = time.time() - self.start_time
        self.total_time += self.duration
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.duration

def conf_error_rate_at_thresh_fn(mask, conf, thresh):
    return np.mean((conf>thresh) != mask)

def rmse_fn(diff, nnz):
    return np.sqrt(np.sum(diff**2)/nnz)

def abs_accuracy_at_thresh_fn(diff, thresh, mask):
    return np.sum((np.abs(diff) < thresh)*mask)/float(np.sum(mask))

def rel_accuracy_at_thresh_fn(pred_ogm, gt_ogm, mask, thresh):
    return np.sum(
        mask * (np.maximum(pred_ogm, gt_ogm) /
                np.minimum(gt_ogm, pred_ogm) < thresh)
        ) / float(np.sum(mask))

def evaluate(label, pred, n_class):
    """Evaluation script to compute pixel level IoU.

    Args:
        label: N-d array of shape [batch, W, H], where each element is a class index.
        pred: N-d array of shape [batch, W, H], the each element is the predicted class index.
        n_class: number of classes
        epsilon: a small value to prevent division by 0

    Returns:
        IoU: array of lengh n_class, where each element is the average IoU for this class.
        tps: same shape as IoU, where each element is the number of TP for each class.
        fps: same shape as IoU, where each element is the number of FP for each class.
        fns: same shape as IoU, where each element is the number of FN for each class.
    """

    assert label.shape == pred.shape, \
        'label and pred shape mismatch: {} vs {}'.format(label.shape, pred.shape)

    label = label.cpu().numpy()
    pred = pred.cpu().numpy()

    tp = np.zeros(n_class)
    fn = np.zeros(n_class)
    fp = np.zeros(n_class)

    for cls_id in range(n_class):
        tp_cls = np.sum(pred[label == cls_id] == cls_id)
        fp_cls = np.sum(label[pred == cls_id] != cls_id)
        fn_cls = np.sum(pred[label == cls_id] != cls_id)

        tp[cls_id] = tp_cls
        fp[cls_id] = fp_cls
        fn[cls_id] = fn_cls

    return tp, fp, fn

def print_evaluate(mc, name, value):
    print(f'{name}:')
    for i in range(1, mc.NUM_CLASS):
        print(f'{mc.CLASSES[i]}: {value[i]}')
    print()

def save_checkpoint(path, epoch, model):
    save_path = os.path.join(path, f"model_path_{epoch}.pkl")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model_dir, epoch, model):
    load_path = os.path.join(model_dir, f"model_epoch_{epoch}.pkl")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded to {load_path}")


def condensing_matrix(in_channel, size_z, size_a):
    assert size_z % 2 == 1 and size_a % 2==1, \
        'size_z and size_a should be odd number'

    half_filter_dim = (size_z*size_a)//2

    # moving neigboring pixels to channel dimension
    nbr2ch_mat = np.zeros(
        (size_z*size_a*in_channel, in_channel, size_z, size_a),
        dtype = np.float32
    )

    for z in range(size_z):
        for a in range(size_a):
            for ch in range(in_channel):
                nbr2ch_mat[z*(size_a*in_channel) + a*in_channel + ch, ch, z, a] = 1

    # exclude the channel index corresponding to the center position
    nbr2ch_mat = np.concatenate(
        [nbr2ch_mat[:in_channel*half_filter_dim, :, :, :],
            nbr2ch_mat[in_channel*(half_filter_dim+1):, :, :, :]],
        axis = 0
    )

    #assert nbr2ch_mat.shape == (size_z, size_a, in_channel, (size_a*size_z-1)*in_channel), \
    #   'error with the shape of nbr2ch_mat after removing center position'

    return nbr2ch_mat

def angular_filter_kernel(in_channel, size_z, size_a, theta_sqs):
    """Compute a gaussian kernel.
    Args:
        size_z: size on the z dimension.
        size_a: size on the a dimension.
        in_channel: input (and output) channel size
        theta_sqs: an array with length == in_channel. Contains variance for
            gaussian kernel for each channel.

    Returns:
        kernel: ND array of size [in_channel, in_channel, size_z, size_a], which is
            just guassian kernel parameters for each channel.
    """

    assert size_z % 2 == 1 and size_a % 2==1, \
        'size_z and size_a should be odd number'

    assert len(theta_sqs) == in_channel, \
        'length of theta_sqs and in_channel does no match'

    # gaussian kernel
    kernel = np.zeros((in_channel, in_channel, size_z, size_a), dtype=np.float32)
    for k in range(in_channel):
        kernel_2d = np.zeros((size_z, size_a), dtype=np.float32)
        for i in range(size_z):
            for j in range(size_a):
                diff = np.sum((np.array([i-size_z//2, j-size_a//2]))**2)
                kernel_2d[i, j] = np.exp(-diff/2/theta_sqs[k])

        # exclude the center position
        kernel_2d[size_z//2, size_a//2] = 0
        kernel[k, k, :, :] = kernel_2d

    return kernel

