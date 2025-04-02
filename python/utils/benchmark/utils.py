from pathlib import Path
import typing
import logging

import numpy as np
from transforms3d.quaternions import qinverse, rotate_vector, qmult

VARIANTS_ANGLE_SIN = 'sin'
VARIANTS_ANGLE_COS = 'cos'

def convert_world2cam_to_cam2world(q, t):
    qinv = qinverse(q)
    tinv = -rotate_vector(t, qinv)
    return qinv, tinv

def load_poses(file: typing.IO, load_confidence: bool = False, is_multi_frame: bool = False):
    """
    Load poses from a text file and convert them to cam2world convention (t is the camera center in world coordinates).

    The text file encodes world2cam poses with the format:
    is_multi_frame = True:
        num_ref_img refimg_path1 ... refimg_pathN tarimg_path qw qx qy qz tx ty tz [confidence]
    is_multi_frame = False:
        imgpath qw qx qy qz tx ty tz [confidence]
    where qw qx qy qz is the quaternion encoding rotation,
    and tx ty tz is the translation vector,
    and confidence is a float encoding confidence, for estimated poses

    Parameters:
        file (typing.IO): Input file object.
        load_confidence (bool): Whether to load confidence/loss values.
        is_multi_frame (bool): If True, use the format for multiple frames as reference images. If False, use the standard format.

    Returns:
        dict: Dictionary of poses where the key is the frame number, and the value is (q, t, confidence/loss).
    """
    poses = dict()

    for line_number, line in enumerate(file.readlines()):
        parts = tuple(line.strip().split(' '))

        if '#' in parts[0]:
            logging.info(f'Ignoring comment line in {file.name} line {line_number}')
            continue

        # Determine the expected number of fields
        if is_multi_frame:
            num_ref = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
            expected_parts = num_ref + 10 if load_confidence else num_ref + 9
        else:
            expected_parts = 9 if load_confidence else 8

        if len(parts) != expected_parts:
            logging.warning(
                f'Invalid number of fields in file {file.name} line {line_number}. '
                f'Expected {expected_parts}, received {len(parts)}. Ignoring line.')
            continue

        try:
            # Parse file based on the format
            if is_multi_frame:
                num_ref = int(parts[0])
                frame_name = ",".join(parts[1:2+num_ref])
                data_offset = 2 + num_ref
            else:
                name = parts[0]
                if '#' in name:
                    logging.info(f'Ignoring comment line in {file.name} line {line_number}')
                    continue
                frame_name = name
                data_offset = 1

        except ValueError:
            logging.warning(
                f'Invalid frame number in file {file.name} line {line_number}. '
                f'Expected formatting like "seq1/frame_00000.jpg". Ignoring line.')
            continue

        try:
            parts_float = tuple(map(float, parts[data_offset:]))
            if any(np.isnan(v) or np.isinf(v) for v in parts_float):
                raise ValueError()
            qw, qx, qy, qz, tx, ty, tz = parts_float[:7]
            score = parts_float[7] if load_confidence else None
        except ValueError:
            logging.warning(
                f'Error parsing pose in file {file.name} line {line_number}. Ignoring line.')
            continue

        q = np.array((qw, qx, qy, qz), dtype=np.float64)
        t = np.array((tx, ty, tz), dtype=np.float64)

        if np.isclose(np.linalg.norm(q), 0):
            logging.warning(
                f'Error parsing pose in file {file.name} line {line_number}. '
                'Quaternion must have non-zero norm. Ignoring line.')
            continue

        # Convert world2cam to cam2world convention
        q, t = convert_world2cam_to_cam2world(q, t)
        poses[frame_name] = (q, t, score)

    return poses


def subsample_poses(poses: dict, subsample: int = 1):
    return {k: v for i, (k, v) in enumerate(poses.items()) if i % subsample == 0}


def load_K(file_path: Path):
    K = dict()
    with file_path.open('r', encoding='utf-8') as f:
        for line in f.readlines():
            if '#' in line:
                continue
            line = line.strip().split(' ')

            frame_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line[1:])
            K[frame_name] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, W, H


def quat_angle_error(label, pred, variant=VARIANTS_ANGLE_SIN) -> np.ndarray:
    assert label.shape == (4,)
    assert pred.shape == (4,)
    assert variant in (VARIANTS_ANGLE_SIN, VARIANTS_ANGLE_COS), \
        f"Need variant to be in ({VARIANTS_ANGLE_SIN}, {VARIANTS_ANGLE_COS})"

    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(label.shape) != 2 or label.shape[0] != 1 or label.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of label: {label.shape}, expected: (1, 4)")

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    if len(pred.shape) != 2 or pred.shape[0] != 1 or pred.shape[1] != 4:
        raise RuntimeError(f"Unexpected shape of pred: {pred.shape}, expected: (1, 4)")

    label = label.astype(np.float64)
    pred = pred.astype(np.float64)

    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    if variant == VARIANTS_ANGLE_COS:
        d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))
        d = np.clip(d, a_min=-1, a_max=1)
        angle = 2. * np.degrees(np.arccos(d))
    elif variant == VARIANTS_ANGLE_SIN:
        if q1.shape[0] != 1 or q2.shape[0] != 1:
            raise NotImplementedError(f"Multiple angles is todo")
        # https://www.researchgate.net/post/How_do_I_calculate_the_smallest_angle_between_two_quaternions/5d6ed4a84f3a3e1ed3656616/citation/download
        sine = qmult(q1[0], qinverse(q2[0]))  # note: takes first element in 2D array
        # 114.59 = 2. * 180. / pi
        angle = np.arcsin(np.linalg.norm(sine[1:], keepdims=True)) * 114.59155902616465
        angle = np.expand_dims(angle, axis=0)

    return angle.astype(np.float64)


def precision_recall(inliers, tp, failures):
    """
    Computes Precision/Recall plot for a set of poses given inliers (confidence) and wether the
    estimated pose error (whatever it may be) is within a threshold.
    Each point in the plot is obtained by choosing a threshold for inliers (i.e. inlier_thr).
    Recall measures how many images have inliers >= inlier_thr
    Precision measures how many images that have inliers >= inlier_thr have 
    estimated pose error <= pose_threshold (measured by counting tps)
    Where pose_threshold is (trans_thr[m], rot_thr[deg])

    Inputs:
        - inliers [N]
        - terr [N]
        - rerr [N]
        - failures (int)
        - pose_threshold (tuple float)
    Output
        - precision [N]
        - recall [N]
        - average_precision (scalar)
    """

    assert len(inliers) == len(tp), 'unequal shapes'

    # sort by inliers (descending order)
    inliers = np.array(inliers)
    sort_idx = np.argsort(inliers)[::-1]
    inliers = inliers[sort_idx]
    tp = np.array(tp).reshape(-1)[sort_idx]

    # get idxs where inliers change (avoid tied up values)
    distinct_value_indices = np.where(np.diff(inliers))[0]
    threshold_idxs = np.r_[distinct_value_indices, inliers.size - 1]

    # compute prec/recall
    N = inliers.shape[0]
    rec = np.arange(N, dtype=np.float32) + 1
    cum_tp = np.cumsum(tp)
    prec = cum_tp[threshold_idxs] / rec[threshold_idxs]
    rec = rec[threshold_idxs] / (float(N) + float(failures))

    # invert order and ensures (prec=1, rec=0) point
    last_ind = rec.searchsorted(rec[-1])
    sl = slice(last_ind, None, -1)
    prec = np.r_[prec[sl], 1]
    rec = np.r_[rec[sl], 0]

    # compute average precision (AUC) as the weighted average of precisions
    average_precision = np.abs(np.sum(np.diff(rec) * np.array(prec)[:-1]))

    return prec, rec, average_precision
