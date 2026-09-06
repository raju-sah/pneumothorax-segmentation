"""Run-Length Encoding (RLE) utilities for SIIM-ACR pneumothorax masks.

SIIM-ACR encodes binary masks in 1-based Fortran (column-major) order:
pairs of (start_position, length).
The sentinel value '-1' or ' -1' indicates an empty mask (no pneumothorax).
"""

from typing import List, Union
import numpy as np


def rle_decode(
    rle_str: Union[str, float],
    shape: tuple = (1024, 1024),
) -> np.ndarray:
    """Decode a SIIM-ACR Run-Length Encoded string into a binary mask.

    Args:
        rle_str: Space-separated RLE string or sentinel ('-1').
        shape: Tuple of (height, width), default (1024, 1024).

    Returns:
        np.ndarray: Binary mask of shape `shape` with dtype uint8 (0 or 1).
    """
    if rle_str is None or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return np.zeros(shape, dtype=np.uint8)

    rle_str = str(rle_str).strip()
    if rle_str == "" or rle_str == "-1":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts = np.asarray([int(float(x)) for x in s[0::2]], dtype=int)
    lengths = np.asarray([int(float(x)) for x in s[1::2]], dtype=int)

    # 1-based indexing in Fortran order
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask into a SIIM-ACR Run-Length Encoded string.

    Args:
        mask: 2D binary array of shape (height, width) with values 0 or 1.

    Returns:
        str: Space-separated RLE string in Fortran order, or '-1' if empty.
    """
    if mask is None or np.sum(mask) == 0:
        return "-1"

    # Flatten in Fortran (column-major) order
    dots = mask.flatten(order="F")
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    runs = np.concatenate([[1] if dots[0] else [], runs, [len(dots) + 1] if dots[-1] else []])

    starts = runs[::2]
    lengths = runs[1::2] - starts

    res = []
    for s, l in zip(starts, lengths):
        res.extend([str(int(s)), str(int(l))])
    return " ".join(res)


def aggregate_rle_masks(
    rle_list: List[Union[str, float]],
    shape: tuple = (1024, 1024),
) -> np.ndarray:
    """Combine multiple RLE masks for an ImageId using bitwise logical OR.

    Args:
        rle_list: List of RLE strings associated with an ImageId.
        shape: Dimensions (height, width) of the target mask.

    Returns:
        np.ndarray: Unified binary mask of shape `shape` with dtype uint8.
    """
    composite = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        mask = rle_decode(rle, shape=shape)
        composite = np.bitwise_or(composite, mask)
    return composite
