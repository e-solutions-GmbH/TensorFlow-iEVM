from typing import Union, List, Generator

import numpy as np


def compute_batch_count(n_samples: int, batch_size: int) -> int:
    """
    Compute amount of batches for given amount of samples and batch size.

    Parameters
    ----------
    n_samples : int, amount of samples.
    batch_size: int, batch size.

    Returns
    -------
    n_batches : int, amount of batches to see all n_samples.
    """
    n_batches = int(n_samples / batch_size)
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches


def batch_generator(*arrs: np.ndarray, batch_size: Union[int, str]) -> Generator[List[np.ndarray], None, None]:
    """
    Make batches of given arrays.

    Parameters
    ----------
    arrs:       *np.ndarray.
                Arbitrary amount of arrays with same size in the first dimension. At least one array is required.
    batch_size: int, str
                int: The constant batch size to use.
                str: Arbitrary/dynamic batch sizes, has to start with 'list-', e.g. 'list-5-10-25-100'.
                     For an array of length 75, this would be returned:
                     1. batch: arr[0:5], 2. batch: arr[5:10], 3. batch: arr[10:25], 4. batch: arr[25:75].

    Returns
    -------
    A generator yielding batches of given arrays.
    """
    if len(arrs) == 0:
        raise ValueError("cannot make batches, at least 1 array ist required")

    n_samples = len(arrs[0])
    for arr in arrs[1:]:
        if len(arr) != n_samples:
            raise ValueError(
                "number of samples has to be equal for all array but is {} and {}".format(len(arr), n_samples))

    if isinstance(batch_size, int):
        for i in range(compute_batch_count(n_samples, batch_size)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            to_return = []
            for arr in arrs:
                to_return.append(arr[start_idx:end_idx])

            yield to_return

    elif isinstance(batch_size, str) and batch_size.startswith("list-"):
        batch_sizes = batch_size.split("-")[1:]
        batch_sizes = list(map(int, batch_sizes))
        start_idx = 0
        for end_idx in batch_sizes:
            end_idx2 = min(n_samples, end_idx)

            to_return = []
            for arr in arrs:
                to_return.append(arr[start_idx:end_idx2])

            yield to_return

            if end_idx >= n_samples:
                break
            start_idx = end_idx
    else:
        raise NotImplementedError("batch_generator not implemented for type '{}'".format(type(batch_size)))
