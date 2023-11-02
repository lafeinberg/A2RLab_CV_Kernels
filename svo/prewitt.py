import numpy as np
from scipy.ndimage import correlate1d
from scipy.ndimage._ni_support import _normalize_sequence

def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Prewitt filter.

    Parameters
    ----------
    input : array_like
        Input array to filter.
    axis : int, optional
        Axis along which to calculate the Prewitt filter. Default is -1.
    output : array, optional
        The array in which to place the output, or None to create a new array.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the input array is extended
        beyond its boundaries. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0.

    Returns
    -------
    output : ndarray
        The result of the Prewitt filter.

    Examples
    --------
    >>> from scipy.ndimage import prewitt
    >>> from scipy import misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = prewitt(ascent)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    axis = normalize_axis_index(axis, input.ndim)
    output = _ni_support._get_output(output, input)
    modes = _normalize_sequence(mode, input.ndim)
    correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [1, 1, 1], ii, output, modes[ii], cval, 0)
    return output
