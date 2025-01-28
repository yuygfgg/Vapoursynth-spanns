import vapoursynth as vs
from vapoursynth import core

import collections.abc
from functools import partial
import logging
from typing import Union, Sequence, Tuple, Union, List, Callable, Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided

import pywt

from scipy.integrate import quad
from scipy import stats, signal, ndimage
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import (_ShapeInfo,)
from scipy import interpolate
from scipy.optimize import minimize


logger = logging.getLogger(__name__)


class mp_gen(rv_continuous):
    """Marchenko-Pastur Distribution class.

    The Marchenko-Pastur Law describes the spectrum of the Wishart random matrices.
    Therefore the spectrum of the random matrices of this ensemble converge
    to the Marchenko-Pastur Law when the matrix size goes to infinity. This
    class provides methods to sample eigenvalues following Marchenko-Pastur
    distribution, computing the PDF and CDF.

    Attributes:
        ratio (float): random matrix size ratio. This is the ratio between the
            number of degrees of freedom 'p' and the sample size 'n'. The value
            of ratio = p/n.
        beta (int): descriptive integer of the Wishart ensemble type.
            For WRE beta=1, for WCE beta=2, for WQE beta=4.
        sigma (float): scale of the distribution. This value also corresponds
            to the standard deviation of the random entries of the sampled matrix.

    Derives:
        lambda_minus (float): lower bound of the support of the Marchenko-Pastur Law.
            It depends on beta, on the scale (sigma) and on the ratio.
        lambda_plus (float): upper bound of the support of the Marchenko-Pastur Law.
            It depends on beta, on the scale (sigma) and on the ratio.

    References:
        - Bar, Z.D. and Silverstain, J.W.
            "Spectral Analysis of Large Dimensional Random Matrices".
            2nd edition. Springer. (2010).

    """

    ARCTAN_OF_INFTY = np.pi/2

    @staticmethod
    def get_lambdas(beta, sigma, ratio):
        lambda_minus = beta * sigma**2 * (1 - np.sqrt(ratio))**2
        lambda_plus = beta * sigma**2 * (1 + np.sqrt(ratio))**2

        return lambda_minus, lambda_plus

    @staticmethod
    def get_var(beta, sigma):
        return beta * sigma**2

    def _get_support(self, beta, sigma, ratio):
        return self.get_lambdas(beta, sigma, ratio)

    def _set_default_interval(self) -> None:
        # computing interval according to the matrix size ratio and support
        if self.ratio <= 1:
            self.default_interval = (self.lambda_minus, self.lambda_plus)
        else:
            self.default_interval = (min(-0.05, self.lambda_minus), self.lambda_plus)

    def _approximate_inv_cdf(self) -> None:
        # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
        x_vals = np.linspace(self.lambda_minus, self.lambda_plus, 1000)
        _pdf = self._pdf(x_vals)
        _cdf = np.cumsum(_pdf)      # approximating CDF
        cdf_y = _cdf/_cdf.max()     # normalizing approximated CDF to 1.0
        self._inv_cdf = interpolate.interp1d(cdf_y, x_vals)

    def _rvs(
        self,
        size: Union[int, Tuple[int]],
        random_state: int,
        _random_state: int = None,
    ) -> np.ndarray:
        # pylint: disable=arguments-differ
        if _random_state is not None:
            np.random.seed(_random_state)

        uniform_samples = np.random.random(size=size)
        return self._inv_cdf(uniform_samples)

    def _argcheck(self, beta, sigma, ratio):
        return beta > 0 and sigma > 0 and 0 < ratio < 1

    def _shape_info(self):
        ibeta = _ShapeInfo("beta", False, (1., 8.), (True, True))
        isigma = _ShapeInfo("sigma", False, (0., np.inf), (False, False))
        iratio = _ShapeInfo("ratio", False, (0., 1.), (False, False))

        return [ibeta, isigma, iratio]

    def _fitstart(self, x):
        lm, lp = np.quantile(x, [0.05, 0.95])
        a = np.sqrt(lm)
        b = np.sqrt(lp)

        sigma = np.sqrt((a + b) ** 2 / 4.)
        ratio = np.sqrt((b - a) / (a + b))

        return 1., sigma, ratio, 0., 1.

    def _pdf(self, x: Union[float, np.ndarray], beta, sigma, ratio) -> Union[float, np.ndarray]:
        """Computes PDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.

        Returns:
            float or numpy array with the computed PDF in the given value(s).

        """
        lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
        var = self.get_var(beta, sigma)

        # pylint: disable=arguments-differ
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(relu(lambda_plus - x) * relu(x - lambda_minus)) / (2.0 * np.pi * ratio * var * x)

    def _cdf(self, x: Union[float, np.ndarray], beta, sigma, ratio) -> Union[float, np.ndarray]:
        """Computes CDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.

        Returns:
            float or numpy array with the computed CDF in the given value(s).

        """

        lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
        # var = self.get_var(beta, sigma)

        # pylint: disable=arguments-differ
        with np.errstate(divide='ignore', invalid='ignore'):
            acum = indicator(x, start=lambda_plus, inclusive="left")
            acum += np.where(
                indicator(
                    x,
                    start=lambda_minus,
                    stop=lambda_plus,
                    inclusive="left",
                ),
                self._cdf_aux_f(x, beta, sigma, ratio),
                0.0,
            )

            if ratio <= 1:
                return acum

            acum += np.where(
                indicator(
                    x,
                    start=lambda_minus,
                    stop=lambda_plus,
                    inclusive="left",
                ),
                (ratio-1)/(2*ratio),
                0.0,
            )

            return acum

    def _cdf_aux_f(self, x: Union[float, np.ndarray], beta, sigma, ratio) -> Union[float, np.ndarray]:

        lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
        var = self.get_var(beta, sigma)

        cdf_aux_r = self._cdf_aux_r(x, lambda_minus, lambda_plus)

        # pylint: disable=line-too-long
        first_arctan_term = np.where(
            x == lambda_minus,
            mp_gen.ARCTAN_OF_INFTY,
            np.arctan((cdf_aux_r**2 - 1)/(2 * cdf_aux_r))
        )

        second_arctan_term = np.where(
            x == lambda_minus,
            mp_gen.ARCTAN_OF_INFTY,
            np.arctan((lambda_minus*cdf_aux_r**2 - lambda_plus) / (2*var*(1-ratio)*cdf_aux_r))
        )

        return 1/(2*np.pi*ratio) * (np.pi*ratio + (1/var)*np.sqrt(relu(lambda_plus-x)*relu(x-lambda_minus)) - (1+ratio)*first_arctan_term + (1-ratio)*second_arctan_term)

    def _cdf_aux_r(self, x: Union[float, np.ndarray], lambda_minus, lambda_plus) -> Union[float, np.ndarray]:
        return np.sqrt((lambda_plus-x)/(x - lambda_minus))


marchenkopastur = mp_gen(name='marchenkopastur')


def mp_fit(x):

    def f(ub):
        xp = x[x < ub]
        theta = marchenkopastur.fit(xp, floc=0., fscale=1.)
        return marchenkopastur.nnlf(theta, xp)

    # This optimization step definitely complicate things, but for the sake of prototyping it's kept to simulate ub cutoffs
    blocks = int(np.round(np.sqrt(x.size)))
    block_size = (np.quantile(x, 0.99) - np.quantile(x, 0.3)) / (blocks)
    guesses = np.linspace(np.quantile(x, 0.3), np.quantile(x, 0.99), blocks + 2)[1: -1]
    ubi = np.argmin([f(x) for x in guesses])
    ub = guesses[ubi]
    # skip fine-grained search
    # guesses = np.linspace(ub - block_size / 2, ub + block_size / 2, blocks)
    # ubi = np.argmin([f(x) for x in guesses])
    # ub = guesses[ubi]
    xp = x[x < ub]
    theta = marchenkopastur.fit(xp)
    return theta


def spanns_core(orig, ref, tolerance=0.7):

    U, sig, Vt = np.linalg.svd(orig, full_matrices=False)
    Un, sig_n, Vnt = np.linalg.svd(ref - orig, full_matrices=False)

    try:
        params = mp_fit(sig_n)
    except Exception as e:
        logger.debug(str(e))
        return orig

    mp = marchenkopastur(*params)

    sig_r = sig - sig * (1 - mp.cdf(sig)) * (1 - tolerance)

    T = U @ np.diag(sig_r) @ Vt
    return T


def get_mask(A, gamma=0.5):

    w1 = 'bior1.1'
    _, D = pywt.dwt2(A, w1)
    B1 = np.abs(pywt.idwt2((None, D), w1))

    w2 = 'coif1'
    _, D = pywt.dwt2(A, w2)
    B2 = np.abs(pywt.idwt2((None, D), w2))

    B = np.maximum(B1, B2)
    B = ndimage.maximum_filter(B, size=3)

    thresh = np.quantile(B.ravel(), gamma)

    if thresh > B.min():
        B = np.clip((B - B.min()) / (thresh - B.min()), 0., 1.)
    else:
        B = np.zeros_like(B)
    return B


def spanns_routine(orig, sigma=1, tol=0.7, gamma=0.5, steps=2, lref=None, dref=None):

    mask = get_mask(orig, gamma)

    # ref = ref if ref is not None else ndimage.median_filter(orig, size=3)
    dref = dref if dref is not None else ndimage.gaussian_filter(orig, sigma=sigma)

    T = orig

    for i in range(steps):
        T = spanns_core(T, dref, tol)
        T = (mask) * T + (1 - mask) * dref

    if lref is not None:
        lb = np.minimum(orig, ref)
        ub = np.maximum(orig, ref)
        orig = np.clip(T, lb, ub)
    else:
        orig = T

    return orig


def frame_spanns_routine(n, f, tol=0.7, gamma=0.5, steps=2, planes=[0, 1, 2], limit=False):

    src = frame_to_matrices(f[0])
    dref = frame_to_matrices(f[1])

    if limit:
        lref = frame_to_matrices(f[2])

    for i in planes:
        s, l, d = src[i], (lref[i] if limit else None), dref[i]
        src[i, :] = spanns_routine(s, tol=tol, gamma=gamma, steps=steps, lref=l, dref=d)

    return matrices_to_frame(src, f[0].copy())


def spanns(clip: vs.VideoNode, sigma: int = 1, tol: float = 0.7, gamma: float = 0.5, passes: int = 2, ref1: vs.VideoNode = None, ref2: vs.VideoNode = None, planes: Sequence[int] = [0, 1, 2], core=vs.core):
    """
    The Survival Probability Adapted Nuclear Norm Shrinkage Denoiser

    Args:
        clip: original video node
        sigma: radius of boxblur (instead of gaussian filter)
        tol: noise tolerance, 0. to 1.
        gamma: texture threshold, 0. to 1., higher value preserves less texture.
        passes: number of denoising steps
        ref1: reference blurred clip, `sigma` will be ignored if `ref1` is provided. default to boxblur with radius sigma
        ref2: reference clip, an approximation for result.
        planes: indices of planes to process
    """

    ref1 = ref1 if ref1 is not None else core.std.BoxBlur(clip, planes, hradius=sigma, hpasses=3, vradius=sigma, vpasses=3)

    if ref2 is not None:
        clips = [clip, ref1, ref2]
        lref = True
    else:
        clips = [clip, ref1]
        lref = False

    routine = partial(frame_spanns_routine, tol=tol, gamma=gamma, steps=passes, planes=planes, limit=lref)

    return core.std.ModifyFrame(clip=clip, clips=clips, selector=routine)


def submatof(mat,rad):
    mat = np.pad(mat, ((rad,rad), (rad,rad)), mode='constant', constant_values=0)
    view_shape = tuple(np.subtract(mat.shape, (rad * 2 + 1,rad * 2 + 1)) + 1) + (rad * 2 + 1,rad * 2 + 1)
    mat_view = as_strided(mat, view_shape, mat.strides * 2)
    # mat_view = mat_view.reshape((-1,) + shape)
    return mat_view


def frame_to_matrices(f: vs.VideoFrame):
    return np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])


def matrices_to_frame(m, f: vs.VideoFrame):
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), m[plane, :, :])
    return f


def relu(x: Union[float,np.ndarray]):
    """
    Element-wise maximum between the value and zero.

    Args:
        x (ndarray): list of numbers to compute its element-wise maximum.

    Returns:
        array_like consisting in the element-wise maximum vector of the given values.
    """
    # return np.maximum(x, np.zeros_like(x))
    return (np.abs(x) + x) / 2


def indicator(x: np.ndarray, start: Optional[float] = None, stop: Optional[float] = None, inclusive: str = "both") -> np.ndarray:
    """
    Element-wise indicator function within a real interval.

    The interval can be left-closed, right-closed, closed, or open.
    For more details, visit:
    https://en.wikipedia.org/wiki/Indicator_function

    Args:
        x (np.ndarray):
            Array (or list-like) of real values for which to compute
            the element-wise indicator image.
        start (float, optional):
            Left boundary of the interval. If not provided,
            the left boundary is treated as -∞.
        stop (float, optional):
            Right boundary of the interval. If not provided,
            the right boundary is treated as +∞.
        inclusive (str, optional):
            Determines which part(s) of the interval are closed.
            - "both": [start, stop]
            - "left": [start, stop)
            - "right": (start, stop]
            - "neither": (start, stop)
            Defaults to "both".

    Returns:
        np.ndarray:
            An array of 1s and 0s, where 1 indicates that
            the corresponding element of `x` lies within
            the specified interval, and 0 otherwise.

    Raises:
        ValueError: If both `start` and `stop` are `None`.
        ValueError: If `inclusive` is an invalid option.
    """

    if start is None and stop is None:
        raise ValueError("Error: provide start and/or stop for the indicator function.")

    valid_inclusive = {"both", "left", "right", "neither"}
    if inclusive not in valid_inclusive:
        raise ValueError(
            f"Error: invalid 'inclusive' parameter: {inclusive}. "
            f"'inclusive' must be one of {valid_inclusive}."
        )

    # Determine whether boundaries are inclusive or exclusive
    left_inclusive = inclusive in {"both", "left"}
    right_inclusive = inclusive in {"both", "right"}

    # Left condition
    if start is not None:
        if left_inclusive:
            left_condition = x >= start
        else:
            left_condition = x > start
    else:
        # No left boundary, so everything passes
        left_condition = np.ones_like(x, dtype=bool)

    # Right condition
    if stop is not None:
        if right_inclusive:
            right_condition = x <= stop
        else:
            right_condition = x < stop
    else:
        # No right boundary, so everything passes
        right_condition = np.ones_like(x, dtype=bool)

    # Combine conditions in a single pass
    condition = left_condition & right_condition

    return np.where(condition, 1.0, 0.0)
