import numpy as np
import autograd.numpy as anp


def g_t_all(
        x: np.ndarray, 
        y: np.ndarray, 
        t: int, 
        I: int, 
        J: int, 
        alpha: float, 
        w: np.ndarray, 
        sinr: np.ndarray
    ) -> anp.ndarray:
    """
    Instantaneous utility per-slot `t` used by the online learning procedure. 
    It is written using `autograd.numpy` so that gradients wrt to `x` and `y` can be computed.
    
    The calculations refer to eq. (9)-(12) of the corresponding paper (see README for the link).

    Parameters
    ----------
    x       : np.ndarray, shape (I, J) 
            Traditional Handover (THO) decision
    y       : np.ndarray, shape (I, J)
            Conditional Handover (CHO) decision
    t       : int
            Time index
    I       : int
            Number of users
    J       : int
            Number of cells
    alpha   : float
            Tightness of approximation, in eq. (11) of the corresponding paper (see README for the link)
    w       : np.ndarray, length J
            Bandwidth of each cell
    sinr    : np.ndarray, shape (T, I, J)
            SINR values in linear scale 

    Returns
    -------
    np.ndarray, shape (I, J)    : utility
    """
    
    x     = x.astype(anp.float64)
    y     = y.astype(anp.float64)
    sinr  = sinr.astype(anp.float64)
    c     = anp.array([[w[j] * anp.log(1 + sinr[t, i, j]) / anp.log(2) for j in range(J)] for i in range(I)]) # Throughput, from eq. (2) and (3) from the corresponding paper

    exp_terms = y * (c ** alpha)
    row_sums_y = anp.sum(exp_terms, axis=1)

    f_y = (1.0 / alpha) * anp.log(row_sums_y + 1)

    l_j   = anp.sum(x, axis=0) + anp.sum(y, axis=0) + 1
    total = anp.sum(x * anp.log(c+1)) + anp.sum(f_y) - anp.sum(l_j * anp.log(l_j))
    return total