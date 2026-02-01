import numpy as np
from typing import Tuple


def discretize_preds(
        z: np.ndarray, 
        rng: np.random.Generator, 
        method: str = 'contra_discr'
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize relaxed (continuous) decisions `z` of the meta-learner into Traditional Handover (THO, `x`) or Conditional Handover (CHO, `y`) decisions.

    Parameters
    ----------
    z       : np.ndarray, shape (2I, J)
            Relaxed decision matrix `z` containing THO and CHO decisions (`x` and `y` respectively).
            First I rows contain the THO variables `x` and next I rows contain the CHO variables `y`, where I is the number of users and J the number of cells.
    rng     : np.random.Generator
            Random number generator used for reproducibility
    method  : str, optional
            Discretization strategy
            - 'none'            : no discretization 
            - 'contra_discr'    : discretization according to eq. (30)-(31) of the corresponding paper (see README for the link)

    Returns
    -------
    np.ndarray, shape (I, J)    : discretized THO decision matrix
    np.ndarray, shape (I, J)    : discretized CHO decision matrix
    
    Raises
    ------
    ValueError  : if discretization method not found
    """

    eps  = 1e-4
    I, J = int(z.shape[0]/2), int(z.shape[1])
    x, y = np.abs(z[:I, :]), np.abs(z[I:, :])
    if method == 'none':
        return x, y

    if method == 'contra_discr':
        x_hat = np.zeros_like(x, dtype=int)
        y_hat = np.zeros_like(y, dtype=int)
        for i in range(I):
            z_i = np.sum(x[i])
            # Bernoulli draw for THO or CHO mode
            z_hat = rng.random() < z_i  
            if z_hat:
                # THO mode
                if z_i < eps:
                    pr = np.full(J, 1.0 / J)    # fallback: uniform distribution
                else:
                    pr = x[i] / (z_i)
                x_hat[i, rng.choice(J, p=pr)] = 1
            else:
                # CHO mode
                if (1 - z_i) < eps:
                    pr = np.full(J, 1.0 / J)    # fallback: uniform distribution
                else:
                    pr = y[i] / (1 - z_i)
                for j in range(J):
                    prob_one  = np.abs(pr[j])
                    prob_one  = np.clip(prob_one,  0.0, 1.0)
                    y_hat[i, j] = rng.choice([0, 1], size=1, p=[1.0-prob_one, prob_one])[0].astype(float)
                # Ensure at least one prepared cell in CHO mode
                if not y_hat[i].any():
                    y_hat[i, np.argmax(pr)] = 1
        return x_hat, y_hat
    
    raise ValueError(f"Unknown discretization method: {method}")

