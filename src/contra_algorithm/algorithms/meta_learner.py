import numpy as np
import autograd.numpy as anp
from typing import Tuple
from autograd import grad
from joblib import Parallel, delayed
from contra_algorithm.discretization.discretize import discretize_preds
from contra_algorithm.objectives.utility import g_t_all
from contra_algorithm.algorithms.experts import OGAExpert, process_expert


def compute_hyperparameters(
        T: int, I: int, J: int, w: np.ndarray, A: np.ndarray, B: np.ndarray, SINRdb_max: float, SINR_scaler: float
    ) -> Tuple[float, np.ndarray]:
    """
    Compute learning rate of meta-learner (`eta`) and learning rates of each expert (`step_sizes`).

    Parameters
    ----------
    T           : int
                Number of slots
    I           : int
                Number of users
    J           : int
                Number of cells
    w           : np.ndarray, length J
                Bandwidth of each cell
    A           : np.ndarray, shape (I, J)
                THO delay matrix
    B           : np.ndarray, shape (I, J)
                CHO delay matrix
    SINRdb_max  : float
                Maximum value of SINR in dB 
    SINR_scaler : float
                Used to scale down SINRs as solvers need smaller numbers. With SINR_scaler=1, max(SINR)~197 which creates huge numbers

    Returns
    -------
    eta         : float
                Learning rate of meta-learner 
    step_sizes  : np.ndarray, length `K` (number of experts)
                Learning rates of experts
    """
              
    max_A   = np.max(A)
    max_B   = np.max(B)
    max_c   = np.max(w) * np.log2(1 + 10**(SINRdb_max / 10) / SINR_scaler) # SINR in dB transformed to linear scale
    # Constants introduced in the paper (see README for the link)
    D       = np.max([np.sqrt(2 * I), np.sqrt(I * (J - 1))])
    D_C     = np.max([np.sqrt(2 * I * max_A), np.sqrt(I * (J - 1) * max_B)])
    M       = np.max([np.log(I) + 1, np.log(max_c) - 1])
    G       = M * np.sqrt(I * J)
    G_C     = M * np.sqrt(I * J * np.max([max_A, max_B]))
    K       = np.ceil(np.log2(np.sqrt(1 + 2 * T))) + 1 # Number of experts
    ni      = (D_C + 1/8) * ((G * D + 2 * D_C)**2)

    eta         = np.sqrt(1 / (T * ni))
    step_sizes  = np.array([np.sqrt((D_C**2) / (T * (G**2 + 2 * G_C))) * (2**(k - 1)) for k in range(1, int(K + 1))])

    return eta, step_sizes


class CONTRA:
    def __init__(
            self, eta, step_sizes, T, I, J, 
            initial_x, initial_y, 
            A, B, 
            discretize_method, 
            alpha, w, sinr, 
            rng, 
            solver
        ):
        # Hyperparameters and problem dimensions
        self.eta = eta                  # Learning rate of meta-learner
        self.step_sizes = step_sizes    # Learning rates of experts
        self.T = T                      # Number of slots
        self.I = I                      # Number of users
        self.J = J                      # Number of cells

        # Initial decisions for THO and CHO decisions 
        self.initial_x = initial_x
        self.initial_y = initial_y

        # Switching costs
        self.A = A                  # THO delay
        self.B = B                  # CHO signaling

        # Used for discretizing relaxed (continuous) decisions to binary
        self.discretize_method = discretize_method

        # Tightness of approximation, in eq. (11) of the corresponding paper (see README for the link)
        self.alpha = alpha

        # Bandwidth of each cell
        self.w = w

        # SINR values in linear scale 
        self.sinr = sinr

        # Random number generator used for reproducibility
        self.rng = rng

        # Solver used
        self.solver = solver

        # Define experts and their weights
        self.experts = np.array([OGAExpert(step_size, self.T, self.I, self.J, self.initial_x, self.initial_y, self.A, self.B, self.solver) for step_size in self.step_sizes]) # create as many experts as the step sizes
        self.weights = np.array([(1 + 1 / len(self.step_sizes)) / (i * (i + 1)) for i in range(1, len(self.experts) + 1)]) # initialize weights
        
        # Used for storing the decisions/results
        self.x_contra = np.empty((self.T, self.I, self.J))
        self.y_contra = np.empty((self.T, self.I, self.J))
        self.g_contra = np.empty(self.T)
    

    def run(self):
        exponential_weighting = np.zeros(len(self.experts))
        gradient_func_x = grad(g_t_all, 0)      # compute the gradient wrt first argument (i.e., `x`)
        gradient_func_y = grad(g_t_all, 1)      # compute the gradient wrt second argument (i.e., `y`)
        for t in range(self.T):
            z_texp = np.array([expert.get_pred_prep(t) for expert in self.experts])
            z_t = np.sum(np.array([self.weights[exp] * z_texp[exp] for exp in range(len(self.experts))]), axis=0) # output (i.e., predict) the weighted average of the experts
            x_t_hat, y_t_hat = discretize_preds(z_t, self.rng, method=self.discretize_method)
            gradients_x = gradient_func_x(anp.array(x_t_hat), anp.array(y_t_hat), t, self.I, self.J, self.alpha, self.w, self.sinr)
            gradients_y = gradient_func_y(anp.array(x_t_hat), anp.array(y_t_hat), t, self.I, self.J, self.alpha, self.w, self.sinr)
            
            # Update each expert's predictions in parallel
            results = Parallel(n_jobs=2*len(self.experts), backend='threading')(
                delayed(process_expert)(
                    expert, self.weights[expert_indx],
                    gradients_x, gradients_y,
                    x_t_hat, y_t_hat, t, self.eta
                )
                for expert_indx, expert in enumerate(self.experts)
            )
            exponential_weighting = np.array(results)
            self.weights = exponential_weighting / np.sum(exponential_weighting)
            self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)

            self.x_contra[t] = x_t_hat 
            self.y_contra[t] = y_t_hat
            self.g_contra[t] =  g_t_all(anp.array(x_t_hat), anp.array(y_t_hat), t, self.I, self.J, self.alpha, self.w, self.sinr) 

        return self.x_contra, self.y_contra, self.g_contra
    

