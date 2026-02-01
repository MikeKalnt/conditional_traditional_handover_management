import numpy as np
from copy import deepcopy
from contra_algorithm.algorithms.meta_learner import compute_hyperparameters, CONTRA
import argparse
import sys
np.set_printoptions(suppress=True, precision=4, threshold=sys.maxsize)


def run_contra(
        T, I, J, 
        initial_x, initial_y,
        A, B,
        discretize_method,
        alpha, w, sinr,
        rng, 
        solver,
        SINRdb_max, SINR_scaler
    ):

    eta, step_sizes = compute_hyperparameters(T, I, J, w, A, B, SINRdb_max, SINR_scaler)

    contra = CONTRA(
        eta, step_sizes, T, I, J,
        deepcopy(initial_x), deepcopy(initial_y),
        A, B,
        discretize_method,
        alpha, w, sinr, 
        rng,
        solver
    )

    x_contra, y_contra, g_contra = contra.run()

    return x_contra, y_contra, g_contra


def parse_args():

    p = argparse.ArgumentParser(description='Evaluating CONTRA algorithm')
    p.add_argument("--sinr_path", type=str, required=True, help="Path to SINR (in dB) .npy file. It should be of shape (T, I, J), where T, I, J are the number of slots, users, and cells, respectively")
    p.add_argument("--THOdelay_path", type=str, required=True, help="Path to THO delay .npy file. It should be of shape (T, I, J), where T, I, J are the number of slots, users, and cells, respectively")
    p.add_argument("--CHOsignaling_path", type=str, required=True, help="Path to CHO signaling .npy file. It should be of shape (T, I, J), where T, I, J are the number of slots, users, and cells, respectively")
    p.add_argument("--w_path", type=str, required=True, help="Path bandwidth of each cell .npy file. It should be of length J, where J is the number of cells.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.24)
    p.add_argument("--discretize", type=str, default="none", choices=["none", "contra_discr"])
    p.add_argument("--solver", type=str, default="MOSEK", help="CVXPY solver name")
    p.add_argument("--SINRdb_max", type=float, default=25.0, help="Maximum value of SINR in dB")
    p.add_argument("--SINR_scaler", type=float, default=10.0, help="Used to scale down SINR values as solvers need smaller numbers")
    return p.parse_args()


def main():
    args = parse_args()

    sinr = np.load(args.sinr_path)
    A    = np.load(args.THOdelay_path)
    B    = np.load(args.CHOsignaling_path)
    w    = np.load(args.w_path) 
    T    = sinr.shape[0]
    I    = sinr.shape[1]
    J    = sinr.shape[2]
    rng  = np.random.default_rng(args.seed)

    # Check dimensions
    if A.shape != (T, I, J):
        raise ValueError(f"THO delay shape mismatch: expected {(T, I, J)}, got {A.shape}")
    if B.shape != (T, I, J):
        raise ValueError(f"CHO signaling shape mismatch: expected {(T, I, J)}, got {B.shape}")
    if w.shape[0] != J:
        raise ValueError(f"Bandwidth vector length mismatch: expected J={J}, got {w.shape[0]}")

    # Example: initialize users in CHO mode randomly
    initial_x = np.zeros((I, J))
    initial_y = rng.integers(0, 2, size=(I, J))
    # Check if whole rows are 0 and fix
    for i in range(I):
        if not initial_y[i].any():
            j = rng.integers(0, J)
            initial_y[i, j] = 1

    x_contra, y_contra, g_contra  = run_contra(
        T, I, J, 
        initial_x, initial_y,
        A, B,
        args.discretize,
        args.alpha, w, sinr,
        rng, 
        args.solver,
        args.SINRdb_max, args.SINR_scaler
    )

    print("CONTRA total score:", np.sum(g_contra))


if __name__ == '__main__':
    main()

