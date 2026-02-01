import numpy as np
import cvxpy as cp


def process_expert(expert, weight, grad_x, grad_y, x_t, y_t, t, eta):
    # For each expert, in CONTRA class:
    # (i) compute the expert's loss at time `t``,
    # (ii) update its exponential weight,
    # (iii) update its expert's internal decision variables via OGA + projection.
    loss = expert.calculate_loss(grad_x, grad_y, x_t, y_t, t)
    new_weight = weight * np.exp(eta * loss)
    expert.update(grad_x, grad_y, t)
    return new_weight


class OGAExpert:
    def __init__(
            self, step_size, T, I, J, 
            initial_x, initial_y,
            A, B, 
            solver
        ):
        # Expert hyperparameters and problem dimensions
        self.step_size = step_size  # Learning rate
        self.T = T                  # Number of slots
        self.I = I                  # Number of users
        self.J = J                  # Number of cells

        # Initial decisions for THO and CHO decisions 
        self.initial_x = initial_x
        self.initial_y = initial_y

        # Switching costs
        self.A = A                  # THO delay
        self.B = B                  # CHO signaling

        # THO and CHO decisions of each expert, initialized
        self.x_exp = np.zeros((self.T+1, self.I, self.J), dtype=np.float64, order="C")
        self.x_exp[0] = self.initial_x
        self.y_exp = np.zeros((self.T+1, self.I, self.J), dtype=np.float64, order="C")
        self.y_exp[0] = self.initial_y

        # Used for projection to the feasible set according to eq. (16) of the paper (see README for the link)
        self.X_param = cp.Parameter((self.I, self.J))
        self.Y_param = cp.Parameter((self.I, self.J))
        X = cp.Variable((self.I, self.J))
        Y = cp.Variable((self.I, self.J))
        objective = cp.Minimize(cp.norm(X - self.X_param, 'fro') + cp.norm(Y - self.Y_param, 'fro'))
        constraints = [
            cp.sum(X, axis=1) <= 1,                         # User is either THO- or CHO-enabled
            Y <= 1 - cp.sum(X, axis=1, keepdims=True),      # CHO-enabled allowed only when the user is not THO-enabled
            cp.sum(X, axis=1) + cp.sum(Y, axis=1) >= 1.0,   # Prevents blocking some users completely
            X >= 0, X <= 1, Y >= 0, Y <= 1]                 # Box constraints            
        self.problem = cp.Problem(objective, constraints)
        self.solver = solver
        self.X_var, self.Y_var = X, Y


    def get_pred_prep(self, t):
        # Create variable `z` by stacking in the first I rows `x` and the next I rows `y`
        return np.concatenate((self.x_exp[t], self.y_exp[t]), axis=0)
    

    def project_to_set(self, X_0, Y_0):
        # Solve the projection to the feasible set
        X_0 = np.ascontiguousarray(np.copy(X_0), dtype=np.float64)
        Y_0 = np.ascontiguousarray(np.copy(Y_0), dtype=np.float64)
        self.X_param.value = X_0
        self.Y_param.value = Y_0
        self.problem.solve(solver=self.solver)
        return self.X_var.value, self.Y_var.value


    def update(self, gradient_x, gradient_y, t):
        # Online Gradient Ascent (OGA) followed by projection
        pred = self.x_exp[t] + self.step_size * gradient_x
        prep = self.y_exp[t] + self.step_size * gradient_y
        self.x_exp[t+1], self.y_exp[t+1] = self.project_to_set(pred, prep)
        return


    def calculate_loss(self, gradients_x, gradients_y, x_t, y_t, t):
        # Surrogate loss used for exponential weighting
        if t == 0:
            return (
                np.sum(gradients_x * (self.x_exp[t] - x_t))
               + np.sum(gradients_y * (self.y_exp[t] - y_t))
               - np.sqrt(np.sum(self.A[t] * self.x_exp[t]**2))
               - np.sqrt(np.sum(self.B[t] * self.y_exp[t]**2))
            )
        else:
            return (
                np.sum(gradients_x * (self.x_exp[t] - x_t))
                + np.sum(gradients_y * (self.y_exp[t] - y_t))
                - np.sqrt(np.sum(self.A[t] * (self.x_exp[t] - self.x_exp[t-1])**2))
                - np.sqrt(np.sum(self.B[t] * (self.y_exp[t] - self.y_exp[t-1])**2))
            )

