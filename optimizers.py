import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, mean_squared_log_error, median_absolute_error, explained_variance_score
import scipy.stats as stats

from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np

class RRAMOptimizer(Optimizer):
    def __init__(self, params, potentiation, depression, tau):
        if not isinstance(potentiation, list):
            raise ValueError("potentiation should be a list")
        if not isinstance(depression, list):
            raise ValueError("depression should be a list")
        
        def map_to_nearest_indices(given_list, reference_list):
            def safe_tensor_conversion(data, dtype=torch.float32):
                if isinstance(data, torch.Tensor):
                    return data.clone().detach().to(dtype)  # Clone if it's already a tensor
                return torch.tensor(data, dtype=dtype)  # Convert if it's a list or other type

            given_tensor = safe_tensor_conversion(given_list)
            reference_tensor = safe_tensor_conversion(reference_list)

            # given_tensor = torch.tensor(given_list, dtype=torch.float32)
            # reference_tensor = torch.tensor(reference_list, dtype=torch.float32)
            
            # Compute absolute differences and find nearest indices
            indices = torch.abs(reference_tensor.unsqueeze(0) - given_tensor.unsqueeze(-1)).argmin(dim=-1)
            
            return torch.tensor(indices.tolist())
        
        def get_weights_from_conductances(finite_set):
            min_conductance, max_conductance = torch.min(torch.tensor([potentiation, depression])), torch.max(torch.tensor([potentiation, depression]))
            # min_conductance, max_conductance = finite_set[0], finite_set[-1]
            mid_conductance = (min_conductance + max_conductance) / 2
            min_weight, max_weight = -1.0, 1.0
            mid_weight = (max_weight + min_weight) / 2

            normalized_conductance = (finite_set - mid_conductance)/(max_conductance - mid_conductance)

            normalized_weight = normalized_conductance

            weights = normalized_weight * (max_weight - mid_weight) + mid_weight

            return weights
        
        potentiation_weights = get_weights_from_conductances(torch.tensor(potentiation))
        depression_weights = get_weights_from_conductances(torch.tensor(depression))

        self.potentiation_to_depression_mapping = map_to_nearest_indices(potentiation_weights, depression_weights)
        self.depression_to_potentiation_mapping = map_to_nearest_indices(depression_weights, potentiation_weights)

        # print(self.potentiation_to_depression_mapping)
        # print(self.depression_to_potentiation_mapping)

        defaults = {"potentiation":potentiation_weights, "depression":depression_weights, "tau":tau, }
        super().__init__(params, defaults)

        if isinstance(potentiation_weights, torch.Tensor):
            self.potentiation = potentiation_weights.clone().detach().to(torch.float32)
        else:
            self.potentiation = torch.tensor(potentiation_weights, dtype=torch.float32)
        
        if isinstance(depression_weights, torch.Tensor):
            self.depression = depression_weights.clone().detach().to(torch.float32)
        else:
            self.depression = torch.tensor(depression_weights, dtype=torch.float32)

        for group in self.param_groups:
            for p in group["params"]:
                self._initialize_param(p)


    def _initialize_param(self, p):
        """snaps param to nearest valid discrete state"""
        self.potentiation = self.potentiation.to(device=p.device)
        self.depression = self.depression.to(device=p.device)
        self.depression_to_potentiation_mapping = self.depression_to_potentiation_mapping.to(p.device)
        self.potentiation_to_depression_mapping = self.potentiation_to_depression_mapping.to(p.device)
        

        distances = torch.cdist(p.data.view(-1,1), self.potentiation.view(-1, 1))

        
        # Get the index of the minimum distance for each element
        min_indices = torch.argmin(distances, dim=1)  # Shape [3*4*5]

        # Reshape back to the original dimensions (without last dim)
        min_indices = min_indices.view(*p.data.shape)

        p.data.copy_(self.potentiation[min_indices])
        self.state[p] = {"state_idx": min_indices, "flags":torch.ones_like(min_indices)}

    def step(self):

        for group in self.param_groups:
            tau = group['tau']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad

                state_idx = self.state[p]['state_idx']
                flags = self.state[p]['flags']

                potentiation_to_decrease_mask = ((grad > tau) & (state_idx < len(self.potentiation) - 1) & (flags)).bool()
                potentiation_to_increase_mask = ((grad < -tau) & (state_idx < len(self.potentiation) - 1) & (flags.bool())).bool()

                depression_to_decrease_mask = ((grad > tau) & (state_idx < len(self.potentiation) - 1) & (1 - flags)).bool()
                depression_to_increase_mask = ((grad < -tau) & (state_idx < len(self.potentiation) - 1) & (~flags.bool())).bool()

                # print(potentiation_to_increase_mask, potentiation_to_decrease_mask)
                # print(depression_to_increase_mask, depression_to_decrease_mask)
                # print(flags, state_idx)
                # print("\n")


                state_idx[potentiation_to_increase_mask] += 1
                state_idx[depression_to_decrease_mask] += 1

                if potentiation_to_decrease_mask.any():
                    state_idx[potentiation_to_decrease_mask] = self.potentiation_to_depression_mapping[   state_idx[potentiation_to_decrease_mask]    ] + 1
                    flags[potentiation_to_decrease_mask] = False

                # Map potentiation to depression correctly
                # if potentiation_to_decrease_mask.any():
                #     mapped_indices = self.potentiation_to_depression_mapping[state_idx[potentiation_to_decrease_mask]]
                #     state_idx[potentiation_to_decrease_mask] = torch.clamp(mapped_indices + 1, max=len(self.depression) - 1)
                #     flags[potentiation_to_decrease_mask] = 0 

                

                if depression_to_increase_mask.any():
                    state_idx[depression_to_increase_mask] = self.depression_to_potentiation_mapping[   state_idx[depression_to_increase_mask]   ] + 1
                    flags[depression_to_increase_mask] = True

                # Map depression to potentiation correctly
                # if depression_to_increase_mask.any():
                #     mapped_indices = self.depression_to_potentiation_mapping[state_idx[depression_to_increase_mask]]
                #     state_idx[depression_to_increase_mask] = torch.clamp(mapped_indices + 1, max=len(self.potentiation) - 1)
                #     flags[depression_to_increase_mask] = 1 

                # Apply updates to parameter data
                updated_values = torch.where(
                    flags.bool(),
                    self.potentiation[state_idx],
                    self.depression[state_idx]
                )
                p.data.copy_(updated_values)



                

                







    


class DiscreteStateOptimizer(Optimizer):
    def __init__(self, params, conductance_list, tau, tie_break="floor"):
        """
        Args:
            params: Iterable of trainable parameters.
            state_list: A single list of allowed discrete states shared by all parameters.
            tau: Threshold for updating states based on gradient magnitude.
            tie_break: Strategy for handling ties ("floor", "ceil", "random").
        """
        if not isinstance(conductance_list, list):
            raise ValueError("state_list should be a list of discrete states.")
        if tie_break not in ["floor", "ceil", "random"]:
            raise ValueError("tie_break must be 'floor', 'ceil', or 'random'.")
        
        def get_weights_from_conductances(finite_set):
            # min_conductance, max_conductance = torch.min(finite_set), torch.max(finite_set)
            min_conductance, max_conductance = finite_set[0], finite_set[-1]
            mid_conductance = (min_conductance + max_conductance) / 2
            min_weight, max_weight = -1.0, 1.0
            mid_weight = (max_weight + min_weight) / 2

            normalized_conductance = (finite_set - mid_conductance)/(max_conductance - mid_conductance)

            normalized_weight = normalized_conductance

            weights = normalized_weight * (max_weight - mid_weight) + mid_weight

            return weights
        
        state_list = get_weights_from_conductances(torch.tensor(conductance_list))
        defaults = {"state_list": state_list, "tau": tau, "tie_break": tie_break}
        super().__init__(params, defaults)

        # Convert state list to a tensor (shared by all params)
        if isinstance(state_list, torch.Tensor):
            self.state_tensor = state_list.clone().detach().to(torch.float32)
        else:
            self.state_tensor = torch.tensor(state_list, dtype=torch.float32)


        # Initialize parameters to the closest allowed state
        for group in self.param_groups:
            for p in group["params"]:
                self._initialize_param(p)

    def _initialize_param(self, p):
        """snaps param to nearest valid discrete state"""
        self.state_tensor = self.state_tensor.to(device=p.device)


        distances = torch.cdist(p.data.view(-1,1), self.state_tensor.view(-1, 1))

        
        # Get the index of the minimum distance for each element
        min_indices = torch.argmin(distances, dim=1)  # Shape [3*4*5]

        # Reshape back to the original dimensions (without last dim)
        min_indices = min_indices.view(*p.data.shape)

        p.data.copy_(self.state_tensor[min_indices])
        self.state[p] = {"state_idx": min_indices}


    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            tau = group["tau"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state_idx = self.state[p]["state_idx"]

                # Create masks for different updates
                increase_mask = (grad > tau) & (state_idx > 0)
                decrease_mask = (grad < -tau) & (state_idx < len(self.state_tensor) - 1)

                # Apply masks to update indices
                state_idx[increase_mask] -= 1
                state_idx[decrease_mask] += 1

                # Update parameter values and store new indices
                p.data.copy_(self.state_tensor[state_idx])
                self.state[p]["state_idx"] = state_idx




# class PolynomialRegressor:
#     def __init__(self, degree=3, device=None):
#         self.degree = degree
#         self.coeffs = None
#         self.x_mean = None
#         self.x_std = None
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

#     def fit(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         # Normalize x
#         self.x_mean = x.mean()
#         self.x_std = x.std()
#         x_norm = (x - self.x_mean) / self.x_std

#         # Create polynomial feature matrix
#         X_poly = torch.vander(x_norm, N=self.degree + 1)
        
#         # Solve least squares on GPU
#         self.coeffs = torch.linalg.lstsq(X_poly, y).solution

#         # Evaluate fit quality
#         y_pred = self.predict(x)
#         mse = torch.mean((y - y_pred) ** 2).item()
#         ss_total = torch.sum((y - y.mean()) ** 2)
#         ss_residual = torch.sum((y - y_pred) ** 2)
#         r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

#         print(f"Fit Quality Metrics:\n - Mean Squared Error (MSE): {mse:.8f}\n - R² Score: {r2_score:.6f}")

#     def predict(self, x):
#         if self.coeffs is None:
#             raise ValueError("Model not fitted yet. Call fit() first.")

#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         x_norm = (x - self.x_mean) / self.x_std
#         X_poly = torch.vander(x_norm, N=self.degree + 1)

#         return X_poly @ self.coeffs

#     def plot_residuals(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         y_pred = self.predict(x)
#         residuals = (y - y_pred).cpu().numpy()  # Move to CPU for plotting

#         plt.figure(figsize=(6, 4))
#         plt.scatter(x.cpu().numpy(), residuals, color='blue', alpha=0.6, label='Residuals')
#         plt.axhline(0, color='red', linestyle='--')
#         plt.xlabel('X values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.legend()
#         plt.show()

#         # Check residual distribution
#         if np.abs(residuals).mean() > 0.01:
#             print("Warning: Residuals are not close to zero. Consider using a different model or transformation.")

#         return residuals

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# class PolynomialRegressor:
#     def __init__(self, degree=3, device=None):
#         self.degree = degree
#         self.coeffs = None
#         self.x_mean = None
#         self.x_std = None
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

#     def fit(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         # Normalize x (feature-wise)
#         self.x_mean = x.mean(dim=0, keepdim=True)
#         self.x_std = x.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
#         x_norm = (x - self.x_mean) / self.x_std

#         # Generate polynomial features for multi-dimensional x
#         X_poly = self._generate_polynomial_features(x_norm)

#         # Solve least squares
#         self.coeffs = torch.linalg.lstsq(X_poly, y).solution

#         # Evaluate fit quality
#         y_pred = self.predict(x)
#         mse = torch.mean((y - y_pred) ** 2).item()
#         ss_total = torch.sum((y - y.mean()) ** 2)
#         ss_residual = torch.sum((y - y_pred) ** 2)
#         r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

#         print(f"Fit Quality Metrics:\n - Mean Squared Error (MSE): {mse:.8f}\n - R² Score: {r2_score:.6f}")

#     def predict(self, x):
#         if self.coeffs is None:
#             raise ValueError("Model not fitted yet. Call fit() first.")

#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         x_norm = (x - self.x_mean) / self.x_std
#         X_poly = self._generate_polynomial_features(x_norm)

#         return X_poly @ self.coeffs

#     def _generate_polynomial_features(self, x):
#         """Generates polynomial features up to the specified degree for multi-dimensional x."""
#         n_samples, n_features = x.shape
#         powers = torch.arange(1, self.degree + 1, device=self.device).repeat(n_features, 1).T
#         X_poly = torch.cat([x ** p for p in powers], dim=1)
#         X_poly = torch.cat([torch.ones(n_samples, 1, device=self.device), X_poly], dim=1)  # Add bias term
#         return X_poly

#     def plot_residuals(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         y_pred = self.predict(x)
#         residuals = (y - y_pred).cpu().numpy()  # Move to CPU for plotting

#         plt.figure(figsize=(6, 4))
#         plt.scatter(x.cpu().numpy(), residuals, color='blue', alpha=0.6, label='Residuals')
#         plt.axhline(0, color='red', linestyle='--')
#         plt.xlabel('X values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.legend()
#         plt.show()

#         # Check residual distribution
#         if np.abs(residuals).mean() > 0.01:
#             print("Warning: Residuals are not close to zero. Consider using a different model or transformation.")

#         return residuals


# class PolynomialRegressor:
#     def __init__(self, degree=3, device=None):
#         self.degree = degree
#         self.coeffs = None
#         self.x_mean = None
#         self.x_std = None
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

#     def fit(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         # Ensure x is at least 2D (n_samples, n_features)
#         if x.dim() == 1:
#             x = x.unsqueeze(1)  # Convert (N,) to (N,1)

#         # Normalize x
#         self.x_mean = x.mean(dim=0, keepdim=True)
#         self.x_std = x.std(dim=0, keepdim=True)
#         x_norm = (x - self.x_mean) / self.x_std

#         # Generate polynomial features
#         X_poly = self._generate_polynomial_features(x_norm)

#         # Solve least squares
#         self.coeffs = torch.linalg.lstsq(X_poly, y).solution

#         # Evaluate fit quality
#         y_pred = self.predict(x)
#         mse = torch.mean((y - y_pred) ** 2).item()
#         ss_total = torch.sum((y - y.mean()) ** 2)
#         ss_residual = torch.sum((y - y_pred) ** 2)
#         r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

#         print(f"Fit Quality Metrics:\n - Mean Squared Error (MSE): {mse:.8f}\n - R² Score: {r2_score:.6f}")

#     def predict(self, x):
#         if self.coeffs is None:
#             raise ValueError("Model not fitted yet. Call fit() first.")

#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         if x.dim() == 1:
#             x = x.unsqueeze(1)

#         x_norm = (x - self.x_mean) / self.x_std
#         X_poly = self._generate_polynomial_features(x_norm)
#         print(f"X shape = {x.shape}, x_norm shape = {x_norm.shape} X_poly shape = {X_poly.shape}, coeff shape = {self.coeffs.shape}")
#         if X_poly.shape[1] != self.coeffs.shape[0]:
#             print("Mismatch detected!")


#         return X_poly @ self.coeffs

#     def _generate_polynomial_features(self, x):
#         """Generates polynomial features up to the specified degree for multi-dimensional x."""
#         n_samples, n_features = x.shape
#         X_poly = [torch.ones(n_samples, 1, device=self.device)]  # Bias term

#         for d in range(1, self.degree + 1):
#             X_poly.append(x ** d)

#         return torch.cat(X_poly, dim=1)

#     def plot_residuals(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)

#         y_pred = self.predict(x)
#         residuals = (y - y_pred).cpu().numpy()  # Move to CPU for plotting

#         plt.figure(figsize=(6, 4))
#         plt.scatter(x.cpu().numpy(), residuals, color='blue', alpha=0.6, label='Residuals')
#         plt.axhline(0, color='red', linestyle='--')
#         plt.xlabel('X values')
#         plt.ylabel('Residuals')
#         plt.title('Residual Plot')
#         plt.legend()
#         plt.show()

#         if np.abs(residuals).mean() > 0.01:
#             print("Warning: Residuals are not close to zero. Consider using a different model or transformation.")

#         return residuals

import torch

class PolynomialRegressor:
    def __init__(self, degree):
        self.degree = degree
        self.coeffs = None
        self.device = torch.device("cpu")  # Default device

    def _generate_polynomial_features(self, x):
        """Generates polynomial features element-wise for scalar input x."""
        self.device = x.device  # Ensure we're using the correct device
        original_shape = x.shape  # Save original shape (a, b, c, ...)
        x_flat = x.view(-1, 1)  # Flatten to (N, 1)

        # Generate polynomial terms for each scalar independently
        powers = torch.arange(1, self.degree + 1, device=self.device)
        X_poly = torch.cat([x_flat ** p for p in powers], dim=1)  # (N, degree)

        return X_poly.view(*original_shape, self.degree)  # Restore shape

    def fit(self, x, y):
        """Fits polynomial regression using least squares."""
        self.device = x.device  # Track the device of input tensors
        X_poly = self._generate_polynomial_features(x).to(self.device)
        X_poly_flat = X_poly.view(-1, self.degree)  # Flatten for regression
        y_flat = y.view(-1, 1).to(self.device)  # Flatten and move to same device

        # Solve least squares problem
        self.coeffs = torch.linalg.lstsq(X_poly_flat, y_flat).solution  # (degree, 1)
        self.coeffs = self.coeffs.to(self.device)  # Ensure coefficients are on the correct device

    def predict(self, x):
        """Predicts output given new input x."""
        X_poly = self._generate_polynomial_features(x).to(self.device)
        return (X_poly @ self.coeffs.to(x.device)).view(x.shape).to(self.device)  # Preserve shape and device
        


class ContinuousOptimizer(Optimizer):
    def __init__(self, params, potentiation:torch.tensor, depression:torch.tensor, potentiation_degree:int, depression_degree:int, tau=0.05):
        # if not isinstance(potentiation, list):
        #     raise ValueError("potentiation should be a list")
        # if not isinstance(depression, list):
        #     raise ValueError("depression should be a list")
        
        potentiation_g_by_g0, depression_g_by_g0 = [], []

        for i in range(len(potentiation)):
            if i == len(potentiation)-1:
                x = potentiation_g_by_g0[-1]
                
            else:
                x = potentiation[i+1]/potentiation[i]    
            potentiation_g_by_g0.append(x)


        for i in range(len(depression)):
            if i == len(depression)-1:
                x = depression_g_by_g0[-1]
            else:
                x = depression[i+1]/depression[i]

            
            depression_g_by_g0.append(x)

        potentiation_g_by_g0 = torch.tensor(potentiation_g_by_g0).to(potentiation.device)
        depression_g_by_g0 = torch.tensor(depression_g_by_g0).to(depression.device)

        self.potentiation_g_by_g0 = potentiation_g_by_g0
        self.depression_g_by_g0 = depression_g_by_g0

        self.potentiation_model = PolynomialRegressor(degree=potentiation_degree)
        self.potentiation_model.fit(potentiation, potentiation_g_by_g0)

        self.depression_model = PolynomialRegressor(degree=depression_degree)
        self.depression_model.fit(depression, depression_g_by_g0)

        # self.min_conductance, self.max_conductance = 3.9e-5, 4.6e-5
        # self.min_conductance, self.max_conductance = torch.max(torch.array(torch.min(potentiation), torch.min(depression))), torch.min(torch.array(torch.max(potentiation), torch.max(depression)))
        # self.min_conductance = min(potentiation.min().item(), depression.min().item())
        # self.max_conductance = max(potentiation.max().item(), depression.max().item())

        self.min_conductance = max(potentiation.min().item(), depression.min().item())
        self.max_conductance = min(potentiation.max().item(), depression.max().item())

        self.min_weight, self.max_weight = -0.5, 0.5

        # Define default hyperparameters for the optimizer
        defaults = {"potentiation": potentiation, "depression": depression, "potentiation_degree":potentiation_degree, "depression_degree":depression_degree, "tau":tau}
        
        # Pass params and defaults to parent class
        super().__init__(params, defaults)

    
    def step(self):
        def map_weights_to_conductance(weight, 
                                       min_weight=self.min_weight, 
                                       max_weight=self.max_weight, 
                                       min_conductance=self.min_conductance, 
                                       max_conductance=self.max_conductance):
            mid_weight = (max_weight+min_weight)/2
            mid_conductance = (max_conductance + min_conductance)/2
            normalized_weight = (weight - mid_weight)/(max_weight-mid_weight)
            # normalized_conductance = 2/torch.pi * torch.arcsin(normalized_weight)
            # normalized_conductance = torch.sign(normalized_weight) * torch.abs(normalized_weight) ** (1/3)
            # normalized_conductance = normalized_weight ** 5
            normalized_conductance = normalized_weight
            conductance = normalized_conductance * (max_conductance - mid_conductance) + mid_conductance
            return conductance
        
        def map_conductance_to_weights(conductance, 
                                            min_weight=self.min_weight, 
                                            max_weight=self.max_weight, 
                                            min_conductance=self.min_conductance, 
                                            max_conductance=self.max_conductance):
            mid_weight = (max_weight+min_weight)/2
            mid_conductance = (max_conductance + min_conductance)/2
            normalized_conductance = (conductance - mid_conductance)/(max_conductance - mid_conductance)
            # normalized_weight = torch.sin(normalized_conductance * torch.pi/2)
            # normalized_weight = torch.sign(normalized_conductance) * torch.abs(normalized_conductance) ** (1/5)
            # normalized_weight = normalized_conductance ** 3
            normalized_weight = normalized_conductance
            weight = normalized_weight*(max_weight - mid_weight) + mid_weight
            return weight
        
        def new_conductance(G0, grad):
            
            # G_pos = G0*1.05
            # G_neg = G0*(0.95)
            # Apply the splines based on gradient
            # tau = 0.05
            # Masks for set, reset, and no change conditions
            set_mask = torch.ge(grad, tau)       # Apply set spline if grad >= tau
            reset_mask = torch.le(grad, -tau)    # Apply reset spline if grad <= -tau

            # Apply the set and reset splines
            # G_pos = G0 * self.reset_splines['1 us'].ev(G0, -1.85)    # Set spline adjustment
            # G_neg = G0 * self.set_splines['1 us'].ev(G0, 1.83) # Reset spline adjustment
            # G_pos = G0 * self.reset_splines(G0)
            # G_neg = G0 * self.set_splines(G0)

            # G_neg = G0 +1.0410955555555556e-09/5
            # G_pos = G0 -2.4080184210526313e-09/5
            G_neg = G0 * self.potentiation_model.predict(G0)
            G_pos = G0 * self.depression_model.predict(G0)
            
            # G_neg = G0 * self.potentiation_g_by_g0.mean()
            # G_pos = G0 * self.depression_g_by_g0.mean()

            # Use torch.where to select G_pos, G_neg, or G0 based on conditions
            G_new = torch.where(set_mask, G_pos, torch.where(reset_mask, G_neg, G0))

            return G_new
            # positive_mask = torch.le(grad, 0)  # Check for grad >= 0
            # G_pos = G0 * (self.set_splines['1 us'].ev(G0, 2.5))
            # G_neg = G0 * (self.reset_splines['1 us'].ev(G0, -2.5))
            
            # # # Combine both positive and negative cases
            # G_new = torch.where(positive_mask, G_pos, G_neg)
            # return G_new
        

        for group in self.param_groups:
            tau = group['tau']
            for p in group['params']:
                if p.grad is None:
                    continue

                weight, grad = p.data, p.grad

                G0 = map_weights_to_conductance(
                    weight=weight,
                )


                G0 = torch.clamp(G0, min=self.min_conductance, max=self.max_conductance)

                G_new = new_conductance(G0=G0, grad=grad)

                new_weight = map_conductance_to_weights(
                    conductance=G_new
                )
                p.data.copy_(new_weight)




        
        



class MLPFitter1:
    def __init__(self, x, y, hidden_layers=[32, 32], lr=0.01, epochs=5000, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(x, dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        
        self.x_mean, self.x_std = x.mean(), x.std()
        self.y_mean, self.y_std = y.mean(), y.std()
        
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        
        self.model = self._build_mlp(1, 1, hidden_layers).to(self.device)
        self.inverse_model = self._build_mlp(1, 1, hidden_layers).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=lr)
        self.epochs = epochs
        
        self.x_train, self.y_train = x, y
        self._train()
        self._train_inverse()

    def _build_mlp(self, input_dim, output_dim, hidden_layers):
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _train(self):
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(self.x_train)
            loss = self.criterion(y_pred, self.y_train)
            loss.backward()
            self.optimizer.step()

    def _train_inverse(self):
        for _ in range(self.epochs):
            self.inverse_optimizer.zero_grad()
            x_pred = self.inverse_model(self.y_train)
            loss = self.criterion(x_pred, self.x_train)
            loss.backward()
            self.inverse_optimizer.step()

    def predict(self, x):
        x_shape = x.shape
        x_device = x.device  # Preserve device
        x = ((x - self.x_mean.to(x_device)) / self.x_std.to(x_device)).view((-1, 1))
        return (self.model(x) * self.y_std.to(x_device) + self.y_mean.to(x_device)).view(x_shape)

    def inverse(self, y):
        y_shape = y.shape
        y_device = y.device  # Preserve device
        y = ((y - self.y_mean.to(y_device)) / self.y_std.to(y_device)).view((-1, 1))
        return (self.inverse_model(y) * self.x_std.to(y_device) + self.x_mean.to(y_device)).view(y_shape)

    def evaluate(self):
        y_pred = self.model(self.x_train) * self.y_std + self.y_mean
        x_pred = self.inverse_model(self.y_train) * self.x_std + self.x_mean
        
        forward_mse = mean_squared_error(self.y_train.cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        forward_r2 = r2_score(self.y_train.cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        
        inverse_mse = mean_squared_error(self.x_train.cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy(), x_pred.cpu().detach().numpy())
        inverse_r2 = r2_score(self.x_train.cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy(), x_pred.cpu().detach().numpy())
        
        print(f"Forward Mapping (x -> y):\n  MSE: {forward_mse:.6f}, R² Score: {forward_r2:.6f}")
        print(f"Inverse Mapping (y -> x):\n  MSE: {inverse_mse:.6f}, R² Score: {inverse_r2:.6f}")

    def plot(self):
        x_vals = self.x_train.cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()
        y_vals = self.y_train.cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        y_preds = self.model(self.x_train).cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        x_preds = self.inverse_model(self.y_train).cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(x_vals, y_vals, label="Data", color="blue")
        axes[0].plot(x_vals, y_preds, label="MLP Fit", color="red")
        axes[0].set_title("Forward Mapping (x -> y)")
        axes[0].legend()
        
        axes[1].scatter(y_vals, x_vals, label="Data", color="blue")
        axes[1].plot(y_vals, x_preds, label="MLP Inverse Fit", color="red")
        axes[1].set_title("Inverse Mapping (y -> x)")
        axes[1].legend()
        
        plt.show()


    
    def plot_residuals_probplot(self):
        # Compute residuals
        y_true = self.y_train.cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        y_pred = self.model(self.x_train).cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        forward_residuals = y_true - y_pred  # Residuals for forward mapping

        x_true = self.x_train.cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()
        x_pred = self.inverse_model(self.y_train).cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()
        inverse_residuals = x_true - x_pred  # Residuals for inverse mapping

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Q-Q plot for forward residuals
        stats.probplot(forward_residuals.flatten(), dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot: Forward Residuals (x -> y)")

        # Q-Q plot for inverse residuals
        stats.probplot(inverse_residuals.flatten(), dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot: Inverse Residuals (y -> x)")

        plt.tight_layout()
        plt.show()


        
    def plot_residuals(self):
        x_vals = self.x_train.cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()
        y_vals = self.y_train.cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        y_preds = self.model(self.x_train).cpu().detach().numpy() * self.y_std.cpu().detach().numpy() + self.y_mean.cpu().detach().numpy()
        x_preds = self.inverse_model(self.y_train).cpu().detach().numpy() * self.x_std.cpu().detach().numpy() + self.x_mean.cpu().detach().numpy()

        forward_residuals = y_vals - y_preds
        inverse_residuals = x_vals - x_preds

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Histogram of residuals
        axes[0, 0].hist(forward_residuals, bins=30, color='blue', alpha=0.7)
        axes[0, 0].set_title("Forward Residuals Histogram (x -> y)")
        axes[0, 0].set_xlabel("Residual Value")
        axes[0, 0].set_ylabel("Frequency")

        axes[0, 1].hist(inverse_residuals, bins=30, color='red', alpha=0.7)
        axes[0, 1].set_title("Inverse Residuals Histogram (y -> x)")
        axes[0, 1].set_xlabel("Residual Value")
        axes[0, 1].set_ylabel("Frequency")

        # Scatter plot of residuals
        axes[1, 0].scatter(x_vals, forward_residuals, color='blue', alpha=0.5)
        axes[1, 0].axhline(y=0, color='black', linestyle='dashed')
        axes[1, 0].set_title("Forward Residuals (x -> y)")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("Residual")

        axes[1, 1].scatter(y_vals, inverse_residuals, color='red', alpha=0.5)
        axes[1, 1].axhline(y=0, color='black', linestyle='dashed')
        axes[1, 1].set_title("Inverse Residuals (y -> x)")
        axes[1, 1].set_xlabel("y")
        axes[1, 1].set_ylabel("Residual")

        plt.tight_layout()
        plt.show()


class NNApproximationBasedOptimizer(Optimizer):
    def __init__(self, params, potentiation:list, depression:list, lr=0.01, tau=0.0):

        defaults = {"potentiation":potentiation, "depression":depression, "lr": lr, "tau":tau}
        super().__init__(params, defaults)

        self.device = self.param_groups[0]['params'][0].device

        self.potentiation_approximator = MLPFitter1(list(range(len(potentiation))), potentiation, device=self.device)
        self.potentiation_approximator.evaluate()
        self.depression_approximator = MLPFitter1(list(range(len(depression))), depression, device=self.device)
        self.depression_approximator.evaluate()

        self.min_conductance = torch.tensor(max(min(potentiation), min(depression)), device=self.device)
        self.max_conductance = torch.tensor(min(max(potentiation), max(depression)), device=self.device)

        # self.min_conductance = torch.tensor(min(min(potentiation), min(depression)), device=self.device)
        # self.max_conductance = torch.tensor(max(max(potentiation), max(depression)), device=self.device)

        self.potentiation = {
            "max" : self.get_state_from_conductance(self.max_conductance),
            "min" : self.get_state_from_conductance(self.min_conductance)
        }
        self.depression = {
            "max": self.get_state_from_conductance(self.max_conductance, False),
            "min":self.get_state_from_conductance(self.min_conductance, False)
        }

        self.min_weight, self.max_weight = -0.5, 0.5


        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:

                    self.state[p] = {
                        "state": self.get_state_from_conductance(self.map_weights_to_conductances(p.data), False)\
                        .clamp(self.depression['max'], self.depression['min']), 
                        "flag": torch.zeros_like(p.data, dtype=torch.bool) 
                    }
                    p.data.copy_(self.map_conductances_to_weights(self.get_conductance_from_state(self.state[p]['state'])))

    def step(self):
        
        for group in self.param_groups:
            lr = group['lr']
            tau = group['tau']
            for p in group['params']:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]['state']
                flags = self.state[p]['flag']

                # print(f"(grad < -tau) : {(grad < -tau)}, flags : {flags}")

                potentiation_to_decrease_mask = ((grad > tau) & flags)
                potentiation_to_increase_mask = ((grad < -tau) & flags)

                depression_to_decrease_mask = ((grad > tau).bool() & (~flags))
                depression_to_increase_mask = ((grad < -tau) & (~flags))

                # print(f"Potentiation to increase : {potentiation_to_increase_mask}")
                # print(f"potentiation to decrease : {potentiation_to_decrease_mask}")
                # print(f"depression to increase : {depression_to_increase_mask}")
                # print(f"depression to decrease : {depression_to_decrease_mask}")
                # print(f"state [potentiation to increase mask] = {state[potentiation_to_increase_mask]}")
                # print(f"state : {state}")
                # print(f"gradient : {grad}")

                state[potentiation_to_increase_mask] -= lr * grad[potentiation_to_increase_mask]
                state[depression_to_decrease_mask] += lr * grad[depression_to_decrease_mask]

                if potentiation_to_decrease_mask.any():
                    conductance = self.map_weights_to_conductances(p.data[potentiation_to_decrease_mask])
                    state[potentiation_to_decrease_mask] = self.get_state_from_conductance(conductance, flag=False) + lr * grad[potentiation_to_decrease_mask]
                    flags[potentiation_to_decrease_mask] = False

                if depression_to_increase_mask.any():
                    conductance = self.map_weights_to_conductances(p.data[depression_to_increase_mask])
                    state[depression_to_increase_mask] = self.get_state_from_conductance(conductance) - lr * grad[depression_to_increase_mask]
                    flags[depression_to_increase_mask] = True

                # print(f"state after update before clamp : {state}")
                if flags.any():
                    state[flags] = state[flags].clamp(self.potentiation['min'], self.potentiation['max'])
                if (~flags).any():
                    state[~flags] = state[~flags].clamp(self.depression['max'], self.depression['min'])

                # print(f"state after clamping : {state} ")
                # print("-------------------------------------------------------------")



                conductance = torch.where(
                    flags.bool(),
                    self.get_conductance_from_state(state),
                    self.get_conductance_from_state(state, False)
                )


                p.data.copy_(self.map_conductances_to_weights(conductance))






    
    def get_conductance_from_state(self, state, flag=True):
        if flag:
            return self.potentiation_approximator.predict(state)
        return self.depression_approximator.predict(state)

    def get_state_from_conductance(self, conductance, flag=True):
        if flag:
            return self.potentiation_approximator.inverse(conductance)
        return self.depression_approximator.inverse(conductance)


    def map_weights_to_conductances(self, weight):

        normalized_weight = (weight - self.min_weight) / (self.max_weight - self.min_weight)

        normalized_conductance = normalized_weight

        return normalized_conductance * (self.max_conductance - self.min_conductance) + self.min_conductance
    
    def map_conductances_to_weights(self, conductance):
        normalized_conductance = (conductance - self.min_conductance) / (self.max_conductance - self.min_conductance)

        normalized_weight = normalized_conductance

        return normalized_weight * (self.max_weight - self.min_weight) + self.min_weight
    

        

        


